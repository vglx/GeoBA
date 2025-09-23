#include "Optimizer.h"
#include "CostFunctions.h"
#include "Projection.h"
#include "BVH.h"

#include <iostream>
#include <limits>
#include <cmath>
#include <numeric>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <omp.h>

namespace {

inline float bilinearSample(const cv::Mat& img, float u, float v){
    int x = (int)std::floor(u), y = (int)std::floor(v);
    int x1 = x + 1, y1 = y + 1;
    if (x < 0 || y < 0 || x1 >= img.cols || y1 >= img.rows) return 0.f;
    float a = u - x, b = v - y;
    float I00 = img.at<float>(y, x);
    float I10 = img.at<float>(y, x1);
    float I01 = img.at<float>(y1, x);
    float I11 = img.at<float>(y1, x1);
    return (1-a)*(1-b)*I00 + a*(1-b)*I10 + (1-a)*b*I01 + a*b*I11;
}

// Area-weighted vertex normals from a *deformed* vertex set and face list.
inline void computeVertexNormals(
    const std::vector<Eigen::Vector3d>& Vdef,
    const std::vector<MeshModel::Triangle>& F,
    std::vector<Eigen::Vector3d>& Nout)
{
    const int N = (int)Vdef.size();
    Nout.assign(N, Eigen::Vector3d::Zero());

    for (const auto& tri : F) {
        const int i0 = tri.v0, i1 = tri.v1, i2 = tri.v2;
        const Eigen::Vector3d& p0 = Vdef[i0];
        const Eigen::Vector3d& p1 = Vdef[i1];
        const Eigen::Vector3d& p2 = Vdef[i2];
        Eigen::Vector3d n = (p1 - p0).cross(p2 - p0); // 2*area * face normal
        if (n.squaredNorm() < 1e-20) continue;
        Nout[i0] += n; Nout[i1] += n; Nout[i2] += n;
    }

    for (int i = 0; i < N; ++i) {
        double nrm = Nout[i].norm();
        if (nrm > 1e-20) Nout[i] /= nrm; else Nout[i].setZero();
    }
}

} // namespace

Optimizer::Optimizer(double w_photo,
                     double w_icp,
                     int maxStages,
                     int maxIterations,
                     double lambda_smooth,
                     double lambda_rot,
                     double lambda_temporal)
    : w_photo_(w_photo), w_icp_(w_icp),
      lambda_smooth_(lambda_smooth), lambda_rot_(lambda_rot), lambda_temporal_(lambda_temporal),
      maxStages_(maxStages), maxIterations_(maxIterations) {}

void Optimizer::optimize(
    const std::vector<MeshModel::Vertex>& mesh_vertices,
    const std::vector<MeshModel::Triangle>& mesh_triangles,
    const Eigen::Matrix3d& K,
    const std::vector<cv::Mat>& observed_rgb,
    const std::vector<cv::Mat>& observed_depth,
    const std::vector<Eigen::Matrix4d>& camera_poses_gt,
    EDGraph& edGraph,
    SaveCallback on_save) {

    const int F  = (int)observed_rgb.size();
    const int Fd = (int)observed_depth.size();
    const int N  = (int)mesh_vertices.size();
    const int G  = edGraph.numNodes();

    if (F <= 1 || N == 0 || G == 0 || F != Fd || (int)camera_poses_gt.size() != F) {
        std::cout << "[Optimizer] Invalid inputs: need RGB+Depth for >=2 frames, mesh & graph non-empty, matching sizes." << std::endl;
        return;
    }

    // ---- grayscale images [0,1]
    std::vector<cv::Mat> imgs_gray; imgs_gray.reserve(F);
    for (const auto& img : observed_rgb) {
        cv::Mat gray;
        if (img.type()==CV_8UC3) {
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            gray.convertTo(gray, CV_32F, 1.0/255.0);
        } else if (img.channels()==3) {
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            gray.convertTo(gray, CV_32F);
        } else if (img.channels()==1) {
            img.convertTo(gray, CV_32F, (img.depth()==CV_8U)? 1.0/255.0 : 1.0);
        } else {
            CV_Assert(false && "Unsupported RGB image type");
        }
        imgs_gray.push_back(gray);
    }

    // =========================
    // Per-frame full ED blocks (12*G). Frame 0 is fixed (no columns for it).
    // =========================
    const int edDimPerFrameFull = 12 * G;
    std::vector<Eigen::VectorXd> Xfull(F, Eigen::VectorXd::Zero(edDimPerFrameFull));
    for (int f=0; f<F; ++f) edGraph.writeToStateVector(Xfull[f], /*offset=*/0);

    const auto& bindings = edGraph.getBindings();
    const auto& edges    = edGraph.getEdges();

    // =========================
    // (1) Build initial BVHs (Evaluate currently ignores them; kept for API)
    // =========================
    std::vector<std::vector<MeshModel::Vertex>> Vdef_init(F, std::vector<MeshModel::Vertex>(N));
    for (int f=0; f<F; ++f) {
        edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);
        auto& Vd = Vdef_init[f];
        #pragma omp parallel for
        for (int i=0; i<N; ++i) {
            Eigen::Vector3d p = edGraph.deformVertex(mesh_vertices[i], i);
            Vd[i].x = (float)p.x(); Vd[i].y = (float)p.y(); Vd[i].z = (float)p.z();
        }
    }
    std::vector<BVH> bvhs; bvhs.reserve(F);
    for (int f=0; f<F; ++f) { bvhs.emplace_back(mesh_triangles, Vdef_init[f]); }

    // =========================
    // (2) Intensity init from ALL VISIBLE FRAMES (mean per vertex)
    //     Replaces the old "frame 0 ONLY" template sampling.
    // =========================
    std::vector<double> I_tmpl(N, std::numeric_limits<double>::quiet_NaN());
    std::vector<int>    I_cnt (N, 0);
    const float init_depth_gate = 5.0f; // mm; consistent with vis_depth_gate below

    for (int f = 0; f < F; ++f) {
        const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
        const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);
        const cv::Mat& img   = imgs_gray[f];
        const cv::Mat& depth = observed_depth[f];

        #pragma omp parallel
        {
            std::vector<double> local_sum(N, 0.0);
            std::vector<int>    local_cnt(N, 0);

            #pragma omp for nowait
            for (int i = 0; i < N; ++i) {
                // Using identity ED (current Xfull) at init stage
                Eigen::Vector3d pw = edGraph.deformVertex(mesh_vertices[i], i);
                Eigen::Vector3d pc = R.transpose() * (pw - t);
                if (pc.z() <= 1e-8) continue;

                float u = (float)(K(0,0)*(pc.x()/pc.z()) + K(0,2));
                float v = (float)(K(1,1)*(pc.y()/pc.z()) + K(1,2));
                if (u < 1 || v < 1 || u > img.cols - 2 || v > img.rows - 2) continue;

                // Depth-consistent gating to avoid occlusion/mismatch
                float z_obs = bilinearSample(depth, u, v);
                if (!(z_obs > 0.f) || !std::isfinite(z_obs)) continue;
                if (std::fabs((float)pc.z() - z_obs) > init_depth_gate) continue;

                float I = bilinearSample(img, u, v);
                local_sum[i] += (double)I;
                local_cnt[i]  += 1;
            }

            #pragma omp critical
            {
                for (int i = 0; i < N; ++i) {
                    if (local_cnt[i] == 0) continue;
                    if (!std::isfinite(I_tmpl[i])) I_tmpl[i] = 0.0;
                    I_tmpl[i] += local_sum[i];
                    I_cnt[i]  += local_cnt[i];
                }
            }
        }
    }

    int assigned_cnt = 0;
    for (int i = 0; i < N; ++i) {
        if (I_cnt[i] > 0) { I_tmpl[i] /= (double)I_cnt[i]; ++assigned_cnt; }
    }
    std::cout << "[Init] Intensity mean assigned for " << assigned_cnt << " / " << N << " vertices." << std::endl;

    // =========================
    // (2.5) Build global intensity mapping (storage index) & initial values
    //        NOTE: we will COMPACT intensity columns PER-ITERATION later.
    // =========================
    std::vector<int> colI(N, -1); int Icount_global=0;
    for (int i=0;i<N;++i) if (std::isfinite(I_tmpl[i])) colI[i] = Icount_global++;
    std::vector<double> I_var(Icount_global, 0.0);
    for (int i=0;i<N;++i) if (colI[i] >= 0) I_var[colI[i]] = I_tmpl[i];

    // =========================
    // Scale variables: global + per-frame micro-adjustments (delta_f for f>=1)
    // =========================
    double gamma_global = 0.0;                // log global scale
    std::vector<double> delta(F, 0.0);        // per-frame micro log-scale, delta[0] kept at 0 by design
    const double lambda_scale_prior   = 0.05; // L2 prior on per-frame deltas (small, keeps deltas tiny)
    const double lambda_scale_temporal= 0.00; // optional temporal smooth (0 -> disabled)

    auto s_eff = [&](int f){ return std::exp(gamma_global + delta[f]); };

    // =========================
    // GN loop with DYNAMIC visibility/active set/row & col layout
    // =========================
    double prev_cost = std::numeric_limits<double>::max();

    for (int it=0; it<maxIterations_; ++it) {
        // 0) write current X to graph
        for (int f=0; f<F; ++f) edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);

        // 0.25) Build *scaled* depth images for this iteration (apply current s_eff)
        std::vector<cv::Mat> depth_scaled(F);
        for (int f=0; f<F; ++f) {
            const float s = static_cast<float>(s_eff(f));
            observed_depth[f].convertTo(depth_scaled[f], CV_32F, s, 0.0);
        }

        // 0.5) Build per-frame deformed vertices and *per-iteration* vertex normals
        std::vector<std::vector<Eigen::Vector3d>> Vdef(F, std::vector<Eigen::Vector3d>(N));
        std::vector<std::vector<Eigen::Vector3d>> normals_w(F, std::vector<Eigen::Vector3d>(N));
        #pragma omp parallel for schedule(static)
        for (int f = 0; f < F; ++f) {
            for (int i = 0; i < N; ++i) {
                Vdef[f][i] = edGraph.deformVertex(mesh_vertices[i], i);
            }
            // compute normals serially per frame (robust & simple)
            computeVertexNormals(Vdef[f], mesh_triangles, normals_w[f]);
        }

        // 1) Recompute visibility (FOV + depth-consistent gate) using *scaled* depth
        const float vis_depth_gate = 5.0f; // mm; tune alongside ICP gate
        std::vector<std::vector<int>> visible_vertices(F);
        for (int f = 0; f < F; ++f) {
            const cv::Mat& img   = imgs_gray[f];
            const cv::Mat& depth = depth_scaled[f];
            const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
            const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);
            std::vector<int> vis; vis.reserve(N/2);

            #pragma omp parallel
            {
                std::vector<int> local; local.reserve(256);
                #pragma omp for nowait
                for (int i = 0; i < N; ++i) {
                    const Eigen::Vector3d& pw = Vdef[f][i];
                    Eigen::Vector3d pc = R.transpose() * (pw - t);
                    if (pc.z() <= 1e-8) continue;
                    float u = static_cast<float>(K(0,0) * (pc.x() / pc.z()) + K(0,2));
                    float v = static_cast<float>(K(1,1) * (pc.y() / pc.z()) + K(1,2));
                    if (u < 1 || v < 1 || u > img.cols - 2 || v > img.rows - 2) continue;

                    // Depth-consistent gating with scaled observed depth
                    float z_obs = bilinearSample(depth, u, v);
                    if (!(z_obs > 0.f) || !std::isfinite(z_obs)) continue;
                    if (std::fabs((float)pc.z() - z_obs) > vis_depth_gate) continue;

                    local.push_back(i);
                }
                #pragma omp critical
                vis.insert(vis.end(), local.begin(), local.end());
            }
            visible_vertices[f].swap(vis);
        }

        // 2) Active nodes/edges (f>=1) & compact index
        std::vector<std::vector<char>> active_node(F, std::vector<char>(G,0));
        std::vector<std::vector<std::pair<int,int>>> active_edges(F);
        std::vector<std::vector<int>> compact_idx(F, std::vector<int>(G,-1));
        std::vector<int> Sf(F,0);

        for (int f=1; f<F; ++f) {
            for (int vid : visible_vertices[f])
                for (int nid : bindings[vid]) active_node[f][nid] = 1;
            int acc=0; for (int j=0;j<G;++j) if (active_node[f][j]) compact_idx[f][j] = acc++;
            Sf[f]=acc;
            std::vector<std::pair<int,int>> Ef; Ef.reserve(edges.size());
            for (const auto& e: edges) if (active_node[f][e.first] && active_node[f][e.second]) Ef.push_back(e);
            active_edges[f].swap(Ef);
        }

        auto offsEDc = [&](int f){ int ofs=0; for (int k=0;k<f; ++k) ofs += 12 * Sf[k]; return ofs; };
        auto colA_c  = [&](int f,int node,int k){ int ci=compact_idx[f][node]; if (ci<0) return -1; return offsEDc(f)+12*ci+k; };
        auto colt_c  = [&](int f,int node,int k){ int ci=compact_idx[f][node]; if (ci<0) return -1; return offsEDc(f)+12*ci+9+k; };

        // 2.5) Intensity columns: per-iteration COMPACT mapping
        std::vector<char> I_active(N, 0);
        for (int f=0; f<F; ++f) {
            for (int vid : visible_vertices[f]) if (colI[vid] >= 0) I_active[vid] = 1;
        }
        std::vector<int> colI_it(N, -1); int Icount_it = 0;
        for (int i=0; i<N; ++i) if (I_active[i]) colI_it[i] = Icount_it++;

        // 2.6) Scale columns (compact layout): 1 global + (F-1) frame deltas for f>=1
        auto scaleColsBase = [&](int edDimCompact_, int Icount_it_){ return edDimCompact_ + Icount_it_; };
        auto colScaleGlob  = [&](int edDimCompact_, int Icount_it_){ return scaleColsBase(edDimCompact_, Icount_it_); };
        auto colScaleFrame = [&](int edDimCompact_, int Icount_it_, int f){ return scaleColsBase(edDimCompact_, Icount_it_) + 1 + (f-1); }; // f>=1

        // 3) Row/col layout
        const bool use_photo = (w_photo_ > 1e-12);
        const bool use_icp   = (w_icp_   > 1e-12);

        std::vector<int> photo_row_ofs(F,0), icp_row_ofs(F,0);
        int photo_rows=0, icp_rows=0;
        for (int f=0; f<F; ++f) { // include f=0 for photo
            photo_row_ofs[f] = photo_rows; if (use_photo) photo_rows += (int)visible_vertices[f].size();
        }
        for (int f=1; f<F; ++f) { // ICP from f=1
            icp_row_ofs[f]   = icp_rows;   if (use_icp)   icp_rows   += (int)visible_vertices[f].size();
        }

        int smooth_rows=0, rot_rows=0, temporal_rows=0;
        for (int f=1; f<F; ++f) { smooth_rows += (int)active_edges[f].size()*(9+3); rot_rows += Sf[f]*9; }
        for (int f=2; f<F; ++f) for (int j=0;j<G;++j) if (active_node[f-1][j] && active_node[f][j]) temporal_rows += 12;

        // scale priors and (optional) temporal smooth rows
        int scale_prior_rows   = (F>=2)? (F-1) : 0;              // delta_f for f=1..F-1
        int scale_tempor_rows  = (F>=3 && lambda_scale_temporal>0.0)? (F-2) : 0; // (delta_f - delta_{f-1}) for f=2..F-1

        const int row_photo_begin       = 0;
        const int row_icp_begin         = row_photo_begin + photo_rows;
        const int row_smooth_begin      = row_icp_begin   + icp_rows;
        const int row_rot_begin         = row_smooth_begin + smooth_rows;
        const int row_temporal_begin    = row_rot_begin    + rot_rows;
        const int row_scale_prior_begin = row_temporal_begin + temporal_rows;
        const int row_scale_temp_begin  = row_scale_prior_begin + scale_prior_rows;
        const int total_rows            = row_scale_temp_begin + scale_tempor_rows;

        int edDimCompact=0; for (int f=0; f<F; ++f) edDimCompact += 12 * Sf[f];
        const int scaleCols = 1 + std::max(0, F-1); // global + (F-1) deltas
        const int stateDimCompact = edDimCompact + Icount_it + scaleCols; // add scale columns

        std::cout << "[Layout it="<<it<<"] rows photo="<<photo_rows
                  << ", icp="<<icp_rows
                  << ", smooth="<<smooth_rows
                  << ", rot="<<rot_rows
                  << ", temporal="<<temporal_rows
                  << ", scale_prior="<<scale_prior_rows
                  << ", scale_temp="<<scale_tempor_rows
                  << ", total="<<total_rows << std::endl;
        std::cout << "[Layout it="<<it<<"] cols edCompact="<< edDimCompact
                  << ", intens="<< Icount_it
                  << ", scale="<< scaleCols
                  << ", total="<< stateDimCompact << std::endl;

        // 4) Assemble J & F
        std::vector<double> Fvec(total_rows, 0.0);
        const double sqrt_w_photo = std::sqrt(std::max(0.0, w_photo_));
        const double sqrt_w_icp   = std::sqrt(std::max(0.0, w_icp_));
        const double sqrt_ls = std::sqrt(std::max(0.0, lambda_smooth_));
        const double sqrt_lr = std::sqrt(std::max(0.0, lambda_rot_));
        const double sqrt_ltp= std::sqrt(std::max(0.0, lambda_temporal_));
        const double sqrt_lsp= std::sqrt(std::max(0.0, lambda_scale_prior));
        const double sqrt_lst= std::sqrt(std::max(0.0, lambda_scale_temporal));

        int num_threads = omp_get_max_threads();
        std::vector<std::vector<Eigen::Triplet<double>>> triplets_thr(num_threads);

        // PHOTO (f=0..F-1)
        if (use_photo) {
            #pragma omp parallel for schedule(static)
            for (int f=0; f<F; ++f) {
                int tid = omp_get_thread_num(); auto& T = triplets_thr[tid];
                edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);
                const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
                const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);
                const cv::Mat& img = imgs_gray[f];
                const BVH& bvh = bvhs[f];
                int r = row_photo_begin + photo_row_ofs[f];
                for (int idx=0; idx<(int)visible_vertices[f].size(); ++idx) {
                    int i = visible_vertices[f][idx];
                    int ci_global = colI[i];
                    int ci_it     = colI_it[i];
                    if (ci_global < 0 || ci_it < 0) { ++r; continue; }
                    const double Icurr = I_var[ci_global];
                    PhotometricError cost(mesh_vertices[i], i, mesh_triangles, K, img, bvh, sqrt_w_photo, &edGraph);

                    double residual = 0.0;
                    Eigen::VectorXd J_ed(12*G); J_ed.setZero();
                    double J_I = 0.0;

                    cost.Evaluate(Icurr, residual, &J_I, (f==0? nullptr : &J_ed), R, t);
                    Fvec[r] = residual;

                    // intensity column (per-iteration compact)
                    const int colIglob = edDimCompact + ci_it;
                    T.emplace_back(r, colIglob, J_I);

                    if (f>=1) {
                        const auto& bnd = bindings[i];
                        for (int nid : bnd) {
                            const int base = 12*nid;
                            for (int c=0;c<9;++c)  { double v=J_ed[base+c];    if (!v) continue; int col=colA_c(f,nid,c); if (col>=0) T.emplace_back(r,col,v); }
                            for (int c=0;c<3;++c)  { double v=J_ed[base+9+c];  if (!v) continue; int col=colt_c(f,nid,c); if (col>=0) T.emplace_back(r,col,v); }
                        }
                    }
                    ++r;
                }
            }
        }

        // ICP (f>=1) — **use per-iteration normals** and **scaled depth** + add scale Jacobians
        if (use_icp) {
            #pragma omp parallel for schedule(static)
            for (int f=1; f<F; ++f) {
                int tid = omp_get_thread_num(); auto& T = triplets_thr[tid];
                edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);
                const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
                const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);
                const cv::Mat& depth = depth_scaled[f];
                int r = row_icp_begin + icp_row_ofs[f];
                for (int idx=0; idx<(int)visible_vertices[f].size(); ++idx) {
                    int i = visible_vertices[f][idx];
                    const Eigen::Vector3d n_w = normals_w[f][i];

                    ProjectiveICPError icpCost(mesh_vertices[i], i, depth, &edGraph, K, n_w, sqrt_w_icp);
                    double residual=0.0; Eigen::VectorXd J_ed(12*G); J_ed.setZero();
                    bool ok = icpCost.Evaluate(residual, &J_ed, R, t);
                    if (!ok) { ++r; continue; }
                    Fvec[r] = residual;

                    // ==== ED Jacobians ====
                    const auto& bnd = bindings[i];
                    for (int nid : bnd) {
                        const int base = 12*nid;
                        for (int c=0;c<9;++c)  { double v=J_ed[base+c];    if (!v) continue; int col=colA_c(f,nid,c); if (col>=0) T.emplace_back(r,col,v); }
                        for (int c=0;c<3;++c)  { double v=J_ed[base+9+c];  if (!v) continue; int col=colt_c(f,nid,c); if (col>=0) T.emplace_back(r,col,v); }
                    }

                    // ==== Scale Jacobians (global + frame) ====
                    // Recompute pixel (u,v) and z_rel at that pixel from *original* relative depth
                    const Eigen::Vector3d& pw = Vdef[f][i];
                    Eigen::Vector3d pc = R.transpose() * (pw - t);
                    if (pc.z() <= 1e-8) { ++r; continue; }
                    float u = static_cast<float>(K(0,0) * (pc.x() / pc.z()) + K(0,2));
                    float v = static_cast<float>(K(1,1) * (pc.y() / pc.z()) + K(1,2));
                    // if out of bound, derivative ~0 (skip)
                    if (!(u >= 1 && v >= 1 && u <= observed_depth[f].cols - 2 && v <= observed_depth[f].rows - 2)) { ++r; continue; }
                    float z_rel = bilinearSample(observed_depth[f], u, v);
                    if (!(z_rel > 0.f) || !std::isfinite(z_rel)) { ++r; continue; }

                    // p_rel in camera (from K^{-1}[u,v,1]^T * z_rel)
                    const double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
                    Eigen::Vector3d p_rel;
                    p_rel.z() = (double)z_rel;
                    p_rel.x() = ((double)u - cx) * p_rel.z() / fx;
                    p_rel.y() = ((double)v - cy) * p_rel.z() / fy;

                    // normal in camera
                    Eigen::Vector3d n_c = R.transpose() * n_w; double nrm = n_c.norm(); if (nrm>1e-20) n_c/=nrm;

                    const double sEf = s_eff(f);
                    const double dr_dseff = - sqrt_w_icp * (p_rel.dot(n_c)); // residual already weighted by sqrt_w_icp
                    const int cGlob = colScaleGlob(edDimCompact, Icount_it);
                    T.emplace_back(r, cGlob, dr_dseff * sEf); // d r / d gamma = d r / d s * s
                    if (f>=1) {
                        const int cFrm = colScaleFrame(edDimCompact, Icount_it, f);
                        T.emplace_back(r, cFrm, dr_dseff * sEf); // d r / d delta_f = same as gamma
                    }

                    ++r;
                }
            }
        }

        // Smoothness
        int row_ptr = row_smooth_begin;
        for (int f=1; f<F; ++f) {
            for (const auto& e: active_edges[f]) {
                int i=e.first, j=e.second;
                for (int m=0;m<9;++m){ int ci=colA_c(f,i,m), cj=colA_c(f,j,m); Fvec[row_ptr]=sqrt_ls*(Xfull[f][12*i+m]-Xfull[f][12*j+m]); if (ci>=0) triplets_thr[0].emplace_back(row_ptr,ci,sqrt_ls); if (cj>=0) triplets_thr[0].emplace_back(row_ptr,cj,-sqrt_ls); ++row_ptr; }
                for (int m=0;m<3;++m){ int ci=colt_c(f,i,m), cj=colt_c(f,j,m); Fvec[row_ptr]=sqrt_ls*(Xfull[f][12*i+9+m]-Xfull[f][12*j+9+m]); if (ci>=0) triplets_thr[0].emplace_back(row_ptr,ci,sqrt_ls); if (cj>=0) triplets_thr[0].emplace_back(row_ptr,cj,-sqrt_ls); ++row_ptr; }
            }
        }

        // Rotation regularization: A^T A - I = 0
        int row_rot = row_rot_begin;
        for (int f=1; f<F; ++f) {
            for (int j=0; j<G; ++j) if (active_node[f][j]) {
                double A_[3][3];
                for (int rr=0; rr<3; ++rr)
                    for (int cc=0; cc<3; ++cc)
                        A_[rr][cc] = Xfull[f][12*j + (rr*3 + cc)];
                for (int p=0; p<3; ++p) {
                    for (int q=0; q<3; ++q) {
                        double AtA_pq = 0.0; for (int k=0; k<3; ++k) AtA_pq += A_[k][p] * A_[k][q];
                        const double target = (p==q)? 1.0 : 0.0;
                        Fvec[row_rot] = sqrt_lr * (AtA_pq - target);
                        for (int rr=0; rr<3; ++rr) {
                            int col_p = colA_c(f, j, rr*3 + p);
                            if (col_p >= 0) triplets_thr[0].emplace_back(row_rot, col_p, sqrt_lr * A_[rr][q]);
                            int col_q = colA_c(f, j, rr*3 + q);
                            if (col_q >= 0) triplets_thr[0].emplace_back(row_rot, col_q, sqrt_lr * A_[rr][p]);
                        }
                        ++row_rot;
                    }
                }
            }
        }

        // Temporal (ED)
        int row_tmp = row_temporal_begin;
        for (int f=2; f<F; ++f) {
            for (int j=0;j<G;++j) if (active_node[f-1][j] && active_node[f][j]) {
                for (int m=0;m<9;++m){ int c1=colA_c(f,j,m), c0=colA_c(f-1,j,m); Fvec[row_tmp]=sqrt_ltp*(Xfull[f][12*j+m]-Xfull[f-1][12*j+m]); if (c1>=0) triplets_thr[0].emplace_back(row_tmp,c1,sqrt_ltp); if (c0>=0) triplets_thr[0].emplace_back(row_tmp,c0,-sqrt_ltp); ++row_tmp; }
                for (int m=0;m<3;++m){ int c1=colt_c(f,j,m), c0=colt_c(f-1,j,m); Fvec[row_tmp]=sqrt_ltp*(Xfull[f][12*j+9+m]-Xfull[f-1][12*j+9+m]); if (c1>=0) triplets_thr[0].emplace_back(row_tmp,c1,sqrt_ltp); if (c0>=0) triplets_thr[0].emplace_back(row_tmp,c0,-sqrt_ltp); ++row_tmp; }
            }
        }

        // Scale priors: sqrt_lsp * delta_f
        int row_sp = row_scale_prior_begin;
        if (scale_prior_rows > 0 && sqrt_lsp > 0.0) {
            const int cGlob = colScaleGlob(edDimCompact, Icount_it);
            (void)cGlob; // not used here, but kept for clarity
            for (int f=1; f<F; ++f) {
                const int cFrm = colScaleFrame(edDimCompact, Icount_it, f);
                Fvec[row_sp] = sqrt_lsp * (delta[f]);
                if (cFrm >= 0) triplets_thr[0].emplace_back(row_sp, cFrm, sqrt_lsp);
                ++row_sp;
            }
        }

        // Scale temporal smooth: sqrt_lst * (delta_f - delta_{f-1})
        int row_st = row_scale_temp_begin;
        if (scale_tempor_rows > 0 && sqrt_lst > 0.0) {
            for (int f=2; f<F; ++f) {
                const int c1 = colScaleFrame(edDimCompact, Icount_it, f);
                const int c0 = colScaleFrame(edDimCompact, Icount_it, f-1);
                Fvec[row_st] = sqrt_lst * (delta[f] - delta[f-1]);
                if (c1 >= 0) triplets_thr[0].emplace_back(row_st, c1, sqrt_lst);
                if (c0 >= 0) triplets_thr[0].emplace_back(row_st, c0, -sqrt_lst);
                ++row_st;
            }
        }

        // Build J and solve
        std::vector<Eigen::Triplet<double>> triplets;
        size_t nnz_est = 0; for (auto& v: triplets_thr) nnz_est += v.size(); triplets.reserve(nnz_est);
        for (auto& v: triplets_thr) { triplets.insert(triplets.end(), v.begin(), v.end()); std::vector<Eigen::Triplet<double>>().swap(v);}        

        Eigen::SparseMatrix<double> J(total_rows, stateDimCompact);
        J.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::VectorXd Fv = Eigen::Map<Eigen::VectorXd>(Fvec.data(), (int)Fvec.size());

        const double cost = 0.5 * Fv.squaredNorm();

        Eigen::SparseMatrix<double> JT = J.transpose();
        Eigen::SparseMatrix<double> H  = JT * J;
        Eigen::VectorXd b = -JT * Fv;

        if (H.rows()>0) {
            Eigen::VectorXd d = H.diagonal();
            double scale = (d.size()>0)? d.cwiseAbs().mean() : 1.0;
            double lm = std::max(1e-12, 1e-3 * std::max(1.0, scale));
            H.diagonal().array() += lm;
        }

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver; solver.compute(H);
        if (solver.info()!=Eigen::Success){ std::cerr << "[Optimizer] LDLT factorization failed.\n"; break; }
        Eigen::VectorXd dx = solver.solve(b); if (solver.info()!=Eigen::Success){ std::cerr << "[Optimizer] Linear solve failed.\n"; break; }

        // Apply update: ED (f>=1)
        for (int f=1; f<F; ++f) {
            const int ofs = offsEDc(f);
            for (int j=0;j<G;++j) if (active_node[f][j]) {
                const int ci = compact_idx[f][j];
                const int base = ofs + 12*ci;
                for (int c=0;c<9;++c)  Xfull[f][12*j + c]     += dx[base + c];
                for (int c=0;c<3;++c)  Xfull[f][12*j + 9 + c] += dx[base + 9 + c];
            }
        }
        // Apply update: intensity variables (per-iteration columns)
        for (int i=0;i<N;++i) {
            int ci_it = colI_it[i];
            int ci_global = colI[i];
            if (ci_it < 0 || ci_global < 0) continue;
            I_var[ci_global] += dx[edDimCompact + ci_it];
            // Optional clamp: I_var[ci_global] = std::min(1.0, std::max(0.0, I_var[ci_global]));
        }

        // Apply update: scales
        const int baseScale = edDimCompact + Icount_it;
        gamma_global += dx[baseScale + 0];
        for (int f=1; f<F; ++f) {
            const int cFrm = baseScale + 1 + (f-1);
            delta[f] += dx[cFrm];
        }

        const double mean_cost = (total_rows>0)? (2.0*cost / (double)total_rows) : cost;
        std::cout << "[GN it="<<it<<"] cost="<<cost<<" (mean "<<mean_cost<<"), |dx|="<<dx.norm() << std::endl;
        if (std::abs(cost - prev_cost) < 1e-6) break; prev_cost = cost;
    }

    if (on_save) {
        for (int f = 1; f < F; ++f) {
            edGraph.updateFromStateVector(Xfull[f], 0);
            on_save(f, edGraph);
        }
    } else {
        if (F > 1) edGraph.updateFromStateVector(Xfull[1], 0);
    }
}