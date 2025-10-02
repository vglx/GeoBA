#include "Optimizer.h"
#include "CostFunctions.h"

#include <iostream>
#include <limits>
#include <cmath>
#include <numeric>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/Geometry>
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

inline bool projectInBounds(
    const Eigen::Vector3d& Pw,
    const Eigen::Matrix3d& Rcw, const Eigen::Vector3d& tcw,
    const Eigen::Matrix3d& K,
    int width, int height,
    float& u, float& v,
    double& z)
{
    Eigen::Vector3d Pc = Rcw * Pw + tcw; // assumes Rcw,tcw are camera-from-world
    z = Pc.z();
    if (z <= 1e-8) return false;
    u = static_cast<float>(K(0,0) * (Pc.x()/z) + K(0,2));
    v = static_cast<float>(K(1,1) * (Pc.y()/z) + K(1,2));
    // strict in-bounds (no clamping): need 1..W-2 to allow gradient sampling
    return (u >= 1 && v >= 1 && u <= width - 2 && v <= height - 2);
}

} // namespace

Optimizer::Optimizer(double w_photo,
                     double w_stereo,
                     int maxStages,
                     int maxIterations,
                     double lambda_smooth,
                     double lambda_rot,
                     double lambda_temporal)
    : w_photo_(w_photo), w_stereo_(w_stereo),
      lambda_smooth_(lambda_smooth), lambda_rot_(lambda_rot), lambda_temporal_(lambda_temporal),
      maxStages_(maxStages), maxIterations_(maxIterations) {}

void Optimizer::optimize(
    const std::vector<MeshModel::Vertex>& mesh_vertices,
    const std::vector<MeshModel::Triangle>& mesh_triangles,
    const Eigen::Matrix3d& K_left,
    const Eigen::Matrix3d& K_right,
    const std::vector<cv::Mat>& rgb_left,
    const std::vector<cv::Mat>& rgb_right,
    const std::vector<Eigen::Matrix4d>& poses_left,   // T_wc (world from cam)
    const std::vector<Eigen::Matrix4d>& poses_right,  // T_wc
    EDGraph& edGraph,
    SaveCallback on_save) {

    const int F  = (int)rgb_left.size();
    const int Fr = (int)rgb_right.size();
    const int N  = (int)mesh_vertices.size();
    const int G  = edGraph.numNodes();

    if (F <= 1 || N == 0 || G == 0 || F != Fr ||
        (int)poses_left.size()  != F ||
        (int)poses_right.size() != F) {
        std::cout << "[Optimizer] Invalid inputs: need L/R RGB for >=2 frames, matching sizes, mesh & graph non-empty." << std::endl;
        return;
    }

    // ---- grayscale images [0,1]
    auto toGray01 = [](const cv::Mat& img){
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
        return gray;
    };

    std::vector<cv::Mat> L(F), R(F);
    for (int f=0; f<F; ++f) { L[f] = toGray01(rgb_left[f]); R[f] = toGray01(rgb_right[f]); }

    // =========================
    // Per-frame full ED blocks (12*G). Frame 0 is fixed (no columns for it).
    // =========================
    const int edDimPerFrameFull = 12 * G;
    std::vector<Eigen::VectorXd> Xfull(F, Eigen::VectorXd::Zero(edDimPerFrameFull));
    for (int f=0; f<F; ++f) edGraph.writeToStateVector(Xfull[f], /*offset=*/0);

    const auto& bindings = edGraph.getBindings();
    const auto& edges    = edGraph.getEdges();

    // =========================
    // (1) Intensity init from FRAME 0 (template view, left camera)
    //     NOTE: We now **only** initialize from f=0. No lazy creation on f>0.
    // =========================
    std::vector<double> I_tmpl(N, std::numeric_limits<double>::quiet_NaN());
    {
        edGraph.updateFromStateVector(Xfull[0], /*offset=*/0); // identity at init
        const Eigen::Matrix3d R0w = poses_left[0].block<3,3>(0,0).transpose(); // R_cw
        const Eigen::Vector3d t0w = -R0w * poses_left[0].block<3,1>(0,3);       // t_cw
        const cv::Mat& img0 = L[0];
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            Eigen::Vector3d pw = edGraph.deformVertex(mesh_vertices[i], i);
            float u, v; double z;
            if (!projectInBounds(pw, R0w, t0w, K_left, img0.cols, img0.rows, u, v, z)) continue;
            I_tmpl[i] = (double)bilinearSample(img0, u, v);
        }
    }

    // Compact intensity storage (global). Only those visible in f=0 are created.
    std::vector<int>    colI(N, -1);
    std::vector<double> I_var; I_var.reserve(N);
    int Icount_global = 0;
    for (int i=0;i<N;++i) if (std::isfinite(I_tmpl[i])) { colI[i] = Icount_global++; I_var.push_back(I_tmpl[i]); }

    // Angle weight toggle (true by default; set false to disable)
    const bool use_angle_weight = true;
    const double cos_thr = 0.2; // backface reject if dot < 0.2

    // =========================
    // GN loop with DYNAMIC visibility/active set/row & col layout
    // =========================
    double prev_cost = std::numeric_limits<double>::max();

    for (int it=0; it<maxIterations_; ++it) {
        // 0) write current X to graph (for completeness)
        for (int f=0; f<F; ++f) edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);

        // 0.5) Build per-frame deformed vertices and normals (world)
        std::vector<std::vector<Eigen::Vector3d>> Vdef(F, std::vector<Eigen::Vector3d>(N));
        std::vector<std::vector<Eigen::Vector3d>> normals_w(F, std::vector<Eigen::Vector3d>(N));
        for (int f = 0; f < F; ++f) {
            edGraph.updateFromStateVector(Xfull[f], /*offset=*/0); // set once per frame (serial)
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) Vdef[f][i] = edGraph.deformVertex(mesh_vertices[i], i);
            computeVertexNormals(Vdef[f], mesh_triangles, normals_w[f]);
        }

        // Precompute camera-from-world (Rcw, tcw) for both eyes
        std::vector<Eigen::Matrix3d> Rcw_L(F), Rcw_R(F);
        std::vector<Eigen::Vector3d> tcw_L(F), tcw_R(F);
        for (int f=0; f<F; ++f) {
            Rcw_L[f] = poses_left[f].block<3,3>(0,0).transpose();
            tcw_L[f] = -Rcw_L[f] * poses_left[f].block<3,1>(0,3);
            Rcw_R[f] = poses_right[f].block<3,3>(0,0).transpose();
            tcw_R[f] = -Rcw_R[f] * poses_right[f].block<3,1>(0,3);
        }

        // 1) Visibility (FOV-only) for LEFT & RIGHT; build intersection for stereo
        std::vector<std::vector<int>> visible_L(F), visible_R(F), visible_both(F);
        for (int f = 0; f < F; ++f) {
            const cv::Mat &imgL = L[f], &imgR = R[f];
            std::vector<int> visL; visL.reserve(N/2);
            std::vector<int> visR; visR.reserve(N/2);

            // LEFT FOV visibility
            #pragma omp parallel
            {
                std::vector<int> local; local.reserve(256);
                #pragma omp for nowait
                for (int i = 0; i < N; ++i) {
                    float u, v; double z;
                    if (!projectInBounds(Vdef[f][i], Rcw_L[f], tcw_L[f], K_left, imgL.cols, imgL.rows, u, v, z)) continue;
                    if (use_angle_weight) {
                        Eigen::Vector3d n = normals_w[f][i];
                        Eigen::Vector3d vdir = (-(Rcw_L[f].transpose()*tcw_L[f]) - Vdef[f][i]).normalized();
                        if (n.dot(vdir) < cos_thr) continue;
                    }
                    local.push_back(i);
                }
                #pragma omp critical
                visL.insert(visL.end(), local.begin(), local.end());
            }

            // RIGHT FOV visibility (strict; simulator: no occlusion needed)
            #pragma omp parallel
            {
                std::vector<int> local; local.reserve(256);
                #pragma omp for nowait
                for (int i = 0; i < N; ++i) {
                    float u, v; double z;
                    if (!projectInBounds(Vdef[f][i], Rcw_R[f], tcw_R[f], K_right, imgR.cols, imgR.rows, u, v, z)) continue;
                    if (use_angle_weight) {
                        Eigen::Vector3d n = normals_w[f][i];
                        Eigen::Vector3d vdir = (-(Rcw_R[f].transpose()*tcw_R[f]) - Vdef[f][i]).normalized();
                        if (n.dot(vdir) < cos_thr) continue;
                    }
                    local.push_back(i);
                }
                #pragma omp critical
                visR.insert(visR.end(), local.begin(), local.end());
            }

            // Intersection for stereo
            std::vector<char> isR(N, 0); for (int id : visR) isR[id] = 1;
            std::vector<int> both; both.reserve(std::min(visL.size(), visR.size()));
            for (int id : visL) if (isR[id]) both.push_back(id);

            visible_L[f].swap(visL);
            visible_R[f].swap(visR);
            visible_both[f].swap(both);
        }

        // 2) Active nodes/edges (f>=1) & compact index
        std::vector<std::vector<char>> active_node(F, std::vector<char>(G,0));
        std::vector<std::vector<std::pair<int,int>>> active_edges(F);
        std::vector<std::vector<int>> compact_idx(F, std::vector<int>(G,-1));
        std::vector<int> Sf(F,0);

        for (int f=1; f<F; ++f) {
            for (int vid : visible_L[f])
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

        // 2.5) Intensity columns: per-iteration COMPACT mapping（只为本轮可见且在f=0有I_var的顶点）
        std::vector<char> I_active(N, 0);
        for (int f=0; f<F; ++f) for (int vid : visible_L[f]) if (colI[vid] >= 0) I_active[vid] = 1;
        std::vector<int> colI_it(N, -1); int Icount_it = 0;
        for (int i=0; i<N; ++i) if (I_active[i]) colI_it[i] = Icount_it++;

        // 3) Row/col layout
        const bool use_photo  = (w_photo_  > 1e-12);
        const bool use_stereo = (w_stereo_ > 1e-12);

        std::vector<int> photo_row_ofs(F,0), stereo_row_ofs(F,0);
        int photo_rows=0, stereo_rows=0;
        for (int f=0; f<F; ++f) { // include f=0 for photo against template intensity
            photo_row_ofs[f] = photo_rows; if (use_photo)  photo_rows  += (int)visible_L[f].size();
            stereo_row_ofs[f]= stereo_rows; if (use_stereo) stereo_rows += (int)visible_both[f].size();
        }

        int smooth_rows=0, rot_rows=0, temporal_rows=0;
        for (int f=1; f<F; ++f) { smooth_rows += (int)active_edges[f].size()*(9+3); rot_rows += Sf[f]*9; }
        for (int f=2; f<F; ++f) for (int j=0;j<G;++j) if (active_node[f-1][j] && active_node[f][j]) temporal_rows += 12;

        const int row_photo_begin    = 0;
        const int row_stereo_begin   = row_photo_begin + photo_rows;
        const int row_smooth_begin   = row_stereo_begin + stereo_rows;
        const int row_rot_begin      = row_smooth_begin + smooth_rows;
        const int row_temporal_begin = row_rot_begin    + rot_rows;
        const int total_rows         = row_temporal_begin + temporal_rows;

        int edDimCompact=0; for (int f=0; f<F; ++f) edDimCompact += 12 * Sf[f];
        const int stateDimCompact = edDimCompact + Icount_it; // ED + intensity only

        std::cout << "[Layout it="<<it<<"] rows photo="<<photo_rows
                  << ", stereo="<<stereo_rows
                  << ", smooth="<<smooth_rows
                  << ", rot="<<rot_rows
                  << ", temporal="<<temporal_rows
                  << ", total="<<total_rows << std::endl;
        std::cout << "[Layout it="<<it<<"] cols edCompact="<< edDimCompact
                  << ", intens="<< Icount_it
                  << ", total="<< stateDimCompact << std::endl;

        // 4) Assemble J & F
        std::vector<double> Fvec(total_rows, 0.0);
        const double sqrt_w_photo  = std::sqrt(std::max(0.0, w_photo_));
        const double sqrt_w_stereo = std::sqrt(std::max(0.0, w_stereo_));
        const double sqrt_ls = std::sqrt(std::max(0.0, lambda_smooth_));
        const double sqrt_lr = std::sqrt(std::max(0.0, lambda_rot_));
        const double sqrt_ltp= std::sqrt(std::max(0.0, lambda_temporal_));

        int num_threads = omp_get_max_threads();
        std::vector<std::vector<Eigen::Triplet<double>>> triplets_thr(num_threads);

        // --- Residual stats (debug)
        size_t photo_kept = 0, photo_skip = 0;
        size_t stereo_kept = 0, stereo_skip = 0;
        double photo_abs_sum = 0.0, stereo_abs_sum = 0.0;

        // PHOTO (f=0..F-1) — Left camera only vs vertex intensity variable
        if (use_photo) {
            for (int f=0; f<F; ++f) {
                edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);

                const Eigen::Matrix3d Rcw = Rcw_L[f];
                const Eigen::Vector3d tcw = tcw_L[f];
                const cv::Mat& img = L[f];
                int row_base = row_photo_begin + photo_row_ofs[f];

                #pragma omp parallel for schedule(static)
                for (int idx=0; idx<(int)visible_L[f].size(); ++idx) {
                    int tid = omp_get_thread_num(); auto& T = triplets_thr[tid];
                    int r = row_base + idx;
                    int i = visible_L[f][idx];

                    int ci_global = colI[i];
                    int ci_it     = colI_it[i];
                    if (ci_global < 0 || ci_it < 0) { photo_skip++; continue; }

                    const double Icurr = I_var[ci_global];
                    PhotometricError cost(mesh_vertices[i], i, mesh_triangles, K_left, img, sqrt_w_photo, &edGraph);

                    double residual = 0.0;
                    Eigen::VectorXd J_ed(12*G); J_ed.setZero();
                    double J_I = 0.0;

                    cost.Evaluate(Icurr, residual, &J_I, (f==0? nullptr : &J_ed), Rcw, tcw);

                    // Optional grazing-angle weight wrt LEFT view
                    double w_ang = 1.0;
                    if (use_angle_weight) {
                        Eigen::Vector3d n = normals_w[f][i];
                        Eigen::Vector3d Cw = -(Rcw.transpose()*tcw); // camera center in world
                        Eigen::Vector3d vdir = (Cw - Vdef[f][i]).normalized();
                        double c = std::max(0.0, n.dot(vdir));
                        if (c < cos_thr) continue; // backface reject
                        w_ang = c*c; // cos^2
                    }

                    Fvec[r] = residual * w_ang;
                    photo_kept++; photo_abs_sum += std::abs(Fvec[r]);

                    // intensity column (per-iteration compact)
                    const int colIglob = edDimCompact + ci_it;
                    T.emplace_back(r, colIglob, J_I * w_ang);

                    if (f>=1) {
                        const auto& bnd = bindings[i];
                        for (int nid : bnd) {
                            const int base = 12*nid;
                            for (int c=0;c<9;++c)  { double v=J_ed[base+c];    if (!v) continue; int col=colA_c(f,nid,c); if (col>=0) T.emplace_back(r,col,v*w_ang); }
                            for (int c=0;c<3;++c)  { double v=J_ed[base+9+c];  if (!v) continue; int col=colt_c(f,nid,c); if (col>=0) T.emplace_back(r,col,v*w_ang); }
                        }
                    }
                }
            }
        }

        // STEREO (f=0..F-1) — Left–Right photometric consistency (no intensity var)
        if (use_stereo) {
            for (int f=0; f<F; ++f) {
                edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);

                const Eigen::Matrix3d RLcw = Rcw_L[f];
                const Eigen::Vector3d tLcw = tcw_L[f];
                const Eigen::Matrix3d RRcw = Rcw_R[f];
                const Eigen::Vector3d tRcw = tcw_R[f];
                int row_base = row_stereo_begin + stereo_row_ofs[f];

                #pragma omp parallel for schedule(static)
                for (int idx=0; idx<(int)visible_both[f].size(); ++idx) {
                    int tid = omp_get_thread_num(); auto& T = triplets_thr[tid];
                    int r = row_base + idx;
                    int i = visible_both[f][idx];

                    // Defensive bounds recheck for right eye
                    float ur, vr; double zr;
                    if (!projectInBounds(Vdef[f][i], RRcw, tRcw, K_right, R[f].cols, R[f].rows, ur, vr, zr)) { stereo_skip++; continue; }

                    StereoPhotometricError scost(
                        mesh_vertices[i], i, &edGraph,
                        K_left, K_right,
                        L[f], R[f],
                        std::sqrt(std::max(0.0, w_stereo_)));

                    double residual=0.0; Eigen::VectorXd J_ed(12*G); J_ed.setZero();
                    bool ok = scost.Evaluate(residual, &J_ed, RLcw, tLcw, RRcw, tRcw);
                    if (!ok) { stereo_skip++; continue; }

                    double w_ang = 1.0;
                    if (use_angle_weight) {
                        Eigen::Vector3d n = normals_w[f][i];
                        Eigen::Vector3d CLw = -(RLcw.transpose()*tLcw);
                        Eigen::Vector3d vdir = (CLw - Vdef[f][i]).normalized();
                        double c = std::max(0.0, n.dot(vdir));
                        if (c < cos_thr) continue;
                        w_ang = c*c;
                    }

                    Fvec[r] = residual * w_ang;
                    stereo_kept++; stereo_abs_sum += std::abs(Fvec[r]);

                    if (f>=1) {
                        const auto& bnd = bindings[i];
                        for (int nid : bnd) {
                            const int base = 12*nid;
                            for (int c=0;c<9;++c)  { double v=J_ed[base+c];    if (!v) continue; int col=colA_c(f,nid,c); if (col>=0) T.emplace_back(r,col,v*w_ang); }
                            for (int c=0;c<3;++c)  { double v=J_ed[base+9+c];  if (!v) continue; int col=colt_c(f,nid,c); if (col>=0) T.emplace_back(r,col,v*w_ang); }
                        }
                    }
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
        }

        const double mean_cost = (total_rows>0)? (2.0*cost / (double)total_rows) : cost;
        const double mean_abs_photo  = (photo_kept>0)?  (photo_abs_sum  / (double)photo_kept)  : 0.0;
        const double mean_abs_stereo = (stereo_kept>0)? (stereo_abs_sum / (double)stereo_kept) : 0.0;
        std::cout << "[GN it="<<it<<"] cost="<<cost<<" (mean "<<mean_cost<<"), |dx|="<<dx.norm() << std::endl;
        std::cout << "          PHOTO kept="<<photo_kept<<" skip="<<photo_skip<<" mean|r|="<<mean_abs_photo
                  << "; STEREO kept="<<stereo_kept<<" skip="<<stereo_skip<<" mean|r|="<<mean_abs_stereo << std::endl;
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