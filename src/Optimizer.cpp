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
}

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
    EDGraph& edGraph) {

    const int F = (int)observed_rgb.size();
    const int Fd= (int)observed_depth.size();
    const int N = (int)mesh_vertices.size();
    const int G = edGraph.numNodes();

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
    // (1) Build BVH per frame from current deformation (initially identity)
    // =========================
    std::vector<std::vector<MeshModel::Vertex>> Vdef_all(F, std::vector<MeshModel::Vertex>(N));
    std::vector<BVH> bvhs; bvhs.reserve(F);
    for (int f=0; f<F; ++f) {
        edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);
        auto& Vd = Vdef_all[f];
        #pragma omp parallel for
        for (int i=0; i<N; ++i) {
            Eigen::Vector3d p = edGraph.deformVertex(mesh_vertices[i], i);
            Vd[i].x = (float)p.x(); Vd[i].y = (float)p.y(); Vd[i].z = (float)p.z();
        }
        bvhs.emplace_back(mesh_triangles, Vd);
    }

    // =========================
    // (2) Visibility (boundary-only; no occlusion test)
    // =========================
    std::vector<std::vector<int>> visible_vertices(F);
    for (int f = 0; f < F; ++f) {
        const cv::Mat& img = imgs_gray[f];
        const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
        const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);

        std::vector<int> vis; vis.reserve(N/2);

        #pragma omp parallel
        {
            std::vector<int> local; local.reserve(256);

            #pragma omp for nowait
            for (int i = 0; i < N; ++i) {
                // Boundary-only visibility: deform -> project -> inside image & in front of camera.
                Eigen::Vector3d pw = edGraph.deformVertex(mesh_vertices[i], i);
                Eigen::Vector3d pc = R.transpose() * (pw - t);
                if (pc.z() <= 1e-8) continue;

                float u = static_cast<float>(K(0,0) * (pc.x() / pc.z()) + K(0,2));
                float v = static_cast<float>(K(1,1) * (pc.y() / pc.z()) + K(1,2));

                if (u >= 0 && u < img.cols && v >= 0 && v < img.rows)
                    local.push_back(i);
            }

            #pragma omp critical
            vis.insert(vis.end(), local.begin(), local.end());
        }

        visible_vertices[f].swap(vis);
    }

    // =========================
    // (2.5) Template intensities from frame 0 ONLY (fixed, no intensity variables)
    // =========================
    std::vector<double> I_tmpl(N, std::numeric_limits<double>::quiet_NaN());
    {
        const int f=0;
        const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
        const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);
        const cv::Mat& img = imgs_gray[f];
        for (int k=0; k<(int)visible_vertices[f].size(); ++k) {
            int i = visible_vertices[f][k];
            Eigen::Vector3d pw = edGraph.deformVertex(mesh_vertices[i], i);
            Eigen::Vector3d pc = R.transpose() * (pw - t);
            if (pc.z() <= 1e-8) continue;
            float u = (float)(K(0,0)*(pc.x()/pc.z()) + K(0,2));
            float v = (float)(K(1,1)*(pc.y()/pc.z()) + K(1,2));
            I_tmpl[i] = (double)bilinearSample(img, u, v);
        }
        int cnt = (int)std::count_if(I_tmpl.begin(), I_tmpl.end(), [](double x){return std::isfinite(x);});
        std::cout << "[Init] Template intensities assigned for " << cnt << " / " << N << " vertices." << std::endl;
    }

    // =========================
    // (3) Active nodes/edges & compact map for f>=1
    // =========================
    std::vector<std::vector<char>> active_node(F, std::vector<char>(G,0));
    std::vector<std::vector<std::pair<int,int>>> active_edges(F);
    std::vector<std::vector<int>> compact_idx(F, std::vector<int>(G,-1));
    std::vector<int> Sf(F,0);

    for (int f=1; f<F; ++f) {
        for (int vid : visible_vertices[f]) {
            if (!std::isfinite(I_tmpl[vid])) continue; // 只让有模板强度的点驱动节点
            for (int nid : bindings[vid]) active_node[f][nid] = 1;
        }
        int acc=0; for (int j=0;j<G;++j) if (active_node[f][j]) compact_idx[f][j] = acc++;
        Sf[f]=acc;
        std::vector<std::pair<int,int>> Ef; Ef.reserve(edges.size());
        for (const auto& e: edges) if (active_node[f][e.first] && active_node[f][e.second]) Ef.push_back(e);
        active_edges[f].swap(Ef);
    }

    auto offsEDc = [&](int f){ int ofs=0; for (int k=0;k<f; ++k) ofs += 12 * Sf[k]; return ofs; };
    auto colA_c  = [&](int f,int node,int k){ int ci=compact_idx[f][node]; if (ci<0) return -1; return offsEDc(f)+12*ci+k; };
    auto colt_c  = [&](int f,int node,int k){ int ci=compact_idx[f][node]; if (ci<0) return -1; return offsEDc(f)+12*ci+9+k; };

    // =========================
    // (4) Row layout: PHOTO rows then ICP rows, then regularizers
    // =========================
    std::vector<int> photo_row_ofs(F,0), icp_row_ofs(F,0);
    int photo_rows=0, icp_rows=0;
    for (int f=1; f<F; ++f) {
        photo_row_ofs[f] = photo_rows; photo_rows += (int)visible_vertices[f].size();
        icp_row_ofs[f]   = icp_rows;   icp_rows   += (int)visible_vertices[f].size(); // upper bound (ICP gating may skip)
    }

    int smooth_rows=0, rot_rows=0, temporal_rows=0;
    for (int f=1; f<F; ++f) { smooth_rows += (int)active_edges[f].size()*(9+3); rot_rows += Sf[f]*9; }
    for (int f=2; f<F; ++f) for (int j=0;j<G;++j) if (active_node[f-1][j] && active_node[f][j]) temporal_rows += 12;

    const int row_photo_begin    = 0;
    const int row_icp_begin      = row_photo_begin + photo_rows;
    const int row_smooth_begin   = row_icp_begin   + icp_rows;
    const int row_rot_begin      = row_smooth_begin + smooth_rows;
    const int row_temporal_begin = row_rot_begin    + rot_rows;
    const int total_rows         = row_temporal_begin + temporal_rows;

    int edDimCompact=0; for (int f=0; f<F; ++f) edDimCompact += 12 * Sf[f];
    const int stateDimCompact = edDimCompact; // no intensity variables

    std::cout << "[Layout] rows photo="<<photo_rows
              << ", icp="<<icp_rows
              << ", smooth="<<smooth_rows
              << ", rot="<<rot_rows
              << ", temporal="<<temporal_rows
              << ", total="<<total_rows << std::endl;
    std::cout << "[Layout] cols edCompact="<< stateDimCompact << std::endl;

    // =========================
    // (5) GN loop
    // =========================
    double prev_cost = std::numeric_limits<double>::max();
    for (int it=0; it<maxIterations_; ++it) {
        std::vector<double> Fvec(total_rows, 0.0);
        const double sqrt_w_photo = std::sqrt(std::max(0.0, w_photo_));
        const double sqrt_w_icp   = std::sqrt(std::max(0.0, w_icp_));
        const double sqrt_ls = std::sqrt(std::max(0.0, lambda_smooth_));
        const double sqrt_lr = std::sqrt(std::max(0.0, lambda_rot_));
        const double sqrt_ltp= std::sqrt(std::max(0.0, lambda_temporal_));

        int num_threads = omp_get_max_threads();
        std::vector<std::vector<Eigen::Triplet<double>>> triplets_thr(num_threads);

        // ---- PHOTO (f>=1)
        #pragma omp parallel for schedule(static)
        for (int f=1; f<F; ++f) {
            int tid = omp_get_thread_num(); auto& T = triplets_thr[tid];
            edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);
            const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
            const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);
            const cv::Mat& img = imgs_gray[f];
            const BVH& bvh = bvhs[f];
            int r = row_photo_begin + photo_row_ofs[f];
            for (int idx=0; idx<(int)visible_vertices[f].size(); ++idx) {
                int i = visible_vertices[f][idx];
                if (!std::isfinite(I_tmpl[i])) { ++r; continue; }
                PhotometricError cost(mesh_vertices[i], i, mesh_triangles, K, img, bvh, sqrt_w_photo, &edGraph);
                double residual=0.0; Eigen::VectorXd J_ed(12*G); J_ed.setZero();
                cost.Evaluate(I_tmpl[i], residual, /*J_I*/nullptr, &J_ed, R, t);
                Fvec[r] = residual;
                const auto& b = bindings[i];
                for (int nid : b) {
                    const int base = 12*nid;
                    for (int c=0;c<9;++c)  { double v=J_ed[base+c];    if (v==0.0) continue; int col=colA_c(f,nid,c); if (col>=0) T.emplace_back(r,col,v); }
                    for (int c=0;c<3;++c)  { double v=J_ed[base+9+c];  if (v==0.0) continue; int col=colt_c(f,nid,c); if (col>=0) T.emplace_back(r,col,v); }
                }
                ++r;
            }
        }

        // ---- ICP (f>=1)
        #pragma omp parallel for schedule(static)
        for (int f=1; f<F; ++f) {
            int tid = omp_get_thread_num(); auto& T = triplets_thr[tid];
            edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);
            const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
            const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);
            const cv::Mat& depth = observed_depth[f];
            const BVH& bvh = bvhs[f];
            int r = row_icp_begin + icp_row_ofs[f];
            for (int idx=0; idx<(int)visible_vertices[f].size(); ++idx) {
                int i = visible_vertices[f][idx];
                
                Eigen::Vector3d n_w(mesh_vertices[i].nx, mesh_vertices[i].ny, mesh_vertices[i].nz);
                ProjectiveICPError icpCost(mesh_vertices[i], i, depth, &edGraph, K, n_w, sqrt_w_icp);

                double residual=0.0; Eigen::VectorXd J_ed(12*G); J_ed.setZero();
                bool ok = icpCost.Evaluate(residual, &J_ed, R, t);
                if (!ok) { ++r; continue; }
                Fvec[r] = residual;
                const auto& b = bindings[i];
                for (int nid : b) {
                    const int base = 12*nid;
                    for (int c=0;c<9;++c)  { double v=J_ed[base+c];    if (v==0.0) continue; int col=colA_c(f,nid,c); if (col>=0) T.emplace_back(r,col,v); }
                    for (int c=0;c<3;++c)  { double v=J_ed[base+9+c];  if (v==0.0) continue; int col=colt_c(f,nid,c); if (col>=0) T.emplace_back(r,col,v); }
                }
                ++r;
            }
        }

        // ---- Smoothness
        int row_ptr = row_smooth_begin;
        for (int f=1; f<F; ++f) {
            for (const auto& e: active_edges[f]) {
                int i=e.first, j=e.second;
                for (int m=0;m<9;++m){ int ci=colA_c(f,i,m), cj=colA_c(f,j,m); Fvec[row_ptr]=sqrt_ls*(Xfull[f][12*i+m]-Xfull[f][12*j+m]); if (ci>=0) triplets_thr[0].emplace_back(row_ptr,ci,sqrt_ls); if (cj>=0) triplets_thr[0].emplace_back(row_ptr,cj,-sqrt_ls); ++row_ptr; }
                for (int m=0;m<3;++m){ int ci=colt_c(f,i,m), cj=colt_c(f,j,m); Fvec[row_ptr]=sqrt_ls*(Xfull[f][12*i+9+m]-Xfull[f][12*j+9+m]); if (ci>=0) triplets_thr[0].emplace_back(row_ptr,ci,sqrt_ls); if (cj>=0) triplets_thr[0].emplace_back(row_ptr,cj,-sqrt_ls); ++row_ptr; }
            }
        }

        // ---- Rotation regularization (rigorous A^T A - I linearization)
        int row_rot = row_rot_begin;
        for (int f=1; f<F; ++f) {
            for (int j=0; j<G; ++j) if (active_node[f][j]) {
                // Extract A (row-major: a00 a01 a02 a10 a11 a12 a20 a21 a22)
                double A_[3][3];
                for (int r=0; r<3; ++r)
                    for (int c=0; c<3; ++c)
                        A_[r][c] = Xfull[f][12*j + (r*3 + c)];

                // r_{pq} = sqrt_lr * ( (A^T A)_{pq} - delta_{pq} ), p,q in {0,1,2}
                // d(A^T A)_{pq}/dA_{rs} = delta_{sp} * A_{rq} + delta_{sq} * A_{rp}
                for (int p=0; p<3; ++p) {
                    for (int q=0; q<3; ++q) {
                        double AtA_pq = 0.0;
                        for (int k=0; k<3; ++k) AtA_pq += A_[k][p] * A_[k][q];
                        const double target = (p==q)? 1.0 : 0.0;
                        Fvec[row_rot] = sqrt_lr * (AtA_pq - target);

                        for (int r=0; r<3; ++r) {
                            // s==p
                            int col_p = colA_c(f, j, r*3 + p);
                            if (col_p >= 0) triplets_thr[0].emplace_back(row_rot, col_p, sqrt_lr * A_[r][q]);
                            // s==q
                            int col_q = colA_c(f, j, r*3 + q);
                            if (col_q >= 0) triplets_thr[0].emplace_back(row_rot, col_q, sqrt_lr * A_[r][p]);
                        }
                        ++row_rot;
                    }
                }
            }
        }

        // ---- Temporal regularization
        int row_tmp = row_temporal_begin;
        for (int f=2; f<F; ++f) {
            for (int j=0;j<G;++j) if (active_node[f-1][j] && active_node[f][j]) {
                for (int m=0;m<9;++m){ int c1=colA_c(f,j,m), c0=colA_c(f-1,j,m); Fvec[row_tmp]=sqrt_ltp*(Xfull[f][12*j+m]-Xfull[f-1][12*j+m]); if (c1>=0) triplets_thr[0].emplace_back(row_tmp,c1,sqrt_ltp); if (c0>=0) triplets_thr[0].emplace_back(row_tmp,c0,-sqrt_ltp); ++row_tmp; }
                for (int m=0;m<3;++m){ int c1=colt_c(f,j,m), c0=colt_c(f-1,j,m); Fvec[row_tmp]=sqrt_ltp*(Xfull[f][12*j+9+m]-Xfull[f-1][12*j+9+m]); if (c1>=0) triplets_thr[0].emplace_back(row_tmp,c1,sqrt_ltp); if (c0>=0) triplets_thr[0].emplace_back(row_tmp,c0,-sqrt_ltp); ++row_tmp; }
            }
        }

        // ---- Build J
        std::vector<Eigen::Triplet<double>> triplets;
        size_t nnz_est = 0; for (auto& v: triplets_thr) nnz_est += v.size(); triplets.reserve(nnz_est);
        for (auto& v: triplets_thr) { triplets.insert(triplets.end(), v.begin(), v.end()); std::vector<Eigen::Triplet<double>>().swap(v);}        

        Eigen::SparseMatrix<double> J(total_rows, stateDimCompact);
        J.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::VectorXd Fv = Eigen::Map<Eigen::VectorXd>(Fvec.data(), (int)Fvec.size());

        const double cost = 0.5 * Fv.squaredNorm();

        // ---- Solve normal equations with light LM damping
        Eigen::SparseMatrix<double> JT = J.transpose();
        Eigen::SparseMatrix<double> H  = JT * J;
        Eigen::VectorXd b = -JT * Fv;

        if (H.rows()>0) {
            Eigen::VectorXd d = H.diagonal();
            double scale = (d.size()>0)? d.cwiseAbs().mean() : 1.0;
            double lm = std::max(1e-12, 1e-6 * std::max(1.0, scale));
            H.diagonal().array() += lm;
        }

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver; solver.compute(H);
        if (solver.info()!=Eigen::Success){ std::cerr << "[Optimizer] LDLT factorization failed.\n"; break; }
        Eigen::VectorXd dx = solver.solve(b); if (solver.info()!=Eigen::Success){ std::cerr << "[Optimizer] Linear solve failed.\n"; break; }

        // ---- Apply update (f>=1)
        for (int f=1; f<F; ++f) {
            const int ofs = offsEDc(f);
            for (int j=0;j<G;++j) if (active_node[f][j]) {
                const int ci = compact_idx[f][j];
                const int base = ofs + 12*ci;
                for (int c=0;c<9;++c)  Xfull[f][12*j + c]     += dx[base + c];
                for (int c=0;c<3;++c)  Xfull[f][12*j + 9 + c] += dx[base + 9 + c];
            }
        }

        std::cout << "[GN it="<<it<<"] cost="<<cost<<", |dx|="<<dx.norm() << std::endl;
        if (std::abs(cost - prev_cost) < 1e-6) break; prev_cost = cost;
    }

    // Export one frame state back to edGraph (for downstream use)
    if (F>1) edGraph.updateFromStateVector(Xfull[1], /*offset=*/0);
}