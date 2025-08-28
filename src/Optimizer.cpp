#include "Optimizer.h"
#include "CostFunctions.h"
#include "Projection.h"
#include "BVH.h"

#include <iostream>
#include <limits>
#include <unordered_set>
#include <cmath>

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <omp.h>

namespace {
inline double sqr(double v){ return v*v; }
static const double kEps = 1e-8;

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

Optimizer::Optimizer(double w_data,
                     int maxStages,
                     int maxIterations,
                     double lambda_smooth,
                     double lambda_rot)
    : w_data_(w_data),
      lambda_smooth_(lambda_smooth),
      lambda_rot_(lambda_rot),
      maxStages_(maxStages),
      maxIterations_(maxIterations) {}

void Optimizer::optimize(
    const std::vector<MeshModel::Vertex>& mesh_vertices,
    const std::vector<MeshModel::Triangle>& mesh_triangles,
    const Eigen::Matrix3d& K,
    const std::vector<cv::Mat>& observed_images,
    const std::vector<Eigen::Matrix4d>& camera_poses_gt,
    EDGraph& edGraph) {

    const size_t F = observed_images.size();
    const size_t N = mesh_vertices.size();
    const int G = edGraph.numNodes();

    if (F <= 1 || N == 0 || G == 0) {
        std::cout << "[Optimizer] Nothing to optimize (need >=2 frames, non‑empty mesh/graph)." << std::endl;
        return;
    }

    // ---- grayscale images in [0,1]
    std::vector<cv::Mat> imgs_gray; imgs_gray.reserve(F);
    for (const auto& img : observed_images) {
        cv::Mat gray;
        if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); else gray = img;
        gray.convertTo(gray, CV_32F, 1.0/255.0);
        imgs_gray.push_back(gray);
    }

    // =========================
    // Full per‑frame ED blocks (12*G) for evaluation/BVH; frame 0 will remain fixed (identity unless caller set otherwise).
    // =========================
    const int edDimPerFrameFull = 12 * G;
    std::vector<Eigen::VectorXd> Xfull(F, Eigen::VectorXd::Zero(edDimPerFrameFull));
    for (int f = 0; f < (int)F; ++f) edGraph.writeToStateVector(Xfull[f], /*offset=*/0);

    // Precompute bindings/edges
    const auto& bindings = edGraph.getBindings();  // [N] -> vector<int> of node ids
    const auto& edges    = edGraph.getEdges();     // vector<pair<int,int>>

    // =========================
    // (1) Deform per frame + build BVH (using current Xfull[f])
    // =========================
    std::vector<std::vector<MeshModel::Vertex>> Vdef_all(F, std::vector<MeshModel::Vertex>(N));
    std::vector<BVH> bvhs; bvhs.reserve(F);
    for (int f = 0; f < (int)F; ++f) {
        edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);
        auto& Vdef = Vdef_all[f];
        #pragma omp parallel for
        for (int vi = 0; vi < (int)N; ++vi) {
            Eigen::Vector3d p = edGraph.deformVertex(mesh_vertices[vi], vi);
            Vdef[vi].x = (float)p.x();
            Vdef[vi].y = (float)p.y();
            Vdef[vi].z = (float)p.z();
        }
        bvhs.emplace_back(mesh_triangles, Vdef);
    }

    // =========================
    // (2) Visibility per frame (we’ll use f=0 for template sampling, f>=1 for data term)
    // =========================
    std::vector<std::vector<int>> visible_vertices(F);
    for (int f = 0; f < (int)F; ++f) {
        edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);
        const auto& img = imgs_gray[f];
        const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
        const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);

        std::vector<int> vis; vis.reserve(N/2);
        #pragma omp parallel
        {
            std::vector<int> vis_local; vis_local.reserve(256);
            #pragma omp for nowait
            for (int i = 0; i < (int)N; ++i) {
                if (Projection::isVertexVisible(mesh_vertices[i], K, R, t,
                                                bvhs[f], img.cols, img.rows, i, &edGraph)) {
                    vis_local.push_back(i);
                }
            }
            #pragma omp critical
            {
                vis.insert(vis.end(), vis_local.begin(), vis_local.end());
            }
        }
        visible_vertices[f].swap(vis);
    }

    // =========================
    // (2.5) Template intensity from frame 0 ONLY (locked, no variable intensities)
    // =========================
    std::vector<double> I_tmpl(N, std::numeric_limits<double>::quiet_NaN());
    {
        const int f = 0; // template frame
        const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
        const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);
        const cv::Mat& img = imgs_gray[f];
        for (int idx = 0; idx < (int)visible_vertices[f].size(); ++idx) {
            const int i = visible_vertices[f][idx];
            const Eigen::Vector3d pw = edGraph.deformVertex(mesh_vertices[i], i); // (should be identity if Xfull[0] is identity)
            const Eigen::Vector3d pc = R.transpose() * (pw - t);
            if (pc.z() <= 1e-8) continue;
            float uf = (float)(K(0,0) * (pc.x()/pc.z()) + K(0,2));
            float vf = (float)(K(1,1) * (pc.y()/pc.z()) + K(1,2));
            I_tmpl[i] = (double)bilinearSample(img, uf, vf);
        }
        const int cnt_tmpl = (int)std::count_if(I_tmpl.begin(), I_tmpl.end(), [](double x){return std::isfinite(x);});
        std::cout << "[Init] template intensity from frame 0: assigned for "
                  << cnt_tmpl << " / " << N << " vertices." << std::endl;
    }

    // =========================
    // (3) Active node/edge sets per frame + compact mapping (FRAME 0 EXCLUDED)
    // =========================
    std::vector<std::vector<char>> active_node(F, std::vector<char>(G, 0));
    std::vector<std::vector<std::pair<int,int>>> active_edges(F);
    std::vector<std::vector<int>> compact_idx(F, std::vector<int>(G, -1));
    std::vector<int> Sf(F, 0);

    for (int f = 1; f < (int)F; ++f) { // NOTE: start from 1 — frame 0 has no variables
        for (int vid : visible_vertices[f]) {
            // Only vertices that have a valid template intensity contribute data; use them to activate nodes
            if (!std::isfinite(I_tmpl[vid])) continue;
            const auto& b = bindings[vid];
            for (int nid : b) active_node[f][nid] = 1;
        }
        int acc = 0;
        for (int j = 0; j < G; ++j) if (active_node[f][j]) compact_idx[f][j] = acc++;
        Sf[f] = acc;
        std::vector<std::pair<int,int>> Ef; Ef.reserve(edges.size());
        for (const auto& e : edges) if (active_node[f][e.first] && active_node[f][e.second]) Ef.push_back(e);
        active_edges[f].swap(Ef);
    }

    auto offsEDc = [&](int f)->int{ int ofs=0; for (int k=0; k< f; ++k) ofs += 12 * Sf[k]; return ofs; };
    auto colA_c  = [&](int f,int node,int k)->int{ int ci = compact_idx[f][node]; if (ci < 0) return -1; return offsEDc(f) + 12*ci + k; };
    auto colt_c  = [&](int f,int node,int k)->int{ int ci = compact_idx[f][node]; if (ci < 0) return -1; return offsEDc(f) + 12*ci + 9 + k; };

    // =========================
    // (4) Row layout  — data from f>=1 only; regs only where variables exist
    // =========================
    std::vector<int> data_row_ofs(F, 0);
    int total_data_rows = 0;
    data_row_ofs[0] = 0; // no data for frame 0
    for (int f = 1; f < (int)F; ++f) { data_row_ofs[f] = total_data_rows; total_data_rows += (int)visible_vertices[f].size(); }

    int smooth_rows = 0, rot_rows = 0;
    for (int f = 1; f < (int)F; ++f) { smooth_rows += (int)active_edges[f].size() * (9 + 3); rot_rows += Sf[f] * 9; }

    int temporal_rows = 0;
    for (int f = 2; f < (int)F; ++f) // need both frames to have variables
        for (int j = 0; j < G; ++j) if (active_node[f-1][j] && active_node[f][j]) temporal_rows += 12;

    const int row_smooth_begin   = total_data_rows;
    const int row_rot_begin      = row_smooth_begin + smooth_rows;
    const int row_temporal_begin = row_rot_begin    + rot_rows;
    const int total_rows         = row_temporal_begin + temporal_rows;

    int edDimCompact = 0; for (int f = 0; f < (int)F; ++f) edDimCompact += 12 * Sf[f];
    const int stateDimCompact = edDimCompact; // NO intensity variables

    std::cout << "[Layout] residual counts  "
              << "data="     << total_data_rows
              << ", smooth=" << smooth_rows
              << ", rot="    << rot_rows
              << ", temporal="<< temporal_rows
              << ", total="  << total_rows << std::endl;
    std::cout << "[Layout] state dims  edCompact=" << edDimCompact << std::endl;

    // =========================
    // (5) Gauss‑Newton (SimplicialLDLT on normal equations)
    // =========================
    double prev_cost = std::numeric_limits<double>::max();
    int inner_no_improve = 0;

    for (int it = 0; it < maxIterations_; ++it) {
        std::vector<double> Fvec(total_rows, 0.0);
        const double sqrt_w   = std::sqrt(std::max(0.0, w_data_));
        const double sqrt_ls  = std::sqrt(std::max(0.0, lambda_smooth_));
        const double sqrt_lr  = std::sqrt(std::max(0.0, lambda_rot_));
        const double sqrt_ltp = std::sqrt(std::max(0.0, lambda_temporal_));

        int num_threads = omp_get_max_threads();
        std::vector<std::vector<Eigen::Triplet<double>>> triplets_thr(num_threads);
        int Kbind = (bindings.empty() ? 0 : (int)bindings[0].size());
        size_t per_thread_est = (size_t)std::max(1, total_data_rows / std::max(1,num_threads)) * (size_t)(12 * std::max(1, Kbind));
        for (auto& v : triplets_thr) v.reserve(per_thread_est);

        // ---- DATA TERM (f >= 1) ----
        #pragma omp parallel for schedule(static)
        for (int f = 1; f < (int)F; ++f) {
            int tid = omp_get_thread_num();
            auto& Tlocal = triplets_thr[tid];

            edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);
            const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
            const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);
            const cv::Mat& img = imgs_gray[f];
            const BVH& bvh = bvhs[f];

            int r = data_row_ofs[f];
            for (int idx = 0; idx < (int)visible_vertices[f].size(); ++idx) {
                const int i = visible_vertices[f][idx];
                if (!std::isfinite(I_tmpl[i])) { ++r; continue; } // no template — skip (row kept as zero)

                PhotometricError cost(mesh_vertices[i], i, mesh_triangles, K, img, bvh, sqrt_w, &edGraph);
                double residual = 0.0; Eigen::VectorXd J_ed(12 * G); J_ed.setZero();
                const double intensity_i = I_tmpl[i];
                cost.Evaluate(intensity_i, residual, /*jacobian_intensity=*/nullptr, &J_ed, R, t);

                Fvec[r] = residual;
                const auto& b = bindings[i];
                for (int nid : b) {
                    int base_src = 12 * nid; // in J_ed
                    for (int c = 0; c < 9; ++c) {
                        const double v = J_ed[base_src + c]; if (v == 0.0) continue;
                        int col = colA_c(f, nid, c); if (col >= 0) Tlocal.emplace_back(r, col, v);
                    }
                    for (int c = 0; c < 3; ++c) {
                        const double v = J_ed[base_src + 9 + c]; if (v == 0.0) continue;
                        int col = colt_c(f, nid, c); if (col >= 0) Tlocal.emplace_back(r, col, v);
                    }
                }
                ++r;
            }
        }

        // ---- SMOOTH REG (per frame f>=1, only active edges)
        int row_ptr = row_smooth_begin;
        for (int f = 1; f < (int)F; ++f) {
            for (const auto& e : active_edges[f]) {
                const int i = e.first, j = e.second;
                for (int m = 0; m < 9; ++m) {
                    const int ci = colA_c(f,i,m), cj = colA_c(f,j,m);
                    Fvec[row_ptr] = sqrt_ls * ( Xfull[f][12*i + m] - Xfull[f][12*j + m] );
                    if (ci >= 0) triplets_thr[0].emplace_back(row_ptr, ci,  sqrt_ls);
                    if (cj >= 0) triplets_thr[0].emplace_back(row_ptr, cj, -sqrt_ls);
                    ++row_ptr;
                }
                for (int m = 0; m < 3; ++m) {
                    const int ci = colt_c(f,i,m), cj = colt_c(f,j,m);
                    Fvec[row_ptr] = sqrt_ls * ( Xfull[f][12*i + 9 + m] - Xfull[f][12*j + 9 + m] );
                    if (ci >= 0) triplets_thr[0].emplace_back(row_ptr, ci,  sqrt_ls);
                    if (cj >= 0) triplets_thr[0].emplace_back(row_ptr, cj, -sqrt_ls);
                    ++row_ptr;
                }
            }
        }

        // ---- ROTATION REG (per frame f>=1)
        int row_rot_ptr = row_rot_begin;
        for (int f = 1; f < (int)F; ++f) {
            for (int j = 0; j < G; ++j) if (active_node[f][j]) {
                Eigen::Matrix3d A; for (int k=0;k<3;++k) for (int l=0;l<3;++l) A(k,l) = Xfull[f][12*j + 3*k + l];
                Eigen::Matrix3d C = A.transpose()*A - Eigen::Matrix3d::Identity();
                for (int k=0;k<3;++k) for (int l=0;l<3;++l) Fvec[row_rot_ptr + 3*k + l] = sqrt_lr * C(k,l);
                for (int i = 0; i < 3; ++i) {
                    for (int jcol = 0; jcol < 3; ++jcol) {
                        const int col_idx = colA_c(f, j, 3*i + jcol); if (col_idx < 0) continue;
                        for (int k = 0; k < 3; ++k) {
                            for (int l = 0; l < 3; ++l) {
                                double d = 0.0; if (jcol == k) d += A(i,l); if (jcol == l) d += A(i,k);
                                if (d != 0.0) triplets_thr[0].emplace_back(row_rot_ptr + 3*k + l, col_idx, sqrt_lr * d);
                            }
                        }
                    }
                }
                row_rot_ptr += 9;
            }
        }

        // ---- TEMPORAL REG (between f-1 and f), only when both frames have variables
        int row_temp_ptr = row_temporal_begin;
        for (int f = 2; f < (int)F; ++f) {
            for (int j = 0; j < G; ++j) if (active_node[f-1][j] && active_node[f][j]) {
                for (int c = 0; c < 9; ++c) {
                    const int c1 = colA_c(f,  j,c); const int c0 = colA_c(f-1,j,c);
                    Fvec[row_temp_ptr] = sqrt_ltp * ( Xfull[f][12*j + c] - Xfull[f-1][12*j + c] );
                    if (c1 >= 0) triplets_thr[0].emplace_back(row_temp_ptr, c1,  sqrt_ltp);
                    if (c0 >= 0) triplets_thr[0].emplace_back(row_temp_ptr, c0, -sqrt_ltp);
                    ++row_temp_ptr;
                }
                for (int c = 0; c < 3; ++c) {
                    const int c1 = colt_c(f,  j,c); const int c0 = colt_c(f-1,j,c);
                    Fvec[row_temp_ptr] = sqrt_ltp * ( Xfull[f][12*j + 9 + c] - Xfull[f-1][12*j + 9 + c] );
                    if (c1 >= 0) triplets_thr[0].emplace_back(row_temp_ptr, c1,  sqrt_ltp);
                    if (c0 >= 0) triplets_thr[0].emplace_back(row_temp_ptr, c0, -sqrt_ltp);
                    ++row_temp_ptr;
                }
            }
        }

        // ---- Gather to J (compact)
        size_t est_data_nnz = (size_t)total_data_rows * (size_t)(12 * std::max(1, (int)(bindings.empty()?0:bindings[0].size())));
        size_t est_smooth_nnz = (size_t)smooth_rows * 2;
        size_t est_rot_nnz = (size_t)rot_rows * 6;
        size_t est_temp_nnz = (size_t)temporal_rows * 2;
        size_t nnz_est = est_data_nnz + est_smooth_nnz + est_rot_nnz + est_temp_nnz;

        std::vector<Eigen::Triplet<double>> triplets; triplets.reserve(nnz_est);
        for (auto& v : triplets_thr) {
            triplets.insert(triplets.end(), v.begin(), v.end());
            std::vector<Eigen::Triplet<double>>().swap(v);
        }

        Eigen::VectorXd Fv = Eigen::Map<Eigen::VectorXd>(Fvec.data(), (int)Fvec.size());
        Eigen::SparseMatrix<double> J(total_rows, stateDimCompact);
        J.setFromTriplets(triplets.begin(), triplets.end());
        std::vector<Eigen::Triplet<double>>().swap(triplets);

        const double cost = Fv.squaredNorm();

        // ========================= Solve Hx = b with SimplicialLDLT =========================
        Eigen::SparseMatrix<double> JT = J.transpose();
        Eigen::VectorXd b = -JT * Fv;
        Eigen::SparseMatrix<double> H = JT * J;

        J.resize(0,0); JT.makeCompressed(); JT.resize(0,0);

        double lm = 1e-6;
        if (H.rows() > 0) {
            Eigen::VectorXd diag = H.diagonal();
            double scale = 1.0; if (diag.size() > 0) { double m = diag.cwiseAbs().maxCoeff(); if (std::isfinite(m) && m > 0) scale = m; }
            lm *= scale;
            H.diagonal().array() += lm;
        }

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>> ldlt;
        ldlt.compute(H);
        if (ldlt.info() != Eigen::Success) {
            std::cout << "[Optimizer] LDLT factorization failed (info=" << (int)ldlt.info() << ")" << std::endl;
            break;
        }
        Eigen::VectorXd delta = ldlt.solve(b);
        if (ldlt.info() != Eigen::Success) {
            std::cout << "[Optimizer] LDLT solve failed (info=" << (int)ldlt.info() << ")" << std::endl;
            break;
        }

        // ---- Apply update back to full ED blocks (only f>=1)
        for (int f = 1; f < (int)F; ++f) {
            const int ofs = offsEDc(f);
            for (int j = 0; j < G; ++j) if (active_node[f][j]) {
                const int ci = compact_idx[f][j];
                const int base_src = ofs + 12*ci;
                for (int c = 0; c < 9; ++c)  Xfull[f][12*j + c]     += delta[base_src + c];
                for (int c = 0; c < 3; ++c)  Xfull[f][12*j + 9 + c] += delta[base_src + 9 + c];
            }
        }

        const double dnorm = delta.norm();
        const double dcost = std::abs(prev_cost - cost);
        std::cout << "[GN it " << it
                  << "] cost=" << cost
                  << ", |delta|=" << dnorm
                  << ", dcost=" << dcost
                  << ", solver=LDLT(lm=" << std::scientific << lm << ")" << std::defaultfloat
                  << std::endl;

        if (dnorm < 1e-6 || dcost < 1e-6) ++inner_no_improve; else inner_no_improve = 0;
        prev_cost = cost;
        if (inner_no_improve >= 3) {
            std::cout << "Early stop (inner) at iter " << it << std::endl;
            break;
        }
    }

    // NOTE: For compatibility with previous behavior, export frame 0 back to edGraph.
    // If you prefer exporting frame 1 (the optimized frame in a 2‑frame setup), change index below to 1.
    if (F > 0) edGraph.updateFromStateVector(Xfull[0], /*offset=*/0);
}