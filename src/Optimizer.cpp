#include "Optimizer.h"
#include "CostFunctions.h"
#include "MeshModel.h"
#include "EDGraph.h"

#include <iostream>
#include <limits>
#include <cmath>
#include <numeric>
#include <unordered_set>

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <omp.h>

namespace {
inline double sqr(double v){ return v*v; }
static const double kEps = 1e-8;

// FOV-only test (no occlusion). Uses CURRENT mesh vertex positions only (no EDGraph),
// and matches the coordinate convention used in ProjectiveICPError::Evaluate.
inline bool inFOVOnly(const MeshModel::Vertex& v,
                      const Eigen::Matrix3d& K,
                      const Eigen::Matrix3d& R_wc,
                      const Eigen::Vector3d& t_wc,
                      int W, int H,
                      int border = 2)
{
    const Eigen::Vector3d pw(v.x, v.y, v.z);              // use already-deformed mesh vertices
    const Eigen::Vector3d pc = R_wc.transpose() * (pw - t_wc);
    if (pc.z() <= 0.0) return false;
    const double invz = 1.0 / pc.z();
    const double u   = K(0,0) * (pc.x()*invz) + K(0,2);
    const double vpx = K(1,1) * (pc.y()*invz) + K(1,2);
    return !(u < border || u >= W - border || vpx < border || vpx >= H - border);
}
}

Optimizer::Optimizer(double w_data,
                     int maxStages,
                     int maxIterations,
                     double lambda_smooth,
                     double lambda_rot,
                     double lambda_temporal)
    : w_data_(w_data),
      lambda_smooth_(lambda_smooth),
      lambda_rot_(lambda_rot),
      lambda_temporal_(lambda_temporal),
      maxStages_(maxStages),
      maxIterations_(maxIterations) {}

void Optimizer::optimize(
    MeshModel& mesh,                                 // we sync deformed verts & recompute normals inside
    const std::vector<cv::Mat>& observed_images,     // depth frames (CV_32F, in mm, NaN invalid)
    const Eigen::Matrix3d& K,
    const std::vector<Eigen::Matrix4d>& camera_poses_gt,
    EDGraph& edGraph)
{
    auto& mesh_vertices  = mesh.getVertices();
    const int F = (int)observed_images.size();
    const int N = (int)mesh_vertices.size();
    const int G = edGraph.numNodes();
    if (F <= 1 || N == 0 || G == 0) {
        std::cout << "[Optimizer] Nothing to optimize (need >=2 frames, non-empty mesh/graph)." << std::endl;
        return;
    }

    // === Per-frame ED state vectors (12*G each). Frame 0 is fixed; we optimize f>=1 ===
    const int edDimPerFrame = 12 * G;
    std::vector<Eigen::VectorXd> Xfull(F, Eigen::VectorXd::Zero(edDimPerFrame));
    for (int f = 0; f < F; ++f) edGraph.writeToStateVector(Xfull[f], /*offset=*/0);

    // Bindings & edges from EDGraph
    const auto& bindings = edGraph.getBindings(); // [N] -> node ids
    const auto& edges    = edGraph.getEdges();    // pair<node_i,node_j>

    // Active node bookkeeping per frame (refreshed every iteration)
    std::vector<std::vector<char>> active_node(F, std::vector<char>(G, 0));
    std::vector<std::vector<std::pair<int,int>>> active_edges(F);
    std::vector<std::vector<int>> compact_idx(F, std::vector<int>(G, -1));
    std::vector<int> Sf(F, 0); // number of active nodes per frame

    // Row/column index helpers (based on active layout)
    auto offsEDc = [&](int f)->int{ int ofs=0; for (int k=0; k< f; ++k) ofs += 12 * Sf[k]; return ofs; };
    auto colA_c  = [&](int f,int node,int k)->int{ int ci = compact_idx[f][node]; if (ci < 0) return -1; return offsEDc(f) + 12*ci + k; };
    auto colt_c  = [&](int f,int node,int k)->int{ int ci = compact_idx[f][node]; if (ci < 0) return -1; return offsEDc(f) + 12*ci + 9 + k; };

    // ==== Gauss-Newton loop ====
    double prev_cost = std::numeric_limits<double>::max();
    int inner_no_improve = 0;

    for (int it = 0; it < maxIterations_; ++it) {
        // 0) (Optional) Sync a reference state — we keep Xfull authoritative per-frame
        for (int f = 0; f < F; ++f) edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);

        // 1) Build FOV lists per frame (no occlusion). No edGraph dependency here.
        std::vector<std::vector<int>> fov_vertices(F);
        #pragma omp parallel for schedule(static)
        for (int f = 0; f < F; ++f) {
            const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
            const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);
            const cv::Mat& depth = observed_images[f];
            std::vector<int> vis; vis.reserve(N/2);
            #pragma omp parallel
            {
                std::vector<int> local; local.reserve(256);
                #pragma omp for nowait
                for (int i = 0; i < N; ++i) {
                    if (inFOVOnly(mesh_vertices[i], K, R, t, depth.cols, depth.rows))
                        local.push_back(i);
                }
                #pragma omp critical
                { vis.insert(vis.end(), local.begin(), local.end()); }
            }
            fov_vertices[f].swap(vis);
        }

        // 2) (Re)build active nodes/edges/compact map based on FOV for f>=1
        for (int f = 0; f < F; ++f) std::fill(active_node[f].begin(), active_node[f].end(), 0);
        for (int f = 1; f < F; ++f) {
            for (int vid : fov_vertices[f]) {
                const auto& b = bindings[vid];
                for (int nid : b) active_node[f][nid] = 1;
            }
            int acc = 0; std::fill(compact_idx[f].begin(), compact_idx[f].end(), -1);
            for (int j = 0; j < G; ++j) if (active_node[f][j]) compact_idx[f][j] = acc++;
            Sf[f] = acc;
            std::vector<std::pair<int,int>> Ef; Ef.reserve(edges.size());
            for (const auto& e : edges) if (active_node[f][e.first] && active_node[f][e.second]) Ef.push_back(e);
            active_edges[f].swap(Ef);
        }

        // 3) Row layout
        std::vector<int> data_row_ofs(F, 0);
        int total_data_rows = 0;
        for (int f = 1; f < F; ++f) { data_row_ofs[f] = total_data_rows; total_data_rows += (int)fov_vertices[f].size(); }

        int smooth_rows = 0, rot_rows = 0;
        for (int f = 1; f < F; ++f) { smooth_rows += (int)active_edges[f].size() * (9 + 3); rot_rows += Sf[f] * 9; }

        int temporal_rows = 0;
        for (int f = 2; f < F; ++f)
            for (int j = 0; j < G; ++j) if (active_node[f-1][j] && active_node[f][j]) temporal_rows += 12;

        const int row_smooth_begin   = total_data_rows;
        const int row_rot_begin      = row_smooth_begin + smooth_rows;
        const int row_temporal_begin = row_rot_begin    + rot_rows;
        const int total_rows         = row_temporal_begin + temporal_rows;

        int edDimCompact = 0; for (int f = 0; f < F; ++f) edDimCompact += 12 * Sf[f];
        const int stateDimCompact = edDimCompact;

        std::cout << "[Layout it=" << it << "] data=" << total_data_rows
                  << ", smooth=" << smooth_rows
                  << ", rot=" << rot_rows
                  << ", temporal=" << temporal_rows
                  << ", total=" << total_rows
                  << "; edDim=" << stateDimCompact << std::endl;

        // 4) Assemble residuals/Jacobians
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

        // ---- DATA TERM (Projective ICP, f >= 1) ----
        // Frame loop is SERIAL (thread-safe). Parallelize over vertices inside each frame.
        for (int f = 1; f < F; ++f) {
            edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);  // safe: serial per-frame
            const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
            const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);
            const cv::Mat& depth = observed_images[f];

            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                auto& Tlocal = triplets_thr[tid];
                #pragma omp for schedule(static)
                for (int idx = 0; idx < (int)fov_vertices[f].size(); ++idx) {
                    const int i = fov_vertices[f][idx];
                    const int row = data_row_ofs[f] + idx;  // fixed row per index (no contention)

                    const auto& v = mesh_vertices[i];
                    Eigen::Vector3d n_w(v.nx, v.ny, v.nz);

                    ProjectiveICPError cost(
                        v, i,
                        K(0,0), K(1,1), K(0,2), K(1,2),
                        depth,
                        sqrt_w, &edGraph,
                        n_w,
                        /*depth_gate=*/0.0);

                    double residual = 0.0; Eigen::VectorXd J_ed(12 * G); J_ed.setZero();
                    if (!cost.Evaluate(residual, &J_ed, R, t)) continue; // keep row allocated but zero (already 0)

                    Fvec[row] = residual;
                    const auto& b = bindings[i];
                    for (int nid : b) {
                        const int base_src = 12 * nid; // in J_ed
                        // rotation 9
                        for (int c = 0; c < 9; ++c) {
                            const double val = J_ed[base_src + c]; if (val == 0.0) continue;
                            const int col = colA_c(f, nid, c); if (col >= 0) Tlocal.emplace_back(row, col, val);
                        }
                        // translation 3
                        for (int c = 0; c < 3; ++c) {
                            const double val = J_ed[base_src + 9 + c]; if (val == 0.0) continue;
                            const int col = colt_c(f, nid, c); if (col >= 0) Tlocal.emplace_back(row, col, val);
                        }
                    }
                }
            }
        }

        // ---- SMOOTHNESS REG (per frame f>=1, only active edges) ----
        int row_ptr = row_smooth_begin;
        for (int f = 1; f < F; ++f) {
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

        // ---- ROTATION ORTHOGONALITY REG (per frame f>=1) ----
        int row_rot_ptr = row_rot_begin;
        for (int f = 1; f < F; ++f) {
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

        // ---- TEMPORAL REG BETWEEN f-1 and f (when both have variables) ----
        int row_temp_ptr = row_temporal_begin;
        for (int f = 2; f < F; ++f) {
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

        // 5) Build J and solve normal equations (with LM damping)
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
        if (ldlt.info() != Eigen::Success) { std::cout << "[Optimizer] LDLT factorization failed (info=" << (int)ldlt.info() << ")" << std::endl; break; }
        Eigen::VectorXd delta = ldlt.solve(b);
        if (ldlt.info() != Eigen::Success) { std::cout << "[Optimizer] LDLT solve failed (info=" << (int)ldlt.info() << ")" << std::endl; break; }

        // 6) Apply update back to per-frame ED states (only f>=1)
        for (int f = 1; f < F; ++f) {
            const int ofs = offsEDc(f);
            for (int j = 0; j < G; ++j) if (active_node[f][j]) {
                const int ci = compact_idx[f][j];
                const int base_src = ofs + 12*ci;
                for (int c = 0; c < 9; ++c)  Xfull[f][12*j + c]     += delta[base_src + c];
                for (int c = 0; c < 3; ++c)  Xfull[f][12*j + 9 + c] += delta[base_src + 9 + c];
            }
        }

        // 7) Report & stopping criteria
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
        if (inner_no_improve >= 3) { std::cout << "Early stop at iter " << it << std::endl; }

        // 8) Sync deformed vertices into MeshModel and recompute normals for next iteration
        if (F > 1) edGraph.updateFromStateVector(Xfull[1], /*offset=*/0); // choose frame-1 as reference
        mesh.updateVerticesByED(edGraph);
        mesh.computeVertexNormals();

        if (inner_no_improve >= 3) break;
    }

    // Export frame 1 state back to EDGraph (typical 2-frame use); change if you prefer a different frame.
    if (F > 1) edGraph.updateFromStateVector(Xfull[1], /*offset=*/0);
}