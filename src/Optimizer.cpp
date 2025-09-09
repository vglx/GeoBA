#include "Optimizer.h"
#include "CostFunctions.h"   // ProjectiveICPError
#include <iostream>
#include <limits>
#include <cmath>
#include <omp.h>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

namespace {
// Deform all vertices with current ED state and compute per-vertex world-space normals
// NOTE: use per-thread local accumulators to avoid undefined behavior with atomics on Eigen scalars.
static void warpVerticesAndComputeNormals(
    const std::vector<MeshModel::Vertex>& V_raw,
    const std::vector<MeshModel::Triangle>& F,
    EDGraph& ed,
    std::vector<Eigen::Vector3d>& Vw,
    std::vector<Eigen::Vector3d>& Nw)
{
    const int N = (int)V_raw.size();
    Vw.resize(N);
    Nw.assign(N, Eigen::Vector3d::Zero());

    // 1) Deform (thread-safe)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        Vw[i] = ed.deformVertex(V_raw[i], i);
    }

    // 2) Thread-local normal accumulation to avoid atomics on Eigen scalars
    const int T = omp_get_max_threads();
    std::vector<std::vector<Eigen::Vector3d>> Nlocal(T, std::vector<Eigen::Vector3d>(N, Eigen::Vector3d::Zero()));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& NL = Nlocal[tid];

        #pragma omp for schedule(static)
        for (int k = 0; k < (int)F.size(); ++k) {
            const auto& tri = F[k];
            const Eigen::Vector3d& p0 = Vw[tri.v0];
            const Eigen::Vector3d& p1 = Vw[tri.v1];
            const Eigen::Vector3d& p2 = Vw[tri.v2];
            Eigen::Vector3d n = (p1 - p0).cross(p2 - p0);
            double ln = n.norm(); if (ln > 1e-20) n /= ln; else n = Eigen::Vector3d(0,0,1);
            NL[tri.v0] += n;
            NL[tri.v1] += n;
            NL[tri.v2] += n;
        }
    }

    // 3) Merge
    for (int t = 0; t < T; ++t) {
        const auto& NL = Nlocal[t];
        for (int i = 0; i < N; ++i) Nw[i] += NL[i];
    }

    // 4) Normalize
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        double nrm = Nw[i].norm();
        if (nrm > 1e-12) Nw[i] /= nrm; else Nw[i] = Eigen::Vector3d(0,0,1);
    }
}
}

void Optimizer::optimize(
    const std::vector<MeshModel::Vertex>& mesh_vertices,
    const std::vector<MeshModel::Triangle>& mesh_triangles,
    const Eigen::Matrix3d& K,
    const std::vector<cv::Mat>& depth_images,
    const std::vector<Eigen::Matrix4d>& camera_poses_gt,
    EDGraph& edGraph)
{
    const int F = (int)depth_images.size();
    const int N = (int)mesh_vertices.size();
    const int G = edGraph.numNodes();
    if (F <= 1 || N == 0 || G == 0) return;

    // Enforce depth mode
    if (!(depth_images[0].type() == CV_32FC1)) {
        std::cerr << "[Optimizer] ERROR: optimize() expects CV_32F single-channel depth images.\n";
        return;
    }
    std::cout << "[Optimizer] Mode = Projective-ICP (depth)" << std::endl;

    // ED state per frame (full 12*G per frame). Frame 0 will remain fixed (no columns)
    const int edDimPerFrameFull = 12 * G;
    std::vector<Eigen::VectorXd> Xfull(F, Eigen::VectorXd::Zero(edDimPerFrameFull));
    for (int f = 0; f < F; ++f) edGraph.writeToStateVector(Xfull[f], /*offset=*/0);

    const auto& bindings = edGraph.getBindings();
    const auto& edges    = edGraph.getEdges();

    // visibility helper (boundary only: Z>0 & in image)
    auto compute_visibility_boundary = [&](const Eigen::VectorXd& Xf,
                                           const cv::Mat& img_like,
                                           const Eigen::Matrix3d& R,
                                           const Eigen::Vector3d& t,
                                           std::vector<int>& vis_out){
        vis_out.clear(); vis_out.reserve(N/2);
        edGraph.updateFromStateVector(Xf, /*offset=*/0);
        #pragma omp parallel
        {
            std::vector<int> vis_local; vis_local.reserve(256);
            #pragma omp for nowait
            for (int i = 0; i < N; ++i) {
                Eigen::Vector3d pw = edGraph.deformVertex(mesh_vertices[i], i);
                Eigen::Vector3d pc = R.transpose() * (pw - t);
                if (pc.z() > 1e-8) {
                    float u = (float)(K(0,0) * (pc.x()/pc.z()) + K(0,2));
                    float v = (float)(K(1,1) * (pc.y()/pc.z()) + K(1,2));
                    if (u >= 0 && u < img_like.cols && v >= 0 && v < img_like.rows)
                        vis_local.push_back(i);
                }
            }
            #pragma omp critical
            vis_out.insert(vis_out.end(), vis_local.begin(), vis_local.end());
        }
    };

    // Gauss-Newton loop
    double prev_cost = std::numeric_limits<double>::max();
    const double sqrt_w = std::sqrt(std::max(0.0, w_data_));
    for (int it = 0; it < maxIterations_; ++it) {
        // (1) visibility per frame
        std::vector<std::vector<int>> visible_vertices(F);
        for (int f = 0; f < F; ++f) {
            compute_visibility_boundary(Xfull[f], depth_images[f],
                                        camera_poses_gt[f].block<3,3>(0,0),
                                        camera_poses_gt[f].block<3,1>(0,3),
                                        visible_vertices[f]);
        }

        // (2) build compact active sets per frame (f>=1)
        std::vector<std::vector<char>> active_node(F, std::vector<char>(G, 0));
        std::vector<std::vector<std::pair<int,int>>> active_edges(F);
        std::vector<std::vector<int>> compact_idx(F, std::vector<int>(G, -1));
        std::vector<int> Sf(F, 0);
        for (int f = 1; f < F; ++f) {
            for (int vid : visible_vertices[f]) {
                const auto& b = bindings[vid];
                for (int nid : b) active_node[f][nid] = 1;
            }
            int acc = 0; for (int j = 0; j < G; ++j) if (active_node[f][j]) compact_idx[f][j] = acc++;
            Sf[f] = acc;
            std::vector<std::pair<int,int>> Ef; Ef.reserve(edges.size());
            for (const auto& e : edges) if (active_node[f][e.first] && active_node[f][e.second]) Ef.push_back(e);
            active_edges[f].swap(Ef);
        }
        auto offsEDc = [&](int f){ int ofs = 0; for (int k = 0; k < f; ++k) ofs += 12 * Sf[k]; return ofs; };
        auto colA_c  = [&](int f,int node,int k){ int ci = compact_idx[f][node]; if (ci < 0) return -1; return offsEDc(f) + 12*ci + k; };
        auto colt_c  = [&](int f,int node,int k){ int ci = compact_idx[f][node]; if (ci < 0) return -1; return offsEDc(f) + 12*ci + 9 + k; };

        // (3) Recompute normals per frame (world)
        std::vector<std::vector<Eigen::Vector3d>> normals_w_per_frame(F);
        for (int f = 0; f < F; ++f) {
            edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);
            std::vector<Eigen::Vector3d> Vw, Nw;
            warpVerticesAndComputeNormals(mesh_vertices, mesh_triangles, edGraph, Vw, Nw);
            normals_w_per_frame[f].swap(Nw);
        }

        // (4) Row layout
        int total_data_rows = 0;
        std::vector<int> data_row_ofs(F, 0);
        for (int f = 0; f < F; ++f) {
            data_row_ofs[f] = total_data_rows;
            if (f >= 1) total_data_rows += (int)visible_vertices[f].size();
        }
        int smooth_rows = 0, rot_rows = 0, temporal_rows = 0;
        for (int f = 1; f < F; ++f) { smooth_rows += (int)active_edges[f].size() * (9 + 3); rot_rows += Sf[f] * 9; }
        for (int f = 2; f < F; ++f) for (int j = 0; j < G; ++j) if (active_node[f-1][j] && active_node[f][j]) temporal_rows += 12;
        const int row_smooth_begin   = total_data_rows;
        const int row_rot_begin      = row_smooth_begin   + smooth_rows;
        const int row_temporal_begin = row_rot_begin      + rot_rows;
        const int total_rows         = row_temporal_begin + temporal_rows;

        int edDimCompact = 0; for (int f = 0; f < F; ++f) edDimCompact += 12 * Sf[f];
        const int stateDimCompact = edDimCompact; // only ED variables

        std::cout << "[Layout] residual counts  data=" << total_data_rows
                  << ", smooth=" << smooth_rows
                  << ", rot=" << rot_rows
                  << ", temporal=" << temporal_rows
                  << ", total=" << total_rows << std::endl;
        std::cout << "[Layout] state dims  edCompact=" << stateDimCompact << std::endl;

        // (5) Assemble
        const double sqrt_ls  = std::sqrt(std::max(0.0, lambda_smooth_));
        const double sqrt_lr  = std::sqrt(std::max(0.0, lambda_rot_));
        const double sqrt_ltp = std::sqrt(std::max(0.0, lambda_temporal_));

        std::vector<double> Fvec(total_rows, 0.0);
        int num_threads = omp_get_max_threads();
        std::vector<std::vector<Eigen::Triplet<double>>> triplets_thr(num_threads);
        for (auto& v : triplets_thr) v.reserve((size_t)std::max(1,total_data_rows/std::max(1,num_threads))*48);

        // ---- DATA (Projective ICP) ----
        #pragma omp parallel for schedule(static)
        for (int f = 1; f < F; ++f) {
            int tid = omp_get_thread_num(); auto& Tlocal = triplets_thr[tid];
            edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);
            const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
            const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);
            int r = data_row_ofs[f];
            for (int idx = 0; idx < (int)visible_vertices[f].size(); ++idx) {
                const int i = visible_vertices[f][idx];
                ProjectiveICPError cost(mesh_vertices[i], i,
                                        depth_images[f], &edGraph, K,
                                        normals_w_per_frame[f][i],
                                        sqrt_w);
                // cost.setDepthGate(/*optional e.g.*/ 5e-3);
                double residual = 0.0; Eigen::VectorXd J_ed(12*G); J_ed.setZero();
                if (cost.Evaluate(residual, &J_ed, R, t)) {
                    Fvec[r] = residual;
                    const auto& b = bindings[i];
                    for (int nid : b) {
                        const int base = 12 * nid;
                        for (int c = 0; c < 9; ++c) { double v = J_ed[base + c]; if (v == 0.0) continue; int col = colA_c(f, nid, c); if (col >= 0) Tlocal.emplace_back(r, col, v); }
                        for (int c = 0; c < 3; ++c) { double v = J_ed[base + 9 + c]; if (v == 0.0) continue; int col = colt_c(f, nid, c); if (col >= 0) Tlocal.emplace_back(r, col, v); }
                    }
                }
                ++r;
            }
        }

        // ---- SMOOTH ----
        int row_ptr = row_smooth_begin;
        for (int f = 1; f < F; ++f) {
            for (const auto& e : active_edges[f]) {
                const int i = e.first, j = e.second;
                for (int m = 0; m < 9;  ++m) { int ci = colA_c(f,i,m), cj = colA_c(f,j,m); Fvec[row_ptr] = sqrt_ls * (Xfull[f][12*i+m] - Xfull[f][12*j+m]); if (ci>=0) triplets_thr[0].emplace_back(row_ptr,ci,sqrt_ls); if (cj>=0) triplets_thr[0].emplace_back(row_ptr,cj,-sqrt_ls); ++row_ptr; }
                for (int m = 0; m < 3;  ++m) { int ci = colt_c(f,i,m), cj = colt_c(f,j,m); Fvec[row_ptr] = sqrt_ls * (Xfull[f][12*i+9+m] - Xfull[f][12*j+9+m]); if (ci>=0) triplets_thr[0].emplace_back(row_ptr,ci,sqrt_ls); if (cj>=0) triplets_thr[0].emplace_back(row_ptr,cj,-sqrt_ls); ++row_ptr; }
            }
        }

        // ---- ROT (orthogonality prior on A) ----
        for (int f = 1; f < F; ++f) {
            for (int j = 0; j < G; ++j) if (active_node[f][j]) {
                for (int k = 0; k < 9; ++k) { int col = colA_c(f,j,k); double target = (k==0||k==4||k==8)?1.0:0.0; Fvec[row_ptr] = sqrt_lr * (Xfull[f][12*j+k] - target); if (col>=0) triplets_thr[0].emplace_back(row_ptr,col,sqrt_lr); ++row_ptr; }
            }
        }

        // ---- TEMPORAL (optional) ----
        for (int f = 2; f < F; ++f) {
            for (int j = 0; j < G; ++j) if (active_node[f-1][j] && active_node[f][j]) {
                for (int m = 0; m < 12; ++m) {
                    int c1 = (m < 9) ? colA_c(f-1,j,m) : colt_c(f-1,j,m-9);
                    int c2 = (m < 9) ? colA_c(f,  j,m) : colt_c(f,  j,m-9);
                    Fvec[row_ptr] = sqrt_ltp * (Xfull[f][12*j+m] - Xfull[f-1][12*j+m]);
                    if (c2>=0) triplets_thr[0].emplace_back(row_ptr,c2,sqrt_ltp);
                    if (c1>=0) triplets_thr[0].emplace_back(row_ptr,c1,-sqrt_ltp);
                    ++row_ptr;
                }
            }
        }

        // (6) Solve normal equations J^T J dx = - J^T F
        std::vector<Eigen::Triplet<double>> triplets; size_t tot = 0; for (auto& t : triplets_thr) tot += t.size(); triplets.reserve(tot); for (auto& t : triplets_thr) { triplets.insert(triplets.end(), t.begin(), t.end()); }
        Eigen::SparseMatrix<double> J(total_rows, stateDimCompact); J.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::VectorXd Fv(total_rows); for (int r = 0; r < total_rows; ++r) Fv[r] = Fvec[r];
        double cost = 0.5 * Fv.squaredNorm();
        std::cout << "[GN it=" << it << "] cost(raw)=" << cost << std::endl;
        if (cost > prev_cost * (1.0 - 1e-9)) break; prev_cost = cost;
        Eigen::SparseMatrix<double> At = J.transpose();
        Eigen::SparseMatrix<double> AtA = At * J; Eigen::VectorXd Atb = -At * Fv;

        // Simple LM-like damping
        Eigen::VectorXd diagA = AtA.diagonal();
        double mean_abs_diag = (diagA.size() > 0) ? diagA.cwiseAbs().mean() : 1.0;
        double damping = std::max(1e-12, 1e-6 * std::max(1.0, mean_abs_diag));
        AtA.diagonal().array() += damping;
        std::cout << "[GN it=" << it << "] damping=" << damping << std::endl;

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver; solver.compute(AtA);
        if (solver.info() != Eigen::Success) { std::cerr << "[Optimizer] LDLT factorization failed.\n"; break; }
        Eigen::VectorXd dx = solver.solve(Atb); if (solver.info() != Eigen::Success) { std::cerr << "[Optimizer] Linear solve failed.\n"; break; }

        // (7) Apply update: ED (f>=1)
        for (int f = 1; f < F; ++f) {
            for (int j = 0; j < G; ++j) if (active_node[f][j]) {
                int base = 12 * j;
                for (int k = 0; k < 9; ++k)  { int col = colA_c(f,j,k); if (col >= 0) Xfull[f][base + k]     += dx[col]; }
                for (int k = 0; k < 3; ++k)  { int col = colt_c(f,j,k); if (col >= 0) Xfull[f][base + 9 + k] += dx[col]; }
            }
        }
        // Next iter: Xfull carries over for deform & normals & visibility
    }
}