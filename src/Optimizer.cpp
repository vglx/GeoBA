#include "Optimizer.h"
#include "CostFunctions.h"
#include "Projection.h"
#include "BVH.h"

#include <iostream>
#include <limits>
#include <unordered_set>
#include <cmath>
#include <numeric>

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

// =====================================================================================
//  New behavior:
//  - Frame 0 remains FIXED (no ED columns), but NOW contributes DATA residual rows.
//  - Per-vertex photometric map intensity I_i is added as a variable (one scalar / vertex).
//  - I_i initialization uses mean/median from frame0 & frame1 (when visible),
//    falling back to single-frame if only one is visible.
//  - A soft prior E_prior = lambda_I * (I_i - I_prior)^2 is added to prevent degeneration.
//  - ED variables are still only built for frames f>=1 (active nodes only).
// =====================================================================================

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
    // Per-frame ED full state (12*G) for evaluation/BVH; frame 0 will remain fixed (identity unless set otherwise).
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
    // (2) Visibility per frame (we’ll now use f=0 ALSO for data rows; f>=1 as before)
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
    // (2.5) Map intensity initialization (I_init) and prior (I_prior) from frame0 & frame1
    // =========================
    std::vector<double> I_init(N, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> I_prior(N, std::numeric_limits<double>::quiet_NaN());
    const double w0 = 0.5, w1 = 0.5; // simple mean; you may switch to median if needed
    {
        // f=0
        const int f0 = 0;
        const Eigen::Matrix3d R0 = camera_poses_gt[f0].block<3,3>(0,0);
        const Eigen::Vector3d t0 = camera_poses_gt[f0].block<3,1>(0,3);
        const cv::Mat& img0 = imgs_gray[f0];
        // f=1 (exists because F>=2 guaranteed above)
        const int f1 = 1;
        const Eigen::Matrix3d R1 = camera_poses_gt[f1].block<3,3>(0,0);
        const Eigen::Vector3d t1 = camera_poses_gt[f1].block<3,1>(0,3);
        const cv::Mat& img1 = imgs_gray[f1];

        std::vector<char> vis0(N, 0), vis1(N, 0);
        for (int idx = 0; idx < (int)visible_vertices[f0].size(); ++idx) vis0[visible_vertices[f0][idx]] = 1;
        for (int idx = 0; idx < (int)visible_vertices[f1].size(); ++idx) vis1[visible_vertices[f1][idx]] = 1;

        #pragma omp parallel for
        for (int i = 0; i < (int)N; ++i) {
            double v0 = std::numeric_limits<double>::quiet_NaN();
            double v1 = std::numeric_limits<double>::quiet_NaN();

            if (vis0[i]) {
                const Eigen::Vector3d pw = edGraph.deformVertex(mesh_vertices[i], i);
                const Eigen::Vector3d pc = R0.transpose() * (pw - t0);
                if (pc.z() > 1e-8) {
                    float uf = (float)(K(0,0) * (pc.x()/pc.z()) + K(0,2));
                    float vf = (float)(K(1,1) * (pc.y()/pc.z()) + K(1,2));
                    v0 = (double)bilinearSample(img0, uf, vf);
                }
            }
            if (vis1[i]) {
                const Eigen::Vector3d pw = edGraph.deformVertex(mesh_vertices[i], i);
                const Eigen::Vector3d pc = R1.transpose() * (pw - t1);
                if (pc.z() > 1e-8) {
                    float uf = (float)(K(0,0) * (pc.x()/pc.z()) + K(0,2));
                    float vf = (float)(K(1,1) * (pc.y()/pc.z()) + K(1,2));
                    v1 = (double)bilinearSample(img1, uf, vf);
                }
            }

            bool f0ok = std::isfinite(v0);
            bool f1ok = std::isfinite(v1);
            if (f0ok && f1ok) {
                I_init[i]  = w0*v0 + w1*v1;
                I_prior[i] = I_init[i]; // use the same as prior target by default
            } else if (f0ok) {
                I_init[i]  = v0;
                I_prior[i] = v0;
            } else if (f1ok) {
                I_init[i]  = v1;
                I_prior[i] = v1;
            } else {
                // leave as NaN (no data); will be skipped unless you want pure prior
            }
        }
    }

    // =========================
    // (3) Active node/edge sets per frame + compact mapping (FRAME 0 EXCLUDED for ED columns)
    //     Activation uses visibility AND having a finite I_prior (to ensure meaningful residuals)
    // =========================
    std::vector<std::vector<char>> active_node(F, std::vector<char>(G, 0));
    std::vector<std::vector<std::pair<int,int>>> active_edges(F);
    std::vector<std::vector<int>> compact_idx(F, std::vector<int>(G, -1));
    std::vector<int> Sf(F, 0);

    for (int f = 1; f < (int)F; ++f) { // NOTE: start from 1 — frame 0 has no ED variables
        for (int vid : visible_vertices[f]) {
            if (!std::isfinite(I_prior[vid])) continue; // only vertices with a valid map prior
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
    // (3.5) Decide which vertices get an intensity variable (use_intensity)
    //       Criterion: appears in at least one data row (f where visible) OR has a finite prior.
    // =========================
    std::vector<char> use_intensity(N, 0);
    for (int f = 0; f < (int)F; ++f) {
        for (int vid : visible_vertices[f]) {
            if (std::isfinite(I_prior[vid])) use_intensity[vid] = 1; // data row will be formed
        }
    }
    for (int i = 0; i < (int)N; ++i) if (std::isfinite(I_prior[i])) use_intensity[i] = 1; // prior-only vertices

    // build compact indices for intensity variables
    std::vector<int> colI(N, -1);
    int Icount = 0; for (int i = 0; i < (int)N; ++i) if (use_intensity[i]) colI[i] = Icount++;

    // current map variables (compact order)
    std::vector<double> I_var(Icount, 0.0);
    for (int i = 0; i < (int)N; ++i) if (colI[i] >= 0) {
        I_var[colI[i]] = std::isfinite(I_init[i]) ? I_init[i] : 0.0; // fallback to 0 if unknown
    }

    // =========================
    // (4) Row layout  — data from f>=0; regs (smooth/rot) where ED variables exist; prior for intensities
    // =========================
    std::vector<int> data_row_ofs(F, 0);
    int total_data_rows = 0;
    for (int f = 0; f < (int)F; ++f) {
        data_row_ofs[f] = total_data_rows;
        // count only vertices that both visible in frame f AND have an intensity variable
        int add = 0; for (int vid : visible_vertices[f]) if (colI[vid] >= 0) ++add;
        total_data_rows += add;
    }

    int smooth_rows = 0, rot_rows = 0;
    for (int f = 1; f < (int)F; ++f) { smooth_rows += (int)active_edges[f].size() * (9 + 3); rot_rows += Sf[f] * 9; }

    int temporal_rows = 0;
    for (int f = 2; f < (int)F; ++f) // need both frames to have variables
        for (int j = 0; j < G; ++j) if (active_node[f-1][j] && active_node[f][j]) temporal_rows += 12;

    // intensity prior rows (one per vertex that has a finite I_prior and a variable)
    int Iprior_rows = 0; for (int i = 0; i < (int)N; ++i) if (colI[i] >= 0 && std::isfinite(I_prior[i])) ++Iprior_rows;

    const int row_smooth_begin   = total_data_rows;
    const int row_rot_begin      = row_smooth_begin + smooth_rows;
    const int row_temporal_begin = row_rot_begin    + rot_rows;
    const int row_Iprior_begin   = row_temporal_begin + temporal_rows;
    const int total_rows         = row_Iprior_begin + Iprior_rows;

    int edDimCompact = 0; for (int f = 0; f < (int)F; ++f) edDimCompact += 12 * Sf[f];
    const int intensDim          = Icount;
    const int stateDimCompact    = edDimCompact + intensDim; // ED (f>=1) + map intensities

    std::cout << "[Layout] residual counts  "
              << "data="     << total_data_rows
              << ", smooth=" << smooth_rows
              << ", rot="    << rot_rows
              << ", temporal="<< temporal_rows
              << ", Iprior=" << Iprior_rows
              << ", total="  << total_rows << std::endl;
    std::cout << "[Layout] state dims  edCompact=" << edDimCompact
              << ", intens=" << intensDim
              << ", total="  << stateDimCompact << std::endl;

    // =========================
    // (5) Gauss‑Newton (SimplicialLDLT on normal equations)
    // =========================
    // Prior weight for intensity (you may expose as CLI/constructor param later)
    const double lambda_I = 0; // heuristic start; tune as needed

    double prev_cost = std::numeric_limits<double>::max();

    for (int it = 0; it < maxIterations_; ++it) {
        std::vector<double> Fvec(total_rows, 0.0);
        const double sqrt_w   = std::sqrt(std::max(0.0, w_data_));
        const double sqrt_ls  = std::sqrt(std::max(0.0, lambda_smooth_));
        const double sqrt_lr  = std::sqrt(std::max(0.0, lambda_rot_));
        const double sqrt_ltp = std::sqrt(std::max(0.0, lambda_temporal_));
        const double sqrt_lI  = std::sqrt(std::max(0.0, lambda_I));

        int num_threads = omp_get_max_threads();
        std::vector<std::vector<Eigen::Triplet<double>>> triplets_thr(num_threads);
        for (auto& v : triplets_thr) v.reserve((size_t)std::max(1, total_data_rows / std::max(1,num_threads)) * 48);

        // ---- DATA TERM (f >= 0; f=0 has NO ED columns) ----
        #pragma omp parallel for schedule(static)
        for (int f = 0; f < (int)F; ++f) {
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
                const int ci = colI[i];
                if (ci < 0) { ++r; continue; } // this vertex has no intensity variable — skipped in counting actually

                const double Icurr = I_var[ci];
                PhotometricError cost(mesh_vertices[i], i, mesh_triangles, K, img, bvh, sqrt_w, &edGraph);
                double residual = 0.0; 
                double J_I = 0.0; // jacobian wrt intensity
                Eigen::VectorXd J_ed(12 * G); J_ed.setZero();

                // Evaluate uses current map intensity; internally clamps sampling & applies Huber
                cost.Evaluate(Icurr, residual, /*jacobian_intensity=*/&J_I,
                              /*jacobian_ed=*/ (f==0 ? nullptr : &J_ed),
                              R, t);

                // residual row value
                Fvec[r] = residual;

                // write intensity column
                if (J_I != 0.0) {
                    const int global_col = edDimCompact + ci; // intensity block placed after ED block
                    Tlocal.emplace_back(r, global_col, J_I);
                }

                // write ED columns (ONLY for f>=1)
                if (f >= 1) {
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

        // ---- ROT ORTHO REG (per-node, per frame f>=1)
        for (int f = 1; f < (int)F; ++f) {
            for (int j = 0; j < G; ++j) if (active_node[f][j]) {
                // Penalize non-orthonormality of A (9 params) using simple L2 to identity rows
                // Here: three rows individually towards [1 0 0], [0 1 0], [0 0 1]
                for (int k = 0; k < 9; ++k) {
                    const int col = colA_c(f, j, k);
                    const double target = (k==0||k==4||k==8) ? 1.0 : 0.0;
                    Fvec[row_ptr] = sqrt_lr * ( Xfull[f][12*j + k] - target );
                    if (col >= 0) triplets_thr[0].emplace_back(row_ptr, col, sqrt_lr);
                    ++row_ptr;
                }
            }
        }

        // ---- TEMPORAL REG (tie same node across consecutive frames where active)
        for (int f = 2; f < (int)F; ++f) {
            for (int j = 0; j < G; ++j) if (active_node[f-1][j] && active_node[f][j]) {
                // penalize difference in 12 params
                for (int m = 0; m < 12; ++m) {
                    const int c1 = (m < 9) ? colA_c(f-1,j,m) : colt_c(f-1,j,m-9);
                    const int c2 = (m < 9) ? colA_c(f  ,j,m) : colt_c(f  ,j,m-9);
                    Fvec[row_ptr] = sqrt_ltp * ( Xfull[f][12*j + m] - Xfull[f-1][12*j + m] );
                    if (c2 >= 0) triplets_thr[0].emplace_back(row_ptr, c2,  sqrt_ltp);
                    if (c1 >= 0) triplets_thr[0].emplace_back(row_ptr, c1, -sqrt_ltp);
                    ++row_ptr;
                }
            }
        }

        // ---- INTENSITY PRIOR (per vertex with prior)
        for (int i = 0, rI = row_Iprior_begin; i < (int)N; ++i) if (colI[i] >= 0 && std::isfinite(I_prior[i])) {
            const int ci = colI[i];
            Fvec[rI] = sqrt_lI * ( I_var[ci] - I_prior[i] );
            const int global_col = edDimCompact + ci;
            triplets_thr[0].emplace_back(rI, global_col, sqrt_lI);
            ++rI;
        }

        // ---- Assemble J and solve normal equations
        std::vector<Eigen::Triplet<double>> triplets;
        size_t total_triplets = 0; for (auto& t : triplets_thr) total_triplets += t.size();
        triplets.reserve(total_triplets);
        for (auto& t : triplets_thr) { triplets.insert(triplets.end(), t.begin(), t.end()); }

        Eigen::SparseMatrix<double> J(total_rows, stateDimCompact);
        J.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::VectorXd Fv(total_rows);
        for (int r = 0; r < total_rows; ++r) Fv[r] = Fvec[r];

        const double cost = 0.5 * Fv.squaredNorm();
        std::cout << "[GN it=" << it << "] cost(raw)=" << cost << std::endl;
        if (cost > prev_cost * (1.0 - 1e-9)) {
            // tiny/no improvement -> stop (you can add LM damping here if desired)
            break;
        }
        prev_cost = cost;

        Eigen::SparseMatrix<double> At = J.transpose();
        Eigen::SparseMatrix<double> AtA = At * J;
        Eigen::VectorXd Atb = -At * Fv;

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(AtA);
        if (solver.info() != Eigen::Success) {
            std::cerr << "[Optimizer] LDLT factorization failed." << std::endl;
            break;
        }
        Eigen::VectorXd dx = solver.solve(Atb);
        if (solver.info() != Eigen::Success) {
            std::cerr << "[Optimizer] Linear solve failed." << std::endl;
            break;
        }

        // ---- Apply update: ED (f>=1) + intensity
        // intensities
        for (int i = 0; i < (int)N; ++i) if (colI[i] >= 0) {
            const int global_col = edDimCompact + colI[i];
            I_var[colI[i]] += dx[global_col];
        }
        // ED
        for (int f = 1; f < (int)F; ++f) {
            for (int j = 0; j < G; ++j) if (active_node[f][j]) {
                const int base = 12*j;
                for (int k = 0; k < 9; ++k) {
                    const int col = colA_c(f,j,k);
                    if (col >= 0) Xfull[f][base + k] += dx[col];
                }
                for (int k = 0; k < 3; ++k) {
                    const int col = colt_c(f,j,k);
                    if (col >= 0) Xfull[f][base + 9 + k] += dx[col];
                }
            }
        }
    }

    // Optionally: write back final ED for caller or export; map intensities I_var if you want to keep them
    // (You can extend Optimizer API to return I_var and/or bake it to MeshModel::Vertex if needed.)
}