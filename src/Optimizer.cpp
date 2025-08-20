#include "Optimizer.h"
#include "CostFunctions.h"
#include "Projection.h"
#include "BVH.h"
#include <iostream>
#include <limits>
#include <unordered_set>
#include <Eigen/Sparse>
#include <omp.h>

namespace {
inline double sqr(double v){ return v*v; }
static const double kEps = 1e-8;
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

    if (F == 0 || N == 0 || G == 0) {
        std::cout << "[Optimizer] Nothing to optimize (empty inputs)." << std::endl;
        return;
    }

    // ---- grayscale images [0,1]
    std::vector<cv::Mat> imgs_gray; imgs_gray.reserve(F);
    for (const auto& img : observed_images) {
        cv::Mat gray;
        if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); else gray = img;
        gray.convertTo(gray, CV_32F, 1.0/255.0);
        imgs_gray.push_back(gray);
    }

    // =========================
    // Full (dense) per-frame ED blocks (12*G), used only for evaluation/BVH
    // Intensities are also stored in a dense vector but will be partially variable.
    // =========================
    const int edDimPerFrameFull = 12 * G;
    std::vector<Eigen::VectorXd> Xfull(F, Eigen::VectorXd::Zero(edDimPerFrameFull));
    for (int f = 0; f < (int)F; ++f) edGraph.writeToStateVector(Xfull[f], /*offset=*/0);
    Eigen::VectorXd Intens = Eigen::VectorXd::Zero((int)N);

    // Precompute bindings once
    const auto& bindings = edGraph.getBindings();  // [N] -> vector<int> of node ids
    const auto& edges    = edGraph.getEdges();     // vector<pair<int,int>>

    double prev_stage_cost = std::numeric_limits<double>::max();
    int outer_no_improve = 0;

    for (int stage = 0; stage < maxStages_; ++stage) {
        // =========================
        // (1) Per-frame deformed meshes + BVH (using current Xfull[f])
        //     IMPORTANT: keep Vdef buffers alive during the whole stage to avoid dangling refs in BVH
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
            bvhs.emplace_back(mesh_triangles, Vdef); // BVH stores refs: safe because Vdef lives for the whole stage
        }

        // =========================
        // (2) Visibility (frame by frame)
        // =========================
        std::vector<std::vector<int>> visible_vertices(F); // per frame list of visible vertex indices
        visible_vertices.assign(F, {});

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
        // (3) Active node/edge sets per frame + compact ED mapping; active intensity set
        // =========================
        std::vector<std::vector<char>> active_node(F, std::vector<char>(G, 0));
        std::vector<std::vector<std::pair<int,int>>> active_edges(F);
        std::vector<std::vector<int>> compact_idx(F, std::vector<int>(G, -1));
        std::vector<int> Sf(F, 0); // active node count per frame

        // intensity activity across all frames this stage
        std::vector<char> int_active(N, 0);

        for (int f = 0; f < (int)F; ++f) {
            // nodes touched by any visible vertex's bindings; mark intensity active too
            for (int vid : visible_vertices[f]) {
                int_active[vid] = 1;
                const auto& b = bindings[vid];
                for (int nid : b) active_node[f][nid] = 1;
            }
            // assign compact indices for ED
            int acc = 0;
            for (int j = 0; j < G; ++j) if (active_node[f][j]) compact_idx[f][j] = acc++;
            Sf[f] = acc;
            // edges whose both ends active
            std::vector<std::pair<int,int>> Ef; Ef.reserve(edges.size());
            for (const auto& e : edges) {
                if (active_node[f][e.first] && active_node[f][e.second]) Ef.push_back(e);
            }
            active_edges[f].swap(Ef);
        }

        // compact column offsets for ED blocks
        auto offsEDc = [&](int f)->int{
            int ofs = 0;
            for (int k = 0; k < f; ++k) ofs += 12 * Sf[k];
            return ofs;
        };
        auto colA_c = [&](int f,int node,int k)->int{
            int ci = compact_idx[f][node];
            if (ci < 0) return -1; // inactive -> no column
            return offsEDc(f) + 12*ci + k; // k in [0..8]
        };
        auto colt_c = [&](int f,int node,int k)->int{
            int ci = compact_idx[f][node];
            if (ci < 0) return -1;
            return offsEDc(f) + 12*ci + 9 + k; // k in [0..2]
        };

        // compact mapping for intensities
        std::vector<int> int_compact_idx(N, -1);
        int N_active = 0;
        for (int i = 0; i < (int)N; ++i) if (int_active[i]) int_compact_idx[i] = N_active++;

        // =========================
        // (4) Row layout
        // =========================
        std::vector<int> data_row_ofs(F, 0);
        int total_data_rows = 0;
        for (int f = 0; f < (int)F; ++f) {
            data_row_ofs[f] = total_data_rows;
            total_data_rows += (int)visible_vertices[f].size();
        }
        int smooth_rows = 0, rot_rows = 0;
        for (int f = 0; f < (int)F; ++f) {
            smooth_rows += (int)active_edges[f].size() * (9 + 3);
            rot_rows    += Sf[f] * 9;
        }
        // temporal rows between f-1 and f (INTERSECTION so both sides exist as variables)
        int temporal_rows = 0;
        for (int f = 1; f < (int)F; ++f) {
            for (int j = 0; j < G; ++j) if (active_node[f-1][j] && active_node[f][j]) temporal_rows += 12;
        }

        const int row_smooth_begin   = total_data_rows;
        const int row_rot_begin      = row_smooth_begin + smooth_rows;
        const int row_temporal_begin = row_rot_begin    + rot_rows;
        const int total_rows         = row_temporal_begin + temporal_rows;

        // compact state dimension = sum_f 12*Sf[f] + N_active (intensities)
        int edDimCompact = 0; for (int f = 0; f < (int)F; ++f) edDimCompact += 12 * Sf[f];
        const int offsInt = edDimCompact; // intensity block offset
        const int stateDimCompact = offsInt + N_active;

        std::cout << "[Stage " << stage << "] residual counts  "
                  << "data="    << total_data_rows
                  << ", smooth=" << smooth_rows
                  << ", rot="    << rot_rows
                  << ", temporal="<< temporal_rows
                  << ", total="  << total_rows << std::endl;
        std::cout << "[Stage " << stage << "] state dims  edCompact=" << edDimCompact
                  << ", intensActive=" << N_active << ", total=" << stateDimCompact << std::endl;

        // =========================
        // (5) Gauss-Newton iterations
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
            for (auto& v : triplets_thr) v.reserve((size_t) (total_data_rows * 24.0 / std::max(1,num_threads)));

            // ---- DATA TERM (frame by frame)
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

                    PhotometricError cost(mesh_vertices[i], i, mesh_triangles, K, img, bvh, sqrt_w, &edGraph);
                    double residual = 0.0;
                    double J_I = 0.0;
                    Eigen::VectorXd J_ed(12 * G);
                    J_ed.setZero();

                    const double intensity_i = Intens(i);
                    cost.Evaluate(intensity_i, residual, &J_I, &J_ed, R, t);

                    Fvec[r] = residual;
                    // intensity column is guaranteed to exist for a visible i in this stage
                    const int icol = offsInt + int_compact_idx[i];
                    Tlocal.emplace_back(r, icol, J_I);

                    // ED columns (only for active bound nodes)
                    const auto& b = bindings[i];
                    for (int nid : b) {
                        int base_src = 12 * nid; // in J_ed
                        // A (9)
                        for (int c = 0; c < 9; ++c) {
                            const double v = J_ed[base_src + c];
                            if (v == 0.0) continue;
                            int col = colA_c(f, nid, c);
                            if (col >= 0) Tlocal.emplace_back(r, col, v);
                        }
                        // t (3)
                        for (int c = 0; c < 3; ++c) {
                            const double v = J_ed[base_src + 9 + c];
                            if (v == 0.0) continue;
                            int col = colt_c(f, nid, c);
                            if (col >= 0) Tlocal.emplace_back(r, col, v);
                        }
                    }
                    ++r;
                }
            }

            // ---- SMOOTH REG  (per frame, only active edges)
            int row_ptr = row_smooth_begin;
            for (int f = 0; f < (int)F; ++f) {
                for (const auto& e : active_edges[f]) {
                    const int i = e.first, j = e.second;

                    // 9 for A diff
                    for (int m = 0; m < 9; ++m) {
                        const int ci = colA_c(f,i,m), cj = colA_c(f,j,m);
                        Fvec[row_ptr] = sqrt_ls * ( Xfull[f][12*i + m] - Xfull[f][12*j + m] );
                        if (ci >= 0) triplets_thr[0].emplace_back(row_ptr, ci,  sqrt_ls);
                        if (cj >= 0) triplets_thr[0].emplace_back(row_ptr, cj, -sqrt_ls);
                        ++row_ptr;
                    }
                    // 3 for t diff
                    for (int m = 0; m < 3; ++m) {
                        const int ci = colt_c(f,i,m), cj = colt_c(f,j,m);
                        Fvec[row_ptr] = sqrt_ls * ( Xfull[f][12*i + 9 + m] - Xfull[f][12*j + 9 + m] );
                        if (ci >= 0) triplets_thr[0].emplace_back(row_ptr, ci,  sqrt_ls);
                        if (cj >= 0) triplets_thr[0].emplace_back(row_ptr, cj, -sqrt_ls);
                        ++row_ptr;
                    }
                }
            }

            // ---- ROTATION REG  (vec(A^T A - I)) — analytic Jacobian, only active nodes
            int row_rot_ptr = row_rot_begin;
            for (int f = 0; f < (int)F; ++f) {
                for (int j = 0; j < G; ++j) if (active_node[f][j]) {
                    // read A_j^f from Xfull
                    Eigen::Matrix3d A;
                    for (int k=0;k<3;++k) for (int l=0;l<3;++l) A(k,l) = Xfull[f][12*j + 3*k + l];
                    Eigen::Matrix3d C = A.transpose()*A - Eigen::Matrix3d::Identity();
                    // residuals
                    for (int k=0;k<3;++k) for (int l=0;l<3;++l) Fvec[row_rot_ptr + 3*k + l] = sqrt_lr * C(k,l);
                    // analytic J
                    for (int i = 0; i < 3; ++i) {
                        for (int jcol = 0; jcol < 3; ++jcol) {
                            const int col_idx = colA_c(f, j, 3*i + jcol);
                            if (col_idx < 0) continue;
                            for (int k = 0; k < 3; ++k) {
                                for (int l = 0; l < 3; ++l) {
                                    double d = 0.0;
                                    if (jcol == k) d += A(i,l);
                                    if (jcol == l) d += A(i,k);
                                    if (d != 0.0) triplets_thr[0].emplace_back(row_rot_ptr + 3*k + l, col_idx, sqrt_lr * d);
                                }
                            }
                        }
                    }
                    row_rot_ptr += 9;
                }
            }

            // ---- TEMPORAL (between f-1 and f), only for nodes active in BOTH frames
            int row_temp_ptr = row_temporal_begin;
            for (int f = 1; f < (int)F; ++f) {
                for (int j = 0; j < G; ++j) if (active_node[f-1][j] && active_node[f][j]) {
                    // A (9)
                    for (int c = 0; c < 9; ++c) {
                        const int c1 = colA_c(f,  j,c);
                        const int c0 = colA_c(f-1,j,c);
                        Fvec[row_temp_ptr] = sqrt_ltp * ( Xfull[f][12*j + c] - Xfull[f-1][12*j + c] );
                        if (c1 >= 0) triplets_thr[0].emplace_back(row_temp_ptr, c1,  sqrt_ltp);
                        if (c0 >= 0) triplets_thr[0].emplace_back(row_temp_ptr, c0, -sqrt_ltp);
                        ++row_temp_ptr;
                    }
                    // t (3)
                    for (int c = 0; c < 3; ++c) {
                        const int c1 = colt_c(f,  j,c);
                        const int c0 = colt_c(f-1,j,c);
                        Fvec[row_temp_ptr] = sqrt_ltp * ( Xfull[f][12*j + 9 + c] - Xfull[f-1][12*j + 9 + c] );
                        if (c1 >= 0) triplets_thr[0].emplace_back(row_temp_ptr, c1,  sqrt_ltp);
                        if (c0 >= 0) triplets_thr[0].emplace_back(row_temp_ptr, c0, -sqrt_ltp);
                        ++row_temp_ptr;
                    }
                }
            }

            // ---- Gather & solve (compact system)
            std::vector<Eigen::Triplet<double>> triplets; triplets.reserve((size_t)total_rows * 24);
            for (auto& v : triplets_thr) triplets.insert(triplets.end(), v.begin(), v.end());

            Eigen::VectorXd Fv = Eigen::Map<Eigen::VectorXd>(Fvec.data(), (int)Fvec.size());
            Eigen::SparseMatrix<double> J(total_rows, stateDimCompact);
            J.setFromTriplets(triplets.begin(), triplets.end());

            const double cost = Fv.squaredNorm();
            Eigen::SparseMatrix<double> H = J.transpose() * J;
            Eigen::VectorXd g = -J.transpose() * Fv;

            const double lambda = 1e-6;
            H += lambda * Eigen::MatrixXd::Identity(H.rows(), H.cols()).sparseView();

            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
            solver.compute(H);
            if (solver.info() != Eigen::Success) { std::cout << "[Optimizer] LDLT failed." << std::endl; break; }
            Eigen::VectorXd delta = solver.solve(g);
            if (solver.info() != Eigen::Success) { std::cout << "[Optimizer] Solve failed." << std::endl; break; }

            // ---- Apply update back to full ED blocks (Xfull) and active Intens
            // ED (per frame)
            for (int f = 0; f < (int)F; ++f) {
                const int ofs = offsEDc(f);
                for (int j = 0; j < G; ++j) if (active_node[f][j]) {
                    const int ci = compact_idx[f][j];
                    const int base_src = ofs + 12*ci;
                    for (int c = 0; c < 9; ++c) Xfull[f][12*j + c]     += delta[base_src + c];
                    for (int c = 0; c < 3; ++c) Xfull[f][12*j + 9 + c] += delta[base_src + 9 + c];
                }
            }
            // intensities (only active ones)
            for (int i = 0; i < (int)N; ++i) if (int_active[i]) {
                const int icol = offsInt + int_compact_idx[i];
                Intens(i) += delta[icol];
            }

            const double dnorm = delta.norm();
            const double dcost = std::abs(prev_cost - cost);
            std::cout << "[Stage " << stage << " | it " << it
                      << "] cost=" << cost
                      << ", |delta|=" << dnorm
                      << ", dcost=" << dcost << std::endl;

            if (dnorm < 1e-6 || dcost < 1e-6) ++inner_no_improve; else inner_no_improve = 0;
            prev_cost = cost;
            if (inner_no_improve >= 3) {
                std::cout << "Early stop (inner) at iter " << it << std::endl;
                break;
            }
        }

        const double stage_cost = prev_cost;
        const double sc = std::abs(prev_stage_cost - stage_cost);
        if (sc < 1e-6) ++outer_no_improve; else outer_no_improve = 0;
        prev_stage_cost = stage_cost;
        if (outer_no_improve >= 3) {
            std::cout << "Early stop (outer) at stage " << stage << std::endl;
            break;
        }
    }

    // Export the final per-frame ED back to edGraph for frame 0 (or leave to caller)
    if (F > 0) {
        edGraph.updateFromStateVector(Xfull[0], /*offset=*/0);
    }
}