#include "Optimizer.h"
#include "CostFunctions.h"
#include "Projection.h"
#include "ImageProcessor.h"
#include <Eigen/Sparse>
#include <iostream>
#include <omp.h>

Optimizer::Optimizer(double weight, int maxStages, int maxIterations,
                     double lambda_smooth,
                     double lambda_temp,
                     double lambda_rigid,
                     int edge_knn)
    : weight_(weight), maxStages_(maxStages), maxIterations_(maxIterations),
      lambda_smooth_(lambda_smooth), lambda_temp_(lambda_temp), lambda_rigid_(lambda_rigid),
      edge_knn_(edge_knn) {}

Sophus::SE3d Optimizer::mat4ToSE3(const Eigen::Matrix4d& Tm) {
    Eigen::Matrix3d R = Tm.block<3,3>(0,0);
    Eigen::Vector3d t = Tm.block<3,1>(0,3);
    return Sophus::SE3d(R, t);
}

cv::Mat Optimizer::ensureGrayFloat(const cv::Mat& img) {
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }
    gray.convertTo(gray, CV_32F, 1.0/255.0);
    return gray;
}

void Optimizer::initializeIntensityByAverage(
    const std::vector<MeshModel::Vertex>& vertices,
    const Eigen::Matrix3d& K,
    const std::vector<cv::Mat>& images_gray,
    const std::vector<Sophus::SE3d>& poses,
    std::vector<double>& I
) const {
    const int V = static_cast<int>(vertices.size());
    const int F = static_cast<int>(images_gray.size());
    I.assign(V, 0.0);
    std::vector<int> cnt(V, 0);

    for (int j = 0; j < F; ++j) {
        const auto& img = images_gray[j];
        const int W = img.cols, H = img.rows;
        const Eigen::Matrix3d& R = poses[j].rotationMatrix();
        const Eigen::Vector3d& t = poses[j].translation();
        for (int vi = 0; vi < V; ++vi) {
            Eigen::Vector3d x(vertices[vi].x, vertices[vi].y, vertices[vi].z);
            // project with pose (no ED in init)
            Eigen::Vector3d Pc = R.transpose() * (x - t);
            if (Pc.z() <= 1e-6) continue;
            double u = K(0,0)*Pc.x()/Pc.z() + K(0,2);
            double v = K(1,1)*Pc.y()/Pc.z() + K(1,2);
            if (u < 0 || u >= W || v < 0 || v >= H) continue;
            float Iuv = ImageProcessor::getBilinearInterpolatedIntensity(img, u, v);
            I[vi] += static_cast<double>(Iuv);
            cnt[vi]++;
        }
    }
    for (int vi = 0; vi < V; ++vi) if (cnt[vi] > 0) I[vi] /= cnt[vi];
}

std::vector<std::pair<int,int>> Optimizer::buildNodeEdges(const std::vector<DeformationNode>& nodes, int k) {
    std::vector<std::pair<int,int>> edges;
    const int N = static_cast<int>(nodes.size());
    if (N == 0 || k <= 0) return edges;
    std::vector<std::pair<double,int>> heap; heap.reserve(N);
    for (int i = 0; i < N; ++i) {
        heap.clear();
        for (int j = 0; j < N; ++j) if (i != j) {
            double d2 = (nodes[i].position - nodes[j].position).squaredNorm();
            heap.emplace_back(d2, j);
        }
        int kk = std::min(k, N-1);
        std::nth_element(heap.begin(), heap.begin()+kk, heap.end());
        for (int m = 0; m < kk; ++m) {
            int j = heap[m].second;
            if (i < j) edges.emplace_back(i,j); // undirected unique
        }
    }
    return edges;
}

void Optimizer::optimize(
    const std::vector<MeshModel::Vertex>& mesh_vertices,
    const std::vector<MeshModel::Triangle>& mesh_triangles,
    const Eigen::Matrix3d& camera_intrinsics,
    const std::vector<cv::Mat>& observed_images,
    const std::vector<Eigen::Matrix4d>& camera_poses,
    EDGraph& edGraph,
    std::vector<EDState>& ed_states_per_frame,
    std::vector<double>& vertex_intensity
) {
    const int F = static_cast<int>(observed_images.size());
    const int V = static_cast<int>(mesh_vertices.size());
    const int G = edGraph.numNodes();

    // 1) Prepare gray images and poses (SE3, but fixed)
    std::vector<cv::Mat> images_gray; images_gray.reserve(F);
    for (const auto& img : observed_images) images_gray.push_back(ensureGrayFloat(img));
    std::vector<Sophus::SE3d> poses(F);
    for (int j = 0; j < F; ++j) poses[j] = mat4ToSE3(camera_poses[j]);

    // 2) Init per-frame ED states to identity
    ed_states_per_frame.assign(F, EDState());
    for (int j = 0; j < F; ++j) ed_states_per_frame[j].resize(G);

    // 3) Init vertex intensity as multi-view average (ED = I)
    initializeIntensityByAverage(mesh_vertices, camera_intrinsics, images_gray, poses, vertex_intensity);

    // 4) State layout: [ frame0(12G), frame1(12G), ..., frameF-1(12G), intensity(V) ]
    const int edBlock = 12 * G;
    const int edDim   = edBlock * F;
    const int intDim  = V;
    const int stateDim = edDim + intDim;

    auto edOffset = [&](int frame){ return frame * edBlock; };
    auto intOffset = [&](int v){ return edDim + v; };

    Eigen::VectorXd X = Eigen::VectorXd::Zero(stateDim);
    for (int j = 0; j < F; ++j) {
        edGraph.writeStateToVector(ed_states_per_frame[j], X, edOffset(j));
    }
    for (int vi = 0; vi < V; ++vi) X(intOffset(vi)) = vertex_intensity[vi];

    // Build a BVH per frame (strict occlusion per frame)
    std::vector<BVH> bvhs; bvhs.reserve(F);
    for (int j = 0; j < F; ++j) bvhs.emplace_back(mesh_triangles, mesh_vertices);

    // Precompute node edges for ARAP-like smoothness
    auto edges = buildNodeEdges(edGraph.getGraphNodes(), edge_knn_);

    double prev_stage_cost = std::numeric_limits<double>::infinity();
    int outer_no_improve = 0;

    for (int stage = 0; stage < maxStages_; ++stage) {
        // 4.1) Refit BVH for each frame using that frame's current ED state
        for (int j = 0; j < F; ++j) {
            std::vector<MeshModel::Vertex> def_vertices(V);
            #pragma omp parallel for
            for (int vi = 0; vi < V; ++vi) {
                Eigen::Vector3d xd = edGraph.deformVertex(mesh_vertices[vi], vi, ed_states_per_frame[j]);
                def_vertices[vi].x = static_cast<float>(xd.x());
                def_vertices[vi].y = static_cast<float>(xd.y());
                def_vertices[vi].z = static_cast<float>(xd.z());
            }
            bvhs[j].refit(def_vertices);
        }

        // 4.2) Visibility table per frame
        std::vector<std::vector<char>> visible(V, std::vector<char>(F, 0));
        std::vector<int> residuals_per_vertex(V, 0);

        #pragma omp parallel for
        for (int vi = 0; vi < V; ++vi) {
            for (int j = 0; j < F; ++j) {
                const auto& T = poses[j];
                Eigen::Matrix3d R = T.rotationMatrix();
                Eigen::Vector3d t = T.translation();
                const cv::Mat& img = images_gray[j];
                if (Projection::isVertexVisible(mesh_vertices[vi], camera_intrinsics, R, t,
                                                bvhs[j], img.cols, img.rows, vi, &edGraph)) {
                    visible[vi][j] = 1;
                    #pragma omp atomic
                    residuals_per_vertex[vi]++;
                }
            }
        }

        // 4.3) Row offsets and regularizer row budget
        std::vector<int> rowOffset(V, 0);
        int total_rows = 0;
        for (int vi = 0; vi < V; ++vi) { rowOffset[vi] = total_rows; total_rows += residuals_per_vertex[vi]; }

        // --- Regularization row budget ---
        // ARAP-like smoothness: per edge and per frame, 12 residuals (9 for A, 3 for b)
        const int num_edges = static_cast<int>(edges.size());
        const int arap_rows  = F * num_edges * 12;
        // Temporal consistency: per node, between consecutive frames, 12 residuals
        const int temp_rows  = (F > 1 ? (F-1) * G * 12 : 0);
        // Near-rigid: per node per frame, 9 residuals on (A - I)
        const int rigid_rows = F * G * 9;
        const int reg_rows_total = arap_rows + temp_rows + rigid_rows;
        const int data_rows_total = total_rows;
        total_rows += reg_rows_total;

        double prev_cost = std::numeric_limits<double>::infinity();
        int inner_no_improve = 0;

        for (int iter = 0; iter < maxIterations_; ++iter) {
            std::vector<double> residuals(total_rows, 0.0);
            int nthreads = omp_get_max_threads();
            std::vector<std::vector<Eigen::Triplet<double>>> triplets_per_thread(nthreads);

            // --- Data term ---
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto& Tvec = triplets_per_thread[tid];

                #pragma omp for
                for (int vi = 0; vi < V; ++vi) {
                    if (residuals_per_vertex[vi] == 0) continue;
                    int row = rowOffset[vi];
                    for (int j = 0; j < F; ++j) {
                        if (!visible[vi][j]) continue;

                        // Construct photometric residual for (vertex vi, frame j)
                        PhotometricError cost(
                            mesh_vertices[vi], vi, mesh_triangles,
                            camera_intrinsics, images_gray[j], bvhs[j],
                            weight_, &edGraph, &ed_states_per_frame[j]
                        );

                        double r = 0.0;
                        const Sophus::SE3d& T = poses[j];
                        Eigen::Matrix<double,6,1> se3 = T.log();
                        double I_v = X(intOffset(vi));
                        // We do NOT optimize pose => jacobian_pose = nullptr
                        double JI = 0.0; Eigen::VectorXd JED; // size = 12*G
                        cost.Evaluate(se3, I_v, r, nullptr, &JI, &JED);
                        residuals[row] = r;

                        // Fill Jacobians: ED block for frame j, and intensity column
                        const int edCol0 = edOffset(j);
                        for (int k = 0; k < JED.size(); ++k) {
                            if (JED(k) != 0.0)
                                Tvec.emplace_back(row, edCol0 + k, JED(k));
                        }
                        if (JI != 0.0) Tvec.emplace_back(row, intOffset(vi), JI);
                        row++;
                    }
                }
            }

            // --- Regularizers ---
            int reg_row = data_rows_total; // start after data rows
            const double sw_arap = std::sqrt(std::max(0.0, lambda_smooth_));
            const double sw_temp = std::sqrt(std::max(0.0, lambda_temp_));
            const double sw_rig  = std::sqrt(std::max(0.0, lambda_rigid_));

            // ARAP-like smoothness: for each frame and edge (i,j), penalize element-wise differences on A and b
            {
                auto& Tvec = triplets_per_thread[0]; // single-thread assemble; cheap vs data term
                for (int j = 0; j < F; ++j) {
                    const int edCol0 = edOffset(j);
                    for (const auto& e : edges) {
                        int i = e.first, l = e.second;
                        // A diff (9)
                        for (int q = 0; q < 9; ++q) {
                            residuals[reg_row] = 0.0;
                            Tvec.emplace_back(reg_row, edCol0 + 12*i + q,  sw_arap);
                            Tvec.emplace_back(reg_row, edCol0 + 12*l + q, -sw_arap);
                            reg_row++;
                        }
                        // b diff (3)
                        for (int q = 0; q < 3; ++q) {
                            residuals[reg_row] = 0.0;
                            Tvec.emplace_back(reg_row, edCol0 + 12*i + 9 + q,  sw_arap);
                            Tvec.emplace_back(reg_row, edCol0 + 12*l + 9 + q, -sw_arap);
                            reg_row++;
                        }
                    }
                }
            }

            // Temporal consistency: between frames j and j-1, per node, element-wise diffs on A and b
            {
                auto& Tvec = triplets_per_thread[0];
                for (int j = 1; j < F; ++j) {
                    const int col_prev = edOffset(j-1);
                    const int col_cur  = edOffset(j);
                    for (int i = 0; i < G; ++i) {
                        for (int q = 0; q < 9; ++q) {
                            residuals[reg_row] = 0.0;
                            Tvec.emplace_back(reg_row, col_cur  + 12*i + q,  sw_temp);
                            Tvec.emplace_back(reg_row, col_prev + 12*i + q, -sw_temp);
                            reg_row++;
                        }
                        for (int q = 0; q < 3; ++q) {
                            residuals[reg_row] = 0.0;
                            Tvec.emplace_back(reg_row, col_cur  + 12*i + 9 + q,  sw_temp);
                            Tvec.emplace_back(reg_row, col_prev + 12*i + 9 + q, -sw_temp);
                            reg_row++;
                        }
                    }
                }
            }

            // Near-rigid prior on A: penalize A - I (9 scalars), small weight
            {
                auto& Tvec = triplets_per_thread[0];
                for (int j = 0; j < F; ++j) {
                    const int edCol0 = edOffset(j);
                    for (int i = 0; i < G; ++i) {
                        for (int q = 0; q < 9; ++q) {
                            double target = (q==0 || q==4 || q==8) ? 1.0 : 0.0; // I3 in column-major vec
                            residuals[reg_row] = sw_rig * (-target);
                            Tvec.emplace_back(reg_row, edCol0 + 12*i + q, sw_rig);
                            reg_row++;
                        }
                    }
                }
            }

            // Assemble and solve
            std::vector<Eigen::Triplet<double>> triplets;
            for (auto& vec : triplets_per_thread) triplets.insert(triplets.end(), vec.begin(), vec.end());

            Eigen::VectorXd Fvec = Eigen::Map<Eigen::VectorXd>(residuals.data(), residuals.size());
            Eigen::SparseMatrix<double> J(total_rows, stateDim);
            J.setFromTriplets(triplets.begin(), triplets.end());

            double cost_val = Fvec.squaredNorm();
            Eigen::SparseMatrix<double> H = J.transpose() * J;
            Eigen::VectorXd g = -J.transpose() * Fvec;

            // LM damping (simple)
            const double lambda = 1e-6;
            H += lambda * Eigen::MatrixXd::Identity(H.rows(), H.cols()).sparseView();

            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
            solver.compute(H);
            if (solver.info() != Eigen::Success) {
                std::cerr << "[Warn] Solver factorization failed. Skipping iter." << std::endl;
                break;
            }
            Eigen::VectorXd dx = solver.solve(g);
            if (solver.info() != Eigen::Success) {
                std::cerr << "[Warn] Solver solve failed. Skipping iter." << std::endl;
                break;
            }

            // Update X
            X += dx;

            // Unpack back to states for next iteration
            for (int j = 0; j < F; ++j) {
                edGraph.readStateFromVector(X, edOffset(j), ed_states_per_frame[j]);
            }
            for (int vi = 0; vi < V; ++vi) vertex_intensity[vi] = X(intOffset(vi));

            double dnorm = dx.norm();
            double dcost = std::abs(prev_cost - cost_val);
            std::cout << "[Stage " << stage << " Iter " << iter << "] cost=" << cost_val
                      << ", |dx|=" << dnorm << ", dcost=" << dcost << std::endl;

            if (dnorm < 1e-6 || dcost < 1e-6) inner_no_improve++;
            else inner_no_improve = 0;
            prev_cost = cost_val;
            if (inner_no_improve >= 3) {
                std::cout << "Early stop (inner) at iter " << iter << std::endl;
                break;
            }
        }

        double stage_cost = std::isfinite(prev_cost) ? prev_cost : 0.0;
        double stage_drop = std::abs(prev_stage_cost - stage_cost);
        if (stage_drop < 1e-6) outer_no_improve++; else outer_no_improve = 0;
        prev_stage_cost = stage_cost;
        if (outer_no_improve >= 3) {
            std::cout << "Early stop (outer) at stage " << stage << std::endl;
            break;
        }
    }
}