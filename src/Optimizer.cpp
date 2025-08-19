#include "Optimizer.h"
#include "CostFunctions.h"
#include "Projection.h"
#include "BVH.h"
#include <iostream>
#include <limits>
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

    BVH bvh(mesh_triangles, mesh_vertices);

    std::vector<cv::Mat> imgs_gray;
    imgs_gray.reserve(F);
    for (const auto& img : observed_images) {
        cv::Mat gray;
        if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); else gray = img;
        gray.convertTo(gray, CV_32F, 1.0/255.0);
        imgs_gray.push_back(gray);
    }

    const int edDim = 12 * G;
    const int intensityDim = static_cast<int>(N);
    const int stateDim = edDim + intensityDim;

    Eigen::VectorXd X = Eigen::VectorXd::Zero(stateDim);
    edGraph.writeToStateVector(X, 0);

    auto colA = [](int node, int k){ return 12*node + k; };
    auto colt = [](int node, int k){ return 12*node + 9 + k; };

    double prev_stage_cost = std::numeric_limits<double>::max();
    int outer_no_improve = 0;

    for (int stage = 0; stage < maxStages_; ++stage) {
        std::vector<MeshModel::Vertex> Vdef(N);
        #pragma omp parallel for
        for (int vi = 0; vi < (int)N; ++vi) {
            Eigen::Vector3d p = edGraph.deformVertex(mesh_vertices[vi], vi);
            Vdef[vi].x = (float)p.x();
            Vdef[vi].y = (float)p.y();
            Vdef[vi].z = (float)p.z();
        }
        bvh.refit(Vdef);

        std::vector<std::vector<bool>> visible(N, std::vector<bool>(F, false));
        std::vector<int> rows_per_vertex(N, 0);

        #pragma omp parallel for
        for (int i = 0; i < (int)N; ++i) {
            for (size_t j = 0; j < F; ++j) {
                const Eigen::Matrix3d R = camera_poses_gt[j].block<3,3>(0,0);
                const Eigen::Vector3d t = camera_poses_gt[j].block<3,1>(0,3);
                const cv::Mat& img = imgs_gray[j];
                if (Projection::isVertexVisible(mesh_vertices[i], K, R, t,
                                                bvh, img.cols, img.rows, i, &edGraph)) {
                    visible[i][j] = true;
                    #pragma omp atomic
                    rows_per_vertex[i]++;
                }
            }
        }

        std::vector<int> row_ofs(N, 0);
        int total_data_rows = 0;
        for (size_t i = 0; i < N; ++i) { row_ofs[i] = total_data_rows; total_data_rows += rows_per_vertex[i]; }

        const auto& edges = edGraph.getEdges();
        const int smooth_rows = (int)edges.size() * (9 + 3);
        const int rot_rows    = G * 9;
        const int total_rows = total_data_rows + smooth_rows + rot_rows;

        std::cout << "[Stage " << stage << "] residuals: data=" << total_data_rows
                  << ", smooth=" << smooth_rows << ", rot=" << rot_rows
                  << ", total=" << total_rows << std::endl;

        double prev_cost = std::numeric_limits<double>::max();
        int inner_no_improve = 0;

        for (int it = 0; it < maxIterations_; ++it) {
            std::vector<double> Fvec(total_rows, 0.0);
            const double sqrt_w = std::sqrt(std::max(0.0, w_data_));
            const double sqrt_ls = std::sqrt(std::max(0.0, lambda_smooth_));
            const double sqrt_lr = std::sqrt(std::max(0.0, lambda_rot_));

            int num_threads = omp_get_max_threads();
            std::vector<std::vector<Eigen::Triplet<double>>> triplets_thr(num_threads);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto& Tlocal = triplets_thr[tid];

                #pragma omp for
                for (int i = 0; i < (int)N; ++i) {
                    if (rows_per_vertex[i] == 0) continue;
                    int r = row_ofs[i];
                    for (size_t j = 0; j < F; ++j) {
                        if (!visible[i][j]) continue;

                        const Eigen::Matrix3d R = camera_poses_gt[j].block<3,3>(0,0);
                        const Eigen::Vector3d t = camera_poses_gt[j].block<3,1>(0,3);
                        const cv::Mat& img = imgs_gray[j];

                        PhotometricError cost(mesh_vertices[i], i, mesh_triangles, K, img, bvh, sqrt_w, &edGraph);
                        double residual = 0.0;
                        double J_I = 0.0;
                        Eigen::VectorXd J_ed(12 * G); J_ed.setZero();

                        const double intensity_i = X(edDim + i);
                        cost.Evaluate(intensity_i, residual, &J_I, &J_ed, R, t);

                        Fvec[r] = residual;
                        if (std::abs(J_I) > 0.0) Tlocal.emplace_back(r, edDim + i, J_I);
                        for (int c = 0; c < J_ed.size(); ++c) {
                            const double v = J_ed[c];
                            if (v != 0.0) Tlocal.emplace_back(r, c, v);
                        }
                        ++r;
                    }
                }
            }

            int row_smooth_begin = total_data_rows;
            for (size_t e = 0; e < edges.size(); ++e) {
                const int i = edges[e].first;
                const int j = edges[e].second;
                Eigen::Matrix3d Ai, Aj; Eigen::Vector3d ti, tj;
                for (int k=0;k<3;++k) for (int l=0;l<3;++l) { Ai(k,l) = X(colA(i,3*k+l)); Aj(k,l) = X(colA(j,3*k+l)); }
                for (int k=0;k<3;++k) { ti(k) = X(colt(i,k)); tj(k) = X(colt(j,k)); }

                for (int m = 0; m < 9; ++m) {
                    const double r = sqrt_ls * (Ai(m/3, m%3) - Aj(m/3, m%3));
                    Fvec[row_smooth_begin] = r;
                    triplets_thr[0].emplace_back(row_smooth_begin, colA(i, m),  sqrt_ls);
                    triplets_thr[0].emplace_back(row_smooth_begin, colA(j, m), -sqrt_ls);
                    ++row_smooth_begin;
                }
                for (int m = 0; m < 3; ++m) {
                    const double r = sqrt_ls * (ti(m) - tj(m));
                    Fvec[row_smooth_begin] = r;
                    triplets_thr[0].emplace_back(row_smooth_begin, colt(i, m),  sqrt_ls);
                    triplets_thr[0].emplace_back(row_smooth_begin, colt(j, m), -sqrt_ls);
                    ++row_smooth_begin;
                }
            }

            int row_rot_begin = total_data_rows + smooth_rows;
            const double eps = 1e-6;
            for (int j = 0; j < G; ++j) {
                Eigen::Matrix3d A;
                for (int k=0;k<3;++k) for (int l=0;l<3;++l) A(k,l) = X(colA(j,3*k+l));
                const Eigen::Matrix3d C = A.transpose()*A - Eigen::Matrix3d::Identity();

                double r0[9];
                for (int k=0;k<3;++k) for (int l=0;l<3;++l) r0[3*k+l] = sqrt_lr * C(k,l);
                for (int m=0;m<9;++m) Fvec[row_rot_begin + m] = r0[m];

                for (int p = 0; p < 9; ++p) {
                    Eigen::Matrix3d A_pert = A;
                    A_pert(p/3, p%3) += eps;
                    const Eigen::Matrix3d C2 = A_pert.transpose()*A_pert - Eigen::Matrix3d::Identity();
                    for (int m = 0; m < 9; ++m) {
                        const double r1 = sqrt_lr * C2(m/3, m%3);
                        const double dr = (r1 - r0[m]) / eps;
                        triplets_thr[0].emplace_back(row_rot_begin + m, colA(j, p), dr);
                    }
                }
                row_rot_begin += 9;
            }

            std::vector<Eigen::Triplet<double>> triplets;
            for (auto& v : triplets_thr) {
                triplets.insert(triplets.end(), v.begin(), v.end());
            }

            Eigen::VectorXd F = Eigen::Map<Eigen::VectorXd>(Fvec.data(), (int)Fvec.size());
            Eigen::SparseMatrix<double> J(total_rows, stateDim);
            J.setFromTriplets(triplets.begin(), triplets.end());

            const double cost = F.squaredNorm();
            Eigen::SparseMatrix<double> H = J.transpose() * J;
            Eigen::VectorXd g = -J.transpose() * F;

            const double lambda = 1e-6;
            H += lambda * Eigen::MatrixXd::Identity(H.rows(), H.cols()).sparseView();

            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
            solver.compute(H);
            if (solver.info() != Eigen::Success) {
                std::cout << "[Optimizer] LDLT failed." << std::endl;
                break;
            }
            Eigen::VectorXd delta = solver.solve(g);
            if (solver.info() != Eigen::Success) {
                std::cout << "[Optimizer] Solve failed." << std::endl;
                break;
            }

            X += delta;
            edGraph.updateFromStateVector(X, 0);

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
}