#include "Optimizer.h"
#include "CostFunctions.h"
#include "Projection.h"
#include "BVH.h"
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <omp.h>
#include "ImageProcessor.h"

Optimizer::Optimizer(double weight, int maxStages, int maxIterations)
    : weight_(weight), maxStages_(maxStages), maxIterations_(maxIterations) {
}

void Optimizer::optimize(
    const std::vector<MeshModel::Vertex>& mesh_vertices,
    const std::vector<MeshModel::Triangle>& mesh_triangles,
    const Eigen::Matrix3d& camera_intrinsics,
    const std::vector<cv::Mat>& observed_images,
    std::vector<Eigen::Matrix4d>& camera_poses) {

    size_t frame_count = observed_images.size();
    size_t vertex_count = mesh_vertices.size();
    int poseDim = static_cast<int>(frame_count * 6);
    int intensityDim = static_cast<int>(vertex_count);
    int stateDim = poseDim + intensityDim;

    BVH bvh(mesh_triangles, mesh_vertices);

    std::vector<cv::Mat> observed_images_gray;
    for (const auto& img : observed_images) {
        cv::Mat gray;
        if (img.channels() == 3) {
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = img;
        }
        gray.convertTo(gray, CV_32F, 1.0/255.0);
        observed_images_gray.push_back(gray);
    }

    Eigen::VectorXd X = Eigen::VectorXd::Zero(stateDim);
    for (size_t i = 0; i < frame_count; i++) {
        Sophus::SE3d T(camera_poses[i].block<3,3>(0,0), camera_poses[i].block<3,1>(0,3));
        X.segment<6>(i*6) = T.log();
    }

    // const int maxStage = 50;
    // const int maxIter = 10;

    for (int stage = 0; stage < maxStages_; ++stage) {
        // --- 1. 重新构造 visible_table 和光度 x2 ---
        std::vector<std::vector<bool>> visible_table(vertex_count, std::vector<bool>(frame_count, false));
        std::vector<int> residuals_per_vertex(vertex_count, 0);
        std::vector<double> x2_values(vertex_count, 0.0);
        std::vector<int> x2_counts(vertex_count, 0);

        #pragma omp parallel for
        for (size_t i = 0; i < vertex_count; ++i) {
            double sum_intensity = 0.0;
            int count = 0;
            for (size_t j = 0; j < frame_count; ++j) {
                Sophus::SE3d T = Sophus::SE3d::exp(X.segment<6>(j * 6));
                Eigen::Matrix3d R = T.rotationMatrix();
                Eigen::Vector3d t = T.translation();
                const cv::Mat& image = observed_images_gray[j];

                if (Projection::isVertexVisible(mesh_vertices[i], camera_intrinsics, R, t,
                                                bvh, image.cols, image.rows)) {
                    Eigen::Vector2d proj = Projection::projectPoint(mesh_vertices[i], camera_intrinsics, R, t);
                    if (proj(0) >= 0 && proj(0) < image.cols && proj(1) >= 0 && proj(1) < image.rows) {
                        float intensity = ImageProcessor::getBilinearInterpolatedIntensity(image, proj(0), proj(1));
                        sum_intensity += intensity;
                        count++;
                        visible_table[i][j] = true;
                        #pragma omp atomic
                        residuals_per_vertex[i]++;
                    }
                }
            }
            if (count > 0) {
                x2_values[i] = sum_intensity / count;
                x2_counts[i] = count;
            }
        }

        for (size_t i = 0; i < vertex_count; ++i) {
            X(poseDim + i) = x2_values[i];
        }

        std::vector<int> row_offset(vertex_count, 0);
        int total_rows = 0;
        for (size_t i = 0; i < vertex_count; ++i) {
            row_offset[i] = total_rows;
            total_rows += residuals_per_vertex[i];
        }

        int total_rows = 0;
        for (size_t i = 0; i < vertex_count; ++i) {
            row_offset[i] = total_rows;
            total_rows += residuals_per_vertex[i];
        }

        // ✅ 加在这里
        std::cout << "[Stage " << stage << "] Total residuals: " << total_rows << std::endl;

        // --- 2. 内层优化：结构固定 ---
        for (int iter = 0; iter < maxIterations_; ++iter) {
            std::vector<double> residuals(total_rows);
            int num_threads = omp_get_max_threads();
            std::vector<std::vector<Eigen::Triplet<double>>> triplets_per_thread(num_threads);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto& local_triplets = triplets_per_thread[tid];

                #pragma omp for
                for (size_t i = 0; i < vertex_count; ++i) {
                    if (x2_counts[i] == 0) continue;

                    int local_rowIndex = row_offset[i];
                    for (size_t j = 0; j < frame_count; ++j) {
                        if (!visible_table[i][j]) continue;

                        Eigen::Matrix<double, 6, 1> se3 = X.segment<6>(j * 6);
                        Sophus::SE3d T = Sophus::SE3d::exp(se3);
                        Eigen::Matrix3d R = T.rotationMatrix();
                        Eigen::Vector3d t = T.translation();
                        const cv::Mat& image = observed_images_gray[j];

                        PhotometricError costFunc(mesh_vertices[i], mesh_triangles,
                                                camera_intrinsics, image, bvh, weight_);
                        double r = 0.0;
                        Eigen::Matrix<double, 1, 6> J_pose;
                        double J_intensity = 0.0;
                        double intensity = X(poseDim + i);

                        costFunc.Evaluate(se3, intensity, r, &J_pose, &J_intensity);
                        residuals[local_rowIndex] = r;

                        for (int k = 0; k < 6; ++k) {
                            local_triplets.emplace_back(local_rowIndex, j * 6 + k, J_pose(k));
                        }
                        local_triplets.emplace_back(local_rowIndex, poseDim + i, J_intensity);
                        local_rowIndex++;
                    }
                }
            }

            std::vector<Eigen::Triplet<double>> triplets;
            for (const auto& vec : triplets_per_thread) {
                triplets.insert(triplets.end(), vec.begin(), vec.end());
            }

            Eigen::VectorXd F = Eigen::Map<Eigen::VectorXd>(residuals.data(), residuals.size());
            Eigen::SparseMatrix<double> J(total_rows, stateDim);
            J.setFromTriplets(triplets.begin(), triplets.end());
            J = J.rightCols(stateDim - 6); // remove first frame

            double cost = F.squaredNorm();
            Eigen::SparseMatrix<double> H = J.transpose() * J;
            Eigen::VectorXd g = -J.transpose() * F;
            double gradNorm = g.norm();

            double lambda = 1e-6;
            H += lambda * Eigen::MatrixXd::Identity(H.rows(), H.cols()).sparseView();

            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
            solver.compute(H);
            if (solver.info() != Eigen::Success) continue;

            Eigen::VectorXd delta = solver.solve(g);
            if (solver.info() != Eigen::Success) continue;

            for (size_t j = 1; j < frame_count; ++j) {
                Eigen::Matrix<double,6,1> d = delta.segment<6>((j-1)*6);
                Sophus::SE3d T = Sophus::SE3d::exp(X.segment<6>(j*6));
                Sophus::SE3d T_up = T * Sophus::SE3d::exp(d);
                X.segment<6>(j*6) = T_up.log();
            }

            X.segment(poseDim, intensityDim) += delta.tail(intensityDim);
            double deltaNorm = delta.norm();

            std::cout << "[Stage " << stage << " Iter " << iter
                      << "] cost=" << cost
                      << ", gradNorm=" << gradNorm
                      << ", deltaNorm=" << deltaNorm << std::endl;

            if (deltaNorm < 1e-6) break;
        }
    }

    // 写回优化后的 pose
    for (size_t i = 0; i < frame_count; i++) {
        Sophus::SE3d T = Sophus::SE3d::exp(X.segment<6>(i*6));
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.block<3,3>(0,0) = T.rotationMatrix();
        pose.block<3,1>(0,3) = T.translation();
        camera_poses[i] = pose;
    }
}