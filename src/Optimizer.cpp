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

Optimizer::Optimizer(double weight, int maxIterations)
    : weight_(weight), maxIterations_(maxIterations) {
    // 如果仍使用 Ceres，这里可初始化 options_ 等，但下面示例为手写优化器
}

void Optimizer::optimize(
    const std::vector<MeshModel::Vertex>& mesh_vertices,
    const std::vector<MeshModel::Triangle>& mesh_triangles,
    const Eigen::Matrix3d& camera_intrinsics,
    const std::vector<cv::Mat>& observed_images,
    std::vector<Eigen::Matrix4d>& camera_poses) {

    size_t frame_count = observed_images.size();
    size_t vertex_count = mesh_vertices.size();
    int poseDim = static_cast<int>(frame_count * 6);         // 每帧6维
    int intensityDim = static_cast<int>(vertex_count);         // 每个顶点1维光度
    int stateDim = poseDim + intensityDim;

    // 构建 BVH 结构（基于 mesh_triangles 和 mesh_vertices）
    BVH bvh(mesh_triangles, mesh_vertices);

    // 将所有观测图像转换为灰度（CV_32F）格式
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

    // 构造全局状态向量 X：
    // 前 poseDim 元素为每帧的 6D 位姿（SE3 对数表示），后 intensityDim 元素为每个顶点的光度值
    Eigen::VectorXd X = Eigen::VectorXd::Zero(stateDim);
    for (size_t i = 0; i < frame_count; i++) {
        // 将 camera_poses[i]（4x4 矩阵）转换为 6D 李代数表示
        Sophus::SE3d T(camera_poses[i].block<3,3>(0,0), camera_poses[i].block<3,1>(0,3));
        X.segment<6>(i*6) = T.log();
    }

    // 利用所有帧对每个顶点计算初始光度均值
    std::vector<double> x2_values(vertex_count, 0.0);
    std::vector<int> x2_counts(vertex_count, 0);
    std::vector<int> visible_vertex_per_frame(frame_count, 0);

    #pragma omp parallel for
    for (size_t i = 0; i < vertex_count; ++i) {
        double sum_intensity = 0.0;
        int count = 0;
        for (size_t j = 0; j < frame_count; ++j) {
            if (Projection::isVertexVisible(mesh_vertices[i], camera_intrinsics,
                camera_poses[j].block<3,3>(0,0), camera_poses[j].block<3,1>(0,3),
                bvh, observed_images_gray[j].cols, observed_images_gray[j].rows)) {

                Eigen::Vector2d proj = Projection::projectPoint(mesh_vertices[i], camera_intrinsics,
                    camera_poses[j].block<3,3>(0,0), camera_poses[j].block<3,1>(0,3));
                int u = static_cast<int>(proj(0));
                int v = static_cast<int>(proj(1));

                if (u >= 0 && u < observed_images_gray[j].cols &&
                    v >= 0 && v < observed_images_gray[j].rows) {
                    sum_intensity += observed_images_gray[j].at<float>(v, u);
                    count++;
                    #pragma omp atomic
                    visible_vertex_per_frame[j]++;
                }
            }
        }
        if (count > 0) {
            x2_values[i] = sum_intensity / count;
            x2_counts[i] = count;
        }
    }

    for (size_t j = 0; j < frame_count; ++j) {
        std::cout << "Frame " << j << " visible vertices: " << visible_vertex_per_frame[j] << std::endl;
    }

    // 将初始光度均值写入状态向量 X（后 intensityDim 部分）
    for (size_t i = 0; i < vertex_count; i++) {
        X(poseDim + i) = x2_values[i];
    }

    // Gauss-Newton 优化迭代
    for (int iter = 0; iter < maxIterations_; iter++) {

        // 对每个顶点和每帧构造残差项
        // 统计每个顶点可见帧数，预估每个顶点会生成多少残差项
        std::vector<int> residuals_per_vertex(vertex_count, 0);
        for (size_t i = 0; i < vertex_count; ++i) {
            if (x2_counts[i] == 0) continue;
            for (size_t j = 0; j < frame_count; ++j) {
                Eigen::Matrix<double, 6, 1> se3 = X.segment<6>(j*6);
                Sophus::SE3d T = Sophus::SE3d::exp(se3);
                Eigen::Matrix3d R = T.rotationMatrix();
                Eigen::Vector3d t = T.translation();
                const cv::Mat& image = observed_images_gray[j];

                if (Projection::isVertexVisible(mesh_vertices[i], camera_intrinsics, R, t,
                                                bvh, image.cols, image.rows)) {
                    residuals_per_vertex[i]++;
                }
            }
        }

        // 计算每个顶点在 residuals 和 triplets 中的起始行号偏移
        std::vector<int> row_offset(vertex_count, 0);
        int total_rows = 0;
        for (size_t i = 0; i < vertex_count; ++i) {
            row_offset[i] = total_rows;
            total_rows += residuals_per_vertex[i];
        }

        // 多线程安全版本
        // 分配全局残差向量（线程共享）
        std::vector<double> residuals(total_rows);

        // 每线程独立 triplets 缓冲区
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
                    Eigen::Matrix<double, 6, 1> se3 = X.segment<6>(j * 6);
                    Sophus::SE3d T = Sophus::SE3d::exp(se3);
                    Eigen::Matrix3d R = T.rotationMatrix();
                    Eigen::Vector3d t = T.translation();
                    const cv::Mat& image = observed_images_gray[j];

                    if (!Projection::isVertexVisible(mesh_vertices[i], camera_intrinsics, R, t,
                                                    bvh, image.cols, image.rows)) {
                        continue;
                    }

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

        // 合并所有线程 triplets
        std::vector<Eigen::Triplet<double>> triplets;
        for (const auto& vec : triplets_per_thread) {
            triplets.insert(triplets.end(), vec.begin(), vec.end());
        }

        Eigen::VectorXd F = Eigen::Map<Eigen::VectorXd>(residuals.data(), residuals.size());
        Eigen::SparseMatrix<double> J(total_rows, stateDim);
        J.setFromTriplets(triplets.begin(), triplets.end());

        // 计算当前 cost 和梯度
        double cost = F.squaredNorm();
        Eigen::SparseMatrix<double> H = J.transpose() * J;
        Eigen::VectorXd g = -J.transpose() * F;
        double gradNorm = g.norm();

        // // 将 H 分块：前 poseDim 为相机位姿，后 intensityDim 为光度
        // Eigen::SparseMatrix<double> U = H.block(0, 0, poseDim, poseDim);
        // Eigen::SparseMatrix<double> W = H.block(0, poseDim, poseDim, intensityDim);
        // Eigen::SparseMatrix<double> V = H.block(poseDim, poseDim, intensityDim, intensityDim);
        // Eigen::VectorXd bp = g.segment(0, poseDim);
        // Eigen::VectorXd bf = g.segment(poseDim, intensityDim);

        // // 这里转换为 dense 求解 Schur 补（实际可用稀疏求解器提高效率）
        // Eigen::MatrixXd U_dense = Eigen::MatrixXd(U);
        // Eigen::MatrixXd W_dense = Eigen::MatrixXd(W);
        // Eigen::MatrixXd V_dense = Eigen::MatrixXd(V);

        // double lambda = 1e-6; // 根据情况调整
        // V_dense += lambda * Eigen::MatrixXd::Identity(V_dense.rows(), V_dense.cols());
        // Eigen::MatrixXd V_inv = V_dense.inverse();

        // Eigen::MatrixXd Schur = U_dense - W_dense * V_inv * W_dense.transpose();
        // Eigen::VectorXd delta_pose = Schur.ldlt().solve(bp - W_dense * V_inv * bf);
        // Eigen::VectorXd delta_intensity = V_inv * (bf - W_dense.transpose() * delta_pose);

        // Eigen::VectorXd delta(stateDim);
        // delta.head(poseDim) = delta_pose;
        // delta.tail(intensityDim) = delta_intensity;

        // 可选 damping
        double lambda = 1e-6;
        H += lambda * Eigen::MatrixXd::Identity(H.rows(), H.cols()).sparseView();

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(H);

        if (solver.info() != Eigen::Success) {
            std::cerr << "H decomposition failed!" << std::endl;
            continue;
        }

        Eigen::VectorXd delta = solver.solve(g);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Linear solve failed!" << std::endl;
            continue;
        }

        // 固定第一帧：将第一帧的 6 个参数更新置零
        delta.segment(0, 6).setZero();

        X += delta;
        double deltaNorm = delta.norm();

        // 打印当前迭代信息：cost, 梯度范数, 更新量范数
        std::cout << "Iteration " << iter
                  << ", cost = " << cost
                  << ", grad norm = " << gradNorm
                  << ", delta norm = " << deltaNorm << std::endl;

        if (deltaNorm < 1e-6) break;
    }

    // 更新优化后的位姿到 camera_poses（从 X 的前 poseDim 部分恢复）
    for (size_t i = 0; i < frame_count; i++) {
        Eigen::Matrix<double, 6, 1> se3 = X.segment<6>(i*6);
        Sophus::SE3d T = Sophus::SE3d::exp(se3);
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.block<3,3>(0,0) = T.rotationMatrix();
        pose.block<3,1>(0,3) = T.translation();
        camera_poses[i] = pose;
    }
}