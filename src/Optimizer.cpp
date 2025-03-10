#include "Optimizer.h"
#include "CostFunctions.h"
#include "Projection.h"
#include "BVH.h"
#include <iostream>
#include <sophus/se3.hpp>
#include <omp.h>

Optimizer::Optimizer(double weight)
    : weight_(weight) {
    options_.linear_solver_type = ceres::SPARSE_SCHUR;
    options_.minimizer_progress_to_stdout = true;
    options_.max_num_iterations = 100;
    options_.num_threads = 4;
}

void Optimizer::optimize(
    const std::vector<MeshModel::Vertex>& mesh_vertices,
    const std::vector<MeshModel::Triangle>& mesh_triangles,
    const Eigen::Matrix3d& camera_intrinsics,
    const std::vector<cv::Mat>& observed_images,
    std::vector<Eigen::Matrix4d>& camera_poses) {

    size_t frame_count = observed_images.size();
    size_t vertex_count = mesh_vertices.size();
    ceres::Problem problem;

    // **构建 BVH 结构**
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

    // **初始化优化变量** (frame_count 个相机位姿，每个 6 维)
    std::vector<double> poses(frame_count * 6);
    for (size_t i = 0; i < frame_count; ++i) {
        Sophus::SE3d pose_SE3(camera_poses[i].block<3,3>(0,0), camera_poses[i].block<3,1>(0,3));
        Eigen::Matrix<double,6,1> se3_vec = pose_SE3.log();
        for (int j = 0; j < 6; ++j) {
            poses[i * 6 + j] = se3_vec[j];
        }
    }

    // **计算每个顶点的光度均值 x2**
    std::vector<double> x2_values(vertex_count, 0.0);
    std::vector<int> x2_counts(vertex_count, 0);

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

                if (u >= 0 && u < observed_images_gray[j].cols && v >= 0 && v < observed_images_gray[j].rows) {
                    sum_intensity += observed_images_gray[j].at<float>(v, u);
                    count++;
                }
            }
        }
        if (count > 0) {
            x2_values[i] = sum_intensity / count;
            x2_counts[i] = count;
        }
    }

    // **添加残差项**
    for (size_t i = 0; i < vertex_count; ++i) {
        if (x2_counts[i] == 0) continue;  // **跳过不可见顶点**
        
        for (size_t j = 0; j < frame_count; ++j) {
            if (Projection::isVertexVisible(mesh_vertices[i], camera_intrinsics,
                camera_poses[j].block<3,3>(0,0), camera_poses[j].block<3,1>(0,3),
                bvh, observed_images_gray[j].cols, observed_images_gray[j].rows)) {

                // **创建残差项**
                ceres::CostFunction* photometric_cf = MultiViewPhotometricError::Create(
                    mesh_vertices[i], mesh_triangles, camera_intrinsics, observed_images_gray[j], bvh, weight_
                );

                // **添加到 Ceres 优化问题**
                problem.AddResidualBlock(photometric_cf, nullptr, &poses[j * 6], &x2_values[i]);
            }
        }
    }

    // // **运行 Ceres 优化前，添加参数块排序**
    // auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();

    // // **相机位姿优先优化**
    // for (size_t i = 0; i < frame_count; ++i) {
    //     ordering->AddElementToGroup(&poses[i * 6], 0);
    // }

    // // **然后优化 x2_values（光度均值）**
    // for (size_t i = 0; i < vertex_count; ++i) {
    //     ordering->AddElementToGroup(&x2_values[i], 1);
    // }

    // // **应用排序**
    // options_.linear_solver_ordering = ordering;

    // **运行 Ceres 优化**
    ceres::Solver::Summary summary;
    ceres::Solve(options_, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
}