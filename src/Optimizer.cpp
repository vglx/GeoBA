#include "Optimizer.h"
#include "CostFunctions.h"
#include <iostream>
#include <sophus/se3.hpp>

Optimizer::Optimizer(double weight_global_depth, double weight_global_gradient,
                     double weight_local_depth, double weight_local_normal)
    : weight_global_depth_(weight_global_depth),
      weight_global_gradient_(weight_global_gradient),
      weight_local_depth_(weight_local_depth),
      weight_local_normal_(weight_local_normal) {
    // 配置 Ceres Solver
    options_.linear_solver_type = ceres::SPARSE_SCHUR;
    options_.minimizer_progress_to_stdout = true;
    options_.max_num_iterations = 10;
}

void Optimizer::optimize(const std::vector<MeshModel::Vertex>& mesh_vertices,
                         const std::vector<MeshModel::Triangle>& mesh_triangles,
                         const Eigen::Matrix3f& camera_intrinsics,
                         const std::vector<cv::Mat>& observed_images,
                         const std::vector<cv::Mat>& observed_depth_maps,
                         const std::vector<Eigen::MatrixXf>& depth_normals,
                         std::vector<Eigen::Matrix4d>& camera_poses) {
    size_t frame_count = observed_depth_maps.size();
    ceres::Problem problem;

    // **1. 绑定优化变量**
    std::vector<Eigen::Matrix<double, 6, 1>> poses(frame_count);  // 每个位置存储一个 SE(3) 李代数变量

    for (size_t i = 0; i < frame_count; ++i) {
        Sophus::SE3d pose_SE3(camera_poses[i].block<3, 3>(0, 0),
                               camera_poses[i].block<3, 1>(0, 3));
        poses[i] = pose_SE3.log();  // 直接存储李代数变量

        ceres::CostFunction* cost_function = LocalGeometricError::Create(mesh_vertices,
                                                                         mesh_triangles,
                                                                         camera_intrinsics,
                                                                         observed_depth_maps[i],
                                                                         depth_normals[i],
                                                                         weight_local_depth_,
                                                                         weight_local_normal_);
        problem.AddResidualBlock(cost_function, nullptr, poses[i].data());
    }

    // // **2. 添加 TripletGlobalError 误差项**
    // for (size_t i = 0; i < frame_count - 2; ++i) {  // 每个误差项连接 3 帧
    //     ceres::CostFunction* triplet_cost_function = TripletGlobalError::Create(
    //         mesh_vertices, mesh_triangles, camera_intrinsics,
    //         observed_depth_maps[i], observed_depth_maps[i + 1], observed_depth_maps[i + 2],
    //         observed_images[i], observed_images[i + 1], observed_images[i + 2],
    //         weight_global_depth_, weight_global_gradient_);

    //     problem.AddResidualBlock(triplet_cost_function, nullptr, poses[i].data(), poses[i + 1].data(), poses[i + 2].data());
    // }

    // **3. 配置 Ceres Solver**
    ceres::Solver::Summary summary;
    ceres::Solve(options_, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    // **4. 更新优化后的相机位姿**
    for (size_t i = 0; i < frame_count; ++i) {
        Sophus::SE3d updated_SE3 = Sophus::SE3d::exp(poses[i]);

        camera_poses[i].block<3, 3>(0, 0) = updated_SE3.rotationMatrix();
        camera_poses[i].block<3, 1>(0, 3) = updated_SE3.translation();
    }
}