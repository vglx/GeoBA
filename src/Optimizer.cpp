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
    options_.max_num_iterations = 100;
}

void Optimizer::optimize(const std::vector<MeshModel::Vertex>& mesh_vertices,
    const std::vector<MeshModel::Triangle>& mesh_triangles,
    const Eigen::Matrix3d& camera_intrinsics,
    const std::vector<cv::Mat>& observed_images,
    const std::vector<cv::Mat>& observed_depth_maps,
    const std::vector<Eigen::MatrixXf>& depth_normals,
    std::vector<Eigen::Matrix4d>& camera_poses) {
    size_t frame_count = observed_depth_maps.size();
    ceres::Problem problem;

    // **1. 绑定优化变量**
    std::vector<double> poses(frame_count * 6);  // 线性数组，每个 SE(3) 变量占 6 个 double

    for (size_t i = 0; i < frame_count; ++i) {
        Sophus::SE3d pose_SE3(camera_poses[i].block<3, 3>(0, 0),
                              camera_poses[i].block<3, 1>(0, 3));
        Eigen::Matrix<double, 6, 1> se3_vec = pose_SE3.log();

        // **存入 `std::vector<double>` 线性数组**
        for (int j = 0; j < 6; ++j) {
            poses[i * 6 + j] = se3_vec[j];
        }

        // **创建 CostFunction**
        ceres::CostFunction* cost_function = LocalGeometricError::Create(
            mesh_vertices, mesh_triangles, camera_intrinsics,
            observed_depth_maps[i], weight_local_depth_);
        problem.AddResidualBlock(cost_function, nullptr, &poses[i * 6]);  // **传入 `double*` 指针**
    }

    for (size_t i = 1; i < frame_count - 1; ++i) {
        std::vector<cv::Mat> depth_maps = {observed_depth_maps[i], observed_depth_maps[i - 1], observed_depth_maps[i + 1]};
    
        ceres::CostFunction* cost_function = NormalConsistencyError::Create(
            mesh_vertices, mesh_triangles, camera_intrinsics, depth_maps, weight_local_normal_);
    
        problem.AddResidualBlock(cost_function, nullptr, &poses[i * 6], &poses[(i - 1) * 6], &poses[(i + 1) * 6]);
    }

        // **3. 配置 Ceres Solver**
    ceres::Solver::Summary summary;
    ceres::Solve(options_, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

        // **4. 更新优化后的相机位姿**
    for (size_t i = 0; i < frame_count; ++i) {
        Eigen::Matrix<double,6,1> se3_vec;
        for (int j = 0; j < 6; ++j) {
            se3_vec[j] = poses[i * 6 + j];
        }

        // **转换回 SE(3)**
        Sophus::SE3d updated_SE3 = Sophus::SE3d::exp(se3_vec);

        camera_poses[i].block<3, 3>(0, 0) = updated_SE3.rotationMatrix();
        camera_poses[i].block<3, 1>(0, 3) = updated_SE3.translation();
        std::cout << "Updated Pose [" << i << "] :\n" << camera_poses[i] << std::endl;
    }
}