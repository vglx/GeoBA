#include "Optimizer.h"
#include "CostFunctions.h"
#include <iostream>

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
                         const std::vector<cv::Mat>& observed_images,
                         const std::vector<cv::Mat>& observed_depth_maps,
                         const std::vector<Eigen::MatrixXf>& depth_normals,
                         const Eigen::Matrix3f& camera_intrinsics,
                         std::vector<Eigen::Matrix4f>& camera_poses) {
    size_t frame_count = observed_images.size();
    ceres::Problem problem;

    // **1. 创建 Ceres 误差项，并传入权重**
    ceres::CostFunction* cost_function = CombinedError::Create(mesh_vertices, mesh_triangles, camera_intrinsics,
                                                               observed_depth_maps, depth_normals, observed_images, 100,
                                                               weight_global_depth_, weight_global_gradient_,
                                                               weight_local_depth_, weight_local_normal_);

    // **2. 绑定优化变量**
    std::vector<double*> pose_ptrs(frame_count);
    std::vector<std::vector<double>> poses(frame_count, std::vector<double>(12));

    for (size_t i = 0; i < frame_count; ++i) {
        Eigen::Matrix3f R = camera_poses[i].block<3, 3>(0, 0);
        Eigen::Vector3f t = camera_poses[i].block<3, 1>(0, 3);

        for (int j = 0; j < 9; ++j) poses[i][j] = static_cast<double>(R(j / 3, j % 3));
        for (int j = 0; j < 3; ++j) poses[i][9 + j] = static_cast<double>(t(j));

        pose_ptrs[i] = poses[i].data();
    }

    problem.AddResidualBlock(cost_function, nullptr, pose_ptrs);

    // **3. 配置 Ceres Solver**
    ceres::Solver::Summary summary;
    ceres::Solve(options_, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;

    // **4. 更新优化后的相机位姿**
    for (size_t i = 0; i < frame_count; ++i) {
        Eigen::Matrix3f R;
        Eigen::Vector3f t;

        for (int j = 0; j < 9; ++j) R(j / 3, j % 3) = static_cast<float>(poses[i][j]);
        for (int j = 0; j < 3; ++j) t(j) = static_cast<float>(poses[i][9 + j]);

        camera_poses[i].block<3, 3>(0, 0) = R;
        camera_poses[i].block<3, 1>(0, 3) = t;
    }
}
