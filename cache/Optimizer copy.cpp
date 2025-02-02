#include "Optimizer.h"
#include "Projection.h"
#include <iostream>

// 构造函数：初始化误差管理器和优化选项
Optimizer::Optimizer(double weight_geometry, double weight_normal, double weight_photometric)
    : error_manager_(weight_geometry, weight_normal, weight_photometric) {
    options_.linear_solver_type = ceres::SPARSE_SCHUR; // 更适合多帧优化
    options_.minimizer_progress_to_stdout = true;
    options_.max_num_iterations = 100; // 默认迭代次数
}

// 优化
void Optimizer::optimize(
    const std::vector<std::vector<Eigen::Vector3f>>& point_clouds,
    const std::vector<std::vector<Eigen::Vector3f>>& point_normals,
    const std::vector<Eigen::Vector3f>& mesh_vertices,
    const std::vector<Eigen::Vector3f>& mesh_normals,
    const std::vector<cv::Mat>& observed_images,
    const std::vector<cv::Mat>& observed_depth_maps,
    const Eigen::Matrix3f& camera_intrinsics,
    std::vector<Eigen::Matrix4f>& camera_poses) {

    std::cout << "Starting multi-frame optimization with occlusion handling...\n";

    // 相机内参
    float fx = camera_intrinsics(0, 0);
    float fy = camera_intrinsics(1, 1);
    float cx = camera_intrinsics(0, 2);
    float cy = camera_intrinsics(1, 2);

    // 创建 Ceres 问题
    ceres::Problem problem;

    // 遍历多帧数据，添加误差项
    for (size_t frame_idx = 0; frame_idx < point_clouds.size(); ++frame_idx) {
        const auto& points = point_clouds[frame_idx];
        const auto& normals = point_normals[frame_idx];
        const auto& image = observed_images[frame_idx];
        const auto& depth_map = observed_depth_maps[frame_idx];
        Eigen::Matrix4f& pose = camera_poses[frame_idx]; // 当前帧的位姿

        // 计算所有顶点的深度
        std::vector<float> depths = Projection::computeVertexDepths(mesh_vertices, camera_intrinsics, pose);

        // 投影网格顶点到图像平面
        std::vector<Eigen::Vector2f> projectedPoints = Projection::projectPoints(
            mesh_vertices, camera_intrinsics, pose.block<3, 3>(0, 0), pose.block<3, 1>(0, 3));

        // 遮挡处理：获取可见性布尔掩码
        std::vector<bool> visibleMask = Projection::handleOcclusion(projectedPoints, depths, image.cols, image.rows);

        // 遍历所有点，检查可见性并添加误差项
        for (size_t i = 0; i < mesh_vertices.size(); ++i) {
            // 跳过不可见点
            if (!visibleMask[i]) continue;

            // 获取当前点及其相关信息
            Eigen::Vector3f point = points[i];
            Eigen::Vector3f point_normal = normals[i];
            Eigen::Vector3f nearest_vertex = mesh_vertices[i];
            Eigen::Vector3f mesh_normal = mesh_normals[i];
            Eigen::Vector2f projected_point = projectedPoints[i];
            float observed_intensity = image.at<uchar>(
                static_cast<int>(projected_point.y()), static_cast<int>(projected_point.x()));

            // 添加几何误差
            error_manager_.addGeometricError(problem, point, nearest_vertex, mesh_vertices[i].data());

            // 添加法向量误差
            error_manager_.addNormalError(problem, point_normal, mesh_normal, mesh_normals[i].data());

            // 添加光度误差
            error_manager_.addPhotometricError(problem, point, observed_intensity, image,
                                            fx, fy, cx, cy, mesh_vertices[i].data(), pose.data());
        }


        // 将相机位姿添加为优化变量
        problem.AddParameterBlock(pose.data(), 6); // SE3 形式优化
    }

    // 配置优化器并运行
    ceres::Solver::Summary summary;
    ceres::Solve(options_, &problem, &summary);

    // 输出优化结果
    std::cout << summary.FullReport() << std::endl;
    std::cout << "Multi-frame optimization with occlusion handling completed.\n";
}
