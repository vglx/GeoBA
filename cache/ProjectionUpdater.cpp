#include "ProjectionUpdater.h"
#include "Projection.h"

ProjectionUpdater::ProjectionUpdater(SharedProjectionData* shared_data,
                                     const std::vector<MeshModel::Vertex>& mesh_vertices, // **更改这里**
                                     const std::vector<MeshModel::Triangle>& mesh_triangles,
                                     const Eigen::Matrix3f& camera_intrinsics,
                                     const std::vector<cv::Mat>& observed_images,
                                     std::vector<Eigen::Matrix4f>& camera_poses)
    : shared_data_(shared_data),
      mesh_vertices_(mesh_vertices), // **更改这里**
      mesh_triangles_(mesh_triangles),
      camera_intrinsics_(camera_intrinsics),
      observed_images_(observed_images),
      camera_poses_(camera_poses) {}

ceres::CallbackReturnType ProjectionUpdater::operator()(const ceres::IterationSummary& summary) {
    // 遍历每帧，读取优化后的相机位姿并更新共享数据
    for (size_t frame_idx = 0; frame_idx < observed_images_.size(); ++frame_idx) {
        Eigen::Matrix4f optimized_pose = camera_poses_[frame_idx]; // 当前轮优化后的位姿

        // 初始化当前帧的数据存储（如果尚未初始化）
        if (shared_data_->projected_points.size() <= frame_idx) {
            shared_data_->projected_points.emplace_back(mesh_vertices_.size());
            shared_data_->projected_depths.emplace_back(mesh_vertices_.size());
            shared_data_->visible_mask.emplace_back(mesh_vertices_.size(), false);
        }

        // 更新投影深度
        shared_data_->projected_depths[frame_idx] = Projection::computeVertexDepths(
            mesh_vertices_, camera_intrinsics_, optimized_pose);

        // 更新投影点
        shared_data_->projected_points[frame_idx] = Projection::projectPoints(
            mesh_vertices_, camera_intrinsics_,
            optimized_pose.block<3, 3>(0, 0), optimized_pose.block<3, 1>(0, 3));

        // 更新可见性掩码（考虑面片遮挡）
        shared_data_->visible_mask[frame_idx] = Projection::handleOcclusion(
            mesh_vertices_, mesh_triangles_, camera_intrinsics_,
            optimized_pose.block<3, 3>(0, 0), optimized_pose.block<3, 1>(0, 3),
            observed_images_[frame_idx].cols, observed_images_[frame_idx].rows);
    }

    return ceres::SOLVER_CONTINUE;
}
