#include "CostFunctions.h"
#include "Projection.h"
#include "ImageProcessor.h"

CombinedError::CombinedError(const std::vector<MeshModel::Vertex>& vertices,
                             const std::vector<MeshModel::Triangle>& triangles,
                             const Eigen::Matrix3f& camera_intrinsics,
                             const std::vector<cv::Mat>& depth_maps,
                             const std::vector<Eigen::MatrixXf>& depth_normals,
                             const std::vector<cv::Mat>& images,
                             int max_vertices,
                             double weight_global_depth,
                             double weight_global_gradient,
                             double weight_local_depth,
                             double weight_local_normal);
    : vertices_(vertices),
      triangles_(triangles),
      camera_intrinsics_(camera_intrinsics),
      depth_maps_(depth_maps),
      depth_normals_(depth_normals),
      images_(images),
      max_vertices_(max_vertices) {}

template <typename T>
bool CombinedError::operator()(T const* const* poses, T* residual) const {
    size_t vertex_count = vertices_.size();
    size_t frame_count = depth_maps_.size();

    std::fill(residual, residual + frame_count * 4, T(0));  // **四类误差（全局深度、一致性梯度、局部深度、局部法向量）**

    // **预计算所有帧的可见性掩码 & 投影点**
    std::vector<std::vector<bool>> visibility_masks(frame_count);
    std::vector<std::vector<Eigen::Vector2f>> projected_points(frame_count);

    for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
        Eigen::Matrix<T, 3, 3> rotation;
        rotation << poses[frame_idx][0], poses[frame_idx][1], poses[frame_idx][2],
                    poses[frame_idx][3], poses[frame_idx][4], poses[frame_idx][5],
                    poses[frame_idx][6], poses[frame_idx][7], poses[frame_idx][8];

        Eigen::Matrix<T, 3, 1> translation(poses[frame_idx][9], poses[frame_idx][10], poses[frame_idx][11]);

        visibility_masks[frame_idx] = Projection::handleOcclusion(
            vertices_, triangles_, camera_intrinsics_, rotation.template cast<float>(), translation.template cast<float>(),
            depth_maps_[frame_idx].cols, depth_maps_[frame_idx].rows);

        projected_points[frame_idx] = Projection::projectPoints(vertices_, camera_intrinsics_, rotation.template cast<float>(), translation.template cast<float>());
    }

    // **计算可见度最高的顶点**
    std::vector<std::pair<int, int>> visibility_count;
    for (size_t i = 0; i < vertex_count; ++i) {
        int count = 0;
        for (size_t j = 0; j < frame_count; ++j) {
            if (visibility_masks[j][i]) count++;
        }
        if (count >= 2) visibility_count.emplace_back(count, i);
    }

    std::sort(visibility_count.begin(), visibility_count.end(), std::greater<>());
    size_t processed_vertices = (max_vertices_ == 0) ? visibility_count.size() : std::min(size_t(max_vertices_), visibility_count.size());

    // **合并局部误差 & 全局误差计算**
    for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
        T global_depth_sum = 0, global_gradient_sum = 0;
        size_t valid_global_count = 0;

        for (size_t vertex_idx = 0; vertex_idx < vertex_count; ++vertex_idx) {
            if (!visibility_masks[frame_idx][vertex_idx]) continue;

            Eigen::Vector2f projected_2d = projected_points[frame_idx][vertex_idx];
            int u = static_cast<int>(projected_2d.x());
            int v = static_cast<int>(projected_2d.y());

            // **局部深度误差**
            if (u >= 0 && u < depth_maps_[frame_idx].cols && v >= 0 && v < depth_maps_[frame_idx].rows) {
                float camera_depth = depth_maps_[frame_idx].at<float>(v, u);
                if (camera_depth > 0) {
                    T depth_error = T(camera_depth) - T(projected_2d.y());
                    residual[frame_idx * 4 + 2] += weight_local_depth * depth_error * depth_error;
                }
            }

            // **局部法向量误差**
            if (u >= 0 && u < depth_normals_[frame_idx].cols() && v >= 0 && v < depth_normals_[frame_idx].rows()) {
                Eigen::Matrix<T, 3, 1> mesh_normal(T(vertices_[vertex_idx].nx), T(vertices_[vertex_idx].ny), T(vertices_[vertex_idx].nz));
                Eigen::Matrix<T, 3, 1> transformed_normal = poses[frame_idx].block<3, 3>(0, 0).template cast<T>() * mesh_normal;

                Eigen::Matrix<T, 3, 1> depth_normal(
                    T(depth_normals_[frame_idx](v, u * 3 + 0)),
                    T(depth_normals_[frame_idx](v, u * 3 + 1)),
                    T(depth_normals_[frame_idx](v, u * 3 + 2)));

                T normal_error = T(1.0) - transformed_normal.dot(depth_normal);
                residual[frame_idx * 4 + 3] += weight_local_normal * normal_error * normal_error;
            }
        }

        // **计算全局误差**
        for (size_t k = 0; k < processed_vertices; ++k) {
            int vertex_idx = visibility_count[k].second;
            if (!visibility_masks[frame_idx][vertex_idx]) continue;

            Eigen::Vector2f projected_2d = projected_points[frame_idx][vertex_idx];
            int u = static_cast<int>(projected_2d.x());
            int v = static_cast<int>(projected_2d.y());

            T depth = T(depth_maps_[frame_idx].at<float>(v, u));
            global_depth_sum += depth;

            T gradient = T(ImageProcessor::computeGradient(images_[frame_idx], u, v));
            global_gradient_sum += gradient;

            valid_global_count++;
        }

        if (valid_global_count > 0) {
            residual[frame_idx * 4] = global_depth_sum / T(valid_global_count);
            residual[frame_idx * 4 + 1] = global_gradient_sum / T(valid_global_count);
        }
    }

    return true;
}

ceres::CostFunction* CombinedError::Create(const std::vector<MeshModel::Vertex>& vertices,
                                            const std::vector<MeshModel::Triangle>& triangles,
                                            const Eigen::Matrix3f& camera_intrinsics,
                                            const std::vector<cv::Mat>& depth_maps,
                                            const std::vector<Eigen::MatrixXf>& depth_normals,
                                            const std::vector<cv::Mat>& images,
                                            int max_vertices) {
    size_t frame_count = depth_maps.size();

    // **创建 Ceres 误差项**
    auto* cost_function = new ceres::DynamicAutoDiffCostFunction<CombinedError>(
        new CombinedError(vertices, triangles, camera_intrinsics, depth_maps, depth_normals, images, max_vertices));

    // **添加优化变量（每个帧的相机位姿，共 12 变量：旋转 9 + 平移 3）**
    for (size_t i = 0; i < frame_count; ++i) {
        cost_function->AddParameterBlock(12);
    }

    // **设置残差维度**
    // 每帧有 4 个误差项：
    // - residual[frame_idx * 4] = 全局深度误差
    // - residual[frame_idx * 4 + 1] = 全局光度梯度误差
    // - residual[frame_idx * 4 + 2] = 局部深度误差
    // - residual[frame_idx * 4 + 3] = 局部法向量误差
    cost_function->SetNumResiduals(frame_count * 4);

    return cost_function;
}