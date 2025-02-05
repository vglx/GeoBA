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
                               double weight_local_normal)
    : vertices_(vertices),
      triangles_(triangles),
      camera_intrinsics_(camera_intrinsics),
      depth_maps_(depth_maps),
      depth_normals_(depth_normals),
      images_(images),
      max_vertices_(max_vertices),
      weight_global_depth_(weight_global_depth),
      weight_global_gradient_(weight_global_gradient),
      weight_local_depth_(weight_local_depth),
      weight_local_normal_(weight_local_normal) {}

template <typename T>
bool CombinedError::operator()(T const* const* poses, T* residual) const {

    size_t vertex_count = vertices_.size();
    size_t frame_count = depth_maps_.size();

    std::fill(residual, residual + frame_count * 4, T(0));  // **深度、法向量、光度梯度误差**

    // **第一步：计算所有帧的可见性掩码 & 投影点**
    std::vector<std::vector<bool>> visibility_masks(frame_count);
    std::vector<std::vector<Eigen::Vector2f>> projected_points(frame_count);
    std::vector<std::vector<float>> vertex_depths_per_frame(frame_count);

    for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
        Eigen::Matrix<T, 3, 3> rotation;
        rotation << poses[frame_idx][0], poses[frame_idx][1], poses[frame_idx][2],
                    poses[frame_idx][3], poses[frame_idx][4], poses[frame_idx][5],
                    poses[frame_idx][6], poses[frame_idx][7], poses[frame_idx][8];

        Eigen::Matrix<T, 3, 1> translation(poses[frame_idx][9], poses[frame_idx][10], poses[frame_idx][11]);

        visibility_masks[frame_idx] = Projection::handleOcclusion(
            vertices_, triangles_, camera_intrinsics_, rotation, translation,
            depth_maps_[frame_idx].cols, depth_maps_[frame_idx].rows);

        projected_points[frame_idx] = Projection::projectPoints(
            vertices_, camera_intrinsics_, rotation, translation);

        vertex_depths_per_frame[frame_idx] = Projection::computeVertexDepths(
            vertices_, camera_intrinsics_, rotation, translation);
    }

    // **第二步：计算所有帧的误差**
    for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
        for (size_t vertex_idx = 0; vertex_idx < vertex_count; ++vertex_idx) {
            if (!visibility_masks[frame_idx][vertex_idx]) continue;

            Eigen::Vector2f& projected_2d = projected_points[frame_idx][vertex_idx];
            int u = static_cast<int>(projected_2d.x());
            int v = static_cast<int>(projected_2d.y());

            // **深度误差**
            if (u >= 1 && u < depth_maps_[frame_idx].cols - 1 && v >= 1 && v < depth_maps_[frame_idx].rows - 1) {
                float camera_depth = depth_maps_[frame_idx].at<float>(v, u);
                if (camera_depth > 0 && std::isfinite(camera_depth)) {
                    T depth_error = T(camera_depth) - T(vertex_depths_per_frame[frame_idx][vertex_idx]); 
                    residual[frame_idx * 4] += weight_local_depth_ * depth_error * depth_error;
                }

            // **法向量误差**
                Eigen::Matrix<T, 3, 1> mesh_normal = Eigen::Vector3f(
                    vertices_[vertex_idx].nx, vertices_[vertex_idx].ny, vertices_[vertex_idx].nz).template cast<T>();
                Eigen::Map<const Eigen::Matrix<T, 3, 3>> rotation(poses[frame_idx]);
                Eigen::Matrix<T, 3, 1> transformed_normal = rotation * mesh_normal;

                Eigen::Matrix<T, 3, 1> depth_normal(
                    T(depth_normals_[frame_idx](v, u * 3 + 0)),
                    T(depth_normals_[frame_idx](v, u * 3 + 1)),
                    T(depth_normals_[frame_idx](v, u * 3 + 2)));

                if (depth_normal.norm() > 1e-6) depth_normal.normalize();
                if (transformed_normal.norm() > 1e-6) transformed_normal.normalize();
                T normal_error = T(1.0) - transformed_normal.dot(depth_normal);
                residual[frame_idx * 4 + 1] += weight_local_normal_ * normal_error * normal_error;
            }
        }
    }

    // **第三步：计算全局光度梯度和深度一致性误差**
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

    for (size_t k = 0; k < processed_vertices; ++k) {
        int vertex_idx = visibility_count[k].second;
        std::vector<T> gradients, depths;

        for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
            if (!visibility_masks[frame_idx][vertex_idx]) continue;

            Eigen::Vector2f& projected_2d = projected_points[frame_idx][vertex_idx];
            int u = static_cast<int>(projected_2d.x());
            int v = static_cast<int>(projected_2d.y());

            if (u >= 1 && u < depth_maps_[frame_idx].cols - 1 && v >= 1 && v < depth_maps_[frame_idx].rows - 1) {
                T depth = T(depth_maps_[frame_idx].at<float>(v, u));
                if (depth > 0.1f) {
                    T gradient = 0.25 * (
                        T(ImageProcessor::computeGradient(images_[frame_idx], u, v)) +
                        T(ImageProcessor::computeGradient(images_[frame_idx], u - 1, v)) +
                        T(ImageProcessor::computeGradient(images_[frame_idx], u + 1, v)) +
                        T(ImageProcessor::computeGradient(images_[frame_idx], u, v - 1))
                    );
                    gradients.push_back(gradient);
                    depths.push_back(depth);
                }
            }
        
        }

        if (gradients.size() < 2 || depths.size() < 2) continue;

        // **计算中位数**
        std::nth_element(gradients.begin(), gradients.begin() + gradients.size() / 2, gradients.end());
        std::nth_element(depths.begin(), depths.begin() + depths.size() / 2, depths.end());

        T gradient_median = gradients[gradients.size() / 2];
        T depth_median = depths[depths.size() / 2];

        // **计算全局误差**
        for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
            if (!visibility_masks[frame_idx][vertex_idx]) continue;

            Eigen::Vector2f projected_2d = projected_points[frame_idx][vertex_idx];
            int u = static_cast<int>(projected_2d.x());
            int v = static_cast<int>(projected_2d.y());

            T gradient = T(ImageProcessor::computeGradient(images_[frame_idx], u, v));
            T gradient_error = gradient - gradient_median;
            residual[frame_idx * 4 + 2] += weight_global_gradient_ * gradient_error * gradient_error;

            T depth = T(depth_maps_[frame_idx].at<float>(v, u));
            T depth_error = depth - depth_median;
            residual[frame_idx * 4 + 3] += weight_global_depth_ * depth_error * depth_error;
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
                                            int max_vertices,
                                            double weight_global_depth,
                                            double weight_global_gradient,
                                            double weight_local_depth,
                                            double weight_local_normal) {
    size_t frame_count = depth_maps.size();

    // **创建 Ceres 误差项**
    auto* cost_function = new ceres::DynamicAutoDiffCostFunction<CombinedError>(
        new CombinedError(vertices, triangles, camera_intrinsics, depth_maps, depth_normals, images, 
                           max_vertices, weight_global_depth, weight_global_gradient, weight_local_depth, weight_local_normal));

    // **添加优化变量（每个帧的相机位姿，共 12 变量：旋转 9 + 平移 3）**
    for (size_t i = 0; i < frame_count; ++i) {
        cost_function->AddParameterBlock(12);
    }

    // **设置残差维度**
    cost_function->SetNumResiduals(frame_count * 4);

    return cost_function;
}