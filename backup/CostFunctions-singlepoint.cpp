#include "CostFunctions.h"

GeometricError::GeometricError(const std::vector<MeshModel::Vertex>& vertices,
                               const std::vector<MeshModel::Triangle>& triangles,
                               const Eigen::Matrix3f& camera_intrinsics,
                               const cv::Mat& depth_map,
                               const Eigen::MatrixXf& depth_normals)
    : vertices_(vertices),
      triangles_(triangles),
      camera_intrinsics_(camera_intrinsics),
      depth_map_(depth_map),
      depth_normals_(depth_normals) {}

template <typename T>
bool GeometricError::operator()(const T* const camera_pose, T* residual) const {
    size_t vertex_count = vertices_.size();
    residual[0] = T(0); // 深度误差
    residual[1] = T(0); // 法向量误差

    // 提取旋转和平移
    Eigen::Matrix<T, 3, 3> rotation;
    rotation << camera_pose[0], camera_pose[1], camera_pose[2],
                camera_pose[3], camera_pose[4], camera_pose[5],
                camera_pose[6], camera_pose[7], camera_pose[8];

    Eigen::Matrix<T, 3, 1> translation(camera_pose[9], camera_pose[10], camera_pose[11]);

    // 遍历所有顶点
    for (size_t vertex_idx = 0; vertex_idx < vertex_count; ++vertex_idx) {
        // 获取顶点信息
        const auto& vertex = vertices_[vertex_idx];
        Eigen::Matrix<T, 3, 1> vertex_pos(T(vertex.x), T(vertex.y), T(vertex.z));

        // 计算投影点
        Eigen::Matrix<T, 3, 1> projected_3d = rotation * vertex_pos + translation;
        T vertex_depth = projected_3d.z();

        Eigen::Matrix<T, 2, 1> projected_2d;
        projected_2d.x() = T(camera_intrinsics_(0, 0)) * projected_3d.x() / projected_3d.z() + T(camera_intrinsics_(0, 2));
        projected_2d.y() = T(camera_intrinsics_(1, 1)) * projected_3d.y() / projected_3d.z() + T(camera_intrinsics_(1, 2));

        // 计算深度误差
        int u = static_cast<int>(projected_2d.x());
        int v = static_cast<int>(projected_2d.y());

        if (u >= 0 && u < depth_map_.cols && v >= 0 && v < depth_map_.rows) {
            float camera_depth = depth_map_.at<float>(v, u);
            if (camera_depth > 0) {
                T depth_error = vertex_depth - T(camera_depth);
                residual[0] += depth_error * depth_error;
            }
        }

        // 计算法向量误差
        if (u >= 0 && u < depth_normals_.cols() && v >= 0 && v < depth_normals_.rows()) {
            Eigen::Matrix<T, 3, 1> mesh_normal(T(vertex.nx), T(vertex.ny), T(vertex.nz));
            Eigen::Matrix<T, 3, 1> transformed_normal = rotation * mesh_normal;

            Eigen::Matrix<T, 3, 1> depth_normal(
                T(depth_normals_(v, u * 3 + 0)),
                T(depth_normals_(v, u * 3 + 1)),
                T(depth_normals_(v, u * 3 + 2)));

            T dot_product = transformed_normal.dot(depth_normal);
            T normal_error = T(1.0) - dot_product;
            residual[1] += normal_error * normal_error;
        }
    }

    return true;
}

ceres::CostFunction* GeometricError::Create(const std::vector<MeshModel::Vertex>& vertices,
                                            const std::vector<MeshModel::Triangle>& triangles,
                                            const Eigen::Matrix3f& camera_intrinsics,
                                            const cv::Mat& depth_map,
                                            const Eigen::MatrixXf& depth_normals) {
    return (new ceres::AutoDiffCostFunction<GeometricError, 2, 12>(
        new GeometricError(vertices, triangles, camera_intrinsics, depth_map, depth_normals)));
}
