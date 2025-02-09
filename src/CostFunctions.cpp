#include "CostFunctions.h"
#include "Projection.h"
#include <sophus/se3.hpp>

LocalGeometricError::LocalGeometricError(const std::vector<MeshModel::Vertex>& vertices,
                                         const std::vector<MeshModel::Triangle>& triangles,
                                         const Eigen::Matrix3f& camera_intrinsics,
                                         const cv::Mat& depth_map,
                                         const Eigen::MatrixXf& depth_normals,
                                         double weight_local_depth,
                                         double weight_local_normal)
    : vertices_(vertices),
      triangles_(triangles),
      camera_intrinsics_(camera_intrinsics),
      depth_map_(depth_map),
      depth_normals_(depth_normals),
      weight_local_depth_(weight_local_depth),
      weight_local_normal_(weight_local_normal) {}

template <typename T>
bool LocalGeometricError::operator()(const T* const se3, T* residual) const {
    Eigen::Map<const Eigen::Matrix<T, 6, 1>> se3_vec(se3);
    Sophus::SE3<T> transform = Sophus::SE3<T>::exp(se3_vec);
    Eigen::Matrix<T, 3, 3> rotation = transform.rotationMatrix();
    Eigen::Matrix<T, 3, 1> translation = transform.translation();

    size_t vertex_count = vertices_.size();
    T depth_residual = T(0);
    T normal_residual = T(0);

    std::vector<bool> visibility_mask = Projection::handleOcclusion(vertices_, triangles_, camera_intrinsics_, rotation, translation, depth_map_.cols, depth_map_.rows);
    std::vector<Eigen::Vector2f> projected_points = Projection::projectPoints(vertices_, camera_intrinsics_, rotation, translation);
    
    std::vector<float> vertex_depths_float = Projection::computeVertexDepths(vertices_, camera_intrinsics_, rotation, translation);
    std::vector<T> vertex_depths;
    vertex_depths.reserve(vertex_depths_float.size());
    for (const auto& d : vertex_depths_float) {
        vertex_depths.push_back(T(d));
    }

    for (size_t vertex_idx = 0; vertex_idx < vertex_count; ++vertex_idx) {
        if (!visibility_mask[vertex_idx]) continue;

        Eigen::Vector2f projected_2d = projected_points[vertex_idx];
        int u = static_cast<int>(projected_2d.x());
        int v = static_cast<int>(projected_2d.y());

        if (u >= 1 && u < depth_map_.cols - 1 && v >= 1 && v < depth_map_.rows - 1) {
            float camera_depth = depth_map_.at<float>(v, u);
            if (camera_depth > 0 && std::isfinite(camera_depth)) {
                T vertex_depth = vertex_depths[vertex_idx];
                depth_residual += weight_local_depth_ * (T(camera_depth) - vertex_depth) * (T(camera_depth) - vertex_depth);
            }

            Eigen::Matrix<T, 3, 1> mesh_normal(T(vertices_[vertex_idx].nx), T(vertices_[vertex_idx].ny), T(vertices_[vertex_idx].nz));
            Eigen::Matrix<T, 3, 1> transformed_normal = rotation * mesh_normal;
            Eigen::Matrix<T, 3, 1> depth_normal(
                T(depth_normals_(v, u * 3)),
                T(depth_normals_(v, u * 3 + 1)),
                T(depth_normals_(v, u * 3 + 2)));

            if (depth_normal.norm() > 1e-6) depth_normal.normalize();
            if (transformed_normal.norm() > 1e-6) transformed_normal.normalize();
            T normal_error = T(1.0) - transformed_normal.dot(depth_normal);
            normal_residual += weight_local_normal_ * normal_error * normal_error;
        }
    }

    residual[0] = depth_residual;
    residual[1] = normal_residual;
    return true;
}

ceres::CostFunction* LocalGeometricError::Create(const std::vector<MeshModel::Vertex>& vertices,
                                                 const std::vector<MeshModel::Triangle>& triangles,
                                                 const Eigen::Matrix3f& camera_intrinsics,
                                                 const cv::Mat& depth_map,
                                                 const Eigen::MatrixXf& depth_normals,
                                                 double weight_local_depth,
                                                 double weight_local_normal) {
    return new ceres::AutoDiffCostFunction<LocalGeometricError, 2, 6>(
        new LocalGeometricError(vertices, triangles, camera_intrinsics, depth_map, depth_normals, weight_local_depth, weight_local_normal));
}

TripletGlobalError::TripletGlobalError(const std::vector<MeshModel::Vertex>& vertices,
                                       const Eigen::Matrix3f& camera_intrinsics,
                                       const cv::Mat& depth_map_i,
                                       const cv::Mat& depth_map_j,
                                       const cv::Mat& depth_map_k,
                                       const cv::Mat& image_i,
                                       const cv::Mat& image_j,
                                       const cv::Mat& image_k,
                                       double weight_global_depth,
                                       double weight_global_gradient)
    : vertices_(vertices),
      camera_intrinsics_(camera_intrinsics),
      depth_map_i_(depth_map_i),
      depth_map_j_(depth_map_j),
      depth_map_k_(depth_map_k),
      image_i_(image_i),
      image_j_(image_j),
      image_k_(image_k),
      weight_global_depth_(weight_global_depth),
      weight_global_gradient_(weight_global_gradient) {}

template <typename T>
bool TripletGlobalError::operator()(const T* const pose_i,
                                    const T* const pose_j,
                                    const T* const pose_k,
                                    T* residual) const {
    size_t vertex_count = vertices_.size();

    // 累加每一帧梯度和深度的平方误差
    T sum_grad_error_i = T(0);
    T sum_depth_error_i = T(0);
    T sum_grad_error_j = T(0);
    T sum_depth_error_j = T(0);
    T sum_grad_error_k = T(0);
    T sum_depth_error_k = T(0);
    size_t valid_count = 0;  // 有效像素数量

    // 将李代数参数转换为 SE(3) 变换
    Sophus::SE3<T> T_i = Sophus::SE3<T>::exp(Eigen::Map<const Eigen::Matrix<T, 6, 1>>(pose_i));
    Sophus::SE3<T> T_j = Sophus::SE3<T>::exp(Eigen::Map<const Eigen::Matrix<T, 6, 1>>(pose_j));
    Sophus::SE3<T> T_k = Sophus::SE3<T>::exp(Eigen::Map<const Eigen::Matrix<T, 6, 1>>(pose_k));

    for (size_t vertex_idx = 0; vertex_idx < vertex_count; ++vertex_idx) {
        Eigen::Vector3f world_point = vertices_[vertex_idx].position;

        // 计算该 3D 点在三帧中的坐标
        Eigen::Vector3<T> proj_i = T_i * world_point.cast<T>();
        Eigen::Vector3<T> proj_j = T_j * world_point.cast<T>();
        Eigen::Vector3<T> proj_k = T_k * world_point.cast<T>();

        // 投影到图像平面（Projection::projectPoint 返回 Eigen::Vector2f）
        Eigen::Vector2f pixel_i = Projection::projectPoint(proj_i, camera_intrinsics_);
        Eigen::Vector2f pixel_j = Projection::projectPoint(proj_j, camera_intrinsics_);
        Eigen::Vector2f pixel_k = Projection::projectPoint(proj_k, camera_intrinsics_);

        // 将投影结果转换为整数像素坐标
        int u_i = static_cast<int>(pixel_i.x());
        int v_i = static_cast<int>(pixel_i.y());
        int u_j = static_cast<int>(pixel_j.x());
        int v_j = static_cast<int>(pixel_j.y());
        int u_k = static_cast<int>(pixel_k.x());
        int v_k = static_cast<int>(pixel_k.y());

        // 检查三个帧的像素是否都在有效范围内
        if (u_i >= 1 && u_i < depth_map_i_.cols - 1 && v_i >= 1 && v_i < depth_map_i_.rows - 1 &&
            u_j >= 1 && u_j < depth_map_j_.cols - 1 && v_j >= 1 && v_j < depth_map_j_.rows - 1 &&
            u_k >= 1 && u_k < depth_map_k_.cols - 1 && v_k >= 1 && v_k < depth_map_k_.rows - 1) {

            // 对深度图直接使用最近邻取值
            T depth_i_val = T(depth_map_i_.at<float>(v_i, u_i));
            T depth_j_val = T(depth_map_j_.at<float>(v_j, u_j));
            T depth_k_val = T(depth_map_k_.at<float>(v_k, u_k));

            // 使用 computeGradient 获取梯度值（整数像素位置）
            T gradient_i_val = T(ImageProcessor::computeGradient(image_i_, u_i, v_i));
            T gradient_j_val = T(ImageProcessor::computeGradient(image_j_, u_j, v_j));
            T gradient_k_val = T(ImageProcessor::computeGradient(image_k_, u_k, v_k));

            // 仅保留深度大于 0.1 的有效像素
            if (depth_i_val > T(0.1) && depth_j_val > T(0.1) && depth_k_val > T(0.1)) {
                // 计算当前像素在三帧中的平均梯度和平均深度
                T grad_mean  = (gradient_i_val + gradient_j_val + gradient_k_val) / T(3.0);
                T depth_mean = (depth_i_val + depth_j_val + depth_k_val) / T(3.0);

                // 分别累加每一帧与均值之差的平方
                sum_grad_error_i += (gradient_i_val - grad_mean) * (gradient_i_val - grad_mean);
                sum_grad_error_j += (gradient_j_val - grad_mean) * (gradient_j_val - grad_mean);
                sum_grad_error_k += (gradient_k_val - grad_mean) * (gradient_k_val - grad_mean);

                sum_depth_error_i += (depth_i_val - depth_mean) * (depth_i_val - depth_mean);
                sum_depth_error_j += (depth_j_val - depth_mean) * (depth_j_val - depth_mean);
                sum_depth_error_k += (depth_k_val - depth_mean) * (depth_k_val - depth_mean);

                valid_count++;
            }
        }
    }

    // 如果存在有效像素，则计算均方根误差 (RMSE) 作为残差输出，否则置为 0
    if (valid_count > 0) {
        T inv_count = T(1.0) / T(valid_count);
        residual[0] = weight_global_gradient_ * sqrt(sum_grad_error_i * inv_count);
        residual[1] = weight_global_depth_    * sqrt(sum_depth_error_i * inv_count);
        residual[2] = weight_global_gradient_ * sqrt(sum_grad_error_j * inv_count);
        residual[3] = weight_global_depth_    * sqrt(sum_depth_error_j * inv_count);
        residual[4] = weight_global_gradient_ * sqrt(sum_grad_error_k * inv_count);
        residual[5] = weight_global_depth_    * sqrt(sum_depth_error_k * inv_count);
    } else {
        residual[0] = residual[1] = residual[2] =
        residual[3] = residual[4] = residual[5] = T(0);
    }

    return true;
}


ceres::CostFunction* TripletGlobalError::Create(
    const std::vector<MeshModel::Vertex>& vertices,
    const Eigen::Matrix3f& camera_intrinsics,
    const cv::Mat& depth_map_i,
    const cv::Mat& depth_map_j,
    const cv::Mat& depth_map_k,
    const cv::Mat& image_i,
    const cv::Mat& image_j,
    const cv::Mat& image_k,
    double weight_global_depth,
    double weight_global_gradient) {
    return new ceres::AutoDiffCostFunction<TripletGlobalError, 6, 6, 6, 6>(
        new TripletGlobalError(vertices, camera_intrinsics,
        depth_map_i, depth_map_j, depth_map_k,
        image_i, image_j, image_k, weight_global_depth, weight_global_gradient));
}