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
    // 1. 计算 SE3 变换（T 类型）
    Eigen::Map<const Eigen::Matrix<T,6,1>> se3_vec(se3);
    Sophus::SE3<T> transform = Sophus::SE3<T>::exp(se3_vec);
    Eigen::Matrix<T,3,3> rotationT = transform.rotationMatrix();
    Eigen::Matrix<T,3,1> translationT = transform.translation();

    // 2. 将摄像机内参转换为 T 类型（保留梯度）
    Eigen::Matrix<T,3,3> intrinsicsT = camera_intrinsics_.template cast<T>();

    // 3. 使用模板化 Projection 接口计算投影、深度和软可见性权重
    std::vector<Eigen::Matrix<T,2,1>> projected_points = 
        Projection::projectPoints<T>(vertices_, intrinsicsT, rotationT, translationT);
    std::vector<T> vertex_depths = 
        Projection::computeVertexDepths<T>(vertices_, intrinsicsT, rotationT, translationT);
    std::vector<T> visWeights = 
        Projection::handleOcclusion<T>(vertices_, triangles_, intrinsicsT, rotationT, translationT,
                                            depth_map_.cols, depth_map_.rows);

    // 4. 累计残差
    T depth_residual = T(0);
    T normal_residual = T(0);
    size_t vertex_count = vertices_.size();

    for (size_t i = 0; i < vertex_count; ++i) {
        // 4.1 检查投影像素是否在图像范围内
        Eigen::Matrix<T,2,1> proj = projected_points[i];
        int u, v;
        if constexpr (std::is_same_v<T, ceres::Jet<double, 6>>) {
            u = static_cast<int>(proj(0).a);  // 如果 T 是 Jet 类型，提取 `a`（标量部分）
            v = static_cast<int>(proj(1).a);
        } 
        else {
            u = static_cast<int>(proj(0));  // 直接转换
            v = static_cast<int>(proj(1));
        }
        
        std::cout << "11111" << std::endl;
        if (u < 0 || u >= depth_map_.cols || v < 0 || v >= depth_map_.rows)
            continue;
        
        std::cout << "22222" << std::endl;
        // 4.2 深度残差计算
        float measured_depth = depth_map_.at<float>(v, u);
        if (measured_depth <= 0 || !std::isfinite(measured_depth))
            continue;
        T diff = T(measured_depth) - vertex_depths[i];
        if constexpr (std::is_same_v<T, ceres::Jet<double, 6>>) {
            std::cout << "measured_depth: " << measured_depth 
                      << ", vertex_depth: " << vertex_depths[i].a  // 提取 `Jet` 的标量部分
                      << ", diff: " << diff.a 
                      << std::endl;
        } else {
            std::cout << "measured_depth: " << measured_depth 
                      << ", vertex_depth: " << vertex_depths[i] 
                      << ", diff: " << diff 
                      << std::endl;
        }
        depth_residual += weight_local_depth_ * visWeights[i] * diff * diff;

        // // 4.3 法向量残差计算
        // // (a) 从网格数据中获取顶点法向量，转换为 T 类型
        // Eigen::Matrix<T,3,1> mesh_normal;
        // mesh_normal << T(vertices_[i].nx), T(vertices_[i].ny), T(vertices_[i].nz);
        // // (b) 将网格法向量通过 SE3 的旋转部分变换到相机坐标系
        // Eigen::Matrix<T,3,1> transformed_normal = rotationT * mesh_normal;
        // // (c) 从深度法向量图中获取测量法向量（cv::Vec3f），转换为 T 类型
        // cv::Vec3f measured_normal_cv = depth_normals_.at<cv::Vec3f>(v, u);
        // Eigen::Matrix<T,3,1> measured_normal;
        // measured_normal << T(measured_normal_cv[0]), T(measured_normal_cv[1]), T(measured_normal_cv[2]);

        // // (d) 归一化法向量（如果范数足够大）
        // if (transformed_normal.norm() > T(1e-6))
        //     transformed_normal.normalize();
        // if (measured_normal.norm() > T(1e-6))
        //     measured_normal.normalize();

        // // (e) 法向量误差定义：1 - 点积，范围 [0,2]
        // T n_error = T(1.0) - transformed_normal.dot(measured_normal);
        // normal_residual += weight_local_normal_ * visWeights[i] * n_error * n_error;
    }

    // 5. 写入残差
    residual[0] = depth_residual;
    residual[1] = normal_residual;
    std::cout << "<---------" << residual[0] << std::endl;
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

// TripletGlobalError::TripletGlobalError(const std::vector<MeshModel::Vertex>& vertices,
//                                        const std::vector<MeshModel::Triangle>& triangles,
//                                        const Eigen::Matrix3f& camera_intrinsics,
//                                        const cv::Mat& depth_map_i,
//                                        const cv::Mat& depth_map_j,
//                                        const cv::Mat& depth_map_k,
//                                        const cv::Mat& image_i,
//                                        const cv::Mat& image_j,
//                                        const cv::Mat& image_k,
//                                        double weight_global_depth,
//                                        double weight_global_gradient)
//     : vertices_(vertices),
//       triangles_(triangles),
//       camera_intrinsics_(camera_intrinsics),
//       depth_map_i_(depth_map_i),
//       depth_map_j_(depth_map_j),
//       depth_map_k_(depth_map_k),
//       image_i_(image_i),
//       image_j_(image_j),
//       image_k_(image_k),
//       weight_global_depth_(weight_global_depth),
//       weight_global_gradient_(weight_global_gradient) {}

// template <typename T>
// bool TripletGlobalError::operator()(const T* const pose_i,
//                                     const T* const pose_j,
//                                     const T* const pose_k,
//                                     T* residual) const {
//     size_t vertex_count = vertices_.size();

//     // 1. 将优化变量转换为 SE(3) 变换
//     Sophus::SE3<T> T_i = Sophus::SE3<T>::exp(Eigen::Map<const Eigen::Matrix<T, 6, 1>>(pose_i));
//     Sophus::SE3<T> T_j = Sophus::SE3<T>::exp(Eigen::Map<const Eigen::Matrix<T, 6, 1>>(pose_j));
//     Sophus::SE3<T> T_k = Sophus::SE3<T>::exp(Eigen::Map<const Eigen::Matrix<T, 6, 1>>(pose_k));

//     Eigen::Matrix<T, 3, 3> rotation_i = T_i.rotationMatrix();
//     Eigen::Matrix<T, 3, 1> translation_i = T_i.translation();
//     Eigen::Matrix<T, 3, 3> rotation_j = T_j.rotationMatrix();
//     Eigen::Matrix<T, 3, 1> translation_j = T_j.translation();
//     Eigen::Matrix<T, 3, 3> rotation_k = T_k.rotationMatrix();
//     Eigen::Matrix<T, 3, 1> translation_k = T_k.translation();

//     // 2. 计算可见性 mask
//     std::vector<bool> visibility_mask_i = Projection::handleOcclusion(vertices_, triangles_, camera_intrinsics_, rotation_i, translation_i, depth_map_i_.cols, depth_map_i_.rows);
//     std::vector<bool> visibility_mask_j = Projection::handleOcclusion(vertices_, triangles_, camera_intrinsics_, rotation_j, translation_j, depth_map_j_.cols, depth_map_j_.rows);
//     std::vector<bool> visibility_mask_k = Projection::handleOcclusion(vertices_, triangles_, camera_intrinsics_, rotation_k, translation_k, depth_map_k_.cols, depth_map_k_.rows);

//     // 3. 计算所有顶点的投影
//     std::vector<Eigen::Vector2f> projected_i = Projection::projectPoints(vertices_, camera_intrinsics_, rotation_i, translation_i);
//     std::vector<Eigen::Vector2f> projected_j = Projection::projectPoints(vertices_, camera_intrinsics_, rotation_j, translation_j);
//     std::vector<Eigen::Vector2f> projected_k = Projection::projectPoints(vertices_, camera_intrinsics_, rotation_k, translation_k);

//     // 4. 计算误差
//     T sum_grad_error_i = T(0);
//     T sum_depth_error_i = T(0);
//     T sum_grad_error_j = T(0);
//     T sum_depth_error_j = T(0);
//     T sum_grad_error_k = T(0);
//     T sum_depth_error_k = T(0);
//     size_t valid_count = 0;

//     for (size_t vertex_idx = 0; vertex_idx < vertex_count; ++vertex_idx) {
//         // 只计算三帧都可见的点
//         if (!visibility_mask_i[vertex_idx] || !visibility_mask_j[vertex_idx] || !visibility_mask_k[vertex_idx]) continue;

//         Eigen::Vector2f pixel_i = projected_i[vertex_idx];
//         Eigen::Vector2f pixel_j = projected_j[vertex_idx];
//         Eigen::Vector2f pixel_k = projected_k[vertex_idx];

//         int u_i = static_cast<int>(pixel_i.x());
//         int v_i = static_cast<int>(pixel_i.y());
//         int u_j = static_cast<int>(pixel_j.x());
//         int v_j = static_cast<int>(pixel_j.y());
//         int u_k = static_cast<int>(pixel_k.x());
//         int v_k = static_cast<int>(pixel_k.y());

//         if (u_i >= 1 && u_i < depth_map_i_.cols - 1 && v_i >= 1 && v_i < depth_map_i_.rows - 1 &&
//             u_j >= 1 && u_j < depth_map_j_.cols - 1 && v_j >= 1 && v_j < depth_map_j_.rows - 1 &&
//             u_k >= 1 && u_k < depth_map_k_.cols - 1 && v_k >= 1 && v_k < depth_map_k_.rows - 1) {

//             T depth_i_val = T(depth_map_i_.at<float>(v_i, u_i));
//             T depth_j_val = T(depth_map_j_.at<float>(v_j, u_j));
//             T depth_k_val = T(depth_map_k_.at<float>(v_k, u_k));

//             T gradient_i_val = T(ImageProcessor::computeGradient(image_i_, u_i, v_i));
//             T gradient_j_val = T(ImageProcessor::computeGradient(image_j_, u_j, v_j));
//             T gradient_k_val = T(ImageProcessor::computeGradient(image_k_, u_k, v_k));

//             if (depth_i_val > T(0.1) && depth_j_val > T(0.1) && depth_k_val > T(0.1)) {
//                 T grad_mean = (gradient_i_val + gradient_j_val + gradient_k_val) / T(3.0);
//                 T depth_mean = (depth_i_val + depth_j_val + depth_k_val) / T(3.0);

//                 sum_grad_error_i += (gradient_i_val - grad_mean) * (gradient_i_val - grad_mean);
//                 sum_grad_error_j += (gradient_j_val - grad_mean) * (gradient_j_val - grad_mean);
//                 sum_grad_error_k += (gradient_k_val - grad_mean) * (gradient_k_val - grad_mean);

//                 sum_depth_error_i += (depth_i_val - depth_mean) * (depth_i_val - depth_mean);
//                 sum_depth_error_j += (depth_j_val - depth_mean) * (depth_j_val - depth_mean);
//                 sum_depth_error_k += (depth_k_val - depth_mean) * (depth_k_val - depth_mean);

//                 valid_count++;
//             }
//         }
//     }

//     // 5. 计算均方根误差 (RMSE)
//     if (valid_count > 0) {
//         T inv_count = T(1.0) / T(valid_count);
//         residual[0] = weight_global_gradient_ * sqrt(sum_grad_error_i * inv_count);
//         residual[1] = weight_global_depth_ * sqrt(sum_depth_error_i * inv_count);
//         residual[2] = weight_global_gradient_ * sqrt(sum_grad_error_j * inv_count);
//         residual[3] = weight_global_depth_ * sqrt(sum_depth_error_j * inv_count);
//         residual[4] = weight_global_gradient_ * sqrt(sum_grad_error_k * inv_count);
//         residual[5] = weight_global_depth_ * sqrt(sum_depth_error_k * inv_count);
//     } else {
//         residual[0] = residual[1] = residual[2] =
//         residual[3] = residual[4] = residual[5] = T(0);
//     }

//     return true;
// }

// ceres::CostFunction* TripletGlobalError::Create(
//     const std::vector<MeshModel::Vertex>& vertices,
//     const std::vector<MeshModel::Triangle>& triangles,
//     const Eigen::Matrix3f& camera_intrinsics,
//     const cv::Mat& depth_map_i,
//     const cv::Mat& depth_map_j,
//     const cv::Mat& depth_map_k,
//     const cv::Mat& image_i,
//     const cv::Mat& image_j,
//     const cv::Mat& image_k,
//     double weight_global_depth,
//     double weight_global_gradient) {
//     return new ceres::AutoDiffCostFunction<TripletGlobalError, 6, 6, 6, 6>(
//         new TripletGlobalError(vertices, triangles, camera_intrinsics,
//         depth_map_i, depth_map_j, depth_map_k,
//         image_i, image_j, image_k, weight_global_depth, weight_global_gradient));
// }