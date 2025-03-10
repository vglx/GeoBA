#include "CostFunctions.h"
#include "Projection.h"
#include <sophus/se3.hpp>
#include "ImageProcessor.h"

LocalGeometricError::LocalGeometricError(const std::vector<MeshModel::Vertex>& vertices,
                                         const std::vector<MeshModel::Triangle>& triangles,
                                         const Eigen::Matrix3d& camera_intrinsics,
                                         const cv::Mat& depth_map,
                                         double weight_local_depth)
    : vertices_(vertices),
      triangles_(triangles),
      camera_intrinsics_(camera_intrinsics),
      depth_map_(depth_map),
      weight_local_depth_(weight_local_depth) {
        // 这里设置残差维度为 vertices_.size()，参数块维度为6
        set_num_residuals(vertices_.size());
        mutable_parameter_block_sizes()->push_back(6);
}

bool LocalGeometricError::Evaluate(double const* const* param, double* residuals, double** jacobians) const {
    const double* se3 = param[0];
    
    // 计算 SE3 变换
    Eigen::Map<const Eigen::Matrix<double,6,1>> se3_vec(se3);
    Sophus::SE3d transform = Sophus::SE3d::exp(se3_vec);
    Eigen::Matrix3d rotation = transform.rotationMatrix();
    Eigen::Vector3d translation = transform.translation();

    // 计算每个顶点是否可见
    std::vector<bool> visFlags = Projection::handleOcclusion(
        vertices_, triangles_, camera_intrinsics_, rotation, translation,
        depth_map_.cols, depth_map_.rows);

    // 预先计算 sqrt(weight) 以便构造残差
    double sqrt_weight = std::sqrt(weight_local_depth_);

    int vis_count = 0;

    // 注意：残差维度应在构造函数中动态设置为 vertices_.size()
    // 遍历每个顶点，分别填入对应的 residuals[i] 以及雅可比矩阵第 i 行
    for (size_t i = 0; i < vertices_.size(); ++i) {
        // 默认对每个顶点先置零（无效或异常情况）
        residuals[i] = 0.0;
        if (jacobians && jacobians[0]) {
            // 用循环将第 i 行全部置零
            for (int j = 0; j < 6; ++j) {
                jacobians[0][i * 6 + j] = 0.0;
            }
        }

        if (!visFlags[i])
            continue;  // 如果该点不可见，保持零残差和雅可比

        vis_count++;
        
        // 计算该点的投影
        Eigen::Vector2d proj = Projection::projectPoint(
            vertices_[i], camera_intrinsics_, rotation, translation);

        int u = static_cast<int>(proj(0));
        int v = static_cast<int>(proj(1));

        // 检查投影是否在图像范围内
        if (u < 0 || u >= depth_map_.cols || v < 0 || v >= depth_map_.rows)
            continue;
            
        float measured_depth = depth_map_.at<float>(v, u);
        if (measured_depth <= 0 || !std::isfinite(measured_depth))
            continue;
        
        double predicted_depth = Projection::computeVertexDepth(
            vertices_[i], camera_intrinsics_, rotation, translation);

        double diff = measured_depth - predicted_depth;

        // std::cout << "diff: " << diff << std::endl;

        // 设置残差：采用 sqrt(weight) 乘以测量误差
        residuals[i] = sqrt_weight * diff;

        // 如果要求雅可比，则计算该顶点的雅可比（注意残差关于预测深度的导数为 -sqrt_weight）
        if (jacobians && jacobians[0]) {
            Eigen::Matrix<double,1,6> J_i = computeAnalyticalJacobian(
                rotation, translation, camera_intrinsics_, vertices_[i], u, v, depth_map_);
            // 由于残差 = sqrt_weight*(measured_depth - predicted_depth)，
            // 而 measured_depth 对参数没有依赖，所以雅可比为 -sqrt_weight * d(predicted_depth)/d(param)
            Eigen::Matrix<double,1,6> row = -sqrt_weight * J_i;
            // 填入雅可比矩阵的第 i 行
            for (int j = 0; j < 6; ++j) {
                jacobians[0][i * 6 + j] = row(j);
            }
        }
    }

    std::cout << "vis_count: " << vis_count << std::endl;

    return true;
}

Eigen::Matrix<double,1,6> LocalGeometricError::computeAnalyticalJacobian(
    const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& translation,
    const Eigen::Matrix3d& intrinsics,
    const MeshModel::Vertex& vertex,
    int u, int v,
    const cv::Mat& image) const {

    Eigen::Matrix<double,1,6> J;
    J.setZero();

    Eigen::Vector3d point_world(vertex.x, vertex.y, vertex.z);

    // **1. 计算相机坐标点**
    Eigen::Vector3d point_cam = rotation.transpose() * (point_world - translation);
    double X = point_cam(0);
    double Y = point_cam(1);
    double Z = point_cam(2);

    // **2. 计算投影中间变量**
    double fx = intrinsics(0, 0);
    double fy = intrinsics(1, 1);
    double cx = intrinsics(0, 2);
    double cy = intrinsics(1, 2);

    double P_c3 = Z;
    double P_c1 = X / Z;
    double P_c2 = Y / Z;

    auto [grad_Z_u, grad_Z_v] = ImageProcessor::computeGradient(image, u, v);

    Eigen::Matrix<double,1,2> J_grad;
    J_grad << grad_Z_u, grad_Z_v; // [∂Z/∂u  ∂Z/∂v]

    // **4. 计算投影雅可比**
    Eigen::Matrix<double,2,3> J_proj;
    J_proj << fx / P_c3, 0, -fx * P_c1 / (P_c3 * P_c3),
              0, fy / P_c3, -fy * P_c2 / (P_c3 * P_c3);

    // **5. 计算相机坐标系到李代数 SE(3) 的雅可比**
    Eigen::Matrix<double,3,6> J_se3;
    J_se3 << 1, 0, 0,  0,  Z, -Y,
             0, 1, 0, -Z,  0,  X,
             0, 0, 1,  Y, -X,  0;

    // **6. 计算解析雅可比部分（针对深度残差）**
    // 提取 R^T 的第三行作为 r3
    Eigen::RowVector3d r3 = rotation.transpose().row(2); // [r_{3,0}, r_{3,1}, r_{3,2}]

    // 根据推导，对 r = measured_depth - Z, 有：
    // ∂r/∂ξ = [ p_{c,y},  -p_{c,x},  0,  r_{3,0},  r_{3,1},  r_{3,2} ]
    Eigen::Matrix<double,6,1> error_term;
    error_term << Y, -X, 0, r3(0), r3(1), r3(2);

    // **7. 计算最终雅可比**
    Eigen::Matrix<double,1,6> J_full = J_grad * J_proj * J_se3 + error_term.transpose();
    J = J_full;
    // std::cout << "Jacobian J: " << J << std::endl;
    return J;
}

ceres::CostFunction* LocalGeometricError::Create(
    const std::vector<MeshModel::Vertex>& vertices,
    const std::vector<MeshModel::Triangle>& triangles,
    const Eigen::Matrix3d& camera_intrinsics,
    const cv::Mat& depth_map,
    double weight_local_depth) {

    return new LocalGeometricError(vertices, triangles, camera_intrinsics, depth_map, weight_local_depth);
}

NormalConsistencyError::NormalConsistencyError(const std::vector<MeshModel::Vertex>& vertices,
                                               const std::vector<MeshModel::Triangle>& triangles,
                                               const Eigen::Matrix3d& camera_intrinsics,
                                               const std::vector<cv::Mat>& depth_maps,
                                               double weight)
    : vertices_(vertices),
      triangles_(triangles),
      camera_intrinsics_(camera_intrinsics),
      depth_maps_(depth_maps),
      weight_(weight) {
        set_num_residuals(vertices_.size() * 3);
        mutable_parameter_block_sizes()->push_back(6);  // 当前帧 SE(3)
        mutable_parameter_block_sizes()->push_back(6);  // 参考帧1 SE(3)
        mutable_parameter_block_sizes()->push_back(6);  // 参考帧2 SE(3)
}

bool NormalConsistencyError::Evaluate(double const* const* param, double* residuals, double** jacobians) const {
    // 解析三帧 SE(3)（优化变量）
    Eigen::Map<const Eigen::Matrix<double,6,1>> se3_vec_0(param[0]);
    Eigen::Map<const Eigen::Matrix<double,6,1>> se3_vec_1(param[1]);
    Eigen::Map<const Eigen::Matrix<double,6,1>> se3_vec_2(param[2]);

    Sophus::SE3d transform_0 = Sophus::SE3d::exp(se3_vec_0);
    Sophus::SE3d transform_1 = Sophus::SE3d::exp(se3_vec_1);
    Sophus::SE3d transform_2 = Sophus::SE3d::exp(se3_vec_2);

    Eigen::Matrix3d rotations[3] = {
        transform_0.rotationMatrix(),
        transform_1.rotationMatrix(),
        transform_2.rotationMatrix()
    };

    Eigen::Vector3d translations[3] = {
        transform_0.translation(),
        transform_1.translation(),
        transform_2.translation()
    };

    int res_index = 0;

    // 检查可见性（此处假设 Projection::handleOcclusion 接受 triangles_ 作为参数）
    std::vector<bool> visibility[3];
    for (int f = 0; f < 3; ++f) {
        visibility[f] = Projection::handleOcclusion(vertices_, triangles_, camera_intrinsics_, rotations[f], translations[f], depth_maps_[f].cols, depth_maps_[f].rows);
    }

    int num_res = this->num_residuals();

    // 先将所有残差初始化为 0
    for (int i = 0; i < num_res; ++i) {
        residuals[i] = 0.0;
    }
    
    // 如果要求雅可比，也将雅可比矩阵的所有元素初始化为 0
    if (jacobians) {
        int param_size = 6; // 参数块大小为 6
        // 假设有 3 个参数块
        for (int b = 0; b < 3; ++b) {
            if (jacobians[b]) {
                for (int i = 0; i < num_res * param_size; ++i) {
                    jacobians[b][i] = 0.0;
                }
            }
        }
    }

    // 遍历所有共视点（每个顶点）
    for (size_t i = 0; i < vertices_.size(); ++i) {
        // 必须三帧都可见
        if (!visibility[0][i] || !visibility[1][i] || !visibility[2][i]) {
            res_index += 3;  // 跳过时也要增加 3
            continue;
        }

        // 对三帧分别计算法向量（利用 computeNormal），记录有效性
        Eigen::Vector3d normals[3];
        bool valid[3] = {false, false, false};

        // 对每一帧 f
        // 注意：这里的 proj 计算的是像素投影位置
        for (int f = 0; f < 3; ++f) {
            Eigen::Vector2d proj = Projection::projectPoint(vertices_[i], camera_intrinsics_, rotations[f], translations[f]);
            int u = static_cast<int>(proj(0));
            int v = static_cast<int>(proj(1));

            if (u >= 0 && u < depth_maps_[f].cols && v >= 0 && v < depth_maps_[f].rows) {
                normals[f] = ImageProcessor::computeNormal(u, v, depth_maps_[f], camera_intrinsics_);
                valid[f] = true;
            }
        }

        if (!valid[0] || !valid[1] || !valid[2]) {
            res_index += 3;
            continue;
        }

        // 计算三个法向量的归一化平均值以及 s = || n0+n1+n2 ||
        Eigen::Vector3d sum_normals = normals[0] + normals[1] + normals[2];
        double s = sum_normals.norm();
        Eigen::Vector3d avg_normal = sum_normals / s;

        // 获取顶点在世界坐标下的位置 p（假设 MeshModel::Vertex 转为 Eigen::Vector3d）
        Eigen::Vector3d p = Eigen::Vector3d(vertices_[i].x, vertices_[i].y, vertices_[i].z);

        // 对每一帧分别计算残差和雅可比
        for (int f = 0; f < 3; ++f) {
            double cos_theta = normals[f].dot(avg_normal);
            cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
            double sin_theta = std::sqrt(1 - cos_theta * cos_theta);
            double angle_error = std::acos(cos_theta);
            residuals[res_index] = weight_ * angle_error;

            if (jacobians) {
                // 对当前帧，计算像素坐标 (u,v) 及调用 ComputeJacobian
                Eigen::Vector2d proj = Projection::projectPoint(vertices_[i], camera_intrinsics_, rotations[f], translations[f]);
                int u = static_cast<int>(proj(0));
                int v = static_cast<int>(proj(1));
                // 从内参矩阵中提取 fx, fy
                double fx = camera_intrinsics_(0, 0);
                double fy = camera_intrinsics_(1, 1);
                
                // 调用 ComputeJacobian，注意传入所有必要参数
                Eigen::Matrix<double, 1, 6> J = ComputeJacobian(u, v, depth_maps_[f], camera_intrinsics_,
                                                                normals[f], avg_normal, s,
                                                                p, rotations[f], translations[f],
                                                                fx, fy);
                // 将雅可比结果写入 jacobians 对应的优化变量块中
                // 此处我们假设 jacobians[f] 指向第 f 个参数块的雅可比数组
                for (int j = 0; j < 6; ++j) {
                    jacobians[f][res_index * 6 + j] = J(j);
                }
            }
            res_index++;
        }
    }

    return true;
}

Eigen::Matrix<double, 1, 6> NormalConsistencyError::ComputeJacobian(
    int u, int v,
    const cv::Mat &depthMap,
    const Eigen::Matrix3d &camera_intrinsics,
    const Eigen::Vector3d& n_i,         // 当前帧法向量 at (u,v)
    const Eigen::Vector3d& n_avg,         // 三帧归一化平均法向量
    double s,                           // s = || n_i + n_j + n_k ||
    const Eigen::Vector3d& p,             // 顶点在世界坐标系下的位置
    const Eigen::Matrix3d& R,             // 当前帧旋转矩阵（相机到世界）
    const Eigen::Vector3d& t,             // 当前帧平移向量
    double fx, double fy) const {

    // 1. 计算 Jₙ,(u,v) = ∂nᵢ/∂(u,v) —— 通过中心差分
    double delta = 1.0; // 像素步长
    Eigen::Vector3d n_right = ImageProcessor::computeNormal(u + 1, v, depthMap, camera_intrinsics);
    Eigen::Vector3d n_left  = ImageProcessor::computeNormal(u - 1, v, depthMap, camera_intrinsics);
    Eigen::Vector3d n_down  = ImageProcessor::computeNormal(u, v + 1, depthMap, camera_intrinsics);
    Eigen::Vector3d n_up    = ImageProcessor::computeNormal(u, v - 1, depthMap, camera_intrinsics);
    Eigen::Vector3d dndu = (n_right - n_left) / (2.0 * delta);
    Eigen::Vector3d dndv = (n_down - n_up) / (2.0 * delta);
    Eigen::Matrix<double, 3, 2> J_n_uv; // 3×2
    J_n_uv.col(0) = dndu;
    J_n_uv.col(1) = dndv;

    // 2. 计算 ∂r/∂nᵢ
    double c = n_i.dot(n_avg);
    // 限制 c 在 [-1,1] 防止数值问题
    c = std::max(-1.0, std::min(1.0, c));
    double denom = 1.0 - c*c;
    double invSqrt = (denom > 1e-8) ? (1.0 / std::sqrt(denom)) : 0.0;
    // 直接项： n_avg^T + (1/s) * n_i^T*(I - n_avg*n_avg^T)
    Eigen::RowVector3d term = n_avg.transpose() + (1.0/s) * n_i.transpose() * (Eigen::Matrix3d::Identity() - n_avg * n_avg.transpose());
    // 投影到 n_i 的切平面： Pₙᵢ(a) = a - (a·nᵢ) nᵢ
    double projComp = term.dot(n_i);
    Eigen::RowVector3d projTerm = term - projComp * n_i.transpose();
    Eigen::RowVector3d dr_dn = -invSqrt * projTerm; // 1×3

    // 3. 计算 ∂(u,v)/∂p_c
    // 先计算 p_c = Rᵀ*(p - t) (位姿定义为相机到世界，则 p_c = Rᵀ(p-t))
    Eigen::Vector3d p_c = R.transpose() * (p - t);
    double X = p_c(0), Y = p_c(1), Z = p_c(2);
    Eigen::Matrix<double, 2, 3> J_uv_pc;
    J_uv_pc << fx / Z,          0,      -fx * X / (Z*Z),
                0,          fy / Z,      -fy * Y / (Z*Z);

    // 4. 计算 ∂p_c/∂ξ
    // 对于 p_c = Rᵀ*(p-t)，有： J_pc,ξ = [ -Rᵀ * hat(p-t) , -Rᵀ ]
    Eigen::Matrix3d hat_pt = Sophus::SO3d::hat(p - t);
    Eigen::Matrix<double, 3, 6> J_pc_xi;
    J_pc_xi << -R.transpose() * hat_pt, -R.transpose();

    // 5. 组合链式导数
    // 最终： ∂r/∂ξ = (∂r/∂nᵢ) (1×3) * J_n_uv (3×2) * J_uv_pc (2×3) * J_pc_xi (3×6)
    Eigen::Matrix<double, 1, 6> J_total = dr_dn * J_n_uv * J_uv_pc * J_pc_xi;
    
    return J_total;
}

ceres::CostFunction* NormalConsistencyError::Create(
    const std::vector<MeshModel::Vertex>& vertices,
    const std::vector<MeshModel::Triangle>& triangles,
    const Eigen::Matrix3d& camera_intrinsics,
    const std::vector<cv::Mat>& depth_maps,
    double weight) {

    return new NormalConsistencyError(vertices, triangles, camera_intrinsics, depth_maps, weight);
}