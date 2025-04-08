#include "CostFunctions.h"

// 实现中需要用到的依赖
#include "Projection.h"
#include "ImageProcessor.h"
#include <sophus/se3.hpp>
#include <cmath>

DepthError::DepthError(const MeshModel::Vertex& vertex,
                                   const std::vector<MeshModel::Triangle>& triangles,
                                   const Eigen::Matrix3d& intrinsics,
                                   const cv::Mat& depth_image,
                                   const BVH& bvh,
                                   double weight)
    : vertex_(vertex),
      triangles_(triangles),
      intrinsics_(intrinsics),
      depth_image_(depth_image),
      bvh_(bvh),
      weight_(weight) {
    // 可在此进行一些预处理
}

bool DepthError::Evaluate(const Eigen::Matrix<double, 6, 1>& se3,
                                double depth,
                                double& residual,
                                Eigen::Matrix<double, 1, 6>* jacobian_pose,
                                double* jacobian_depth) const {
    // 将 se3 转换为 SE3 变换
    Sophus::SE3d transform = Sophus::SE3d::exp(se3);
    Eigen::Matrix3d R = transform.rotationMatrix();
    Eigen::Vector3d t = transform.translation();

    // 判断顶点是否可见：直接调用 Projection::isVertexVisible
    if (!Projection::isVertexVisible(vertex_, intrinsics_, R, t, bvh_, depth_image_.cols, depth_image_.rows)) {
        residual = 0.0;
        if (jacobian_pose) jacobian_pose->setZero();
        if (jacobian_depth) *jacobian_depth = 0.0;
        return true;
    }

    // 投影计算：直接调用 Projection::projectPoint
    Eigen::Vector2d proj = Projection::projectPoint(vertex_, intrinsics_, R, t);
    int u = static_cast<int>(proj(0));
    int v = static_cast<int>(proj(1));
    if (u < 0 || u >= depth_image_.cols || v < 0 || v >= depth_image_.rows) {
        residual = 0.0;
        if (jacobian_pose) jacobian_pose->setZero();
        if (jacobian_depth) *jacobian_depth = 0.0;
        return true;
    }

    // 获取图像像素值（假设图像为 CV_32F 类型）
    float pixel_value = ImageProcessor::getBilinearInterpolatedValue(depth_image_, proj(0), proj(1));
    double sqrt_weight = std::sqrt(weight_);
    residual = sqrt_weight * (pixel_value - depth);

    // 计算雅可比：调用 computeJacobian 封装函数
    if (jacobian_pose || jacobian_depth) {
        // Eigen::Matrix<double, 1, 6> J_total = computeJacobian(vertex_, intrinsics_, R, t, current_image_, u, v);
        Eigen::Matrix<double, 1, 6> J_total = computeNumericalJacobian(se3, depth);
        if (jacobian_pose) {
            *jacobian_pose = sqrt_weight * J_total;
        }
        if (jacobian_depth) {
            *jacobian_depth = -sqrt_weight;
        }
    }

    return true;
}

Eigen::Matrix<double, 1, 6> DepthError::computeAnalyticalJacobian(const MeshModel::Vertex& vertex,
                                                                        const Eigen::Matrix3d& intrinsics,
                                                                        const Eigen::Matrix3d& R,
                                                                        const Eigen::Vector3d& t,
                                                                        const cv::Mat& image,
                                                                        int u, int v) const {
    Eigen::Matrix<double, 1, 6> J;
    J.setZero();

    // 计算图像梯度
    auto grad = ImageProcessor::computeGradient(image, u, v);
    double grad_u = grad.first, grad_v = grad.second;
    Eigen::Matrix<double, 1, 2> J_grad;
    J_grad << grad_u, grad_v;

    // 计算点在相机坐标系下的坐标
    Eigen::Vector3d point_world(vertex.x, vertex.y, vertex.z);
    Eigen::Vector3d point_cam = R.transpose() * (point_world - t);
    double X = point_cam(0), Y = point_cam(1), Z = point_cam(2);
    double fx = intrinsics(0, 0), fy = intrinsics(1, 1);
    double P_c1 = X / Z, P_c2 = Y / Z, P_c3 = Z;

    // 计算投影雅可比，参考相机模型导数公式
    Eigen::Matrix<double, 2, 3> J_proj;
    J_proj << fx / P_c3,      0, -fx * P_c1 / (P_c3 * P_c3),
                  0,    fy / P_c3, -fy * P_c2 / (P_c3 * P_c3);

    // 计算 SE3 对投影点的影响（基于李代数求导）
    // 此处采用与之前 MultiViewPhotometricError::computeJacobian 类似的实现
    Eigen::Matrix<double, 3, 6> J_se3;
    Eigen::Vector3d p_diff = point_world - t;  // p_w - t
    Eigen::Matrix3d skew;
    skew << 0,           -p_diff(2),  p_diff(1),
            p_diff(2),    0,         -p_diff(0),
            -p_diff(1),   p_diff(0),   0;
    J_se3 << R.transpose() * skew, -R.transpose();

    // 最终雅可比为链式法则相乘
    J = J_grad * J_proj * J_se3;
    return J;
}

Eigen::Matrix<double, 1, 6> DepthError::computeNumericalJacobian(const Eigen::Matrix<double, 6, 1>& se3,
                                                                        double depth) const {
    double epsilon = 1e-6;
    double sqrt_weight = std::sqrt(weight_);
    
    // 先计算当前 se3 参数下的光度误差 error0
    Sophus::SE3d transform = Sophus::SE3d::exp(se3);
    Eigen::Matrix3d R = transform.rotationMatrix();
    Eigen::Vector3d t = transform.translation();

    // 检查顶点是否可见
    if (!Projection::isVertexVisible(vertex_, intrinsics_, R, t, bvh_, depth_image__.cols, depth_image__.rows)) {
        return Eigen::Matrix<double, 1, 6>::Zero();
    }

    // 投影计算
    Eigen::Vector2d proj = Projection::projectPoint(vertex_, intrinsics_, R, t);
    int u = static_cast<int>(proj(0));
    int v = static_cast<int>(proj(1));
    if (u < 0 || u >= depth_image_.cols || v < 0 || v >= depth_image_.rows) {
        return Eigen::Matrix<double, 1, 6>::Zero();
    }

    // 获取当前像素值（假设图像为 CV_32F 类型）
    float pixel_value = pixel_value = ImageProcessor::getBilinearInterpolatedValue(depth_image_, proj(0), proj(1));;
    double error0 = sqrt_weight * (pixel_value - depth);

    // 数值雅可比
    Eigen::Matrix<double, 1, 6> J_num;
    J_num.setZero();

    // 对 se3 中的每个自由度施加微小扰动，计算有限差分
    for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double, 6, 1> se3_perturbed = se3;
        se3_perturbed(i) += epsilon;

        Sophus::SE3d transform_perturbed = Sophus::SE3d::exp(se3_perturbed);
        Eigen::Matrix3d R_perturbed = transform_perturbed.rotationMatrix();
        Eigen::Vector3d t_perturbed = transform_perturbed.translation();

        // 检查扰动后的顶点是否可见
        if (!Projection::isVertexVisible(vertex_, intrinsics_, R_perturbed, t_perturbed, bvh_, depth_image_.cols, depth_image_.rows)) {
            continue;
        }
        Eigen::Vector2d proj_perturbed = Projection::projectPoint(vertex_, intrinsics_, R_perturbed, t_perturbed);
        int u_pert = static_cast<int>(proj_perturbed(0));
        int v_pert = static_cast<int>(proj_perturbed(1));
        if (u_pert < 0 || u_pert >= depth_image_.cols || v_pert < 0 || v_pert >= depth_image_.rows) {
            continue;
        }

        float pixel_value_perturbed = ImageProcessor::getBilinearInterpolatedValue(depth_image_, proj_perturbed(0), proj_perturbed(1));
        double error_perturbed = sqrt_weight * (pixel_value_perturbed - depth);

        // 计算有限差分
        J_num(i) = (error_perturbed - error0) / epsilon;
    }
    return J_num;
}