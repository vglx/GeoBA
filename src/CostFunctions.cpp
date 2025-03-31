#include "CostFunctions.h"

// 实现中需要用到的依赖
#include "Projection.h"
#include "ImageProcessor.h"
#include <sophus/se3.hpp>
#include <cmath>

PhotometricError::PhotometricError(const MeshModel::Vertex& vertex,
                                   const std::vector<MeshModel::Triangle>& triangles,
                                   const Eigen::Matrix3d& intrinsics,
                                   const cv::Mat& current_image,
                                   const BVH& bvh,
                                   double weight)
    : vertex_(vertex),
      triangles_(triangles),
      intrinsics_(intrinsics),
      current_image_(current_image),
      bvh_(bvh),
      weight_(weight) {
    // 可在此进行一些预处理
}

bool PhotometricError::Evaluate(const Eigen::Matrix<double, 6, 1>& se3,
                                double intensity,
                                double& residual,
                                Eigen::Matrix<double, 1, 6>* jacobian_pose,
                                double* jacobian_intensity) const {
    // 将 se3 转换为 SE3 变换
    Sophus::SE3d transform = Sophus::SE3d::exp(se3);
    Eigen::Matrix3d R = transform.rotationMatrix();
    Eigen::Vector3d t = transform.translation();

    // 判断顶点是否可见：直接调用 Projection::isVertexVisible
    if (!Projection::isVertexVisible(vertex_, intrinsics_, R, t, bvh_, current_image_.cols, current_image_.rows)) {
        residual = 0.0;
        if (jacobian_pose) jacobian_pose->setZero();
        if (jacobian_intensity) *jacobian_intensity = 0.0;
        return true;
    }

    // 投影计算：直接调用 Projection::projectPoint
    Eigen::Vector2d proj = Projection::projectPoint(vertex_, intrinsics_, R, t);
    int u = static_cast<int>(proj(0));
    int v = static_cast<int>(proj(1));
    if (u < 0 || u >= current_image_.cols || v < 0 || v >= current_image_.rows) {
        residual = 0.0;
        if (jacobian_pose) jacobian_pose->setZero();
        if (jacobian_intensity) *jacobian_intensity = 0.0;
        return true;
    }

    // 获取图像像素值（假设图像为 CV_32F 类型）
    float pixel_value = current_image_.at<float>(v, u);
    double sqrt_weight = std::sqrt(weight_);
    residual = sqrt_weight * (pixel_value - intensity);

    // 计算雅可比：调用 computeJacobian 封装函数
    if (jacobian_pose || jacobian_intensity) {
        Eigen::Matrix<double, 1, 6> J_total = computeJacobian(vertex_, intrinsics_, R, t, current_image_, u, v);
        if (jacobian_pose) {
            *jacobian_pose = sqrt_weight * J_total;
        }
        if (jacobian_intensity) {
            *jacobian_intensity = -sqrt_weight;
        }
    }

    return true;
}

Eigen::Matrix<double, 1, 6> PhotometricError::computeJacobian(const MeshModel::Vertex& vertex,
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
    J_se3 << -R.transpose()(0,0), -R.transpose()(0,1), -R.transpose()(0,2),  0,         -Z,  Y,
             -R.transpose()(1,0), -R.transpose()(1,1), -R.transpose()(1,2),  Z,          0, -X,
             -R.transpose()(2,0), -R.transpose()(2,1), -R.transpose()(2,2), -Y,          X,  0;

    // 最终雅可比为链式法则相乘
    J = J_grad * J_proj * J_se3;
    return J;
}