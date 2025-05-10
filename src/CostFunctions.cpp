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
    if (proj(0) < 0 || proj(0) >= current_image_.cols || proj(1) < 0 || proj(1) >= current_image_.rows) {
        residual = 0.0;
        if (jacobian_pose) jacobian_pose->setZero();
        if (jacobian_intensity) *jacobian_intensity = 0.0;
        return true;
    }

    // 获取图像像素值（假设图像为 CV_32F 类型）
    float pixel_value = ImageProcessor::getBilinearInterpolatedIntensity(current_image_, proj(0), proj(1));
    double sqrt_weight = std::sqrt(weight_);
    residual = sqrt_weight * (pixel_value - intensity);

    // 计算雅可比：调用 computeJacobian 封装函数
    if (jacobian_pose || jacobian_intensity) {
        Eigen::Matrix<double, 1, 6> J_total = computeAnalyticalJacobian(vertex_, intrinsics_, R, t, current_image_, proj(0), proj(1));
        // Eigen::Matrix<double, 1, 6> J_total = computeNumericalJacobian(se3, intensity);

        if (jacobian_pose) {
            *jacobian_pose = sqrt_weight * J_total;
        }
        if (jacobian_intensity) {
            *jacobian_intensity = -sqrt_weight;
        }
    }

    return true;
}

Eigen::Matrix<double, 1, 6> PhotometricError::computeAnalyticalJacobian(
    const MeshModel::Vertex& vertex,
    const Eigen::Matrix3d& intrinsics,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    const cv::Mat& image,
    double u, double v) const
{
    const double epsilon = 1e-6;
    const double sqrt_weight = std::sqrt(weight_);

    // —— 一：纯解析雅可比 —— 
    // 1) 图像梯度
    auto grad = ImageProcessor::computeGradient(image, u, v);
    Eigen::Matrix<double,1,2> J_grad;
    J_grad << grad.first, grad.second;

    // 2) 相机坐标（与你原来一致）
    Eigen::Vector3d pw(vertex.x, vertex.y, vertex.z);
    Eigen::Vector3d pc = R.transpose() * (pw - t);
    double X = pc.x(), Y = pc.y(), Z = std::max(pc.z(), 1e-6);

    // 3) 投影雅可比
    double fx = intrinsics(0,0), fy = intrinsics(1,1);
    Eigen::Matrix<double,2,3> J_proj;
    J_proj << fx/Z,      0, -fx*X/(Z*Z),
                 0, fy/Z, -fy*Y/(Z*Z);

    // 4) SE3 对 pc 的雅可比（左乘扰动 + camera→world 约定）
    Eigen::Matrix<double, 3, 6> J_se3;
    Eigen::Vector3d p_diff = R.transpose() * (pw - t);
    Eigen::Matrix3d skew;
    skew << 0,           -p_diff(2),  p_diff(1),
            p_diff(2),    0,         -p_diff(0),
            -p_diff(1),   p_diff(0),   0;
    J_se3 << -Eigen::Matrix<double, 3, 3>::Identity(), skew;

    // 5) 链式相乘得到纯解析
    Eigen::Matrix<double,1,6> J = J_grad * J_proj * J_se3;

    // —— 二：搬入“可见性+越界”门控 —— 
    // 用同样的 ±ε 扰动方式去检测每一维
    Sophus::SE3d Twc(R, t);  // 注意：R,t 这里是你的 camera→world
    int W = image.cols, H = image.rows;
    for (int i = 0; i < 6; ++i) {
        // 构造扰动
        Eigen::Matrix<double,6,1> d = Eigen::Matrix<double,6,1>::Zero();
        d(i) = epsilon;
        Sophus::SE3d Tp = Twc * Sophus::SE3d::exp( d);
        Sophus::SE3d Tm = Twc * Sophus::SE3d::exp(-d);

        // 拆出 R+, t+
        Eigen::Matrix3d Rp = Tp.rotationMatrix();
        Eigen::Vector3d tp = Tp.translation();
        bool vis_p = Projection::isVertexVisible(vertex_, intrinsics, Rp, tp,
                                                 bvh_, W, H);
        Eigen::Vector2d uv_p = Projection::projectPoint(vertex_, intrinsics, Rp, tp);
        bool in_p = uv_p.x() >= 0 && uv_p.x() < W && uv_p.y() >= 0 && uv_p.y() < H;

        // 拆出 R-, t-
        Eigen::Matrix3d Rm = Tm.rotationMatrix();
        Eigen::Vector3d tm = Tm.translation();
        bool vis_m = Projection::isVertexVisible(vertex_, intrinsics, Rm, tm,
                                                 bvh_, W, H);
        Eigen::Vector2d uv_m = Projection::projectPoint(vertex_, intrinsics, Rm, tm);
        bool in_m = uv_m.x() >= 0 && uv_m.x() < W && uv_m.y() >= 0 && uv_m.y() < H;

        // 如果任一端不可见或越界，就把 J(i) 置零
        if (!(vis_p && in_p && vis_m && in_m)) {
            J(0, i) = 0.0;
        }
    }

    return J;
}

Eigen::Matrix<double, 1, 6> PhotometricError::computeNumericalJacobian(
    const Eigen::Matrix<double, 6, 1>& se3,
    double intensity) const {
    const double epsilon     = 1e-6;

    // 基准变换 T
    Sophus::SE3d transform = Sophus::SE3d::exp(se3);

    Eigen::Matrix<double, 1, 6> J_num = Eigen::Matrix<double, 1, 6>::Zero();
    for (int i = 0; i < 6; ++i) {
        // +ε 扰动
        Eigen::Matrix<double, 6, 1> d_plus = Eigen::Matrix<double, 6, 1>::Zero();
        d_plus(i) =  epsilon;
        Sophus::SE3d T_plus = transform * Sophus::SE3d::exp(d_plus);
        // Sophus::SE3d T_plus = Sophus::SE3d::exp(d_plus) * transform;

        // –ε 扰动
        Eigen::Matrix<double, 6, 1> d_minus = Eigen::Matrix<double, 6, 1>::Zero();
        d_minus(i) = -epsilon;
        Sophus::SE3d T_minus = transform * Sophus::SE3d::exp(d_minus);
        // Sophus::SE3d T_minus = Sophus::SE3d::exp(d_minus) * transform;

        // 计算两端扰动后的残差
        auto computeResidual = [&](const Sophus::SE3d& T)->double {
            Eigen::Matrix3d R = T.rotationMatrix();
            Eigen::Vector3d t = T.translation();

            // 可见性及边界检查
            if (!Projection::isVertexVisible(vertex_, intrinsics_, R, t,
                                            bvh_, current_image_.cols, current_image_.rows))
                return NAN;
            Eigen::Vector2d p = Projection::projectPoint(vertex_, intrinsics_, R, t);
            if (p.x() < 0 || p.x() >= current_image_.cols ||
                p.y() < 0 || p.y() >= current_image_.rows)
                return NAN;

            float pv = ImageProcessor::getBilinearInterpolatedIntensity(
                           current_image_, p.x(), p.y());
            return pv - intensity;
        };

        double r_plus  = computeResidual(T_plus);
        double r_minus = computeResidual(T_minus);
        if (std::isnan(r_plus) || std::isnan(r_minus)) {
            // 如果任一端不可见／越界，就跳过该分量，保持为 0
            continue;
        }

        // 中心差分
        J_num(i) = (r_plus - r_minus) / (2.0 * epsilon);
    }

    return J_num;
}