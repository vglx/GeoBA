#include "CostFunctions.h"
#include "Projection.h"
#include <sophus/se3.hpp>
#include <algorithm>
#include <cmath>
#include "ImageProcessor.h"

int g_nonzero_jacobian_residuals = 0;

MultiViewPhotometricError::MultiViewPhotometricError(
    const MeshModel::Vertex& vertex,
    const std::vector<MeshModel::Triangle>& triangles,
    const Eigen::Matrix3d& camera_intrinsics,
    const cv::Mat& current_image,
    const BVH& bvh,
    double weight_photometric)
    : vertex_(vertex),
      triangles_(triangles),
      camera_intrinsics_(camera_intrinsics),
      current_image_(current_image),
      bvh_(bvh),
      weight_photometric_(weight_photometric) {

    set_num_residuals(1);
    mutable_parameter_block_sizes()->push_back(6);
    mutable_parameter_block_sizes()->push_back(1);
}

bool MultiViewPhotometricError::Evaluate(double const* const* parameters,
                                         double* residuals,
                                         double** jacobians) const {

    extern int g_nonzero_jacobian_residuals;    

    Eigen::Map<const Eigen::Matrix<double,6,1>> se3_current(parameters[0]); // 访问 x1（相机位姿）
    double intensity_avg = parameters[1][0]; // 访问 x2（光度均值） 

    Sophus::SE3d transform_current = Sophus::SE3d::exp(se3_current);
    Eigen::Matrix3d R_current = transform_current.rotationMatrix();
    Eigen::Vector3d t_current = transform_current.translation();

    double sqrt_weight = std::sqrt(weight_photometric_);

    // 先检查顶点可见性
    bool visible = Projection::isVertexVisible(
        vertex_, camera_intrinsics_,
        transform_current.rotationMatrix(), t_current,
        bvh_, current_image_.cols, current_image_.rows
    );

    if (!visible) {
        residuals[0] = 0.0;
        
        if (jacobians) {
            if (jacobians[0]) { 
                std::fill(jacobians[0], jacobians[0] + 6, 0.0);
            }
            if (jacobians[1]) { 
                jacobians[1][0] = 0.0;
            }
        }
        
        return true;
    }

    // 计算投影误差
    Eigen::Vector2d proj = Projection::projectPoint(
        vertex_, camera_intrinsics_, 
        transform_current.rotationMatrix(), t_current
    );

    if (proj(0) < 0 || proj(0) >= current_image_.cols || proj(1) < 0 || proj(1) >= current_image_.rows) {
        residuals[0] = 0.0;
        if (jacobians && jacobians[0]) {
            std::fill(jacobians[0], jacobians[0] + 6, 0.0);
        }
        return true;
    }

    float I_proj = ImageProcessor::getBilinearInterpolatedValue(current_image_, proj(0), proj(1));
    residuals[0] = sqrt_weight * (I_proj - intensity_avg);

    if (jacobians) {
        Eigen::Matrix<double,1,6> J_current = computeNumericalJacobian(se3_current, intensity_avg);
        // Eigen::Matrix<double, 1, 6> J_current = computeJacobian(R_current, t_current, proj(0), proj(1));
        // Eigen::Matrix<double,1,6> J_1 = computeNumericalJacobian(se3_current, intensity_avg);
        
        // std::cout << "Analytic: " << J_current << std::endl;
        // std::cout << "Numerical: " << J_1 << std::endl;

        if (jacobians[0]) { // 6D 位姿的 Jacobian
            for (int j = 0; j < 6; ++j) {
                jacobians[0][j] = sqrt_weight * J_current(j);
            }
        }
        if (jacobians[1]) { // 1D 光度的 Jacobian
            jacobians[1][0] = -sqrt_weight; // ✅ 正确
        }
    }

    if (jacobians) {
        bool nonzero = false;
        if (jacobians[0]) {
            for (int j = 0; j < 6; ++j) {
                if (std::abs(jacobians[0][j]) > 1e-10) {
                    nonzero = true;
                    break;
                }
            }
        }
        if (nonzero) {
            #pragma omp atomic
            ++g_nonzero_jacobian_residuals;
        }
    }
    
    return true;
}

Eigen::Matrix<double,1,6> MultiViewPhotometricError::computeJacobian(
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    double u, double v) const {
    Eigen::Matrix<double,1,6> J;
    J.setZero();

    Eigen::Vector3d point_world(vertex_.x, vertex_.y, vertex_.z);
    Eigen::Vector3d point_cam = R.transpose() * (point_world - t);
    double X = point_cam(0), Y = point_cam(1), Z = point_cam(2);
    Z = std::max(Z, 1e-3);
    double fx = camera_intrinsics_(0, 0), fy = camera_intrinsics_(1, 1);
    double cx = camera_intrinsics_(0, 2), cy = camera_intrinsics_(1, 2);

    // 计算图像梯度
    auto grad = ImageProcessor::computeGradient(current_image_, u, v);
    double grad_u = grad.first, grad_v = grad.second;
    Eigen::Matrix<double,1,2> J_grad;
    J_grad << grad_u, grad_v;

    Eigen::Matrix<double,2,3> J_proj;
    J_proj << fx / Z, 0, -fx * X / (Z * Z),
              0, fy / Z, -fy * Y / (Z * Z);

    Eigen::Matrix<double,3,6> J_se3;
    Eigen::Vector3d p_diff = R.transpose() * (point_world - t);  // p_w - t
    Eigen::Matrix3d skew;
    skew << 0,           -p_diff(2),  p_diff(1),
            p_diff(2),    0,         -p_diff(0),
            -p_diff(1),   p_diff(0),   0;
    // J_se3 << R.transpose() * skew, -R.transpose();
    J_se3 << -Eigen::Matrix<double, 3, 3>::Identity(), skew;

    Eigen::Matrix<double,1,6> J_current = J_grad * J_proj * J_se3;
    // std::cout << "J_grad: " << J_grad << " J_proj: " << J_proj << " J_se3: "<< J_se3 << std::endl;
    J = J_current;
    return J;
}

Eigen::Matrix<double, 1, 6> MultiViewPhotometricError::computeNumericalJacobian(const Eigen::Matrix<double, 6, 1>& se3,
                                                                        double intensity) const {
    double epsilon = 1e-6;
    double sqrt_weight = std::sqrt(weight_photometric_);
    
    // 先计算当前 se3 参数下的光度误差 error0
    Sophus::SE3d transform = Sophus::SE3d::exp(se3);
    Eigen::Matrix3d R = transform.rotationMatrix();
    Eigen::Vector3d t = transform.translation();

    // 检查顶点是否可见
    if (!Projection::isVertexVisible(vertex_, camera_intrinsics_, R, t, bvh_, current_image_.cols, current_image_.rows)) {
        return Eigen::Matrix<double, 1, 6>::Zero();
    }

    // 投影计算
    Eigen::Vector2d proj = Projection::projectPoint(vertex_, camera_intrinsics_, R, t);
    if (proj(0) < 0 || proj(0) >= current_image_.cols || proj(1) < 0 || proj(1) >= current_image_.rows) {
        return Eigen::Matrix<double, 1, 6>::Zero();
    }

    // 获取当前像素值（假设图像为 CV_32F 类型）
    float pixel_value = ImageProcessor::getBilinearInterpolatedValue(current_image_, proj(0), proj(1));;
    double error0 = sqrt_weight * (pixel_value - intensity);

    // 数值雅可比
    Eigen::Matrix<double, 1, 6> J_num;
    J_num.setZero();

    // 对 se3 中的每个自由度施加微小扰动，计算有限差分
    for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double, 6, 1> se3_perturbed = se3;
        se3_perturbed(i) += epsilon;
        Sophus::SE3d transform_perturbed = Sophus::SE3d::exp(se3_perturbed);

        // Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
        // delta(i) = epsilon;
        // Sophus::SE3d transform_perturbed = transform * Sophus::SE3d::exp(delta);
        // Sophus::SE3d transform_perturbed = Sophus::SE3d::exp(delta) * transform;

        Eigen::Matrix3d R_perturbed = transform_perturbed.rotationMatrix();
        Eigen::Vector3d t_perturbed = transform_perturbed.translation();

        // 检查扰动后的顶点是否可见
        if (!Projection::isVertexVisible(vertex_, camera_intrinsics_, R_perturbed, t_perturbed, bvh_, current_image_.cols, current_image_.rows)) {
            continue;
        }
        Eigen::Vector2d proj_perturbed = Projection::projectPoint(vertex_, camera_intrinsics_, R_perturbed, t_perturbed);
        if (proj_perturbed(0) < 0 || proj_perturbed(0) >= current_image_.cols || proj_perturbed(1) < 0 || proj_perturbed(1) >= current_image_.rows) {
            continue;
        }

        float pixel_value_perturbed = ImageProcessor::getBilinearInterpolatedValue(current_image_, proj_perturbed(0), proj_perturbed(1));
        double error_perturbed = sqrt_weight * (pixel_value_perturbed - intensity);

        // 计算有限差分
        J_num(i) = (error_perturbed - error0) / epsilon;
    }
    return J_num;
}

ceres::CostFunction* MultiViewPhotometricError::Create(
    const MeshModel::Vertex& vertex,
    const std::vector<MeshModel::Triangle>& triangles,
    const Eigen::Matrix3d& camera_intrinsics,
    const cv::Mat& current_image,
    const BVH& bvh,
    double weight_photometric) {
    return new MultiViewPhotometricError(
        vertex, triangles, camera_intrinsics, current_image, bvh, weight_photometric
    );
}
