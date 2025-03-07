#include "CostFunctions.h"
#include "Projection.h"
#include <sophus/se3.hpp>
#include "ImageProcessor.h"
#include <omp.h>

MultiViewPhotometricError::MultiViewPhotometricError(const std::vector<MeshModel::Vertex>& vertices,
                                                     const std::vector<MeshModel::Triangle>& triangles,
                                                     const Eigen::Matrix3d& camera_intrinsics,
                                                     const cv::Mat& current_image,
                                                     const std::vector<cv::Mat>& other_images,
                                                     const std::vector<const double*>& other_se3,
                                                     double weight_photometric)
    : vertices_(vertices),
      triangles_(triangles),
      camera_intrinsics_(camera_intrinsics),
      current_image_(current_image),
      other_images_(other_images),
      other_se3_(other_se3),
      weight_photometric_(weight_photometric) {
    // 设置残差数量为每个顶点一个
    set_num_residuals(vertices_.size());
    // 参数块设置：这里只有当前帧作为优化变量，因此只设置一个6维参数块
    mutable_parameter_block_sizes()->push_back(6);
}

bool MultiViewPhotometricError::Evaluate(double const* const* parameters,
                                         double* residuals,
                                         double** jacobians) const {
    // 当前帧位姿（优化变量），从参数块中读取
    Eigen::Map<const Eigen::Matrix<double,6,1>> se3_current(parameters[0]);
    Sophus::SE3d transform_current = Sophus::SE3d::exp(se3_current);
    Eigen::Matrix3d R_current = transform_current.rotationMatrix();
    Eigen::Vector3d t_current = transform_current.translation();

    double sqrt_weight = std::sqrt(weight_photometric_);
    
    // 对当前帧进行可见性检测
    std::vector<bool> vis_current = Projection::handleOcclusion(vertices_,
                                                                triangles_,
                                                                camera_intrinsics_,
                                                                R_current, t_current,
                                                                current_image_.cols,
                                                                current_image_.rows);
    // 对其他帧依次检测可见性，并存入二维 bool 向量（顺序与 other_images_ 对应）
    std::vector<std::vector<bool>> vis_all;
    for (size_t k = 0; k < other_images_.size(); ++k) {
        // 从固定其他帧位姿指针中读取数据
        Eigen::Map<const Eigen::Matrix<double,6,1>> se3_other(other_se3_[k]);
        Sophus::SE3d transform_other = Sophus::SE3d::exp(se3_other);
        Eigen::Matrix3d R_other = transform_other.rotationMatrix();
        Eigen::Vector3d t_other = transform_other.translation();
        std::vector<bool> vis = Projection::handleOcclusion(vertices_,
                                                            triangles_,
                                                            camera_intrinsics_,
                                                            R_other, t_other,
                                                            other_images_[k].cols,
                                                            other_images_[k].rows);
        vis_all.push_back(vis);
    }

    // 遍历每个顶点计算残差
    #pragma omp parallel for
    for (size_t i = 0; i < vertices_.size(); ++i) {
        residuals[i] = 0.0;
        // 初始化当前帧雅可比为零
        if (jacobians && jacobians[0]) {
            for (int j = 0; j < 6; ++j) {
                jacobians[0][i * 6 + j] = 0.0;
            }
        }
        // 仅当当前帧中该点可见时，才进行残差计算
        if (!vis_current[i])
            continue;
        // 当前帧的投影及光度值
        Eigen::Vector2d proj_current = Projection::projectPoint(vertices_[i],
                                                                camera_intrinsics_,
                                                                R_current, t_current);
        int u_current = static_cast<int>(proj_current(0));
        int v_current = static_cast<int>(proj_current(1));
        if (u_current < 0 || u_current >= current_image_.cols ||
            v_current < 0 || v_current >= current_image_.rows)
            continue;
        float I_current = current_image_.at<float>(v_current, u_current);

        // 将当前帧的观测计入平均（包含当前帧）
        double sum_intensity = I_current;
        int count = 1;
        
        // 遍历其他帧：仅当该点在该帧中可见时提取光度
        for (size_t k = 0; k < other_images_.size(); ++k) {
            const cv::Mat& img = other_images_[k];
            if (!vis_all[k][i])
                continue;
            // 通过 other_se3_ 指针读取其他帧位姿
            Eigen::Map<const Eigen::Matrix<double,6,1>> se3_other(other_se3_[k]);
            Sophus::SE3d transform_other = Sophus::SE3d::exp(se3_other);
            Eigen::Matrix3d R_other = transform_other.rotationMatrix();
            Eigen::Vector3d t_other = transform_other.translation();
            Eigen::Vector2d proj_other = Projection::projectPoint(vertices_[i],
                                                                    camera_intrinsics_,
                                                                    R_other, t_other);
            int u_other = static_cast<int>(proj_other(0));
            int v_other = static_cast<int>(proj_other(1));
            if (u_other < 0 || u_other >= img.cols || v_other < 0 || v_other >= img.rows)
                continue;
            
            float I_other = img.at<float>(v_other, u_other);
            sum_intensity += I_other;
            count++;
        }

        // if (count < 10)
        //     continue;

        double I_avg = sum_intensity / count;
        residuals[i] = sqrt_weight * (I_current - I_avg);
        
        // 计算当前帧雅可比（仅当前帧为优化变量）
        double factor = sqrt_weight * (1.0 - 1.0 / count);
        if (jacobians && jacobians[0]) {
            Eigen::Matrix<double,1,6> J_current = computeJacobianForVertex(vertices_[i],
                                                                            camera_intrinsics_,
                                                                            R_current, t_current,
                                                                            current_image_,
                                                                            u_current, v_current);
            Eigen::Matrix<double,1,6> row = factor * J_current;
            for (int j = 0; j < 6; ++j) {
                jacobians[0][i * 6 + j] = row(j);
            }
        }
    }

    return true;
}

Eigen::Matrix<double,1,6> MultiViewPhotometricError::computeJacobianForVertex(
    const MeshModel::Vertex& vertex,
    const Eigen::Matrix3d& intrinsics,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    const cv::Mat& image,
    int u, int v) const {
    Eigen::Matrix<double,1,6> J;
    J.setZero();

    Eigen::Vector3d point_world(vertex.x, vertex.y, vertex.z);
    Eigen::Vector3d point_cam = R.transpose() * (point_world - t);
    double X = point_cam(0), Y = point_cam(1), Z = point_cam(2);
    double fx = intrinsics(0, 0), fy = intrinsics(1, 1);
    double cx = intrinsics(0, 2), cy = intrinsics(1, 2);

    double P_c1 = X / Z;
    double P_c2 = Y / Z;
    double P_c3 = Z;

    // 计算图像梯度
    auto grad = ImageProcessor::computeGradient(image, u, v);
    double grad_u = grad.first, grad_v = grad.second;
    Eigen::Matrix<double,1,2> J_grad;
    J_grad << grad_u, grad_v;

    Eigen::Matrix<double,2,3> J_proj;
    J_proj << fx / P_c3, 0, -fx * P_c1 / (P_c3 * P_c3),
              0, fy / P_c3, -fy * P_c2 / (P_c3 * P_c3);

    Eigen::Matrix<double,3,6> J_se3;
    J_se3 << 1, 0, 0,  0, -Z,  Y,
             0, 1, 0,  Z,  0, -X,
             0, 0, 1, -Y,  X,  0;

    Eigen::Matrix<double,1,6> J_current = J_grad * J_proj * J_se3;
    J = J_current;
    return J;
}

ceres::CostFunction* MultiViewPhotometricError::Create(
    const std::vector<MeshModel::Vertex>& vertices,
    const std::vector<MeshModel::Triangle>& triangles,
    const Eigen::Matrix3d& camera_intrinsics,
    const cv::Mat& current_image,
    const std::vector<cv::Mat>& other_images,
    const std::vector<const double*>& other_se3,
    double weight_photometric) {
    return new MultiViewPhotometricError(vertices, triangles, camera_intrinsics,
                                       current_image, other_images,
                                       other_se3, weight_photometric);
}