#include "Evaluation.h"
#include <iostream>
#include <cmath>
#include <algorithm>

void Evaluation::ComputeRMSE(const std::vector<Eigen::Matrix4d>& gt,
                             const std::vector<Eigen::Matrix4d>& poses) {
    if (gt.size() != poses.size()) {
        std::cerr << "Error: Ground truth and poses must have the same size!" << std::endl;
        return;
    }

    double sum_squared_trans_error = 0.0;
    double sum_squared_rot_error = 0.0;
    int count = gt.size();

    for (size_t i = 0; i < count; ++i) {
        Eigen::Matrix4d diff = gt[i].inverse() * poses[i];

        // 平移误差（3D 向量范数）
        Eigen::Vector3d trans_error = diff.block<3,1>(0,3);
        sum_squared_trans_error += trans_error.squaredNorm();

        // 旋转误差：计算旋转矩阵的角度误差
        Eigen::Matrix3d R_diff = diff.block<3,3>(0,0);
        // 防止浮点误差导致 acos 参数超出 [-1, 1]
        double trace = R_diff.trace();
        double cos_angle = std::min(std::max((trace - 1.0) / 2.0, -1.0), 1.0);
        double angle = std::acos(cos_angle);
        sum_squared_rot_error += angle * angle;
    }

    double trans_rmse = std::sqrt(sum_squared_trans_error / count);
    double rot_rmse = std::sqrt(sum_squared_rot_error / count);

    std::cout << "Translation RMSE: " << trans_rmse << std::endl;
    std::cout << "Rotation RMSE (radians): " << rot_rmse << std::endl;
    // 如有需要，也可以转换为度数：
    std::cout << "Rotation RMSE (degrees): " << (rot_rmse * 180.0 / M_PI) << std::endl;
}

void Evaluation::ComputeRMSE(const std::vector<Eigen::Matrix4d>& ground_truth,
                             const std::vector<Eigen::Matrix4d>& initial_poses,
                             const std::vector<Eigen::Matrix4d>& optimized_poses) {
    std::cout << "Initial poses error:" << std::endl;
    ComputeRMSE(ground_truth, initial_poses);
    std::cout << "Optimized poses error:" << std::endl;
    ComputeRMSE(ground_truth, optimized_poses);
}