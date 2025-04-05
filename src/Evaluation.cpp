#include "Evaluation.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>

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
}

void Evaluation::ComputeRMSE(const std::vector<Eigen::Matrix4d>& ground_truth,
                             const std::vector<Eigen::Matrix4d>& initial_poses,
                             const std::vector<Eigen::Matrix4d>& optimized_poses) {
    std::cout << "Initial poses error:" << std::endl;
    ComputeRMSE(ground_truth, initial_poses);
    std::cout << "Optimized poses error:" << std::endl;
    ComputeRMSE(ground_truth, optimized_poses);
}

// 将旋转矩阵转换为欧拉角（ZYX顺序，对应 yaw-pitch-roll，即 alpha-beta-gamma）
Eigen::Vector3d Evaluation::RotationMatrixToEulerZYX(const Eigen::Matrix3d& R) {
    double beta = std::asin(-R(2,0));
    double alpha = std::atan2(R(2,1), R(2,2));
    double gamma = std::atan2(R(1,0), R(0,0));
    return Eigen::Vector3d(alpha, beta, gamma);
}

// 计算 RMSE（逐轴误差 & 总误差）
void Evaluation::ComputeEulerRMSE(const std::vector<Eigen::Matrix4d>& gt,
                                  const std::vector<Eigen::Matrix4d>& poses,
                                  const std::string& label) {
    if (gt.size() != poses.size()) {
        std::cerr << "Error: Ground truth and poses must have the same size!" << std::endl;
        return;
    }

    std::vector<double> euler_errors[3];       // alpha, beta, gamma
    std::vector<double> translation_errors[3]; // x, y, z

    for (size_t i = 0; i < gt.size(); ++i) {
        Eigen::Matrix3d R_gt = gt[i].block<3,3>(0,0);
        Eigen::Vector3d t_gt = gt[i].block<3,1>(0,3);

        Eigen::Matrix3d R_est = poses[i].block<3,3>(0,0);
        Eigen::Vector3d t_est = poses[i].block<3,1>(0,3);

        Eigen::Vector3d euler_gt = RotationMatrixToEulerZYX(R_gt);
        Eigen::Vector3d euler_est = RotationMatrixToEulerZYX(R_est);

        Eigen::Vector3d euler_diff = euler_est - euler_gt;
        Eigen::Vector3d trans_diff = t_est - t_gt;

        for (int j = 0; j < 3; ++j) {
            euler_errors[j].push_back(std::abs(euler_diff[j]));
            translation_errors[j].push_back(std::abs(trans_diff[j]));
        }
    }

    auto compute_rmse = [](const std::vector<double>& v) {
        double sum_sq = 0.0;
        for (double val : v) sum_sq += val * val;
        return std::sqrt(sum_sq / v.size());
    };

    double rmse_euler[3], rmse_trans[3];
    for (int i = 0; i < 3; ++i) {
        rmse_euler[i] = compute_rmse(euler_errors[i]);
        rmse_trans[i] = compute_rmse(translation_errors[i]);
    }

    double rmse_euler_total = std::sqrt(
        rmse_euler[0]*rmse_euler[0] +
        rmse_euler[1]*rmse_euler[1] +
        rmse_euler[2]*rmse_euler[2]) / std::sqrt(3.0);

    double rmse_trans_total = std::sqrt(
        rmse_trans[0]*rmse_trans[0] +
        rmse_trans[1]*rmse_trans[1] +
        rmse_trans[2]*rmse_trans[2]) / std::sqrt(3.0);

    // 输出
    std::cout << "====================== " << label << " ======================\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Euler RMSE (alpha, beta, gamma) [rad] : "
              << rmse_euler[0] << ", " << rmse_euler[1] << ", " << rmse_euler[2] << "\n";
    std::cout << "Euler RMSE Total [rad]              : " << rmse_euler_total << "\n";

    std::cout << "Translation RMSE (x, y, z) [mm]     : "
              << rmse_trans[0] << ", " << rmse_trans[1] << ", " << rmse_trans[2] << "\n";
    std::cout << "Translation RMSE Total [mm]         : " << rmse_trans_total << "\n";
    std::cout << "=================================================================\n";
}

void Evaluation::ComputeEulerRMSE(const std::vector<Eigen::Matrix4d>& ground_truth,
                                  const std::vector<Eigen::Matrix4d>& initial_poses,
                                  const std::vector<Eigen::Matrix4d>& optimized_poses) {
    ComputeEulerTranslationRMSE(ground_truth, initial_poses, "Initial Pose Error");
    ComputeEulerTranslationRMSE(ground_truth, optimized_poses, "Optimized Pose Error");
}