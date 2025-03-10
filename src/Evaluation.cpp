#include "Evaluation.h"
#include <iostream>
#include <cmath>

double Evaluation::ComputeRMSE(const std::vector<Eigen::Matrix4d>& gt, 
                   const std::vector<Eigen::Matrix4d>& poses) {
    if (gt.size() != poses.size()) {
        std::cerr << "Error: Ground truth and poses must have the same size!" << std::endl;
        return -1.0;
    }

    double sum_squared_error = 0.0;
    int count = gt.size();

    for (size_t i = 0; i < count; ++i) {
        Eigen::Matrix4d diff = gt[i].inverse() * poses[i];
        Eigen::Vector3d translation_error = diff.block<3,1>(0,3);
        sum_squared_error += translation_error.squaredNorm();
    }

    return std::sqrt(sum_squared_error / count);
}

void Evaluation::ComputeRMSE(const std::vector<Eigen::Matrix4d>& ground_truth,
                             const std::vector<Eigen::Matrix4d>& initial_poses,
                             const std::vector<Eigen::Matrix4d>& optimized_poses) {
    double initial_rmse = ComputeRMSE(ground_truth, initial_poses);
    double optimized_rmse = ComputeRMSE(ground_truth, optimized_poses);

    if (initial_rmse >= 0.0) {
        std::cout << "Initial RMSE: " << initial_rmse << std::endl;
    }
    if (optimized_rmse >= 0.0) {
        std::cout << "Optimized RMSE: " << optimized_rmse << std::endl;
    }
}