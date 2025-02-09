#include "ErrorManager.h"
#include "CostFunctions.h"
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <iostream>

// 测试几何误差
void testGeometricError(ErrorManager& error_manager, ceres::Problem& problem) {
    std::cout << "Testing Geometric Error...\n";

    // 示例数据
    Eigen::Vector3f point(1.0f, 2.0f, 3.0f);
    Eigen::Vector3f nearest_vertex(1.1f, 2.1f, 3.1f);
    double vertex_params[3] = {1.1, 2.1, 3.1};

    // 添加几何误差
    error_manager.addGeometricError(problem, point, nearest_vertex, vertex_params);

    std::cout << "Geometric error added.\n";
}

// 测试法向量误差
void testNormalError(ErrorManager& error_manager, ceres::Problem& problem) {
    std::cout << "Testing Normal Error...\n";

    // 示例数据
    Eigen::Vector3f point_normal(0.0f, 0.0f, 1.0f);
    Eigen::Vector3f mesh_normal(0.1f, 0.1f, 0.9f);
    double normal_params[3] = {0.1, 0.1, 0.9};

    // 添加法向量误差
    error_manager.addNormalError(problem, point_normal, mesh_normal, normal_params);

    std::cout << "Normal error added.\n";
}

// 测试光度误差
void testPhotometricError(ErrorManager& error_manager, ceres::Problem& problem) {
    std::cout << "Testing Photometric Error...\n";

    // 示例数据
    Eigen::Vector3f point(0.0f, 0.0f, 1.0f);
    float observed_intensity = 128.0f;
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC1);
    image.at<uchar>(240, 320) = 128; // 灰度值设置为 128
    double vertex_params[3] = {0.0, 0.0, 1.0};
    double camera_params[12] = {1.0, 0.0, 0.0, 0.0,
                                0.0, 1.0, 0.0, 0.0,
                                0.0, 0.0, 1.0, 0.0};

    // 添加光度误差
    error_manager.addPhotometricError(problem, point, observed_intensity, image,
                                      500.0f, 500.0f, 320.0f, 240.0f, vertex_params, camera_params);

    std::cout << "Photometric error added.\n";
}

// 测试综合优化
void testOptimization() {
    std::cout << "Testing Combined Optimization...\n";

    // 创建 Ceres Problem
    ceres::Problem problem;

    // 创建 ErrorManager，设置权重
    ErrorManager error_manager(1.0, 0.5, 0.3); // 几何误差、法向量误差、光度误差的权重

    // 添加误差
    testGeometricError(error_manager, problem);
    testNormalError(error_manager, problem);
    testPhotometricError(error_manager, problem);

    // 配置优化器
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;

    // 运行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;
    std::cout << "Combined optimization test completed.\n";
}

int main() {
    testOptimization();
    return 0;
}
