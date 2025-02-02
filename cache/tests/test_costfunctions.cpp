#include "CostFunctions.h"
#include <ceres/ceres.h>
#include <iostream>
#include <opencv2/opencv.hpp>

void testGeometricError() {
    std::cout << "Testing Geometric Error...\n";

    // 定义点云点和最近邻网格顶点
    Eigen::Vector3f point(1.0f, 2.0f, 3.0f);
    Eigen::Vector3f nearest_vertex(1.1f, 2.1f, 3.1f);

    // 创建 Ceres Problem
    ceres::Problem problem;

    // 添加几何误差
    double vertex_params[3] = {1.1, 2.1, 3.1};
    ceres::CostFunction* cost_function = GeometricError::Create(point, nearest_vertex);
    problem.AddResidualBlock(cost_function, nullptr, vertex_params);

    // 配置和运行优化器
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "Optimization results:\n" << summary.FullReport() << std::endl;
    std::cout << "Optimized Vertex: [" << vertex_params[0] << ", " << vertex_params[1] << ", " << vertex_params[2] << "]\n\n";
}

void testNormalError() {
    std::cout << "Testing Normal Error...\n";

    // 定义点云法向量和网格法向量
    Eigen::Vector3f point_normal(0.0f, 0.0f, 1.0f);
    Eigen::Vector3f mesh_normal(0.1f, 0.1f, 0.9f);

    // 创建 Ceres Problem
    ceres::Problem problem;

    // 添加法向量误差
    double normal_params[3] = {0.1, 0.1, 0.9};
    ceres::CostFunction* cost_function = NormalError::Create(point_normal, mesh_normal);
    problem.AddResidualBlock(cost_function, nullptr, normal_params);

    // 配置和运行优化器
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "Optimization results:\n" << summary.FullReport() << std::endl;
    std::cout << "Optimized Normal: [" << normal_params[0] << ", " << normal_params[1] << ", " << normal_params[2] << "]\n\n";
}

void testPhotometricError() {
    std::cout << "Testing Photometric Error...\n";

    // 定义相机内参
    float fx = 500.0f, fy = 500.0f, cx = 320.0f, cy = 240.0f;

    // 创建图像
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC1);
    image.at<uchar>(240, 320) = 128; // 灰度值设置为 128

    // 定义点云点和观测灰度值
    float observed_intensity = 128.0f;
    Eigen::Vector3f point(0.0f, 0.0f, 1.0f);

    // 创建 Ceres Problem
    ceres::Problem problem;

    // 添加光度误差
    double vertex_params[3] = {0.0, 0.0, 1.0};
    double camera_params[12] = {1.0, 0.0, 0.0, 0.0,
                                0.0, 1.0, 0.0, 0.0,
                                0.0, 0.0, 1.0, 0.0};
    ceres::CostFunction* cost_function = PhotometricError::Create(observed_intensity, image, fx, fy, cx, cy);
    problem.AddResidualBlock(cost_function, nullptr, vertex_params, camera_params);

    // 配置和运行优化器
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "Optimization results:\n" << summary.FullReport() << std::endl;
    std::cout << "Optimized Vertex: [" << vertex_params[0] << ", " << vertex_params[1] << ", " << vertex_params[2] << "]\n\n";
}

int main() {
    testGeometricError();
    testNormalError();
    testPhotometricError();
    return 0;
}
