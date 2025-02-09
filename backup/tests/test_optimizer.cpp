#include "Optimizer.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <vector>
#include <iostream>

void testOptimizer() {
    std::cout << "Testing Optimizer...\n";

    // 示例数据
    std::vector<Eigen::Vector3f> point_cloud = {
        {1.0f, 2.0f, 3.0f}, 
        {2.0f, 3.0f, 4.0f}
    }; // 点云点

    std::vector<Eigen::Vector3f> point_normals = {
        {0.0f, 0.0f, 1.0f}, 
        {0.0f, 1.0f, 0.0f}
    }; // 点云法向量

    std::vector<Eigen::Vector3f> mesh_vertices = {
        {1.1f, 2.1f, 3.1f}, 
        {2.2f, 3.2f, 4.2f}
    }; // 网格顶点

    std::vector<Eigen::Vector3f> mesh_normals = {
        {0.0f, 0.1f, 0.9f}, 
        {0.1f, 0.9f, 0.0f}
    }; // 网格法向量

    cv::Mat observed_image = cv::Mat::zeros(480, 640, CV_8UC1); // 灰度图像
    observed_image.at<uchar>(240, 320) = 128; // 示例灰度值

    Eigen::Matrix3f camera_intrinsics; // 相机内参
    camera_intrinsics << 500.0, 0.0, 320.0,
                         0.0, 500.0, 240.0,
                         0.0, 0.0, 1.0;

    // 创建优化器
    Optimizer optimizer(1.0, 0.5, 0.3); // 权重：几何 1.0，法向量 0.5，光度 0.3

    // 调用优化器执行优化
    optimizer.optimize(point_cloud, point_normals, mesh_vertices, mesh_normals, observed_image, camera_intrinsics);

    // 输出优化后结果
    std::cout << "Optimization Results:\n";
    for (size_t i = 0; i < mesh_vertices.size(); ++i) {
        std::cout << "Vertex " << i << ": [" 
                  << mesh_vertices[i].x() << ", " 
                  << mesh_vertices[i].y() << ", " 
                  << mesh_vertices[i].z() << "]\n";
    }
}

int main() {
    testOptimizer();
    return 0;
}
