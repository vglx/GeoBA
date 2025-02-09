#include "ImageProcessor.h"
#include <iostream>

int main() {
    // 创建 ImageProcessor 实例
    ImageProcessor processor;

    // 测试加载 RGB 图像
    cv::Mat rgbImage;
    if (!processor.loadRGBImage("path/to/rgb_image.jpg", rgbImage)) {
        std::cerr << "Failed to load RGB image!" << std::endl;
        return -1;
    }
    std::cout << "Successfully loaded RGB image: "
              << rgbImage.cols << "x" << rgbImage.rows << std::endl;

    // 测试加载深度图
    cv::Mat depthImage;
    if (!processor.loadDepthImage("path/to/depth_image.png", depthImage)) {
        std::cerr << "Failed to load depth image!" << std::endl;
        return -1;
    }
    std::cout << "Successfully loaded depth image: "
              << depthImage.cols << "x" << depthImage.rows << std::endl;

    // 测试无效深度值过滤
    processor.filterInvalidDepth(depthImage, 0.3f, 5.0f);
    std::cout << "Filtered depth values out of range [0.3, 5.0] meters." << std::endl;

    // 测试深度图插值
    processor.interpolateDepth(depthImage);
    std::cout << "Interpolated invalid depth values." << std::endl;

    // 测试深度图法向量计算
    float fx = 525.0f, fy = 525.0f, cx = 320.0f, cy = 240.0f; // 假设相机内参
    Eigen::MatrixXf normals = processor.computeDepthNormals(depthImage, fx, fy, cx, cy);
    std::cout << "Computed depth normals. First 5 normals:" << std::endl;
    for (int i = 0; i < std::min(5, (int)normals.rows()); ++i) {
        std::cout << "Normal " << i << ": " << normals.row(i) << std::endl;
    }

    // 测试光度补偿
    cv::Mat compensatedImage = processor.photometricCompensation(rgbImage);
    std::cout << "Performed photometric compensation." << std::endl;

    // 显示图像（可选）
    cv::imshow("Original RGB Image", rgbImage);
    cv::imshow("Compensated Image", compensatedImage);
    cv::waitKey(0);

    return 0;
}
