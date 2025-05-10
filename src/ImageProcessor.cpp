#include "ImageProcessor.h"
#include <iostream>
#include <cmath>
#include <Eigen/Geometry>

ImageProcessor::ImageProcessor() {}

std::pair<float, float> ImageProcessor::computeGradient(const cv::Mat& image, double u, double v) {
    // 先做一次浮点范围检查
    int W = image.cols, H = image.rows;
    // if (u < 0.0 || u > W - 1.0 || v < 0.0 || v > H - 1.0) {
    //     return {0.f, 0.f};
    // }
    // 如果在最外圈 1 像素范围内，直接返回零梯度
    if (u < 1.0 || u > W - 2.0 ||
        v < 1.0 || v > H - 2.0) {
        return {0.f, 0.f};
    }

    // 计算四个邻点坐标并 clamp
    int u0 = static_cast<int>(std::floor(u));
    int v0 = static_cast<int>(std::floor(v));
    int u1 = std::min(u0 + 1, W - 1);
    int v1 = std::min(v0 + 1, H - 1);

    // 取出四个像素值
    float I00 = image.at<float>(v0, u0);
    float I01 = image.at<float>(v0, u1);
    float I10 = image.at<float>(v1, u0);
    float I11 = image.at<float>(v1, u1);

    // 计算局部偏移
    float du = static_cast<float>(u - u0);
    float dv = static_cast<float>(v - v0);

    // 双线性插值对 u、v 的解析偏导
    // ∂I/∂u = (I01 - I00)*(1 - dv) + (I11 - I10)*dv
    float grad_u = (I01 - I00) * (1.0f - dv)
                 + (I11 - I10) * dv;

    // ∂I/∂v = (I10 - I00)*(1 - du) + (I11 - I01)*du
    float grad_v = (I10 - I00) * (1.0f - du)
                 + (I11 - I01) * du;

    return {grad_u, grad_v};
}

float ImageProcessor::getBilinearInterpolatedIntensity(const cv::Mat& image, double u, double v) {
    // 获取整数坐标
    int u0 = static_cast<int>(std::floor(u));
    int v0 = static_cast<int>(std::floor(v));
    int u1 = u0 + 1;
    int v1 = v0 + 1;

    // 判断边界
    if (u0 < 0 || u1 >= image.cols || v0 < 0 || v1 >= image.rows) {
        return 0.f; // 或者返回其他适当的值
    }

    // 计算插值权重
    double du = u - u0;
    double dv = v - v0;
    float I00 = image.at<float>(v0, u0);
    float I01 = image.at<float>(v0, u1);
    float I10 = image.at<float>(v1, u0);
    float I11 = image.at<float>(v1, u1);

    // 双线性插值
    float intensity = (1 - du) * (1 - dv) * I00 +
                      du * (1 - dv) * I01 +
                      (1 - du) * dv * I10 +
                      du * dv * I11;
    return intensity;
}

std::vector<cv::Mat> ImageProcessor::applyGaussianBlur(std::vector<cv::Mat>& rgb_images, int kernel_size, double sigma) {
    if (kernel_size % 2 == 0 || kernel_size < 1) {
        std::cerr << "Error: kernel_size must be an odd number greater than 1.\n";
        return {};
    }

    std::vector<cv::Mat> processed_images;
    processed_images.reserve(rgb_images.size());

    for (const cv::Mat& img : rgb_images) {
        if (img.empty()) {
            std::cerr << "Warning: Empty image encountered, skipping.\n";
            processed_images.push_back(cv::Mat());  // 保持原有图像索引一致，插入空图像
            continue;
        }

        cv::Mat blurred;
        cv::GaussianBlur(img, blurred, cv::Size(kernel_size, kernel_size), sigma);
        processed_images.push_back(blurred);
    }

    return processed_images;
}

std::vector<cv::Mat> ImageProcessor::downsampleImages(std::vector<cv::Mat>& rgb_images, double scale_factor) {
    if (scale_factor <= 0.0 || scale_factor >= 1.0) {
        std::cerr << "Error: scale_factor must be in the range (0, 1).\n";
        return {};
    }

    std::vector<cv::Mat> downsampled_images;
    downsampled_images.reserve(rgb_images.size());

    for (const cv::Mat& img : rgb_images) {
        if (img.empty()) {
            std::cerr << "Warning: Empty image encountered, skipping.\n";
            downsampled_images.push_back(cv::Mat());  // 保持索引一致
            continue;
        }

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
        downsampled_images.push_back(resized);
    }

    return downsampled_images;
}