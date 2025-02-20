#include "ImageProcessor.h"
#include <iostream>
#include <cmath>
#include <Eigen/Geometry>

ImageProcessor::ImageProcessor() {}

void ImageProcessor::filterInvalidDepth(cv::Mat& depthImage, float minDepth, float maxDepth) {
    for (int y = 0; y < depthImage.rows; ++y) {
        for (int x = 0; x < depthImage.cols; ++x) {
            float& depth = depthImage.at<float>(y, x);
            if (depth < minDepth || depth > maxDepth || std::isnan(depth)) {
                depth = 0.0f; // 将无效值设置为零
            }
        }
    }
}

void ImageProcessor::interpolateDepth(cv::Mat& depthImage) {
    cv::Mat mask = (depthImage == 0.0f); // 标记无效区域
    cv::Mat filled;
    cv::inpaint(depthImage, mask, filled, 3, cv::INPAINT_TELEA);
    depthImage = filled;
}

void ImageProcessor::filterExDepth(cv::Mat& depthImage, float minDepth, float maxDepth) {
    depthImage.setTo(0, depthImage < minDepth); // 去除过小深度
    depthImage.setTo(0, depthImage > maxDepth); // 去除过大深度
}

void ImageProcessor::computeGradients(const cv::Mat& depthImage, cv::Mat& gradX, cv::Mat& gradY) {
    // 使用 Sobel 算子计算梯度
    cv::Sobel(depthImage, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(depthImage, gradY, CV_32F, 0, 1, 3);
}

std::pair<float, float> ImageProcessor::computeGradient(const cv::Mat& image, int u, int v) {
    // **检查边界**
    if (u <= 0 || u >= image.cols - 1 || v <= 0 || v >= image.rows - 1) return {0.0f, 0.0f};

    // **计算 Sobel 梯度**
    cv::Mat grad_x, grad_y;
    cv::Sobel(image, grad_x, CV_32F, 1, 0, 3);  // x 方向梯度
    cv::Sobel(image, grad_y, CV_32F, 0, 1, 3);  // y 方向梯度

    // **取 (u, v) 处梯度值**
    float dx = grad_x.at<float>(v, u);  // ∂Z/∂u
    float dy = grad_y.at<float>(v, u);  // ∂Z/∂v

    return {dx, dy};  // 返回水平和垂直梯度
}

Eigen::MatrixXf ImageProcessor::computeDepthNormals(const cv::Mat& depthImage, float fx, float fy, float cx, float cy) {
    // 深度图的尺寸
    int rows = depthImage.rows;
    int cols = depthImage.cols;

    // 创建一个矩阵存储法向量
    Eigen::MatrixXf normals(rows * cols, 3);

    // 计算深度梯度
    cv::Mat gradX, gradY;
    computeGradients(depthImage, gradX, gradY);

    // 遍历每个像素计算法向量
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float Z = depthImage.at<float>(y, x);

            // 如果深度为零，跳过
            if (Z <= 0.0f) {
                normals.row(y * cols + x) << 0, 0, 0;
                continue;
            }

            // 计算相邻像素的梯度值
            float dZdx = gradX.at<float>(y, x);
            float dZdy = gradY.at<float>(y, x);

            // 转换为法向量
            Eigen::Vector3f normal(-dZdx / fx, -dZdy / fy, 1.0f);
            normal.normalize();

            // 存储法向量
            normals.row(y * cols + x) = normal;
        }
    }

    return normals;
}

void ImageProcessor::buildGaussianPyramid(const cv::Mat& image, std::vector<cv::Mat>& pyramid, int levels) {
    pyramid.clear();
    pyramid.push_back(image);
    for (int i = 1; i < levels; ++i) {
        cv::Mat down;
        cv::pyrDown(pyramid[i - 1], down);
        pyramid.push_back(down);
    }
}

void ImageProcessor::buildLaplacianPyramid(const std::vector<cv::Mat>& gaussianPyramid, std::vector<cv::Mat>& laplacianPyramid) {
    laplacianPyramid.clear();
    for (size_t i = 0; i < gaussianPyramid.size() - 1; ++i) {
        cv::Mat up, laplacian;
        cv::pyrUp(gaussianPyramid[i + 1], up, gaussianPyramid[i].size());
        cv::subtract(gaussianPyramid[i], up, laplacian);
        laplacianPyramid.push_back(laplacian);
    }
    laplacianPyramid.push_back(gaussianPyramid.back());
}

cv::Mat ImageProcessor::reconstructFromPyramid(const std::vector<cv::Mat>& pyramid) {
    cv::Mat image = pyramid.back();
    for (int i = pyramid.size() - 2; i >= 0; --i) {
        cv::Mat up;
        cv::pyrUp(image, up, pyramid[i].size());
        image = up + pyramid[i];
    }
    return image;
}

cv::Mat ImageProcessor::photometricCompensation(const cv::Mat& image) {
    // 构建高斯和拉普拉斯金字塔
    std::vector<cv::Mat> gaussianPyramid, laplacianPyramid;
    int levels = 5; // 金字塔层数
    buildGaussianPyramid(image, gaussianPyramid, levels);
    buildLaplacianPyramid(gaussianPyramid, laplacianPyramid);

    // 平滑图像（基于高斯金字塔）
    cv::Mat smoothed = gaussianPyramid.back();

    // 增强局部对比度（基于拉普拉斯金字塔）
    cv::Mat compensated = reconstructFromPyramid(laplacianPyramid);

    // 融合结果
    cv::Mat result;
    cv::addWeighted(smoothed, 0.5, compensated, 0.5, 0, result);

    return result;
}

Eigen::Vector3d ImageProcessor::computeNormal(int u, int v, const cv::Mat& depthMap, const Eigen::Matrix3d& camera_intrinsics) {
    // 边界检查，确保能够取到邻域像素
    if(u <= 0 || u >= depthMap.cols - 1 || v <= 0 || v >= depthMap.rows - 1) {
        return Eigen::Vector3d(0, 0, 0);
    }
    
    // 从内参矩阵中提取 fx, fy, cx, cy
    double fx = camera_intrinsics(0, 0);
    double fy = camera_intrinsics(1, 1);
    double cx = camera_intrinsics(0, 2);
    double cy = camera_intrinsics(1, 2);
    
    // 获取中心点及其邻域像素的深度值
    float d_center = depthMap.at<float>(v, u);
    float d_left   = depthMap.at<float>(v, u - 1);
    float d_right  = depthMap.at<float>(v, u + 1);
    float d_up     = depthMap.at<float>(v - 1, u);
    float d_down   = depthMap.at<float>(v + 1, u);
    
    // 如果深度值无效，则返回零向量
    if(d_center <= 0 || d_left <= 0 || d_right <= 0 || d_up <= 0 || d_down <= 0) {
        return Eigen::Vector3d(0, 0, 0);
    }
    
    // 内嵌将像素点转换为三维点的代码
    auto to3D = [&](int u_coord, int v_coord, float depth) -> Eigen::Vector3d {
        return Eigen::Vector3d((u_coord - cx) * depth / fx,
                               (v_coord - cy) * depth / fy,
                               depth);
    };
    
    // 将邻域像素转换为 3D 点
    Eigen::Vector3d p_left  = to3D(u - 1, v, d_left);
    Eigen::Vector3d p_right = to3D(u + 1, v, d_right);
    Eigen::Vector3d p_up    = to3D(u, v - 1, d_up);
    Eigen::Vector3d p_down  = to3D(u, v + 1, d_down);
    
    // 使用中心差分计算局部 3D 坐标梯度
    Eigen::Vector3d dpdx = (p_right - p_left) * 0.5;
    Eigen::Vector3d dpdy = (p_down  - p_up)   * 0.5;
    
    // 叉乘得到法向量，注意叉乘顺序决定法向量方向
    Eigen::Vector3d normal = dpdx.cross(dpdy);
    
    // 如果法向量模长为零，返回零向量，否则归一化后返回
    double norm = normal.norm();
    if(norm == 0)
        return Eigen::Vector3d(0, 0, 0);
    
    normal.normalize();
    return normal;
}