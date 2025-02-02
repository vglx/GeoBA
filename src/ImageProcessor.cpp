#include "ImageProcessor.h"
#include <iostream>

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

float ImageProcessor::computeGradient(const cv::Mat& image, int u, int v) {
    // **检查边界**
    if (u <= 0 || u >= image.cols - 1 || v <= 0 || v >= image.rows - 1) return 0.0f;

    // **计算 Sobel 梯度**
    cv::Mat grad_x, grad_y;
    cv::Sobel(image, grad_x, CV_32F, 1, 0, 3);  // x 方向梯度
    cv::Sobel(image, grad_y, CV_32F, 0, 1, 3);  // y 方向梯度

    // **取 (u, v) 处梯度值**
    float dx = grad_x.at<float>(v, u);
    float dy = grad_y.at<float>(v, u);

    return std::sqrt(dx * dx + dy * dy);  // **计算梯度强度**
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