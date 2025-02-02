#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <Eigen/Core>

class ImageProcessor {
public:
    ImageProcessor();

    // 深度图处理
    void filterInvalidDepth(cv::Mat& depthImage, float minDepth, float maxDepth);
    void interpolateDepth(cv::Mat& depthImage);
    void filterExDepth(cv::Mat& depthImage, float minDepth, float maxDepth);

    // 光度补偿
    cv::Mat photometricCompensation(const cv::Mat& image);

    // 计算深度图法向量
    Eigen::MatrixXf computeDepthNormals(const cv::Mat& depthImage, float fx, float fy, float cx, float cy);

    float computeGradient(const cv::Mat& image, int u, int v);

private:
    // 构建金字塔
    void buildGaussianPyramid(const cv::Mat& image, std::vector<cv::Mat>& pyramid, int levels);
    void buildLaplacianPyramid(const std::vector<cv::Mat>& gaussianPyramid, std::vector<cv::Mat>& laplacianPyramid);
    cv::Mat reconstructFromPyramid(const std::vector<cv::Mat>& pyramid);

    // 计算像素点的梯度
    void computeGradients(const cv::Mat& depthImage, cv::Mat& gradX, cv::Mat& gradY);
};

#endif // IMAGE_PROCESSOR_H
