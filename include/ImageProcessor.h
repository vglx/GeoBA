#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <Eigen/Core>

class ImageProcessor {
public:
    ImageProcessor();

    static std::pair<float, float> computeGradient(const cv::Mat& image, double u, double v);

    static float getBilinearInterpolatedValue(const cv::Mat& image, double u, double v);

    static std::vector<cv::Mat> applyGaussianBlur(std::vector<cv::Mat>& rgb_images, int kernel_size, double sigma);

    static std::vector<cv::Mat> downsampleImages(std::vector<cv::Mat>& rgb_images, double scale_factor);

    static std::vector<cv::Mat> applyGammaCorrection(const std::vector<cv::Mat>& rgb_images, double gamma);
    
    static std::vector<cv::Mat> applyCLAHE(const std::vector<cv::Mat>& rgb_images);

    static std::vector<cv::Mat> suppressHighlights(const std::vector<cv::Mat>& rgb_images);
};

#endif // IMAGE_PROCESSOR_H
