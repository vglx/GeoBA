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
};

#endif // IMAGE_PROCESSOR_H
