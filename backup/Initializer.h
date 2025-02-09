#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

class Initializer {
public:
    // 直接调用的静态方法，使用 DSO 估计相机位姿
    static bool estimateWithDSO(const std::vector<cv::Mat>& rgb_images, 
                                const Eigen::Matrix3f& camera_intrinsics, 
                                std::vector<Eigen::Matrix4f>& estimated_poses, 
                                const std::string& config_file);
};

#endif // INITIALIZER_H
