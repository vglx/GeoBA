#ifndef ERROR_MANAGER_H
#define ERROR_MANAGER_H

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

class ErrorManager {
public:
    ErrorManager(double weight_geometry, double weight_normal, double weight_photometric);

    // 添加几何误差
    void addGeometricError(ceres::Problem& problem,
                           const Eigen::Vector3f& point,
                           const Eigen::Vector3f& nearest_vertex,
                           double* vertex_params);

    // 添加法向量误差
    void addNormalError(ceres::Problem& problem,
                        const Eigen::Vector3f& point_normal,
                        const Eigen::Vector3f& mesh_normal,
                        double* normal_params);

    // 添加光度误差
    void addPhotometricError(ceres::Problem& problem,
                             const Eigen::Vector3f& point,
                             float observed_intensity,
                             const cv::Mat& image,
                             float fx, float fy, float cx, float cy,
                             double* vertex_params, double* camera_params);

private:
    double weight_geometry_;     // 几何误差权重
    double weight_normal_;       // 法向量误差权重
    double weight_photometric_;  // 光度误差权重
};

#endif // ERROR_MANAGER_H
