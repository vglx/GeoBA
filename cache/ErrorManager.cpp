#include "ErrorManager.h"
#include "CostFunctions.h"

// 构造函数
ErrorManager::ErrorManager(double weight_geometry, double weight_normal, double weight_photometric)
    : weight_geometry_(weight_geometry), weight_normal_(weight_normal), weight_photometric_(weight_photometric) {}

// 添加几何误差
void ErrorManager::addGeometricError(ceres::Problem& problem,
                                     const Eigen::Vector3f& point,
                                     const Eigen::Vector3f& nearest_vertex,
                                     double* vertex_params) {
    ceres::CostFunction* cost_function = GeometricError::Create(point, nearest_vertex);
    ceres::LossFunction* loss_function = new ceres::ScaledLoss(nullptr, weight_geometry_, ceres::TAKE_OWNERSHIP);
    problem.AddResidualBlock(cost_function, loss_function, vertex_params);
}

// 添加法向量误差
void ErrorManager::addNormalError(ceres::Problem& problem,
                                  const Eigen::Vector3f& point_normal,
                                  const Eigen::Vector3f& mesh_normal,
                                  double* normal_params) {
    ceres::CostFunction* cost_function = NormalError::Create(point_normal, mesh_normal);
    ceres::LossFunction* loss_function = new ceres::ScaledLoss(nullptr, weight_normal_, ceres::TAKE_OWNERSHIP);
    problem.AddResidualBlock(cost_function, loss_function, normal_params);
}

// 添加光度误差
void ErrorManager::addPhotometricError(ceres::Problem& problem,
                                       const Eigen::Vector3f& point,
                                       float observed_intensity,
                                       const cv::Mat& image,
                                       float fx, float fy, float cx, float cy,
                                       double* vertex_params, double* camera_params) {
    ceres::CostFunction* cost_function = PhotometricError::Create(observed_intensity, image, fx, fy, cx, cy);
    ceres::LossFunction* loss_function = new ceres::ScaledLoss(nullptr, weight_photometric_, ceres::TAKE_OWNERSHIP);
    problem.AddResidualBlock(cost_function, loss_function, vertex_params, camera_params);
}
