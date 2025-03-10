#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include "MeshModel.h"

class LocalGeometricError : public ceres::CostFunction {
public:
    LocalGeometricError(const std::vector<MeshModel::Vertex>& vertices,
                        const std::vector<MeshModel::Triangle>& triangles,
                        const Eigen::Matrix3d& camera_intrinsics,
                        const cv::Mat& depth_map,
                        double weight_local_depth);

    bool Evaluate(double const* const* param, double* residuals, double** jacobians) const override;

    static ceres::CostFunction* Create(const std::vector<MeshModel::Vertex>& vertices,
                                       const std::vector<MeshModel::Triangle>& triangles,
                                       const Eigen::Matrix3d& camera_intrinsics,
                                       const cv::Mat& depth_map,
                                       double weight_local_depth);

private:
    Eigen::Matrix<double,1,6> computeAnalyticalJacobian(
        const Eigen::Matrix3d& rotation,
        const Eigen::Vector3d& translation,
        const Eigen::Matrix3d& intrinsics,
        const MeshModel::Vertex& vertex,
        int u, int v,
        const cv::Mat& image) const;

    const std::vector<MeshModel::Vertex>& vertices_;
    const std::vector<MeshModel::Triangle>& triangles_;
    Eigen::Matrix3d camera_intrinsics_;
    cv::Mat depth_map_;
    double weight_local_depth_;
};

class NormalConsistencyError : public ceres::CostFunction {
public:
    NormalConsistencyError(const std::vector<MeshModel::Vertex>& vertices,
                           const std::vector<MeshModel::Triangle>& triangles,
                           const Eigen::Matrix3d& camera_intrinsics,
                           const std::vector<cv::Mat>& depth_maps,  // 传入所有深度图
                           double weight);

    bool Evaluate(double const* const* param, double* residuals, double** jacobians) const override;

    static ceres::CostFunction* Create(const std::vector<MeshModel::Vertex>& vertices,
                                       const std::vector<MeshModel::Triangle>& triangles,
                                       const Eigen::Matrix3d& camera_intrinsics,
                                       const std::vector<cv::Mat>& depth_maps,  // 传入所有深度图
                                       double weight);

private:
    Eigen::Matrix<double, 1, 6> ComputeJacobian(
        int u, int v,
        const cv::Mat &depthMap,
        const Eigen::Matrix3d &camera_intrinsics,
        const Eigen::Vector3d& n_i,         // 当前帧法向量 at (u,v)
        const Eigen::Vector3d& n_avg,         // 三帧归一化平均法向量
        double s,                           // s = || n_i + n_j + n_k ||
        const Eigen::Vector3d& p,             // 顶点在世界坐标系下的位置
        const Eigen::Matrix3d& R,             // 当前帧旋转矩阵（相机到世界）
        const Eigen::Vector3d& t,             // 当前帧平移向量
        double fx, double fy) const;

    const std::vector<MeshModel::Vertex>& vertices_;
    const std::vector<MeshModel::Triangle>& triangles_;
    Eigen::Matrix3d camera_intrinsics_;
    std::vector<cv::Mat> depth_maps_;  // 存储三帧深度图
    double weight_;
};

class MultiViewPhotometricError : public ceres::CostFunction {
public:
    MultiViewPhotometricError(const std::vector<MeshModel::Vertex>& vertices,
                                const std::vector<MeshModel::Triangle>& triangles,
                                const Eigen::Matrix3d& camera_intrinsics,
                                const cv::Mat& current_image,
                                const std::vector<cv::Mat>& other_images,
                                const std::vector<const double*>& other_se3,
                                double weight_photometric);

    // Evaluate 的参数块中，仅第一个参数块为当前帧的 SE(3)（优化变量）。
    // 其他帧的位姿即使传入参数块，也忽略，直接使用构造时固定传入的 other_se3_。
    virtual bool Evaluate(double const* const* parameters,
                            double* residuals,
                            double** jacobians) const;

    // 工厂函数
    static ceres::CostFunction* Create(
        const std::vector<MeshModel::Vertex>& vertices,
        const std::vector<MeshModel::Triangle>& triangles,
        const Eigen::Matrix3d& camera_intrinsics,
        const cv::Mat& current_image,
        const std::vector<cv::Mat>& other_images,
        const std::vector<const double*>& other_se3,
        double weight_photometric);

private:
    // 计算当前帧关于单个顶点的雅可比（针对光度残差）
    Eigen::Matrix<double,1,6> computeJacobianForVertex(
        const MeshModel::Vertex& vertex,
        const Eigen::Matrix3d& intrinsics,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t,
        const cv::Mat& image,
        int u, int v) const;

    const std::vector<MeshModel::Vertex>& vertices_;
    const std::vector<MeshModel::Triangle>& triangles_;
    const Eigen::Matrix3d camera_intrinsics_;
    const cv::Mat current_image_;
    const std::vector<cv::Mat> other_images_;
    // 固定其他帧的位姿，以指针形式传入，每个指向 6 个 double
    const std::vector<const double*> other_se3_;
    double weight_photometric_;
};

#endif // COSTFUNCTIONS_H
