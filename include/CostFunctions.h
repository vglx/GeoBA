#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#include "MeshModel.h"
#include "BVH.h"
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

// 仅包含接口声明中需要的依赖，其他实现依赖放在 .cpp 中
class PhotometricError {
public:
    // 构造函数
    PhotometricError(const MeshModel::Vertex& vertex,
                     const std::vector<MeshModel::Triangle>& triangles,
                     const Eigen::Matrix3d& intrinsics,
                     const cv::Mat& current_image,
                     const BVH& bvh,
                     double weight);

    // Evaluate 方法：传入当前的 6D 位姿和 1D 光度均值，计算残差和雅可比
    bool Evaluate(const Eigen::Matrix<double, 6, 1>& se3,
                  double intensity,
                  double& residual,
                  Eigen::Matrix<double, 1, 6>* jacobian_pose,
                  double* jacobian_intensity) const;

private:
    // 成员变量
    MeshModel::Vertex vertex_;
    std::vector<MeshModel::Triangle> triangles_;
    Eigen::Matrix3d intrinsics_;
    cv::Mat current_image_;
    BVH bvh_;
    double weight_;

    // 计算雅可比：封装图像梯度、投影雅可比和 SE3 导数的计算
    Eigen::Matrix<double, 1, 6> computeAnalyticalJacobian(const MeshModel::Vertex& vertex,
                                                const Eigen::Matrix3d& intrinsics,
                                                const Eigen::Matrix3d& R,
                                                const Eigen::Vector3d& t,
                                                const cv::Mat& image,
                                                double u, double v) const;

    Eigen::Matrix<double, 1, 6> computeNumericalJacobian(const Eigen::Matrix<double, 6, 1>& se3,
                                                                           double intensity) const;                                                

};

#endif // COSTFUNCTIONS_H