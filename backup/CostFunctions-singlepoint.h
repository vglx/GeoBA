#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "MeshModel.h" // 需要包含 MeshModel 以访问顶点信息

class GeometricError {
public:
    GeometricError(const std::vector<MeshModel::Vertex>& vertices,
                   const std::vector<MeshModel::Triangle>& triangles,
                   const Eigen::Matrix3f& camera_intrinsics,
                   const cv::Mat& depth_map,
                   const Eigen::MatrixXf& depth_normals);

    template <typename T>
    bool operator()(const T* const camera_pose, T* residual) const;

    static ceres::CostFunction* Create(const std::vector<MeshModel::Vertex>& vertices,
                                       const std::vector<MeshModel::Triangle>& triangles,
                                       const Eigen::Matrix3f& camera_intrinsics,
                                       const cv::Mat& depth_map,
                                       const Eigen::MatrixXf& depth_normals);

private:
    const std::vector<MeshModel::Vertex>& vertices_;
    const std::vector<MeshModel::Triangle>& triangles_;
    const Eigen::Matrix3f camera_intrinsics_;
    const cv::Mat& depth_map_;
    const Eigen::MatrixXf& depth_normals_;
};

#endif // COST_FUNCTIONS_H
