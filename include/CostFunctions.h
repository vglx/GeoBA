#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "MeshModel.h"
#include "Projection.h"
#include "ImageProcessor.h"

class CombinedError {
public:
    CombinedError(const std::vector<MeshModel::Vertex>& vertices,
                  const std::vector<MeshModel::Triangle>& triangles,
                  const Eigen::Matrix3f& camera_intrinsics,
                  const std::vector<cv::Mat>& depth_maps,
                  const std::vector<Eigen::MatrixXf>& depth_normals,
                  const std::vector<cv::Mat>& images,
                  int max_vertices,
                  double weight_global_depth,
                  double weight_global_gradient,
                  double weight_local_depth,
                  double weight_local_normal);

    template <typename T>
    bool operator()(T const* const* poses, T* residual) const;

    static ceres::CostFunction* Create(const std::vector<MeshModel::Vertex>& vertices,
                                       const std::vector<MeshModel::Triangle>& triangles,
                                       const Eigen::Matrix3f& camera_intrinsics,
                                       const std::vector<cv::Mat>& depth_maps,
                                       const std::vector<Eigen::MatrixXf>& depth_normals,
                                       const std::vector<cv::Mat>& images,
                                       int max_vertices,
                                       double weight_global_depth,
                                       double weight_global_gradient,
                                       double weight_local_depth,
                                       double weight_local_normal);

private:
    const std::vector<MeshModel::Vertex>& vertices_;
    const std::vector<MeshModel::Triangle>& triangles_;
    const Eigen::Matrix3f camera_intrinsics_;
    const std::vector<cv::Mat>& depth_maps_;
    const std::vector<Eigen::MatrixXf>& depth_normals_;
    const std::vector<cv::Mat>& images_;
    int max_vertices_;
    double weight_global_depth_;
    double weight_global_gradient_;
    double weight_local_depth_;
    double weight_local_normal_;
};

#endif // COST_FUNCTIONS_H
