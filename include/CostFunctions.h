#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "MeshModel.h"
#include "BVH.h"

class MultiViewPhotometricError : public ceres::CostFunction {
public:
    MultiViewPhotometricError(
        const MeshModel::Vertex& vertex,
        const std::vector<MeshModel::Triangle>& triangles,
        const Eigen::Matrix3d& camera_intrinsics,
        const cv::Mat& current_image,
        const BVH& bvh,
        double weight_photometric);

    virtual bool Evaluate(
        double const* const* parameters,
        double* residuals,
        double** jacobians) const override;

    static ceres::CostFunction* Create(
        const MeshModel::Vertex& vertex,
        const std::vector<MeshModel::Triangle>& triangles,
        const Eigen::Matrix3d& camera_intrinsics,
        const cv::Mat& current_image,
        const BVH& bvh,
        double weight_photometric);

private:
    const MeshModel::Vertex vertex_;
    const std::vector<MeshModel::Triangle>& triangles_;
    const Eigen::Matrix3d camera_intrinsics_;
    const cv::Mat current_image_;
    const BVH& bvh_;
    double weight_photometric_;

    Eigen::Matrix<double,1,6> computeJacobian(
        const MeshModel::Vertex& vertex,
        const Eigen::Matrix3d& intrinsics,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t,
        const cv::Mat& image,
        int u, int v) const;
};

#endif // COST_FUNCTIONS_H