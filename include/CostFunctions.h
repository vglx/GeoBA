#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#include "MeshModel.h"
#include "BVH.h"
#include "EDGraph.h"
#include "Projection.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

// Photometric error struct with joint pose and EDGraph parameter optimization
class PhotometricError {
public:
    PhotometricError(
        const MeshModel::Vertex& vertex,
        int vertex_index,
        const std::vector<MeshModel::Triangle>& triangles,
        const Eigen::Matrix3d& intrinsics,
        const cv::Mat& current_image,
        const BVH& bvh,
        double weight,
        const EDGraph* ed = nullptr
    );

    // Compute residual and Jacobians for pose(6), intensity(1), and ED params (6*G)
    bool Evaluate(
        const Eigen::Matrix<double,6,1>& se3,
        double intensity,
        double& residual,
        Eigen::Matrix<double,1,6>* jacobian_pose,
        double* jacobian_intensity,
        Eigen::VectorXd* jacobian_ed = nullptr
    ) const;

private:
    MeshModel::Vertex vertex_;
    int vidx_;
    std::vector<MeshModel::Triangle> triangles_;
    Eigen::Matrix3d intrinsics_;
    cv::Mat current_image_;
    BVH bvh_;
    double weight_;
    const EDGraph* ed_;

    // Helper: compute pose Jacobian using deformed vertex
    Eigen::Matrix<double,1,6> computePoseJacobian(
        const Eigen::Vector3d& v_def,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t,
        double u,
        double v
    ) const;

    // Helper: compute ED param Jacobian for one node
    Eigen::Matrix<double,1,6> computeEDJacobianAtNode(
        const Eigen::Matrix<double,1,2>& J_grad,
        const Eigen::Matrix<double,2,3>& J_proj,
        const DeformationNode& node,
        const Eigen::Vector3d& v0,
        double weight_k
    ) const;
};

#endif // COSTFUNCTIONS_H