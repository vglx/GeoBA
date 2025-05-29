#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include "MeshModel.h"
#include "BVH.h"

class Optimizer {
public:
    explicit Optimizer(double weight);
    
    void optimize(const std::vector<MeshModel::Vertex>& mesh_vertices,
                  const std::vector<MeshModel::Triangle>& mesh_triangles,
                  const Eigen::Matrix3d& camera_intrinsics,
                  const std::vector<cv::Mat>& observed_images,
                  std::vector<Eigen::Matrix4d>& camera_poses);

    void optimizePhotometryOnly(
        const std::vector<MeshModel::Vertex>& mesh_vertices,
        const std::vector<MeshModel::Triangle>& mesh_triangles,
        const Eigen::Matrix3d& camera_intrinsics,
        const std::vector<cv::Mat>& observed_images,
        const std::vector<Eigen::Matrix4d>& camera_poses,
        std::vector<double>& x2_values_out);

    void optimizeWithInitialPhotometry(
        const std::vector<MeshModel::Vertex>& mesh_vertices,
        const std::vector<MeshModel::Triangle>& mesh_triangles,
        const Eigen::Matrix3d& camera_intrinsics,
        const std::vector<cv::Mat>& observed_images,
        std::vector<Eigen::Matrix4d>& camera_poses,
        std::vector<double>& x2_values_inout);

private:
    double weight_;
    ceres::Solver::Options options_;
};

#endif // OPTIMIZER_H