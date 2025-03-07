#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "MeshModel.h"

class Optimizer {
public:
   
    Optimizer(double weight = 1.0);

    /**
     * @brief 运行优化，调整相机位姿
     * @param mesh_vertices 3D 网格顶点
     * @param observed_images RGB 图像序列
     * @param observed_depth_maps 深度图序列
     * @param camera_intrinsics 相机内参矩阵
     * @param camera_poses 初始相机位姿（优化变量）
     */
    void optimize(const std::vector<MeshModel::Vertex>& mesh_vertices,
                  const std::vector<MeshModel::Triangle>& mesh_triangles,
                  const Eigen::Matrix3d& camera_intrinsics,
                  const std::vector<cv::Mat>& observed_images,
                  const std::vector<cv::Mat>& observed_depth_maps,
                  const std::vector<Eigen::MatrixXf>& depth_normals,
                  std::vector<Eigen::Matrix4d>& camera_poses);

private:
    ceres::Solver::Options options_;  // Ceres 迭代优化参数

    // **误差项权重**
    double weight_;
};

#endif // OPTIMIZER_H
