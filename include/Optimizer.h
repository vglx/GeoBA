#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "MeshModel.h"
#include "BVH.h"
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

class Optimizer {
public:
    // 构造函数接收权重和最大迭代次数
    Optimizer(double weight, int maxStages, int maxIterations);

    // 优化函数接口，与之前保持一致
    void optimize(const std::vector<MeshModel::Vertex>& mesh_vertices,
                  const std::vector<MeshModel::Triangle>& mesh_triangles,
                  const Eigen::Matrix3d& camera_intrinsics,
                  const std::vector<cv::Mat>& observed_images,
                  std::vector<Eigen::Matrix4d>& camera_poses);
private:
    double weight_;
    int maxStages_;
    int maxIterations_;
};

#endif // OPTIMIZER_H