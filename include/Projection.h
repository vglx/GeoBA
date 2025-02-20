#ifndef PROJECTION_H
#define PROJECTION_H

#include <Eigen/Dense>
#include <vector>
#include "MeshModel.h"

class Projection {
public:
    static Eigen::Vector2d projectPoint(
        const MeshModel::Vertex& vertex, 
        const Eigen::Matrix3d& intrinsics, 
        const Eigen::Matrix3d& rotation, 
        const Eigen::Vector3d& translation);

    static double computeVertexDepth(
        const MeshModel::Vertex& vertex, 
        const Eigen::Matrix3d& intrinsics,
        const Eigen::Matrix3d& rotation, 
        const Eigen::Vector3d& translation);

    static std::vector<bool> handleOcclusion(
        const std::vector<MeshModel::Vertex>& vertices,
        const std::vector<MeshModel::Triangle>& triangles,
        const Eigen::Matrix3d& intrinsics,
        const Eigen::Matrix3d& rotation,
        const Eigen::Vector3d& translation,
        int imageWidth,
        int imageHeight);
};

#endif // PROJECTION_H