#ifndef PROJECTION_H
#define PROJECTION_H

#include <Eigen/Dense>
#include <vector>
#include "MeshModel.h"
#include "BVH.h"

class Projection {
public:
    static Eigen::Vector2d projectPoint(
        const MeshModel::Vertex& vertex, 
        const Eigen::Matrix3d& intrinsics, 
        const Eigen::Matrix3d& rotation, 
        const Eigen::Vector3d& translation);
    
    static bool isVertexVisible(
        const MeshModel::Vertex& vertex,
        const Eigen::Matrix3d& intrinsics,
        const Eigen::Matrix3d& rotation,
        const Eigen::Vector3d& translation,
        const BVH& bvh,
        int imageWidth,
        int imageHeight);
};

#endif // PROJECTION_H