#ifndef PROJECTION_H
#define PROJECTION_H

#include <Eigen/Dense>
#include "MeshModel.h"
#include "BVH.h"
#include "EDGraph.h" 

class Projection {
public:
    // 投影：必须提供 ED 变形状态
    static Eigen::Vector2d projectPoint(
        const MeshModel::Vertex& vertex,
        const Eigen::Matrix3d& intrinsics,
        const Eigen::Matrix3d& rotation,
        const Eigen::Vector3d& translation,
        int vidx,
        const EDGraph* ed,
        const EDState& state);

    // 可见性检测：必须提供 ED 变形状态
    static bool isVertexVisible(
        const MeshModel::Vertex& vertex,
        const Eigen::Matrix3d& intrinsics,
        const Eigen::Matrix3d& rotation,
        const Eigen::Vector3d& translation,
        const BVH& bvh,
        int imageWidth,
        int imageHeight,
        int vidx,
        const EDGraph* ed,
        const EDState& state);
};

#endif // PROJECTION_H