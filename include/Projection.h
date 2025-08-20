#ifndef PROJECTION_H
#define PROJECTION_H

#include <Eigen/Dense>
#include "MeshModel.h"
#include "BVH.h"
#include "EDGraph.h"

class Projection {
public:
    // 投影（保持与现有实现一致）：先可选 ED 变形，再用 T_wc 约定投影
    // p_c = R^T (p_w - t),  u = fx * X/Z + cx,  v = fy * Y/Z + cy
    static Eigen::Vector2d projectPoint(
        const MeshModel::Vertex& vertex,
        const Eigen::Matrix3d& intrinsics,
        const Eigen::Matrix3d& rotation,
        const Eigen::Vector3d& translation,
        int vidx,
        const EDGraph* ed = nullptr);

    // 兼容原签名的可见性检测（默认带 2px 边界安全带、0.5% 深度容差）
    static bool isVertexVisible(
        const MeshModel::Vertex& vertex,
        const Eigen::Matrix3d& intrinsics,
        const Eigen::Matrix3d& rotation,
        const Eigen::Vector3d& translation,
        const BVH& bvh,
        int imageWidth,
        int imageHeight,
        int vidx,
        const EDGraph* ed = nullptr);

    // 带可调参数版本：borderBand（像素），zEpsRel（相对深度容差，例如 0.005）
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
        int borderBand,
        double zEpsRel);
};

#endif // PROJECTION_H