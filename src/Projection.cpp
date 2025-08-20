#include "Projection.h"
#include <limits>
#include <algorithm>
#include <Eigen/Geometry>

Eigen::Vector2d Projection::projectPoint(
    const MeshModel::Vertex& vertex,
    const Eigen::Matrix3d& intrinsics,
    const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& translation,
    int vidx,
    const EDGraph* ed) {
    // 原始顶点 -> 可选 ED 变形
    const Eigen::Vector3d pos0(vertex.x, vertex.y, vertex.z);
    const Eigen::Vector3d pw = ed ? ed->deformVertex(vertex, vidx) : pos0;

    // T_wc 约定：p_c = R^T (p_w - t)
    const Eigen::Vector3d pc = rotation.transpose() * (pw - translation);
    const Eigen::Vector3d ip = intrinsics * pc;
    return { ip.x() / ip.z(), ip.y() / ip.z() };
}

bool Projection::isVertexVisible(
    const MeshModel::Vertex& vertex,
    const Eigen::Matrix3d& intrinsics,
    const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& translation,
    const BVH& bvh,
    int imageWidth,
    int imageHeight,
    int vidx,
    const EDGraph* ed) {
    // 默认：2px 边界安全带，0.5% 深度容差
    return isVertexVisible(vertex, intrinsics, rotation, translation,
                           bvh, imageWidth, imageHeight, vidx, ed,
                           /*borderBand=*/2, /*zEpsRel=*/0.005);
}

bool Projection::isVertexVisible(
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
    double zEpsRel) {

    // 1) 可选 ED 变形后的世界坐标
    const Eigen::Vector3d pos0(vertex.x, vertex.y, vertex.z);
    const Eigen::Vector3d pw = ed ? ed->deformVertex(vertex, vidx) : pos0;

    // 2) 相机前方性：Z>0
    const Eigen::Vector3d pc = rotation.transpose() * (pw - translation);
    if (pc.z() <= 0.0) return false;

    // 3) 像素坐标 + 边界安全带
    const double fx = intrinsics(0,0), fy = intrinsics(1,1);
    const double cx = intrinsics(0,2), cy = intrinsics(1,2);
    const double invZ = 1.0 / pc.z();
    const double u = fx * (pc.x() * invZ) + cx;
    const double v = fy * (pc.y() * invZ) + cy;
    if (u < borderBand || u >= imageWidth  - borderBand ||
        v < borderBand || v >= imageHeight - borderBand) return false;

    // 4) BVH 遮挡检测（射线从相机指向该点）
    const Eigen::Vector3d cam = translation;
    const Eigen::Vector3d ray = (pw - cam).normalized();
    double tHit = std::numeric_limits<double>::max();
    if (!bvh.traverse(cam, ray, tHit)) return false; // 无命中，保守为不可见（与原实现一致）

    const double tV = (pw - cam).norm();
    const double eps = std::max(1e-9, zEpsRel * tHit);
    // 若最近命中深度与顶点深度足够接近，则认为“没有更近遮挡物”，可见
    return std::abs(tHit - tV) < eps;
}