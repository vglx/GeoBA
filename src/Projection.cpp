#include "Projection.h"
#include <limits>
#include <Eigen/Geometry>
#include <iostream>

Eigen::Vector2d Projection::projectPoint(
    const MeshModel::Vertex& vertex, 
    const Eigen::Matrix3d& intrinsics, 
    const Eigen::Matrix3d& rotation, 
    const Eigen::Vector3d& translation) {

    Eigen::Vector3d point(vertex.x, vertex.y, vertex.z);
    Eigen::Vector3d cameraPoint = rotation.transpose() * (point - translation);
    Eigen::Vector3d imagePoint = intrinsics * cameraPoint;
    return Eigen::Vector2d(imagePoint(0) / imagePoint(2), imagePoint(1) / imagePoint(2));
}

bool Projection::isVertexVisible(
    const MeshModel::Vertex& vertex,
    const Eigen::Matrix3d& intrinsics,
    const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& translation,
    const BVH& bvh,
    int imageWidth,
    int imageHeight) {

    Eigen::Vector3d cameraCenter = translation;
    Eigen::Vector3d vertexPos(vertex.x, vertex.y, vertex.z);
    Eigen::Vector3d camPoint = rotation.transpose() * (vertexPos - translation);

    // 如果点在相机后面，则不可见
    if (camPoint.z() <= 0) {
        return false;
    }
    
    // 投影到图像平面
    Eigen::Vector3d imagePoint = intrinsics * camPoint;
    double imgX = imagePoint(0) / imagePoint(2);
    double imgY = imagePoint(1) / imagePoint(2);

    // 检查是否超出图像范围
    if (imgX < 0 || imgX >= imageWidth || imgY < 0 || imgY >= imageHeight) {
        return false;
    }

    // 计算射线方向
    Eigen::Vector3d rayDir = (vertexPos - cameraCenter).normalized();
    double tHit = std::numeric_limits<double>::max();

    // 调用 BVH 进行遮挡检测
    return bvh.traverse(cameraCenter, rayDir, tHit) && 
           std::abs(tHit - (vertexPos - cameraCenter).norm()) < 0.005 * tHit;
}