#include "Projection.h"
#include <limits>
#include <Eigen/Geometry>

Eigen::Vector2d Projection::projectPoint(
    const MeshModel::Vertex& vertex,
    const Eigen::Matrix3d& intrinsics,
    const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& translation,
    int vidx,
    const EDGraph* ed,
    const EDState& state) {
    Eigen::Vector3d pos0(vertex.x, vertex.y, vertex.z);
    Eigen::Vector3d point = ed ? ed->deformVertex(vertex, vidx, state) : pos0;
    Eigen::Vector3d cameraPoint = rotation.transpose() * (point - translation);
    Eigen::Vector3d imagePoint = intrinsics * cameraPoint;
    return Eigen::Vector2d(
        imagePoint(0) / imagePoint(2),
        imagePoint(1) / imagePoint(2)
    );
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
    const EDState& state) {
    Eigen::Vector3d cameraCenter = translation;
    Eigen::Vector3d pos0(vertex.x, vertex.y, vertex.z);
    Eigen::Vector3d vertexPos = ed ? ed->deformVertex(vertex, vidx, state) : pos0;

    Eigen::Vector3d camPoint = rotation.transpose() * (vertexPos - translation);
    if (camPoint.z() <= 0) return false;

    Eigen::Vector3d imagePoint = intrinsics * camPoint;
    double imgX = imagePoint(0) / imagePoint(2);
    double imgY = imagePoint(1) / imagePoint(2);
    if (imgX < 0 || imgX >= imageWidth || imgY < 0 || imgY >= imageHeight)
        return false;

    Eigen::Vector3d rayDir = (vertexPos - cameraCenter).normalized();
    double tHit = std::numeric_limits<double>::max();
    return bvh.traverse(cameraCenter, rayDir, tHit)
        && std::abs(tHit - (vertexPos - cameraCenter).norm()) < 0.005 * tHit;
}