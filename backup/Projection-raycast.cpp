#include "Projection.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <Eigen/Geometry>

Eigen::Vector2f Projection::projectPoint(
    const MeshModel::Vertex& vertex, 
    const Eigen::Matrix3f& intrinsics, 
    const Eigen::Matrix3f& rotation, 
    const Eigen::Vector3f& translation) {

    Eigen::Vector3f point(vertex.x, vertex.y, vertex.z);
    Eigen::Vector3f cameraPoint = rotation * point + translation;
    Eigen::Vector3f imagePoint = intrinsics * cameraPoint;

    return Eigen::Vector2f(imagePoint(0) / imagePoint(2), imagePoint(1) / imagePoint(2));
}

// 批量投影 3D 点到 2D 图像平面
std::vector<Eigen::Vector2f> Projection::projectPoints(
    const std::vector<MeshModel::Vertex>& vertices, 
    const Eigen::Matrix3f& intrinsics, 
    const Eigen::Matrix3f& rotation, 
    const Eigen::Vector3f& translation) {
    
    std::vector<Eigen::Vector2f> projectedPoints;
    projectedPoints.reserve(vertices.size());

    for (const auto& vertex : vertices) {
        projectedPoints.push_back(projectPoint(vertex, intrinsics, rotation, translation));
    }
    
    return projectedPoints;
}

// 计算每个顶点的深度
std::vector<float> Projection::computeVertexDepths(
    const std::vector<MeshModel::Vertex>& vertices, 
    const Eigen::Matrix3f& intrinsics,
    const Eigen::Matrix3f& rotation, 
    const Eigen::Vector3f& translation) {
    
    std::vector<float> depths;
    depths.reserve(vertices.size());

    for (const auto& vertex : vertices) {
        Eigen::Vector3f point(vertex.x, vertex.y, vertex.z);
        Eigen::Vector3f cameraPoint = rotation * point + translation;
        depths.push_back(cameraPoint.z());
    }

    return depths;
}

// 使用光线投射计算可见性
std::vector<bool> Projection::handleOcclusion(
    const std::vector<MeshModel::Vertex>& vertices,
    const std::vector<MeshModel::Triangle>& triangles,
    const Eigen::Matrix3f& intrinsics,
    const Eigen::Matrix3f& rotation,
    const Eigen::Vector3f& translation,
    int imageWidth,
    int imageHeight) {

    std::vector<bool> visibleMask(vertices.size(), false);

    // 计算相机光心
    Eigen::Vector3f cameraCenter = -(rotation.transpose() * translation);

    // 遍历所有顶点，检测可见性
    for (size_t i = 0; i < vertices.size(); ++i) {
        Eigen::Vector3f vertex(vertices[i].x, vertices[i].y, vertices[i].z);
        Eigen::Vector3f rayDir = (vertex - cameraCenter).normalized();

        if (rayIntersectsMesh(cameraCenter, rayDir, vertex, triangles, vertices)) {
            visibleMask[i] = true;
        }
    }

    return visibleMask;
}

bool Projection::rayIntersectsMesh(
    const Eigen::Vector3f& rayOrigin,
    const Eigen::Vector3f& rayDir,
    const Eigen::Vector3f& targetVertex,
    const std::vector<MeshModel::Triangle>& triangles,
    const std::vector<MeshModel::Vertex>& vertices) {

    float minHitDistance = std::numeric_limits<float>::max();
    bool targetHit = false;

    for (const auto& tri : triangles) {
        Eigen::Vector3f v0(vertices[tri.v0].x, vertices[tri.v0].y, vertices[tri.v0].z);
        Eigen::Vector3f v1(vertices[tri.v1].x, vertices[tri.v1].y, vertices[tri.v1].z);
        Eigen::Vector3f v2(vertices[tri.v2].x, vertices[tri.v2].y, vertices[tri.v2].z);

        // 计算三角形法向量，过滤掉背面三角形
        Eigen::Vector3f normal = (v1 - v0).cross(v2 - v0).normalized();
        if (rayDir.dot(normal) < 0) continue; 

        float t;
        if (rayTriangleIntersection(rayOrigin, rayDir, v0, v1, v2, t)) {
            if (t > 0 && t < minHitDistance) {
                minHitDistance = t;
                float targetDepth = (targetVertex - rayOrigin).norm();
                if (std::abs(t - targetDepth) < 1e-3) {
                    targetHit = true;
                }
            }
        }
    }

    return targetHit;
}

// 计算射线与三角形的交点（Möller–Trumbore算法）
bool Projection::rayTriangleIntersection(
    const Eigen::Vector3f& rayOrigin,
    const Eigen::Vector3f& rayDir,
    const Eigen::Vector3f& v0,
    const Eigen::Vector3f& v1,
    const Eigen::Vector3f& v2,
    float& t) {

    const float EPSILON = 1e-6;
    Eigen::Vector3f edge1 = v1 - v0;
    Eigen::Vector3f edge2 = v2 - v0;
    Eigen::Vector3f h = rayDir.cross(edge2);
    float a = edge1.dot(h);

    if (std::abs(a) < EPSILON) return false; // 射线与三角形平行

    float f = 1.0 / a;
    Eigen::Vector3f s = rayOrigin - v0;
    float u = f * s.dot(h);

    if (u < 0.0 || u > 1.0) return false;

    Eigen::Vector3f q = s.cross(edge1);
    float v = f * rayDir.dot(q);

    if (v < 0.0 || u + v > 1.0) return false;

    t = f * edge2.dot(q);

    if (t > EPSILON) return true; // 交点在射线上

    return false;
}