#include "Projection.h"
#include <algorithm>
#include <cmath>
#include <limits>

Eigen::Vector2f Projection::projectPoint(
    const MeshModel::Vertex& vertex, 
    const Eigen::Matrix3f& intrinsics, 
    const Eigen::Matrix3f& rotation, 
    const Eigen::Vector3f& translation) {
    // 顶点坐标转为 Eigen::Vector3f
    Eigen::Vector3f point(vertex.x, vertex.y, vertex.z);

    // 世界坐标转相机坐标
    Eigen::Vector3f cameraPoint = rotation * point + translation;

    // 相机坐标投影到图像平面
    Eigen::Vector3f imagePoint = intrinsics * cameraPoint;

    // 归一化
    return Eigen::Vector2f(imagePoint(0) / imagePoint(2), imagePoint(1) / imagePoint(2));
}

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

std::vector<float> Projection::computeVertexDepths(
    const std::vector<MeshModel::Vertex>& vertices, 
    const Eigen::Matrix3f& cameraIntrinsics,
    const Eigen::Matrix3f& rotation, 
    const Eigen::Vector3f& translation) {
    // 计算每个顶点的深度
    std::vector<float> depths;
    depths.reserve(vertices.size());
    for (const auto& vertex : vertices) {
        Eigen::Vector3f point(vertex.x, vertex.y, vertex.z);
        Eigen::Vector3f cameraPoint = rotation * point + translation;
        depths.push_back(cameraPoint.z());
    }
    return depths;
}

std::vector<bool> Projection::handleOcclusion(
    const std::vector<MeshModel::Vertex>& vertices, 
    const std::vector<MeshModel::Triangle>& triangles,
    const Eigen::Matrix3f& intrinsics,
    const Eigen::Matrix3f& rotation,
    const Eigen::Vector3f& translation,
    int imageWidth,
    int imageHeight) {

    // 初始化深度缓冲区
    std::vector<std::vector<float>> depthBuffer(imageHeight, std::vector<float>(imageWidth, std::numeric_limits<float>::max()));
    std::vector<bool> visibleMask(vertices.size(), false);

    // 遍历每个三角形
    for (const auto& triangle : triangles) {
        // 获取三角形的顶点
        const auto& v0 = vertices[triangle.v0];
        const auto& v1 = vertices[triangle.v1];
        const auto& v2 = vertices[triangle.v2];

        // 投影到图像平面
        Eigen::Vector2f p0 = projectPoint(v0, intrinsics, rotation, translation);
        Eigen::Vector2f p1 = projectPoint(v1, intrinsics, rotation, translation);
        Eigen::Vector2f p2 = projectPoint(v2, intrinsics, rotation, translation);

        // 计算顶点深度
        float z0 = (rotation * Eigen::Vector3f(v0.x, v0.y, v0.z) + translation).z();
        float z1 = (rotation * Eigen::Vector3f(v1.x, v1.y, v1.z) + translation).z();
        float z2 = (rotation * Eigen::Vector3f(v2.x, v2.y, v2.z) + translation).z();

        // 光栅化三角形并更新深度缓冲区
        rasterizeTriangle(p0, p1, p2, z0, z1, z2, depthBuffer, imageWidth, imageHeight);
    }

    // 根据深度缓冲区确定顶点的可见性
    for (size_t i = 0; i < vertices.size(); ++i) {
        Eigen::Vector2f projected = projectPoint(vertices[i], intrinsics, rotation, translation);
        if (!isPointInImage(projected, imageWidth, imageHeight)) continue;
        int x = static_cast<int>(std::round(projected.x()));
        int y = static_cast<int>(std::round(projected.y()));

        if (std::abs(depthBuffer[y][x] - (rotation * Eigen::Vector3f(vertices[i].x, vertices[i].y, vertices[i].z) + translation).z()) < 1e-6) {
            visibleMask[i] = true;
        }
    }

    return visibleMask;
}

bool Projection::isPointInImage(
    const Eigen::Vector2f& point, 
    int imageWidth, 
    int imageHeight) {
    return point.x() >= 0 && point.x() < imageWidth && 
           point.y() >= 0 && point.y() < imageHeight;
}

void Projection::rasterizeTriangle(
    const Eigen::Vector2f& p0, const Eigen::Vector2f& p1, const Eigen::Vector2f& p2,
    float z0, float z1, float z2,
    std::vector<std::vector<float>>& depthBuffer,
    int imageWidth, int imageHeight) {
    // 获取三角形的边界框
    int minX = std::max(0, static_cast<int>(std::floor(std::min({p0.x(), p1.x(), p2.x()}))));
    int maxX = std::min(imageWidth - 1, static_cast<int>(std::ceil(std::max({p0.x(), p1.x(), p2.x()}))));
    int minY = std::max(0, static_cast<int>(std::floor(std::min({p0.y(), p1.y(), p2.y()}))));
    int maxY = std::min(imageHeight - 1, static_cast<int>(std::ceil(std::max({p0.y(), p1.y(), p2.y()}))));

    // 遍历边界框内的像素
    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            Eigen::Vector2f pixel(x + 0.5f, y + 0.5f);

            // 使用重心坐标法判断像素是否在三角形内部
            float w0 = (p1.x() - p0.x()) * (pixel.y() - p0.y()) - (p1.y() - p0.y()) * (pixel.x() - p0.x());
            float w1 = (p2.x() - p1.x()) * (pixel.y() - p1.y()) - (p2.y() - p1.y()) * (pixel.x() - p1.x());
            float w2 = (p0.x() - p2.x()) * (pixel.y() - p2.y()) - (p0.y() - p2.y()) * (pixel.x() - p2.x());

            if ((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0)) {
                // 计算深度值（插值）
                float area = std::abs(w0 + w1 + w2);
                float alpha = w1 / area, beta = w2 / area, gamma = w0 / area;
                float z = alpha * z0 + beta * z1 + gamma * z2;

                // 更新深度缓冲区
                if (z < depthBuffer[y][x]) {
                    depthBuffer[y][x] = z;
                }
            }
        }
    }
}
