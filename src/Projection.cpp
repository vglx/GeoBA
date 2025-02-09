#include "Projection.h"
#include <ceres/ceres.h>
#include <algorithm>
#include <cmath>
#include <limits>

template <typename T>
Eigen::Vector2f Projection::projectPoint(
    const MeshModel::Vertex& vertex, 
    const Eigen::Matrix3f& intrinsics, 
    const Eigen::Matrix<T, 3, 3>& rotation, 
    const Eigen::Matrix<T, 3, 1>& translation) {

    Eigen::Matrix<T, 3, 1> point(T(vertex.x), T(vertex.y), T(vertex.z));  
    Eigen::Matrix<T, 3, 1> cameraPoint = rotation * point + translation;
    Eigen::Matrix<double, 3, 3> intrinsics_double = intrinsics.cast<double>();  // 先转 double
    Eigen::Matrix<T, 3, 3> intrinsicsT = intrinsics_double.template cast<T>();  // 再转 T
    Eigen::Matrix<T, 3, 1> imagePoint = intrinsicsT * cameraPoint;

    if constexpr (std::is_same<T, ceres::Jet<double, 6>>::value) {
        return Eigen::Vector2f(static_cast<float>(imagePoint(0).a) / static_cast<float>(imagePoint(2).a),
                               static_cast<float>(imagePoint(1).a) / static_cast<float>(imagePoint(2).a));
    } 
    else {
        return Eigen::Vector2f(static_cast<float>(imagePoint(0)) / static_cast<float>(imagePoint(2)),
                               static_cast<float>(imagePoint(1)) / static_cast<float>(imagePoint(2)));
    }
}

template <typename T>
std::vector<Eigen::Vector2f> Projection::projectPoints(
    const std::vector<MeshModel::Vertex>& vertices, 
    const Eigen::Matrix3f& intrinsics, 
    const Eigen::Matrix<T, 3, 3>& rotation, 
    const Eigen::Matrix<T, 3, 1>& translation) {
    std::vector<Eigen::Vector2f> projectedPoints;
    projectedPoints.reserve(vertices.size());
    for (const auto& vertex : vertices) {
        projectedPoints.push_back(projectPoint(vertex, intrinsics, rotation, translation));
    }
    return projectedPoints;
}

template <typename T>
std::vector<float> Projection::computeVertexDepths(
    const std::vector<MeshModel::Vertex>& vertices, 
    const Eigen::Matrix3f& intrinsics,
    const Eigen::Matrix<T, 3, 3>& rotation, 
    const Eigen::Matrix<T, 3, 1>& translation) {
    // 计算每个顶点的深度
    std::vector<float> depths;
    depths.reserve(vertices.size());
    for (const auto& vertex : vertices) {
        Eigen::Matrix<T, 3, 1> point(T(vertex.x), T(vertex.y), T(vertex.z));  // ✅ 统一 `T`
        Eigen::Matrix<T, 3, 1> cameraPoint = rotation * point + translation;  // ✅ 类型匹配
        if constexpr (std::is_same<T, ceres::Jet<double, 6>>::value) {
            depths.push_back(static_cast<float>(cameraPoint.z().a));  // 提取 `a` 值
        } 
        else {
            depths.push_back(static_cast<float>(cameraPoint.z()));  // 直接转换
        }
    }
    return depths;
}

template <typename T>
std::vector<bool> Projection::handleOcclusion(
    const std::vector<MeshModel::Vertex>& vertices, 
    const std::vector<MeshModel::Triangle>& triangles,
    const Eigen::Matrix3f& intrinsics,
    const Eigen::Matrix<T, 3, 3>& rotation,
    const Eigen::Matrix<T, 3, 1>& translation,
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
        Eigen::Matrix<T, 3, 1> p0_world(T(v0.x), T(v0.y), T(v0.z));
        Eigen::Matrix<T, 3, 1> p1_world(T(v1.x), T(v1.y), T(v1.z));
        Eigen::Matrix<T, 3, 1> p2_world(T(v2.x), T(v2.y), T(v2.z));

        T z0 = (rotation * p0_world + translation).z();  // ✅ `T` 统一
        T z1 = (rotation * p1_world + translation).z();
        T z2 = (rotation * p2_world + translation).z();
        // 光栅化三角形并更新深度缓冲区
        if constexpr (std::is_same<T, ceres::Jet<double, 6>>::value) {
            rasterizeTriangle(p0, p1, p2, 
                static_cast<float>(z0.a), 
                static_cast<float>(z1.a), 
                static_cast<float>(z2.a), 
                depthBuffer, imageWidth, imageHeight);
        } 
        else {
            rasterizeTriangle(p0, p1, p2, 
                static_cast<float>(z0), 
                static_cast<float>(z1), 
                static_cast<float>(z2), 
                depthBuffer, imageWidth, imageHeight);
        }
    }

    // 根据深度缓冲区确定顶点的可见性
    for (size_t i = 0; i < vertices.size(); ++i) {
        Eigen::Vector2f projected = projectPoint(vertices[i], intrinsics, rotation, translation);
        if (!isPointInImage(projected, imageWidth, imageHeight)) continue;
        int x = static_cast<int>(std::round(projected.x()));
        int y = static_cast<int>(std::round(projected.y()));

        Eigen::Matrix<T, 3, 1> point_world(T(vertices[i].x), T(vertices[i].y), T(vertices[i].z));
        T point_depth = (rotation * point_world + translation).z();

        if constexpr (std::is_same<T, ceres::Jet<double, 6>>::value) {
            if (Eigen::numext::abs(depthBuffer[y][x] - point_depth.a) < 1e-6f) {
                visibleMask[i] = true;
            }
        } 
        else {
            if (Eigen::numext::abs(depthBuffer[y][x] - point_depth) < T(1e-6)) {
                visibleMask[i] = true;
            }
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
                float area = std::abs(w0 + w1 + w2) + 1e-6f;
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

// 显式实例化模板（对于 double）
template std::vector<bool> Projection::handleOcclusion<double>(
    const std::vector<MeshModel::Vertex>&, const std::vector<MeshModel::Triangle>&,
    const Eigen::Matrix3f&, const Eigen::Matrix<double, 3, 3>&,
    const Eigen::Matrix<double, 3, 1>&, int, int);

template std::vector<Eigen::Vector2f> Projection::projectPoints<double>(
    const std::vector<MeshModel::Vertex>&, const Eigen::Matrix3f&,
    const Eigen::Matrix<double, 3, 3>&, const Eigen::Matrix<double, 3, 1>&);

template std::vector<float> Projection::computeVertexDepths<double>(
    const std::vector<MeshModel::Vertex>&, const Eigen::Matrix3f&,
    const Eigen::Matrix<double, 3, 3>&, const Eigen::Matrix<double, 3, 1>&);

template std::vector<bool> Projection::handleOcclusion<ceres::Jet<double, 6>>(
    const std::vector<MeshModel::Vertex>&, const std::vector<MeshModel::Triangle>&,
    const Eigen::Matrix3f&, const Eigen::Matrix<ceres::Jet<double, 6>, 3, 3>&,
    const Eigen::Matrix<ceres::Jet<double, 6>, 3, 1>&, int, int);

template std::vector<Eigen::Vector2f> Projection::projectPoints<ceres::Jet<double, 6>>(
    const std::vector<MeshModel::Vertex>&, const Eigen::Matrix3f&,
    const Eigen::Matrix<ceres::Jet<double, 6>, 3, 3>&, const Eigen::Matrix<ceres::Jet<double, 6>, 3, 1>&);

template std::vector<float> Projection::computeVertexDepths<ceres::Jet<double, 6>>(
    const std::vector<MeshModel::Vertex>&, const Eigen::Matrix3f&,
    const Eigen::Matrix<ceres::Jet<double, 6>, 3, 3>&, const Eigen::Matrix<ceres::Jet<double, 6>, 3, 1>&);
