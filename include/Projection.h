#ifndef PROJECTION_H
#define PROJECTION_H

#include <Eigen/Core>
#include <vector>
#include <limits>
#include <cmath>
#include <ceres/ceres.h>
#include "MeshModel.h"


class Projection {
public:
    // 3D 点投影到 2D 图像平面
    template <typename T>
    static Eigen::Vector2f projectPoint(
        const MeshModel::Vertex& vertex, 
        const Eigen::Matrix3f& intrinsics, 
        const Eigen::Matrix<T, 3, 3>& rotation, 
        const Eigen::Matrix<T, 3, 1>& translation) {

        Eigen::Matrix<T, 3, 1> point(T(vertex.x), T(vertex.y), T(vertex.z));
        Eigen::Matrix<T, 3, 1> cameraPoint = rotation * point + translation;
        Eigen::Matrix<T, 3, 1> imagePoint = intrinsics.cast<T>() * cameraPoint;

        if constexpr (std::is_same<T, ceres::Jet<double, 6>>::value || std::is_same<T, ceres::Jet<double, 18>>::value) {
            return Eigen::Vector2f(static_cast<float>(imagePoint(0).a) / static_cast<float>(imagePoint(2).a),
                                   static_cast<float>(imagePoint(1).a) / static_cast<float>(imagePoint(2).a));
        } else {
            return Eigen::Vector2f(static_cast<float>(imagePoint(0)) / static_cast<float>(imagePoint(2)),
                                   static_cast<float>(imagePoint(1)) / static_cast<float>(imagePoint(2)));
        }
    }

    // 批量投影 3D 点到 2D 图像平面
    template <typename T>
    static std::vector<Eigen::Vector2f> projectPoints(
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

    // 计算每个顶点的深度
    template <typename T>
    static std::vector<float> computeVertexDepths(
        const std::vector<MeshModel::Vertex>& vertices, 
        const Eigen::Matrix3f& intrinsics,
        const Eigen::Matrix<T, 3, 3>& rotation, 
        const Eigen::Matrix<T, 3, 1>& translation) {
        
        std::vector<float> depths;
        depths.reserve(vertices.size());
        for (const auto& vertex : vertices) {
            Eigen::Matrix<T, 3, 1> point(T(vertex.x), T(vertex.y), T(vertex.z));
            Eigen::Matrix<T, 3, 1> cameraPoint = rotation * point + translation;
            if constexpr (std::is_same<T, ceres::Jet<double, 6>>::value || std::is_same<T, ceres::Jet<double, 18>>::value) {
                depths.push_back(static_cast<float>(cameraPoint.z().a));  // 取 `.a` 成员
            } else {
                depths.push_back(static_cast<float>(cameraPoint.z()));  // 普通情况
            }
        }
        return depths;
    }

    // 遮挡处理：考虑面片遮挡
    template <typename T>
    static std::vector<bool> handleOcclusion(
        const std::vector<MeshModel::Vertex>& vertices, 
        const std::vector<MeshModel::Triangle>& triangles,
        const Eigen::Matrix3f& intrinsics,
        const Eigen::Matrix<T, 3, 3>& rotation,
        const Eigen::Matrix<T, 3, 1>& translation,
        int imageWidth,
        int imageHeight) {

        std::vector<std::vector<float>> depthBuffer(imageHeight, std::vector<float>(imageWidth, std::numeric_limits<float>::max()));
        std::vector<bool> visibleMask(vertices.size(), false);

        for (const auto& triangle : triangles) {
            const auto& v0 = vertices[triangle.v0];
            const auto& v1 = vertices[triangle.v1];
            const auto& v2 = vertices[triangle.v2];

            Eigen::Vector2f p0 = projectPoint(v0, intrinsics, rotation, translation);
            Eigen::Vector2f p1 = projectPoint(v1, intrinsics, rotation, translation);
            Eigen::Vector2f p2 = projectPoint(v2, intrinsics, rotation, translation);

            Eigen::Matrix<T, 3, 1> p0_world(T(v0.x), T(v0.y), T(v0.z));
            Eigen::Matrix<T, 3, 1> p1_world(T(v1.x), T(v1.y), T(v1.z));
            Eigen::Matrix<T, 3, 1> p2_world(T(v2.x), T(v2.y), T(v2.z));

            T z0 = (rotation * p0_world + translation).z();
            T z1 = (rotation * p1_world + translation).z();
            T z2 = (rotation * p2_world + translation).z();

            if constexpr (std::is_same<T, ceres::Jet<double, 6>>::value || std::is_same<T, ceres::Jet<double, 18>>::value) {
                rasterizeTriangle(p0, p1, p2, static_cast<float>(z0.a), static_cast<float>(z1.a), static_cast<float>(z2.a), depthBuffer, imageWidth, imageHeight);
            } 
            else {
                rasterizeTriangle(p0, p1, p2, static_cast<float>(z0), static_cast<float>(z1), static_cast<float>(z2), depthBuffer, imageWidth, imageHeight);
            }
        }

        for (size_t i = 0; i < vertices.size(); ++i) {
            Eigen::Vector2f projected = projectPoint(vertices[i], intrinsics, rotation, translation);
            if (!isPointInImage(projected, imageWidth, imageHeight)) continue;
            int x = static_cast<int>(std::round(projected.x()));
            int y = static_cast<int>(std::round(projected.y()));

            Eigen::Matrix<T, 3, 1> point_world(T(vertices[i].x), T(vertices[i].y), T(vertices[i].z));
            T point_depth = (rotation * point_world + translation).z();

            if constexpr (std::is_same<T, ceres::Jet<double, 6>>::value || std::is_same<T, ceres::Jet<double, 18>>::value) {
                if (std::abs(depthBuffer[y][x] - static_cast<float>(point_depth.a)) < 1e-6f) {
                    visibleMask[i] = true;
                }
            } 
            else {
                if (std::abs(depthBuffer[y][x] - static_cast<float>(point_depth)) < 1e-6f) {
                    visibleMask[i] = true;
                }
            }
        }

        return visibleMask;
    }

    // 验证投影点是否在图像范围内
    static bool isPointInImage(
        const Eigen::Vector2f& point, 
        int imageWidth, 
        int imageHeight);

private:
    // 辅助方法：光栅化三角形并更新深度缓冲区
    static void rasterizeTriangle(
        const Eigen::Vector2f& p0, const Eigen::Vector2f& p1, const Eigen::Vector2f& p2,
        float z0, float z1, float z2,
        std::vector<std::vector<float>>& depthBuffer,
        int imageWidth, int imageHeight);
};

#endif // PROJECTION_H
