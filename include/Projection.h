#ifndef PROJECTION_H
#define PROJECTION_H

#include <Eigen/Core>
#include <vector>
#include "MeshModel.h" // 引入 MeshModel 用于三角形和顶点的定义

class Projection {
public:
    // 3D 点投影到 2D 图像平面
    static Eigen::Vector2f projectPoint(
        const MeshModel::Vertex& vertex, 
        const Eigen::Matrix3f& intrinsics, 
        const Eigen::Matrix3f& rotation, 
        const Eigen::Vector3f& translation);

    // 批量投影 3D 点到 2D 图像平面
    static std::vector<Eigen::Vector2f> projectPoints(
        const std::vector<MeshModel::Vertex>& vertices, 
        const Eigen::Matrix3f& intrinsics, 
        const Eigen::Matrix3f& rotation, 
        const Eigen::Vector3f& translation);

    // 遮挡处理：考虑面片遮挡
    static std::vector<bool> handleOcclusion(
        const std::vector<MeshModel::Vertex>& vertices, 
        const std::vector<MeshModel::Triangle>& triangles,
        const Eigen::Matrix3f& intrinsics,
        const Eigen::Matrix3f& rotation,
        const Eigen::Vector3f& translation,
        int imageWidth,
        int imageHeight);

    // 计算每个顶点的深度
    static std::vector<float> computeVertexDepths(
        const std::vector<MeshModel::Vertex>& vertices, 
        const Eigen::Matrix3f& cameraIntrinsics,
        const Eigen::Matrix3f& rotation, 
        const Eigen::Vector3f& translation);

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
