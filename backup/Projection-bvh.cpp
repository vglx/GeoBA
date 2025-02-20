#include "Projection.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <Eigen/Geometry>
#include <vector>
#include <omp.h>

// -------------------------
// BVH 数据结构与辅助函数
// -------------------------
struct BVHNode {
    Eigen::Vector3f bbox_min;
    Eigen::Vector3f bbox_max;
    int left;   // 如果为叶子节点，则 left == -1
    int right;  // 如果为叶子节点，则 right == -1
    std::vector<int> triangleIndices; // 叶子节点存储对应的三角形索引
};

namespace {

// 计算射线与 AABB 的相交（slab 方法）
bool intersectAABB(const Eigen::Vector3f& rayOrigin, const Eigen::Vector3f& rayDir,
                   const Eigen::Vector3f& bbox_min, const Eigen::Vector3f& bbox_max,
                   float& tmin, float& tmax) {
    tmin = 0.0f;
    tmax = std::numeric_limits<float>::max();
    for (int i = 0; i < 3; ++i) {
        float invD = 1.0f / rayDir[i];
        float t0 = (bbox_min[i] - rayOrigin[i]) * invD;
        float t1 = (bbox_max[i] - rayOrigin[i]) * invD;
        if (invD < 0.0f) std::swap(t0, t1);
        tmin = std::max(tmin, t0);
        tmax = std::min(tmax, t1);
        if (tmax <= tmin)
            return false;
    }
    return true;
}

// 递归构建 BVH 节点
int buildBVH(const std::vector<int>& triIndices,
             const std::vector<MeshModel::Triangle>& triangles,
             const std::vector<MeshModel::Vertex>& vertices,
             std::vector<BVHNode>& nodes) {
    BVHNode node;
    // 计算当前节点内所有三角形的包围盒与质心
    Eigen::Vector3f bbox_min( std::numeric_limits<float>::max(),
                              std::numeric_limits<float>::max(),
                              std::numeric_limits<float>::max());
    Eigen::Vector3f bbox_max(-std::numeric_limits<float>::max(),
                             -std::numeric_limits<float>::max(),
                             -std::numeric_limits<float>::max());
    std::vector<Eigen::Vector3f> centroids;
    centroids.reserve(triIndices.size());

    for (int idx : triIndices) {
        const MeshModel::Triangle& tri = triangles[idx];
        Eigen::Vector3f v0(vertices[tri.v0].x, vertices[tri.v0].y, vertices[tri.v0].z);
        Eigen::Vector3f v1(vertices[tri.v1].x, vertices[tri.v1].y, vertices[tri.v1].z);
        Eigen::Vector3f v2(vertices[tri.v2].x, vertices[tri.v2].y, vertices[tri.v2].z);
        Eigen::Vector3f tri_min = v0.cwiseMin(v1).cwiseMin(v2);
        Eigen::Vector3f tri_max = v0.cwiseMax(v1).cwiseMax(v2);
        bbox_min = bbox_min.cwiseMin(tri_min);
        bbox_max = bbox_max.cwiseMax(tri_max);
        Eigen::Vector3f centroid = (v0 + v1 + v2) / 3.0f;
        centroids.push_back(centroid);
    }
    node.bbox_min = bbox_min;
    node.bbox_max = bbox_max;
    node.left = -1;
    node.right = -1;

    // 如果三角形数较少，则设为叶子节点
    if (triIndices.size() <= 4) {
        node.triangleIndices = triIndices;
        nodes.push_back(node);
        return nodes.size() - 1;
    }

    // 选择分裂轴：取包围盒最长的维度
    Eigen::Vector3f extent = bbox_max - bbox_min;
    int axis = 0;
    if (extent[1] > extent[0]) axis = 1;
    if (extent[2] > extent[axis]) axis = 2;

    // 根据质心在该轴上的值进行分裂：先求中位数
    std::vector<int> leftIndices, rightIndices;
    std::vector<float> centroidVals;
    centroidVals.reserve(centroids.size());
    for (const auto& c : centroids) {
        centroidVals.push_back(c[axis]);
    }
    std::nth_element(centroidVals.begin(), centroidVals.begin() + centroidVals.size()/2, centroidVals.end());
    float median = centroidVals[centroidVals.size()/2];

    for (size_t i = 0; i < triIndices.size(); ++i) {
        if (centroids[i][axis] < median) {
            leftIndices.push_back(triIndices[i]);
        } else {
            rightIndices.push_back(triIndices[i]);
        }
    }
    // 防止一侧为空则均分
    if (leftIndices.empty() || rightIndices.empty()) {
        leftIndices.clear();
        rightIndices.clear();
        for (size_t i = 0; i < triIndices.size(); ++i) {
            if (i < triIndices.size()/2)
                leftIndices.push_back(triIndices[i]);
            else
                rightIndices.push_back(triIndices[i]);
        }
    }

    int leftChild  = buildBVH(leftIndices, triangles, vertices, nodes);
    int rightChild = buildBVH(rightIndices, triangles, vertices, nodes);
    node.left = leftChild;
    node.right = rightChild;
    nodes.push_back(node);
    return nodes.size() - 1;
}

// 计算射线与三角形的交点（Möller–Trumbore 算法）
bool rayTriangleIntersection(
    const Eigen::Vector3f& rayOrigin,
    const Eigen::Vector3f& rayDir,
    const Eigen::Vector3f& v0,
    const Eigen::Vector3f& v1,
    const Eigen::Vector3f& v2,
    float& t) {

    const float EPSILON = 1e-6f;
    Eigen::Vector3f edge1 = v1 - v0;
    Eigen::Vector3f edge2 = v2 - v0;
    Eigen::Vector3f h = rayDir.cross(edge2);
    float a = edge1.dot(h);

    if (std::abs(a) < EPSILON) return false; // 射线与三角形平行

    float f = 1.0f / a;
    Eigen::Vector3f s = rayOrigin - v0;
    float u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f) return false;

    Eigen::Vector3f q = s.cross(edge1);
    float v = f * rayDir.dot(q);
    if (v < 0.0f || u + v > 1.0f) return false;

    t = f * edge2.dot(q);
    if (t > EPSILON) return true; // 交点在射线上

    return false;
}

// 遍历 BVH，返回射线沿方向的最近交点距离（存入 tHit）；若无交点则返回 false
bool traverseBVH(const std::vector<BVHNode>& nodes,
                 const std::vector<MeshModel::Triangle>& triangles,
                 const std::vector<MeshModel::Vertex>& vertices,
                 int nodeIndex,
                 const Eigen::Vector3f& rayOrigin,
                 const Eigen::Vector3f& rayDir,
                 float& tHit) {
    const BVHNode& node = nodes[nodeIndex];
    float tmin, tmax;
    if (!intersectAABB(rayOrigin, rayDir, node.bbox_min, node.bbox_max, tmin, tmax))
        return false;

    bool hit = false;
    // 如果是叶子节点，则遍历所有三角形
    if (node.left == -1 && node.right == -1) {
        for (int idx : node.triangleIndices) {
            const MeshModel::Triangle& tri = triangles[idx];
            Eigen::Vector3f v0(vertices[tri.v0].x, vertices[tri.v0].y, vertices[tri.v0].z);
            Eigen::Vector3f v1(vertices[tri.v1].x, vertices[tri.v1].y, vertices[tri.v1].z);
            Eigen::Vector3f v2(vertices[tri.v2].x, vertices[tri.v2].y, vertices[tri.v2].z);
            float t;
            if (rayTriangleIntersection(rayOrigin, rayDir, v0, v1, v2, t)) {
                if (t > 0 && t < tHit) {
                    tHit = t;
                    hit = true;
                }
            }
        }
    } else {
        // 内部节点：递归遍历左右子节点
        if (node.left != -1)
            if (traverseBVH(nodes, triangles, vertices, node.left, rayOrigin, rayDir, tHit))
                hit = true;
        if (node.right != -1)
            if (traverseBVH(nodes, triangles, vertices, node.right, rayOrigin, rayDir, tHit))
                hit = true;
    }
    return hit;
}

} // end anonymous namespace

// -------------------------
// Projection 类实现
// -------------------------
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

// 修改后的使用 BVH 加速的遮挡判断函数
std::vector<bool> Projection::handleOcclusion(
    const std::vector<MeshModel::Vertex>& vertices,
    const std::vector<MeshModel::Triangle>& triangles,
    const Eigen::Matrix3f& intrinsics,
    const Eigen::Matrix3f& rotation,
    const Eigen::Vector3f& translation,
    int imageWidth,
    int imageHeight) {

    std::vector<bool> visibleMask(vertices.size(), false);

    // 构建三角形索引列表
    std::vector<int> triIndices(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) {
        triIndices[i] = i;
    }
    // 构建 BVH（这里每次调用时构建，若网格不变可考虑缓存）
    std::vector<BVHNode> nodes;
    nodes.reserve(triangles.size());
    int rootIndex = buildBVH(triIndices, triangles, vertices, nodes);

    // 计算相机光心
    Eigen::Vector3f cameraCenter = -(rotation.transpose() * translation);

    // 对每个顶点做遮挡判断
    #pragma omp parallel for
    for (size_t i = 0; i < vertices.size(); ++i) {
        Eigen::Vector3f vertex(vertices[i].x, vertices[i].y, vertices[i].z);
        Eigen::Vector3f rayDir = (vertex - cameraCenter).normalized();
        float tHit = std::numeric_limits<float>::max();
        bool hit = traverseBVH(nodes, triangles, vertices, rootIndex, cameraCenter, rayDir, tHit);
        float targetDepth = (vertex - cameraCenter).norm();
        // 如果最接近的交点与目标距离接近（容差 1e-3），认为该顶点可见
        if (hit && std::abs(tHit - targetDepth) < 1e-3) {
            visibleMask[i] = true;
        } else {
            visibleMask[i] = false;
        }
    }

    return visibleMask;
}