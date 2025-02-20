#ifndef PROJECTION_H
#define PROJECTION_H

#include <algorithm>
#include <limits>
#include <cmath>
#include <vector>
#include <omp.h>
#include <Eigen/Geometry>
#include <ceres/jet.h>
#include "MeshModel.h"

namespace ProjectionDetail {

// 内部使用的 BVH 节点结构体
struct BVHNode {
    Eigen::Vector3f bbox_min;
    Eigen::Vector3f bbox_max;
    int left;   // 如果为叶子节点，则 left == -1
    int right;  // 如果为叶子节点，则 right == -1
    std::vector<int> triangleIndices; // 叶子节点存储对应的三角形索引
};

// 计算射线与 AABB 的交点（slab 方法）
inline bool intersectAABB(const Eigen::Vector3f& rayOrigin,
                          const Eigen::Vector3f& rayDir,
                          const Eigen::Vector3f& bbox_min,
                          const Eigen::Vector3f& bbox_max,
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
inline int buildBVH(const std::vector<int>& triIndices,
                    const std::vector<MeshModel::Triangle>& triangles,
                    const std::vector<MeshModel::Vertex>& vertices,
                    std::vector<BVHNode>& nodes) {
    BVHNode node;
    // 计算当前节点中所有三角形的包围盒与质心
    Eigen::Vector3f bbox_min(std::numeric_limits<float>::max(),
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
    // 当三角形数较少时作为叶子节点
    if (triIndices.size() <= 4) {
        node.triangleIndices = triIndices;
        nodes.push_back(node);
        return static_cast<int>(nodes.size()) - 1;
    }
    // 选择分裂轴：选包围盒最长的轴
    Eigen::Vector3f extent = bbox_max - bbox_min;
    int axis = 0;
    if (extent[1] > extent[0]) axis = 1;
    if (extent[2] > extent[axis]) axis = 2;
    // 根据质心在该轴上的值进行分裂：求中位数
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
    // 防止分裂后一侧为空，则均分
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
    int leftChild = buildBVH(leftIndices, triangles, vertices, nodes);
    int rightChild = buildBVH(rightIndices, triangles, vertices, nodes);
    node.left = leftChild;
    node.right = rightChild;
    nodes.push_back(node);
    return static_cast<int>(nodes.size()) - 1;
}

// 计算射线与三角形交点（Möller–Trumbore 算法）
inline bool rayTriangleIntersection(const Eigen::Vector3f& rayOrigin,
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
    if (std::abs(a) < EPSILON) return false;
    float f = 1.0f / a;
    Eigen::Vector3f s = rayOrigin - v0;
    float u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f) return false;
    Eigen::Vector3f q = s.cross(edge1);
    float v = f * rayDir.dot(q);
    if (v < 0.0f || u + v > 1.0f) return false;
    t = f * edge2.dot(q);
    return (t > EPSILON);
}

// 遍历 BVH：返回射线沿方向的最近交点距离（存入 tHit）；若无交点则返回 false
inline bool traverseBVH(const std::vector<BVHNode>& nodes,
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
        if (node.left != -1)
            if (traverseBVH(nodes, triangles, vertices, node.left, rayOrigin, rayDir, tHit))
                hit = true;
        if (node.right != -1)
            if (traverseBVH(nodes, triangles, vertices, node.right, rayOrigin, rayDir, tHit))
                hit = true;
    }
    return hit;
}

} // end namespace ProjectionDetail

// -------------------------
// Projection 类声明
// -------------------------
class Projection {
public:

    template <typename T>
    static Eigen::Matrix<T,2,1> projectPoint(const MeshModel::Vertex &vertex,
                                              const Eigen::Matrix<T,3,3> &intrinsics,
                                              const Eigen::Matrix<T,3,3> &rotation,
                                              const Eigen::Matrix<T,3,1> &translation) {
        Eigen::Matrix<T,3,1> point;
        point << T(vertex.x), T(vertex.y), T(vertex.z);
        Eigen::Matrix<T,3,1> cameraPoint = rotation * point + translation;
        Eigen::Matrix<T,3,1> imagePoint = intrinsics * cameraPoint;
        return Eigen::Matrix<T,2,1>( imagePoint(0) / imagePoint(2),
                                      imagePoint(1) / imagePoint(2) );
    }

    template <typename T>
    static std::vector<Eigen::Matrix<T,2,1>> projectPoints(const std::vector<MeshModel::Vertex> &vertices,
                                                            const Eigen::Matrix<T,3,3> &intrinsics,
                                                            const Eigen::Matrix<T,3,3> &rotation,
                                                            const Eigen::Matrix<T,3,1> &translation) {
        std::vector<Eigen::Matrix<T,2,1>> projectedPoints;
        projectedPoints.reserve(vertices.size());
        for (const auto &vertex : vertices) {
            projectedPoints.push_back(projectPoint(vertex, intrinsics, rotation, translation));
        }
        return projectedPoints;
    }

    template <typename T>
    static std::vector<T> computeVertexDepths(const std::vector<MeshModel::Vertex> &vertices,
                                               const Eigen::Matrix<T,3,3> &intrinsics,
                                               const Eigen::Matrix<T,3,3> &rotation,
                                               const Eigen::Matrix<T,3,1> &translation) {
        std::vector<T> depths;
        depths.reserve(vertices.size());
        for (const auto &vertex : vertices) {
            Eigen::Matrix<T,3,1> point;
            point << T(vertex.x), T(vertex.y), T(vertex.z);
            Eigen::Matrix<T,3,1> cameraPoint = rotation * point + translation;
            depths.push_back(cameraPoint(2));
        }
        return depths;
    }

    // 模板版本的软可见性函数：返回每个顶点的连续可见性权重
    // 为了使可见性计算保持对参数的依赖，我们用一个软函数（例如高斯函数）将 BVH 结果平滑化。
    template <typename T>
    static std::vector<T> handleOcclusion(const std::vector<MeshModel::Vertex> &vertices,
                                           const std::vector<MeshModel::Triangle> &triangles,
                                           const Eigen::Matrix<T,3,3> &intrinsics,
                                           const Eigen::Matrix<T,3,3> &rotation,
                                           const Eigen::Matrix<T,3,1> &translation,
                                           int imageWidth,
                                           int imageHeight) {
        std::vector<T> visibilityWeights(vertices.size(), T(0));
        // 构建三角形索引列表
        std::vector<int> triIndices(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) {
            triIndices[i] = static_cast<int>(i);
        }
        // 构建 BVH（使用 float 版，因为 BVH 部分基于 if 判断不容易完全模板化）
        std::vector<ProjectionDetail::BVHNode> nodes;
        nodes.reserve(triangles.size());
        int rootIndex = ProjectionDetail::buildBVH(triIndices, triangles, vertices, nodes);

        // 计算相机光心：先转换为 float，再转为 T
        Eigen::Matrix<float, 3, 3> rotation_f;
        Eigen::Vector3f translation_f;

        if constexpr (std::is_same_v<T, ceres::Jet<double, 6>>) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    rotation_f(i, j) = static_cast<float>(rotation(i, j).a);
                }
                translation_f(i) = static_cast<float>(translation(i).a);
            }
        } 
        else {
            rotation_f = rotation.template cast<float>();
            translation_f = translation.template cast<float>();
        }
        Eigen::Vector3f cameraCenter_f = -(rotation_f.transpose() * translation_f);
        Eigen::Matrix<T,3,1> cameraCenter;
        cameraCenter << T(cameraCenter_f(0)), T(cameraCenter_f(1)), T(cameraCenter_f(2));

        // 平滑尺度（需根据实际情况调整）
        T sigma = T(1e-3);

        #pragma omp parallel for
        for (size_t i = 0; i < vertices.size(); ++i) {
            Eigen::Matrix<T,3,1> vertexT;
            vertexT << T(vertices[i].x), T(vertices[i].y), T(vertices[i].z);
            Eigen::Matrix<T,3,1> rayDir = (vertexT - cameraCenter).normalized();
            T targetDepth = (vertexT - cameraCenter).norm();
            float tHit_f = std::numeric_limits<float>::max();
            if constexpr (std::is_same_v<T, ceres::Jet<double, 6>>) {
                Eigen::Vector3f rayDir_f(rayDir(0).a, rayDir(1).a, rayDir(2).a);
                ProjectionDetail::traverseBVH(nodes, triangles, vertices, rootIndex, 
                                                    cameraCenter_f, rayDir_f, tHit_f);
            } 
            else {
                Eigen::Vector3f rayDir_f(rayDir(0), rayDir(1), rayDir(2));
                ProjectionDetail::traverseBVH(nodes, triangles, vertices, rootIndex, 
                                                    cameraCenter_f, rayDir_f, tHit_f);
            }
            T tHit = T(tHit_f);
            // 计算软权重：使用高斯函数 weight = exp(-((tHit - targetDepth)^2)/sigma)
            T weight = ceres::exp(- ((tHit - targetDepth) * (tHit - targetDepth)) / sigma);
            visibilityWeights[i] = weight;
        }
        return visibilityWeights;
    }
};

#endif // PROJECTION_H
