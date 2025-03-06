#include "Projection.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <Eigen/Geometry>
#include <omp.h>

// -------------------------
// BVH 数据结构与辅助函数
// -------------------------
struct BVHNode {
    Eigen::Vector3d bbox_min;
    Eigen::Vector3d bbox_max;
    int left;   // 如果为叶子节点，则 left == -1
    int right;  // 如果为叶子节点，则 right == -1
    std::vector<int> triangleIndices; // 叶子节点存储对应的三角形索引
};

namespace {

// 计算射线与 AABB 的相交（slab 方法）
bool intersectAABB(const Eigen::Vector3d& rayOrigin, const Eigen::Vector3d& rayDir,
                   const Eigen::Vector3d& bbox_min, const Eigen::Vector3d& bbox_max,
                   double& tmin, double& tmax) {
    tmin = 0.0;
    tmax = std::numeric_limits<double>::max();
    for (int i = 0; i < 3; ++i) {
        double invD = 1.0 / rayDir[i];
        double t0 = (bbox_min[i] - rayOrigin[i]) * invD;
        double t1 = (bbox_max[i] - rayOrigin[i]) * invD;
        if (invD < 0.0) std::swap(t0, t1);
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
    
    // **使用 double 以匹配 Eigen::Vector3d**
    Eigen::Vector3d bbox_min( std::numeric_limits<double>::max(),
                              std::numeric_limits<double>::max(),
                              std::numeric_limits<double>::max());
    Eigen::Vector3d bbox_max(-std::numeric_limits<double>::max(),
                             -std::numeric_limits<double>::max(),
                             -std::numeric_limits<double>::max());

    std::vector<Eigen::Vector3d> centroids;
    centroids.reserve(triIndices.size());

    // **计算三角形包围盒**
    for (int idx : triIndices) {
        const MeshModel::Triangle& tri = triangles[idx];

        Eigen::Vector3d v0(vertices[tri.v0].x, vertices[tri.v0].y, vertices[tri.v0].z);
        Eigen::Vector3d v1(vertices[tri.v1].x, vertices[tri.v1].y, vertices[tri.v1].z);
        Eigen::Vector3d v2(vertices[tri.v2].x, vertices[tri.v2].y, vertices[tri.v2].z);

        Eigen::Vector3d tri_min = v0.cwiseMin(v1).cwiseMin(v2);
        Eigen::Vector3d tri_max = v0.cwiseMax(v1).cwiseMax(v2);

        bbox_min = bbox_min.cwiseMin(tri_min);
        bbox_max = bbox_max.cwiseMax(tri_max);

        Eigen::Vector3d centroid = (v0 + v1 + v2) / 3.0;
        centroids.push_back(centroid);
    }

    node.bbox_min = bbox_min;
    node.bbox_max = bbox_max;
    node.left = -1;
    node.right = -1;

    // **如果三角形数较少，则设为叶子节点**
    if (triIndices.size() <= 4) {
        node.triangleIndices = triIndices;
        nodes.push_back(node);
        return nodes.size() - 1;
    }

    // **选择分裂轴**
    Eigen::Vector3d extent = bbox_max - bbox_min;
    int axis = 0;
    if (extent[1] > extent[0]) axis = 1;
    if (extent[2] > extent[axis]) axis = 2;

    // **使用中位数划分**
    std::vector<int> leftIndices, rightIndices;
    std::vector<double> centroidVals;
    centroidVals.reserve(centroids.size());

    for (const auto& c : centroids) {
        centroidVals.push_back(c[axis]);
    }
    std::nth_element(centroidVals.begin(), centroidVals.begin() + centroidVals.size()/2, centroidVals.end());
    double median = centroidVals[centroidVals.size()/2];

    for (size_t i = 0; i < triIndices.size(); ++i) {
        if (centroids[i][axis] < median) {
            leftIndices.push_back(triIndices[i]);
        } else {
            rightIndices.push_back(triIndices[i]);
        }
    }

    // **防止某一侧为空**
    if (leftIndices.empty() || rightIndices.empty()) {
        leftIndices.clear();
        rightIndices.clear();
        for (size_t i = 0; i < triIndices.size(); ++i) {
            if (i < triIndices.size() / 2)
                leftIndices.push_back(triIndices[i]);
            else
                rightIndices.push_back(triIndices[i]);
        }
    }

    // **递归构建左右子树**
    int leftChild  = buildBVH(leftIndices, triangles, vertices, nodes);
    int rightChild = buildBVH(rightIndices, triangles, vertices, nodes);

    node.left = leftChild;
    node.right = rightChild;
    nodes.push_back(node);
    return nodes.size() - 1;
}

// 计算射线与三角形的交点（Möller–Trumbore 算法）
bool rayTriangleIntersection(
    const Eigen::Vector3d& rayOrigin,
    const Eigen::Vector3d& rayDir,
    const Eigen::Vector3d& v0,
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2,
    double& t) {

    const double EPSILON = 1e-6;
    Eigen::Vector3d edge1 = v1 - v0;
    Eigen::Vector3d edge2 = v2 - v0;
    Eigen::Vector3d h = rayDir.cross(edge2);
    double a = edge1.dot(h);

    if (std::abs(a) < EPSILON) return false; // 射线与三角形平行

    double f = 1.0 / a;
    Eigen::Vector3d s = rayOrigin - v0;
    double u = f * s.dot(h);
    if (u < 0.0 || u > 1.0) return false;

    Eigen::Vector3d q = s.cross(edge1);
    double v = f * rayDir.dot(q);
    if (v < 0.0 || u + v > 1.0) return false;

    t = f * edge2.dot(q);
    return (t > EPSILON);
}

// 遍历 BVH，返回射线沿方向的最近交点距离（存入 tHit）；若无交点则返回 false
bool traverseBVH(const std::vector<BVHNode>& nodes,
                 const std::vector<MeshModel::Triangle>& triangles,
                 const std::vector<MeshModel::Vertex>& vertices,
                 int nodeIndex,
                 const Eigen::Vector3d& rayOrigin,
                 const Eigen::Vector3d& rayDir,
                 double& tHit) {
    const BVHNode& node = nodes[nodeIndex];
    double tmin, tmax;
    if (!intersectAABB(rayOrigin, rayDir, node.bbox_min.cast<double>(), node.bbox_max.cast<double>(), tmin, tmax))
        return false;

    bool hit = false;
    // 如果是叶子节点，则遍历所有三角形
    if (node.left == -1 && node.right == -1) {
        for (int idx : node.triangleIndices) {
            const MeshModel::Triangle& tri = triangles[idx];
            Eigen::Vector3d v0(vertices[tri.v0].x, vertices[tri.v0].y, vertices[tri.v0].z);
            Eigen::Vector3d v1(vertices[tri.v1].x, vertices[tri.v1].y, vertices[tri.v1].z);
            Eigen::Vector3d v2(vertices[tri.v2].x, vertices[tri.v2].y, vertices[tri.v2].z);
            double t;
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

double Projection::computeVertexDepth(
    const MeshModel::Vertex& vertex, 
    const Eigen::Matrix3d& intrinsics,
    const Eigen::Matrix3d& rotation, 
    const Eigen::Vector3d& translation) {
    
    Eigen::Vector3d point = Eigen::Vector3d(vertex.x, vertex.y, vertex.z);
    Eigen::Vector3d cameraPoint = rotation.transpose() * (point - translation);
    return cameraPoint.z();  // 返回 Z 深度
}

// 使用 BVH 加速的遮挡判断函数
std::vector<bool> Projection::handleOcclusion(
    const std::vector<MeshModel::Vertex>& vertices,
    const std::vector<MeshModel::Triangle>& triangles,
    const Eigen::Matrix3d& intrinsics,
    const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& translation,
    int imageWidth,
    int imageHeight) {

    std::vector<bool> visibleMask(vertices.size(), false);

    // 构建三角形索引列表
    std::vector<int> triIndices(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) {
        triIndices[i] = i;
    }
    
    // 构建 BVH
    std::vector<BVHNode> nodes;
    nodes.reserve(triangles.size());
    int rootIndex = buildBVH(triIndices, triangles, vertices, nodes);

    // 计算相机光心
    Eigen::Vector3d cameraCenter = translation;

    // 遍历所有顶点
    #pragma omp parallel for
    for (size_t i = 0; i < vertices.size(); ++i) {
        Eigen::Vector3d vertex(vertices[i].x, vertices[i].y, vertices[i].z);
        // 先将顶点转换到相机坐标系
        Eigen::Vector3d camPoint = rotation.transpose() * (vertex - translation);
        // 如果 Z <= 0，则该点在相机后面，不可见
        if (camPoint.z() <= 0) {
            visibleMask[i] = false;
            continue;
        }
        Eigen::Vector3d rayDir = (vertex - cameraCenter).normalized();
        double tHit = std::numeric_limits<double>::max();
        bool hit = traverseBVH(nodes, triangles, vertices, rootIndex, cameraCenter, rayDir, tHit);
        double targetDepth = (vertex - cameraCenter).norm();
        
        double threshold = 0.005 * tHit;  // 允许误差 0.5% 的相对深度误差
        if (hit && std::abs(tHit - targetDepth) < threshold) {
            visibleMask[i] = true;
        } else {
            visibleMask[i] = false;
        }
    }

    return visibleMask;
}