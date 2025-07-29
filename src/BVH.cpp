#include "BVH.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <Eigen/Dense>

BVH::BVH(const std::vector<MeshModel::Triangle>& triangles,
         const std::vector<MeshModel::Vertex>& vertices)
    : triangles_(triangles), vertices_(vertices) {
    std::vector<int> triIndices(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) triIndices[i] = i;
    rootIndex = buildBVH(triIndices, triangles, vertices);
}

int BVH::buildBVH(const std::vector<int>& triIndices,
                  const std::vector<MeshModel::Triangle>& triangles,
                  const std::vector<MeshModel::Vertex>& vertices) {
    BVHNode node;
    Eigen::Vector3d bbox_min(std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::max());
    Eigen::Vector3d bbox_max(-std::numeric_limits<double>::max(),
                             -std::numeric_limits<double>::max(),
                             -std::numeric_limits<double>::max());

    std::vector<Eigen::Vector3d> centroids;
    centroids.reserve(triIndices.size());
    for (int idx : triIndices) {
        const auto& tri = triangles[idx];
        Eigen::Vector3d v0(vertices[tri.v0].x, vertices[tri.v0].y, vertices[tri.v0].z);
        Eigen::Vector3d v1(vertices[tri.v1].x, vertices[tri.v1].y, vertices[tri.v1].z);
        Eigen::Vector3d v2(vertices[tri.v2].x, vertices[tri.v2].y, vertices[tri.v2].z);
        bbox_min = bbox_min.cwiseMin(v0.cwiseMin(v1).cwiseMin(v2));
        bbox_max = bbox_max.cwiseMax(v0.cwiseMax(v1).cwiseMax(v2));
        centroids.push_back((v0 + v1 + v2) * (1.0/3.0));
    }
    node.bbox_min = bbox_min;
    node.bbox_max = bbox_max;
    node.left = node.right = -1;

    if (triIndices.size() <= 4) {
        node.triangleIndices = triIndices;
        nodes.push_back(node);
        return int(nodes.size() - 1);
    }

    Eigen::Vector3d extents = bbox_max - bbox_min;
    int axis; extents.maxCoeff(&axis);
    std::vector<std::pair<int,double>> order;
    for (size_t i = 0; i < triIndices.size(); ++i)
        order.emplace_back(triIndices[i], centroids[i][axis]);
    std::sort(order.begin(), order.end(), [](auto& a, auto& b){ return a.second < b.second; });

    size_t mid = order.size()/2;
    std::vector<int> leftIdx, rightIdx;
    for (size_t i = 0; i < order.size(); ++i) {
        if (i < mid) leftIdx.push_back(order[i].first);
        else         rightIdx.push_back(order[i].first);
    }
    if (leftIdx.empty() || rightIdx.empty()) {
        node.triangleIndices = triIndices;
        nodes.push_back(node);
        return int(nodes.size() - 1);
    }
    int left  = buildBVH(leftIdx,  triangles, vertices);
    int right = buildBVH(rightIdx, triangles, vertices);
    node.left = left;
    node.right = right;
    nodes.push_back(node);
    return int(nodes.size() - 1);
}

bool BVH::traverse(const Eigen::Vector3d& rayOrigin,
                   const Eigen::Vector3d& rayDir,
                   double& tHit) const {
    return traverseBVH(rootIndex, rayOrigin, rayDir, tHit);
}

bool BVH::traverseBVH(int nodeIndex,
                      const Eigen::Vector3d& rayOrigin,
                      const Eigen::Vector3d& rayDir,
                      double& tHit) const {
    const BVHNode& node = nodes[nodeIndex];
    double tmin, tmax;
    if (!intersectAABB(rayOrigin, rayDir, node.bbox_min, node.bbox_max, tmin, tmax))
        return false;
    bool hit = false;
    if (node.left < 0 && node.right < 0) {
        for (int idx : node.triangleIndices) {
            const auto& tri = triangles_[idx];
            Eigen::Vector3d v0(vertices_[tri.v0].x, vertices_[tri.v0].y, vertices_[tri.v0].z);
            Eigen::Vector3d v1(vertices_[tri.v1].x, vertices_[tri.v1].y, vertices_[tri.v1].z);
            Eigen::Vector3d v2(vertices_[tri.v2].x, vertices_[tri.v2].y, vertices_[tri.v2].z);
            double t;
            if (rayTriangleIntersection(rayOrigin, rayDir, v0, v1, v2, t) && t>0 && t<tHit) {
                tHit = t;
                hit = true;
            }
        }
    } else {
        hit |= traverseBVH(node.left,  rayOrigin, rayDir, tHit);
        hit |= traverseBVH(node.right, rayOrigin, rayDir, tHit);
    }
    return hit;
}

bool BVH::intersectAABB(const Eigen::Vector3d& rayOrigin,
                        const Eigen::Vector3d& rayDir,
                        const Eigen::Vector3d& bbox_min,
                        const Eigen::Vector3d& bbox_max,
                        double& tmin,
                        double& tmax) {
    tmin = 0;
    tmax = std::numeric_limits<double>::max();
    for (int i = 0; i < 3; ++i) {
        double invD = 1.0 / rayDir[i];
        double t0 = (bbox_min[i] - rayOrigin[i]) * invD;
        double t1 = (bbox_max[i] - rayOrigin[i]) * invD;
        if (invD < 0) std::swap(t0, t1);
        tmin = std::max(tmin, t0);
        tmax = std::min(tmax, t1);
        if (tmax <= tmin) return false;
    }
    return true;
}

bool BVH::rayTriangleIntersection(const Eigen::Vector3d& rayOrigin,
                                  const Eigen::Vector3d& rayDir,
                                  const Eigen::Vector3d& v0,
                                  const Eigen::Vector3d& v1,
                                  const Eigen::Vector3d& v2,
                                  double& t) {
    Eigen::Vector3d edge1 = v1 - v0;
    Eigen::Vector3d edge2 = v2 - v0;
    Eigen::Vector3d h = rayDir.cross(edge2);
    double a = edge1.dot(h);
    if (std::abs(a) < 1e-6) return false;
    double f = 1.0 / a;
    Eigen::Vector3d s = rayOrigin - v0;
    double u = f * s.dot(h);
    if (u < 0 || u > 1) return false;
    Eigen::Vector3d q = s.cross(edge1);
    double v = f * rayDir.dot(q);
    if (v < 0 || u + v > 1) return false;
    t = f * edge2.dot(q);
    return t > 1e-6;
}

void BVH::computeLeafAABB(BVHNode& node,
                          const std::vector<MeshModel::Vertex>& deformedVertices) {
    Eigen::Vector3d minPt(std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max());
    Eigen::Vector3d maxPt(-std::numeric_limits<double>::max(),
                          -std::numeric_limits<double>::max(),
                          -std::numeric_limits<double>::max());
    for (int idx : node.triangleIndices) {
        const auto& tri = triangles_[idx];
        Eigen::Vector3d v0(deformedVertices[tri.v0].x,
                           deformedVertices[tri.v0].y,
                           deformedVertices[tri.v0].z);
        Eigen::Vector3d v1(deformedVertices[tri.v1].x,
                           deformedVertices[tri.v1].y,
                           deformedVertices[tri.v1].z);
        Eigen::Vector3d v2(deformedVertices[tri.v2].x,
                           deformedVertices[tri.v2].y,
                           deformedVertices[tri.v2].z);
        minPt = minPt.cwiseMin(v0.cwiseMin(v1).cwiseMin(v2));
        maxPt = maxPt.cwiseMax(v0.cwiseMax(v1).cwiseMax(v2));
    }
    node.bbox_min = minPt;
    node.bbox_max = maxPt;
}

void BVH::refit(const std::vector<MeshModel::Vertex>& deformedVertices) {
    // 1. 更新叶子节点
    for (auto& node : nodes) {
        if (node.left < 0 && node.right < 0) {
            computeLeafAABB(node, deformedVertices);
        }
    }
    // 2. 自底向上更新内部节点
    for (int i = int(nodes.size()) - 1; i >= 0; --i) {
        auto& node = nodes[i];
        if (node.left >= 0 && node.right >= 0) {
            const auto& L = nodes[node.left];
            const auto& R = nodes[node.right];
            node.bbox_min = L.bbox_min.cwiseMin(R.bbox_min);
            node.bbox_max = L.bbox_max.cwiseMax(R.bbox_max);
        }
    }
}