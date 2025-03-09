#include "BVH.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <Eigen/Dense>

BVH::BVH(const std::vector<MeshModel::Triangle>& triangles,
         const std::vector<MeshModel::Vertex>& vertices)
         : triangles_(triangles), vertices_(vertices) {
       std::vector<int> triIndices(triangles.size());
       for (size_t i = 0; i < triangles.size(); ++i) {
              triIndices[i] = i;
       }
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

    if (triIndices.size() <= 4) {
        node.triangleIndices = triIndices;
        nodes.push_back(node);
        return nodes.size() - 1;
    }

    int axis = (bbox_max - bbox_min).maxCoeff();

    std::vector<int> leftIndices, rightIndices;
    double median = centroids[triIndices.size() / 2][axis];

    for (size_t i = 0; i < triIndices.size(); ++i) {
        if (centroids[i][axis] < median) {
            leftIndices.push_back(triIndices[i]);
        } else {
            rightIndices.push_back(triIndices[i]);
        }
    }

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

    int leftChild = buildBVH(leftIndices, triangles, vertices);
    int rightChild = buildBVH(rightIndices, triangles, vertices);

    node.left = leftChild;
    node.right = rightChild;
    nodes.push_back(node);
    return nodes.size() - 1;
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
    if (node.left == -1 && node.right == -1) {
        for (int idx : node.triangleIndices) {
            const MeshModel::Triangle& tri = triangles_[idx];  // **使用成员变量 triangles_**
            Eigen::Vector3d v0(vertices_[tri.v0].x, vertices_[tri.v0].y, vertices_[tri.v0].z);
            Eigen::Vector3d v1(vertices_[tri.v1].x, vertices_[tri.v1].y, vertices_[tri.v1].z);
            Eigen::Vector3d v2(vertices_[tri.v2].x, vertices_[tri.v2].y, vertices_[tri.v2].z);

            double t;
            if (rayTriangleIntersection(rayOrigin, rayDir, v0, v1, v2, t)) {
                if (t > 0 && t < tHit) {
                    tHit = t;
                    hit = true;
                }
            }
        }
    } else {
        hit |= traverseBVH(node.left, rayOrigin, rayDir, tHit);
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

// **静态方法：三角形相交检测**
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
    if (u < 0.0 || u > 1.0) return false;
    Eigen::Vector3d q = s.cross(edge1);
    double v = f * rayDir.dot(q);
    if (v < 0.0 || u + v > 1.0) return false;
    t = f * edge2.dot(q);
    return t > 1e-6;
}