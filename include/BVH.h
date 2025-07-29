#ifndef BVH_H
#define BVH_H

#include <vector>
#include <Eigen/Core>
#include "MeshModel.h"

class BVH {
public:
    // 构造 BVH（只需构建一次）
    BVH(const std::vector<MeshModel::Triangle>& triangles,
        const std::vector<MeshModel::Vertex>& vertices);

    // 重新根据变形后的顶点更新 AABB（Refit 操作）
    void refit(const std::vector<MeshModel::Vertex>& deformedVertices);

    // 进行 BVH 遮挡检测
    bool traverse(const Eigen::Vector3d& rayOrigin,
                  const Eigen::Vector3d& rayDir,
                  double& tHit) const;

private:
    struct BVHNode {
        Eigen::Vector3d bbox_min;
        Eigen::Vector3d bbox_max;
        int left;
        int right;
        std::vector<int> triangleIndices;
    };

    std::vector<BVHNode> nodes;
    int rootIndex;

    const std::vector<MeshModel::Triangle>& triangles_;
    const std::vector<MeshModel::Vertex>& vertices_;

    int buildBVH(const std::vector<int>& triIndices,
                 const std::vector<MeshModel::Triangle>& triangles,
                 const std::vector<MeshModel::Vertex>& vertices);

    bool traverseBVH(int nodeIndex,
                     const Eigen::Vector3d& rayOrigin,
                     const Eigen::Vector3d& rayDir,
                     double& tHit) const;

    static bool intersectAABB(const Eigen::Vector3d& rayOrigin,
                              const Eigen::Vector3d& rayDir,
                              const Eigen::Vector3d& bbox_min,
                              const Eigen::Vector3d& bbox_max,
                              double& tmin,
                              double& tmax);

    static bool rayTriangleIntersection(const Eigen::Vector3d& rayOrigin,
                                        const Eigen::Vector3d& rayDir,
                                        const Eigen::Vector3d& v0,
                                        const Eigen::Vector3d& v1,
                                        const Eigen::Vector3d& v2,
                                        double& t);

    // 更新叶子节点的 AABB
    void computeLeafAABB(BVHNode& node,
                         const std::vector<MeshModel::Vertex>& deformedVertices);
};

#endif // BVH_H