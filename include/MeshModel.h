#ifndef MESH_MODEL_H
#define MESH_MODEL_H

#include <vector>
#include <Eigen/Core>

class MeshModel {
public:
    struct Vertex {
        float x, y, z;    // 顶点坐标
        float nx, ny, nz; // 顶点法向量
    };

    struct Triangle {
        int v0, v1, v2;   // 顶点索引
    };

    MeshModel();

    // 设置网格数据
    void setVertices(const std::vector<Vertex>& vertices);
    void setTriangles(const std::vector<Triangle>& triangles);

    // 获取网格数据
    const std::vector<Vertex>& getVertices() const;
    const std::vector<Triangle>& getTriangles() const;

    // 计算顶点法向量
    void computeVertexNormals();

private:
    std::vector<Vertex> vertices_;   // 顶点列表
    std::vector<Triangle> triangles_; // 三角形列表

    // 辅助方法：计算三角形法向量
    Eigen::Vector3f computeSurfaceNormal(const Triangle& triangle) const;
};

#endif // MESH_MODEL_H
