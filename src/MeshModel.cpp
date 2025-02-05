#include "MeshModel.h"
#include "tiny_obj_loader.h" // 使用 TinyObjLoader 解析 OBJ 文件
#include <Eigen/Geometry>
#include <iostream>

MeshModel::MeshModel() {}

void MeshModel::setVertices(const std::vector<Vertex>& vertices) {
    vertices_ = vertices;
}

void MeshModel::setTriangles(const std::vector<Triangle>& triangles) {
    triangles_ = triangles;
}

const std::vector<MeshModel::Vertex>& MeshModel::getVertices() const {
    return vertices_;
}

const std::vector<MeshModel::Triangle>& MeshModel::getTriangles() const {
    return triangles_;
}

Eigen::Vector3f MeshModel::computeSurfaceNormal(const Triangle& triangle) const {
    const auto& v0 = vertices_[triangle.v0];
    const auto& v1 = vertices_[triangle.v1];
    const auto& v2 = vertices_[triangle.v2];

    Eigen::Vector3f p0(v0.x, v0.y, v0.z);
    Eigen::Vector3f p1(v1.x, v1.y, v1.z);
    Eigen::Vector3f p2(v2.x, v2.y, v2.z);

    Eigen::Vector3f edge1 = p1 - p0;
    Eigen::Vector3f edge2 = p2 - p0;
    Eigen::Vector3f normal = edge1.cross(edge2);

    return normal.normalized();
}

void MeshModel::computeVertexNormals() {
    for (const auto& triangle : triangles_) {
        if (triangle.v0 >= vertices_.size() || triangle.v1 >= vertices_.size() || triangle.v2 >= vertices_.size()) {
            std::cerr << "Error: Triangle index out of bounds! v0: " << triangle.v0
                    << " v1: " << triangle.v1 << " v2: " << triangle.v2 
                    << " but vertices_ size is " << vertices_.size() << std::endl;
            return;
        }
    }

    // 初始化每个顶点的法向量为零
    for (auto& vertex : vertices_) {
        vertex.nx = vertex.ny = vertex.nz = 0.0f;
    }

    // 遍历每个三角形，计算法向量并分配给顶点
    for (const auto& triangle : triangles_) {
        Eigen::Vector3f normal = computeSurfaceNormal(triangle);

        // 将法向量累加到每个顶点
        auto& v0 = vertices_[triangle.v0];
        auto& v1 = vertices_[triangle.v1];
        auto& v2 = vertices_[triangle.v2];

        v0.nx += normal.x();
        v0.ny += normal.y();
        v0.nz += normal.z();

        v1.nx += normal.x();
        v1.ny += normal.y();
        v1.nz += normal.z();

        v2.nx += normal.x();
        v2.ny += normal.y();
        v2.nz += normal.z();
    }

    // 对每个顶点的法向量进行归一化
    for (auto& vertex : vertices_) {
        Eigen::Vector3f normal(vertex.nx, vertex.ny, vertex.nz);
        normal.normalize();

        vertex.nx = normal.x();
        vertex.ny = normal.y();
        vertex.nz = normal.z();
    }
}
