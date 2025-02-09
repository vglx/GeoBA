#include "MeshModel.h"
#include <iostream>

int main() {
    MeshModel model;

    // 加载 OBJ 文件
    if (!model.loadFromOBJ("path/to/your/model.obj")) {
        std::cerr << "Failed to load OBJ file.\n";
        return -1;
    }

    // 计算顶点法向量
    model.computeVertexNormals();

    // 打印每个顶点的法向量
    const auto& vertices = model.getVertices();
    for (size_t i = 0; i < vertices.size(); ++i) {
        const auto& vertex = vertices[i];
        std::cout << "Vertex " << i + 1 << " Normal: ("
                  << vertex.nx << ", " << vertex.ny << ", " << vertex.nz << ")\n";
    }

    return 0;
}
