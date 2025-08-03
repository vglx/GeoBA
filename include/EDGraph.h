#ifndef EDGRAPH_H
#define EDGRAPH_H

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>
#include "MeshModel.h"  // 顶点定义

// 单个变形图节点结构
struct DeformationNode {
    Eigen::Vector3d position;       // 节点中心（rest position）
    Sophus::SE3d transform;         // 当前 SE3 变换
    std::vector<int> neighbors;     // 可选：用于平滑约束
};

class EDGraph {
public:
    explicit EDGraph(int K = 4);

    void initializeGraph(const std::vector<MeshModel::Vertex>& mesh_vertices, int sampling_step = 10);

    // 设置节点集合（rest positions + initial transforms）
    void setGraphNodes(const std::vector<DeformationNode>& nodes);
    // 绑定每个顶点到 K 个最近节点（预计算）
    void bindVertices(const std::vector<MeshModel::Vertex>& vertices);
    // 返回变形后的顶点，vidx 为顶点索引
    Eigen::Vector3d deformVertex(const MeshModel::Vertex& vertex, int vidx) const;

    // 将 state vector 中的 SE3 变量写入 graph
    void updateFromStateVector(const Eigen::VectorXd& x, int offset);
    // 将 graph 当前 SE3 写入 state vector
    void writeToStateVector(Eigen::VectorXd& x, int offset) const;

    // 获取节点数
    int numNodes() const { return static_cast<int>(graph_.size()); }
    // 获取节点列表
    const std::vector<DeformationNode>& getGraphNodes() const { return graph_; }

    // 获取绑定关系和权重
    const std::vector<std::vector<int>>& getBindings() const { return bindings_; }
    const std::vector<std::vector<double>>& getWeights() const { return weights_; }

private:
    int K_;  // 每个顶点绑定的节点数量
    std::vector<DeformationNode> graph_;                  // 变形图节点
    std::vector<std::vector<int>> bindings_;              // 顶点 -> 节点索引
    std::vector<std::vector<double>> weights_;            // 顶点 -> 权重
};

#endif // EDGRAPH_H