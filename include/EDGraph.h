#ifndef EDGRAPH_H
#define EDGRAPH_H

#include <Eigen/Core>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "MeshModel.h"  // 顶点定义: struct MeshModel::Vertex { double x,y,z; ... }

// -------------------------------
// Deformation node (Affine)
// Each node j stores:
//   - position g_j (rest center)
//   - affine A_j (3x3)
//   - translation t_j (3x1)
//   - neighbors: indices used for smooth regularization (optional)
// -------------------------------
struct DeformationNode {
    Eigen::Vector3d position;        // g_j
    Eigen::Matrix3d A;               // 3x3 affine (init = I)
    Eigen::Vector3d t;               // translation (init = 0)
    std::vector<int> neighbors;      // neighbor node indices for smoothness (size ~ neighborK)
};

class EDGraph {
public:
    // 采样模式
    enum class SamplingMode { Stride, Voxel, FPS };

    struct BuildParams {
        SamplingMode mode = SamplingMode::Stride;
        int    stride      = 10;        // Stride: 每隔 stride 个顶点采 1 个
        double voxel_size  = 0.01;      // Voxel: 体素边长（模型单位）
        int    fps_target  = 1500;      // FPS: 目标控制点数
        int    K_bind      = 4;         // 顶点 -> 控制点 KNN 绑定数
        int    neighborK   = 6;         // 每控制点邻居数（用于平滑正则）
    };

    // K: 每个顶点绑定的节点数；neighborK: 每个节点的邻接数量（用于平滑正则）
    explicit EDGraph(int K = 4, int neighborK = 6);

    // 旧接口（向后兼容）：按索引步长采样
    void initializeGraph(const std::vector<MeshModel::Vertex>& mesh_vertices,
                         int sampling_step = 10,
                         bool build_neighbors = true);

    // 新接口：可插拔采样策略
    bool initializeGraph(const std::vector<MeshModel::Vertex>& mesh_vertices,
                         const BuildParams& params,
                         bool build_neighbors = true);

    // 外部直接设置节点（提供 g/A/t），随后可 bindVertices / buildNeighbors
    void setGraphNodes(const std::vector<DeformationNode>& nodes);

    // 绑定每个顶点到 K 个最近节点（预计算）
    void bindVertices(const std::vector<MeshModel::Vertex>& vertices);

    // 根据预绑定（vidx）对顶点进行仿射 ED 变形
    // p = sum_k w_k [ A_k (v-g_k) + g_k + t_k ]
    Eigen::Vector3d deformVertex(const MeshModel::Vertex& vertex, int vidx) const;

    // ------- 状态向量交互（与 Optimizer 对接） -------
    // 约定：每个节点写入 12 维 —— A(行主序 9 项) + t(3 项)
    void writeToStateVector(Eigen::VectorXd& x, int offset) const;
    void updateFromStateVector(const Eigen::VectorXd& x, int offset);

    // ------- 邻接（用于 smooth 正则） -------
    void setNeighborsForSmoothing(int neighborK);       // 重新按 neighborK 构建邻接
    const std::vector<std::pair<int,int>>& getEdges() const { return edges_; }

    // ------- 访问器 -------
    int numNodes() const { return static_cast<int>(graph_.size()); }
    const std::vector<DeformationNode>& getGraphNodes() const { return graph_; }
    const std::vector<std::vector<int>>& getBindings() const { return bindings_; }
    const std::vector<std::vector<double>>& getWeights() const { return weights_; }

private:
    // 构建节点邻接（内部使用；返回边集）
    void buildNeighbors_();

    // 采样策略实现
    void buildNodesStride(const std::vector<MeshModel::Vertex>&, int stride);

    struct VKey { int32_t x, y, z; };
    struct VKeyHash { size_t operator()(const VKey& k) const noexcept {
            // 3D 哈希
            uint64_t h = (uint64_t)(uint32_t)k.x * 73856093ull
                        ^ (uint64_t)(uint32_t)k.y * 19349663ull
                        ^ (uint64_t)(uint32_t)k.z * 83492791ull;
            return (size_t)h;
        }
    };
    struct VKeyEq { bool operator()(const VKey& a, const VKey& b) const noexcept { return a.x==b.x && a.y==b.y && a.z==b.z; } };

    void buildNodesVoxel(const std::vector<MeshModel::Vertex>&, double voxel_size);
    void buildNodesFPS(const std::vector<MeshModel::Vertex>&, int target);

    int K_;                 // 顶点绑定的节点数
    int neighborK_;         // 每节点的邻接数量

    std::vector<DeformationNode> graph_;                // 节点数组
    std::vector<std::vector<int>> bindings_;            // 顶点 -> 节点索引(长度 K_ 或更小)
    std::vector<std::vector<double>> weights_;          // 顶点 -> 权重(与 bindings_ 对齐)
    std::vector<std::pair<int,int>> edges_;             // 无向边 (i<j) 列表
};

#endif // EDGRAPH_H