#include "EDGraph.h"
#include <algorithm>
#include <limits>
#include <cmath>

namespace {
inline double sqr(double v) { return v * v; }
}

EDGraph::EDGraph(int K, int neighborK)
    : K_(K), neighborK_(neighborK) {}

void EDGraph::initializeGraph(const std::vector<MeshModel::Vertex>& mesh_vertices,
                              int sampling_step,
                              bool build_neighbors) {
    // 1) 采样生成节点
    std::vector<DeformationNode> nodes;
    nodes.reserve(mesh_vertices.size() / std::max(1, sampling_step) + 1);

    for (size_t i = 0; i < mesh_vertices.size(); i += std::max(1, sampling_step)) {
        const auto& v = mesh_vertices[i];
        DeformationNode node{};
        node.position = Eigen::Vector3d(v.x, v.y, v.z);
        node.A.setIdentity();
        node.t.setZero();
        nodes.push_back(node);
    }
    setGraphNodes(nodes);

    // 2) 绑定顶点 -> K 近邻节点
    bindVertices(mesh_vertices);

    // 3) 可选构建邻接边
    if (build_neighbors) buildNeighbors_();
}

void EDGraph::setGraphNodes(const std::vector<DeformationNode>& nodes) {
    graph_ = nodes;
    edges_.clear();
}

void EDGraph::bindVertices(const std::vector<MeshModel::Vertex>& vertices) {
    const size_t nV = vertices.size();
    const int G = numNodes();
    bindings_.assign(nV, {});
    weights_.assign(nV, {});
    if (G == 0 || nV == 0) return;

    for (size_t vid = 0; vid < nV; ++vid) {
        const Eigen::Vector3d v(vertices[vid].x, vertices[vid].y, vertices[vid].z);
        std::vector<std::pair<int, double>> dists;
        dists.reserve(G);
        for (int j = 0; j < G; ++j) {
            double dist = (v - graph_[j].position).norm();
            dists.emplace_back(j, dist);
        }
        std::nth_element(dists.begin(), dists.begin() + std::min(K_, (int)dists.size()) - 1, dists.end(),
                         [](const auto& a, const auto& b){ return a.second < b.second; });
        std::sort(dists.begin(), dists.begin() + std::min(K_, (int)dists.size()),
                  [](const auto& a, const auto& b){ return a.second < b.second; });

        const int kth = std::min(K_, (int)dists.size());
        bindings_[vid].resize(kth);
        weights_[vid].resize(kth);

        double sumW = 0.0;
        constexpr double eps = 1e-8;
        for (int k = 0; k < kth; ++k) {
            bindings_[vid][k] = dists[k].first;
            // 反距离权重，并做归一化
            double w = 1.0 / (dists[k].second + eps);
            weights_[vid][k] = w;
            sumW += w;
        }
        if (sumW <= eps) {
            // 退化情形：把第一个权重设为 1
            if (kth > 0) {
                std::fill(weights_[vid].begin(), weights_[vid].end(), 0.0);
                weights_[vid][0] = 1.0;
            }
        } else {
            for (double& w : weights_[vid]) w /= sumW;
        }
    }
}

Eigen::Vector3d EDGraph::deformVertex(const MeshModel::Vertex& vertex, int vidx) const {
    const Eigen::Vector3d v(vertex.x, vertex.y, vertex.z);
    Eigen::Vector3d out = Eigen::Vector3d::Zero();

    if ((size_t)vidx >= bindings_.size()) return v; // 安全保护

    const auto& node_ids = bindings_[vidx];
    const auto& node_ws  = weights_[vidx];

    for (size_t k = 0; k < node_ids.size(); ++k) {
        const int nid = node_ids[k];
        const double w = node_ws[k];
        const auto& node = graph_[nid];

        const Eigen::Vector3d& g = node.position;
        const Eigen::Vector3d p = node.A * (v - g) + g + node.t; // 仿射 ED
        out += w * p;
    }
    return out;
}

void EDGraph::writeToStateVector(Eigen::VectorXd& x, int offset) const {
    const int G = numNodes();
    if (G == 0) return;
    // 确保 x 尺寸足够由调用方负责；此处仅写入
    for (int i = 0; i < G; ++i) {
        const auto& A = graph_[i].A; // 行主序: A(0,0)..A(2,2)
        x(offset + 12*i + 0) = A(0,0);
        x(offset + 12*i + 1) = A(0,1);
        x(offset + 12*i + 2) = A(0,2);
        x(offset + 12*i + 3) = A(1,0);
        x(offset + 12*i + 4) = A(1,1);
        x(offset + 12*i + 5) = A(1,2);
        x(offset + 12*i + 6) = A(2,0);
        x(offset + 12*i + 7) = A(2,1);
        x(offset + 12*i + 8) = A(2,2);
        x(offset + 12*i + 9) = graph_[i].t(0);
        x(offset + 12*i +10) = graph_[i].t(1);
        x(offset + 12*i +11) = graph_[i].t(2);
    }
}

void EDGraph::updateFromStateVector(const Eigen::VectorXd& x, int offset) {
    const int G = numNodes();
    if (G == 0) return;
    for (int i = 0; i < G; ++i) {
        Eigen::Matrix3d A;
        A(0,0) = x(offset + 12*i + 0);
        A(0,1) = x(offset + 12*i + 1);
        A(0,2) = x(offset + 12*i + 2);
        A(1,0) = x(offset + 12*i + 3);
        A(1,1) = x(offset + 12*i + 4);
        A(1,2) = x(offset + 12*i + 5);
        A(2,0) = x(offset + 12*i + 6);
        A(2,1) = x(offset + 12*i + 7);
        A(2,2) = x(offset + 12*i + 8);
        graph_[i].A = A;
        graph_[i].t(0) = x(offset + 12*i + 9);
        graph_[i].t(1) = x(offset + 12*i +10);
        graph_[i].t(2) = x(offset + 12*i +11);
    }
}

void EDGraph::setNeighborsForSmoothing(int neighborK) {
    neighborK_ = std::max(0, neighborK);
    buildNeighbors_();
}

void EDGraph::buildNeighbors_() {
    // 基于节点坐标的暴力 KNN（无第三方索引依赖）
    const int G = numNodes();
    edges_.clear();
    if (G <= 1 || neighborK_ <= 0) {
        // 清空 neighbors 字段
        for (auto& n : graph_) n.neighbors.clear();
        return;
    }

    for (int i = 0; i < G; ++i) {
        std::vector<std::pair<int,double>> dists;
        dists.reserve(G-1);
        const Eigen::Vector3d gi = graph_[i].position;
        for (int j = 0; j < G; ++j) if (j != i) {
            double d = (gi - graph_[j].position).squaredNorm();
            dists.emplace_back(j, d);
        }
        const int k = std::min(neighborK_, (int)dists.size());
        std::nth_element(dists.begin(), dists.begin()+k, dists.end(),
                         [](const auto& a, const auto& b){ return a.second < b.second; });
        std::sort(dists.begin(), dists.begin()+k,
                  [](const auto& a, const auto& b){ return a.second < b.second; });

        graph_[i].neighbors.clear();
        graph_[i].neighbors.reserve(k);
        for (int t = 0; t < k; ++t) {
            const int j = dists[t].first;
            graph_[i].neighbors.push_back(j);
            int a = std::min(i, j), b = std::max(i, j);
            edges_.emplace_back(a, b);
        }
    }

    // 去重 (i<j) 边
    std::sort(edges_.begin(), edges_.end());
    edges_.erase(std::unique(edges_.begin(), edges_.end()), edges_.end());
}