#include "EDGraph.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <random>

namespace {
inline double sqr(double v) { return v * v; }
}

EDGraph::EDGraph(int K, int neighborK)
    : K_(K), neighborK_(neighborK) {}

// 旧接口：保持向后兼容（按步长采样）
void EDGraph::initializeGraph(const std::vector<MeshModel::Vertex>& mesh_vertices,
                              int sampling_step,
                              bool build_neighbors) {
    BuildParams p; p.mode = SamplingMode::Stride; p.stride = std::max(1, sampling_step);
    p.K_bind = K_; p.neighborK = neighborK_;
    initializeGraph(mesh_vertices, p, build_neighbors);
}

// 新接口：可插拔采样策略
bool EDGraph::initializeGraph(const std::vector<MeshModel::Vertex>& mesh_vertices,
                              const BuildParams& params,
                              bool build_neighbors) {
    if (mesh_vertices.empty()) return false;

    // 1) 采样生成节点
    graph_.clear(); edges_.clear();
    switch (params.mode) {
        case SamplingMode::Stride: buildNodesStride(mesh_vertices, std::max(1, params.stride)); break;
        case SamplingMode::Voxel:  buildNodesVoxel (mesh_vertices, std::max(1e-9, params.voxel_size)); break;
        case SamplingMode::FPS:    buildNodesFPS   (mesh_vertices, std::max(1, params.fps_target)); break;
    }

    // 2) 绑定顶点 -> K 近邻节点
    K_ = params.K_bind;            // 覆盖构造时的默认 K_
    neighborK_ = params.neighborK; // 同步邻接度
    bindVertices(mesh_vertices);

    // 3) 可选构建邻接边
    if (build_neighbors) buildNeighbors_();
    return true;
}

void EDGraph::setGraphNodes(const std::vector<DeformationNode>& nodes) {
    graph_ = nodes; edges_.clear();
}

void EDGraph::buildNodesStride(const std::vector<MeshModel::Vertex>& V, int stride) {
    graph_.clear(); graph_.reserve(V.size() / std::max(1, stride) + 1);
    for (size_t i = 0; i < V.size(); i += (size_t)std::max(1, stride)) {
        const auto& v = V[i];
        DeformationNode node{};
        node.position = Eigen::Vector3d(v.x, v.y, v.z);
        node.A.setIdentity(); node.t.setZero();
        graph_.push_back(node);
    }
}

void EDGraph::buildNodesVoxel(const std::vector<MeshModel::Vertex>& V, double s) {
    // 体素网格降采样：每个体素选一个代表点（靠近体素中心的顶点）
    graph_.clear();
    if (V.empty()) return;

    // 1) 计算包围盒（用 Eigen 的逐元素 min/max，避免 std::min 类型推导冲突）
    Eigen::Vector3d bbmin(  std::numeric_limits<double>::max(),
                            std::numeric_limits<double>::max(),
                            std::numeric_limits<double>::max());
    Eigen::Vector3d bbmax(- std::numeric_limits<double>::max(),
                          - std::numeric_limits<double>::max(),
                          - std::numeric_limits<double>::max());

    for (const auto& v : V) {
        const Eigen::Vector3d p(v.x, v.y, v.z);
        bbmin = bbmin.cwiseMin(p);
        bbmax = bbmax.cwiseMax(p);
    }

    // 2) 体素散列
    const Eigen::Vector3d invS(1.0 / s, 1.0 / s, 1.0 / s);
    std::unordered_map<VKey, int, VKeyHash, VKeyEq> rep;
    rep.reserve(V.size() / 8);

    for (size_t i = 0; i < V.size(); ++i) {
        const auto& v = V[i];
        const Eigen::Vector3d p(v.x, v.y, v.z);

        const Eigen::Vector3d rel = (p - bbmin).cwiseProduct(invS);
        const VKey k{
            static_cast<int>(std::floor(rel.x())),
            static_cast<int>(std::floor(rel.y())),
            static_cast<int>(std::floor(rel.z()))
        };

        auto it = rep.find(k);
        if (it == rep.end()) {
            rep.emplace(k, static_cast<int>(i));
        } else {
            // 选更靠近体素中心的顶点
            const int old = it->second;
            const Eigen::Vector3d pc = bbmin + Eigen::Vector3d((k.x + 0.5) * s,
                                                               (k.y + 0.5) * s,
                                                               (k.z + 0.5) * s);
            const double d_old = (Eigen::Vector3d(V[old].x, V[old].y, V[old].z) - pc).squaredNorm();
            const double d_new = (p - pc).squaredNorm();
            if (d_new < d_old) it->second = static_cast<int>(i);
        }
    }

    // 3) 输出节点
    graph_.reserve(rep.size());
    for (const auto& kv : rep) {
        const auto& v = V[kv.second];
        DeformationNode node;
        node.position = Eigen::Vector3d(v.x, v.y, v.z);
        node.A.setIdentity();
        node.t.setZero();
        graph_.push_back(node);
    }
}

void EDGraph::buildNodesFPS(const std::vector<MeshModel::Vertex>& V, int target) {
    // 最远点采样：O(N * target)
    graph_.clear();
    if (V.empty() || target <= 0) return;
    target = std::min<int>(target, (int)V.size());

    // 距离表初始化为 +inf
    std::vector<double> mindist(V.size(), std::numeric_limits<double>::infinity());
    std::vector<int> chosen; chosen.reserve(target);

    // 选一个种子（这里用包围盒中心最近的点，稳健）
    Eigen::Vector3d bbmin( std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    Eigen::Vector3d bbmax(-std::numeric_limits<double>::max(),-std::numeric_limits<double>::max(),-std::numeric_limits<double>::max());
    for (const auto& v : V) { bbmin.x() = std::min(bbmin.x(), v.x); bbmin.y() = std::min(bbmin.y(), v.y); bbmin.z() = std::min(bbmin.z(), v.z);
                              bbmax.x() = std::max(bbmax.x(), v.x); bbmax.y() = std::max(bbmax.y(), v.y); bbmax.z() = std::max(bbmax.z(), v.z); }
    Eigen::Vector3d center = 0.5*(bbmin+bbmax);
    int seed = 0; double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < (int)V.size(); ++i) {
        double d = (Eigen::Vector3d(V[i].x, V[i].y, V[i].z) - center).squaredNorm();
        if (d < best) { best = d; seed = i; }
    }

    auto add_point = [&](int idx){
        chosen.push_back(idx);
        Eigen::Vector3d p(V[idx].x, V[idx].y, V[idx].z);
        for (int i = 0; i < (int)V.size(); ++i) {
            Eigen::Vector3d q(V[i].x, V[i].y, V[i].z);
            double d = (q - p).squaredNorm();
            if (d < mindist[i]) mindist[i] = d;
        }
    };

    add_point(seed);
    while ((int)chosen.size() < target) {
        int next = 0; double far2 = -1.0;
        for (int i = 0; i < (int)V.size(); ++i) {
            if (mindist[i] > far2) { far2 = mindist[i]; next = i; }
        }
        add_point(next);
    }

    graph_.reserve(chosen.size());
    for (int idx : chosen) {
        const auto& v = V[idx];
        DeformationNode node{}; node.position = Eigen::Vector3d(v.x, v.y, v.z);
        node.A.setIdentity(); node.t.setZero();
        graph_.push_back(node);
    }
}

void EDGraph::buildNodesFPS(const std::vector<MeshModel::Vertex>& V, int target) {
    // 最远点采样：O(N * target)
    graph_.clear();
    if (V.empty() || target <= 0) return;
    target = std::min<int>(target, static_cast<int>(V.size()));

    // 距离表初始化为 +inf
    std::vector<double> mindist(V.size(), std::numeric_limits<double>::infinity());
    std::vector<int> chosen; 
    chosen.reserve(target);

    // 1) 用逐元素 min/max 求包围盒（避免 std::min/max 的类型推导问题）
    Eigen::Vector3d bbmin(  std::numeric_limits<double>::max(),
                            std::numeric_limits<double>::max(),
                            std::numeric_limits<double>::max());
    Eigen::Vector3d bbmax( -std::numeric_limits<double>::max(),
                           -std::numeric_limits<double>::max(),
                           -std::numeric_limits<double>::max());
    for (const auto& v : V) {
        const Eigen::Vector3d p(v.x, v.y, v.z);
        bbmin = bbmin.cwiseMin(p);
        bbmax = bbmax.cwiseMax(p);
    }

    // 2) 选种子：离包围盒中心最近的点
    const Eigen::Vector3d center = 0.5 * (bbmin + bbmax);
    int seed = 0; 
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(V.size()); ++i) {
        const Eigen::Vector3d p(V[i].x, V[i].y, V[i].z);
        const double d = (p - center).squaredNorm();
        if (d < best) { best = d; seed = i; }
    }

    // 3) FPS 主循环
    auto add_point = [&](int idx){
        chosen.push_back(idx);
        const Eigen::Vector3d p(V[idx].x, V[idx].y, V[idx].z);
        for (int i = 0; i < static_cast<int>(V.size()); ++i) {
            const Eigen::Vector3d q(V[i].x, V[i].y, V[i].z);
            const double d = (q - p).squaredNorm();
            if (d < mindist[i]) mindist[i] = d;
        }
    };

    add_point(seed);
    while (static_cast<int>(chosen.size()) < target) {
        int next = 0; 
        double far2 = -1.0;
        for (int i = 0; i < static_cast<int>(V.size()); ++i) {
            if (mindist[i] > far2) { far2 = mindist[i]; next = i; }
        }
        add_point(next);
    }

    // 4) 输出节点
    graph_.reserve(chosen.size());
    for (int idx : chosen) {
        const auto& v = V[idx];
        DeformationNode node;
        node.position = Eigen::Vector3d(v.x, v.y, v.z);
        node.A.setIdentity();
        node.t.setZero();
        graph_.push_back(node);
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
    for (int i = 0; i < G; ++i) {
        const auto& A = graph_[i].A;
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
        for (auto& n : graph_) n.neighbors.clear();
        return;
    }

    for (int i = 0; i < G; ++i) {
        std::vector<std::pair<int,double>> dists; dists.reserve(G-1);
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

        graph_[i].neighbors.clear(); graph_[i].neighbors.reserve(k);
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