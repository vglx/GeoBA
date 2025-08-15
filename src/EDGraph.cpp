#include "EDGraph.h"

EDGraph::EDGraph(int K) : K_(K) {}

void EDGraph::initializeGraph(const std::vector<MeshModel::Vertex>& mesh_vertices,
                              int sampling_step) {
    if (sampling_step <= 0) sampling_step = 1;
    std::vector<DeformationNode> nodes;
    nodes.reserve(mesh_vertices.size() / sampling_step + 1);
    for (size_t i = 0; i < mesh_vertices.size(); i += static_cast<size_t>(sampling_step)) {
        const auto& v = mesh_vertices[i];
        DeformationNode node;
        node.position = Eigen::Vector3d(v.x, v.y, v.z);
        nodes.push_back(std::move(node));
    }
    setGraphNodes(nodes);
    bindVertices(mesh_vertices);
}

void EDGraph::setGraphNodes(const std::vector<DeformationNode>& nodes) {
    graph_ = nodes;
}

void EDGraph::bindVertices(const std::vector<MeshModel::Vertex>& vertices) {
    const size_t V = vertices.size();
    bindings_.assign(V, {});
    weights_.assign(V, {});
    if (graph_.empty()) return;
    const int K_eff = std::min<int>(K_, static_cast<int>(graph_.size()));
    std::vector<std::pair<double,int>> heap;
    heap.reserve(graph_.size());
    for (size_t vi = 0; vi < V; ++vi) {
        const Eigen::Vector3d p(vertices[vi].x, vertices[vi].y, vertices[vi].z);
        heap.clear();
        for (int j = 0; j < (int)graph_.size(); ++j) {
            double d2 = (p - graph_[j].position).squaredNorm();
            heap.emplace_back(d2, j);
        }
        std::nth_element(heap.begin(), heap.begin() + K_eff, heap.end());
        heap.resize(K_eff);
        const double eps = 1e-12;
        double wsum = 0.0;
        std::vector<double> w(K_eff, 0.0);
        for (int k = 0; k < K_eff; ++k) {
            double inv = 1.0 / std::max(heap[k].first, eps);
            w[k] = inv; wsum += inv;
        }
        for (int k = 0; k < K_eff; ++k) w[k] /= wsum;
        bindings_[vi].resize(K_eff);
        weights_[vi].resize(K_eff);
        for (int k = 0; k < K_eff; ++k) {
            bindings_[vi][k] = heap[k].second;
            weights_[vi][k]  = w[k];
        }
    }
}

Eigen::Vector3d EDGraph::deformVertex(const MeshModel::Vertex& vertex, int vidx,
                                      const EDState& state) const {
    const Eigen::Vector3d x(vertex.x, vertex.y, vertex.z);
    return deformPosition(x, vidx, state);
}

Eigen::Vector3d EDGraph::deformPosition(const Eigen::Vector3d& x, int vidx,
                                        const EDState& state) const {
    const auto& idxs = bindings_.at(vidx);
    const auto& wts  = weights_.at(vidx);
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (size_t k = 0; k < idxs.size(); ++k) {
        const int ni = idxs[k];
        const double w = wts[k];
        const Eigen::Vector3d& g = graph_[ni].position;
        const Eigen::Matrix3d& A = state.A[ni];
        const Eigen::Vector3d& b = state.b[ni];
        sum += w * (A * (x - g) + g + b);
    }
    return sum;
}

void EDGraph::writeStateToVector(const EDState& st, Eigen::VectorXd& x, int offset) const {
    const int N = numNodes();
    for (int i = 0; i < N; ++i) {
        Eigen::Map<const Eigen::Matrix<double,9,1>> Avec(st.A[i].data());
        x.segment<9>(offset + 12*i) = Avec;
        x.segment<3>(offset + 12*i + 9) = st.b[i];
    }
}

void EDGraph::readStateFromVector(const Eigen::VectorXd& x, int offset, EDState& st) const {
    const int N = numNodes();
    if ((int)st.size() != N) st.resize(N);
    for (int i = 0; i < N; ++i) {
        Eigen::Map<const Eigen::Matrix<double,9,1>> Avec(x.segment<9>(offset + 12*i).data());
        st.A[i] = Eigen::Map<const Eigen::Matrix3d>(Avec.data());
        st.b[i] = x.segment<3>(offset + 12*i + 9);
    }
}