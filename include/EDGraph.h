#ifndef EDGRAPH_H
#define EDGRAPH_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>
#include "MeshModel.h"

struct EDState {
    std::vector<Eigen::Matrix3d> A;
    std::vector<Eigen::Vector3d> b;

    void resize(size_t N) {
        A.assign(N, Eigen::Matrix3d::Identity());
        b.assign(N, Eigen::Vector3d::Zero());
    }
    size_t size() const { return A.size(); }
};

struct DeformationNode {
    Eigen::Vector3d position;
    std::vector<int> neighbors;
};

class EDGraph {
public:
    explicit EDGraph(int K = 4);

    void initializeGraph(const std::vector<MeshModel::Vertex>& mesh_vertices,
                         int sampling_step = 10);
    void setGraphNodes(const std::vector<DeformationNode>& nodes);
    void bindVertices(const std::vector<MeshModel::Vertex>& vertices);

    Eigen::Vector3d deformVertex(const MeshModel::Vertex& vertex, int vidx,
                                 const EDState& state) const;
    Eigen::Vector3d deformPosition(const Eigen::Vector3d& x, int vidx,
                                   const EDState& state) const;

    void writeStateToVector(const EDState& st, Eigen::VectorXd& x, int offset) const;
    void readStateFromVector(const Eigen::VectorXd& x, int offset, EDState& st) const;

    int numNodes() const { return static_cast<int>(graph_.size()); }
    const std::vector<DeformationNode>& getGraphNodes() const { return graph_; }
    const std::vector<std::vector<int>>&  getBindings() const { return bindings_; }
    const std::vector<std::vector<double>>& getWeights() const { return weights_; }

    int K() const { return K_; }
    void setK(int K) { K_ = K; }

private:
    int K_;
    std::vector<DeformationNode> graph_;
    std::vector<std::vector<int>>    bindings_;
    std::vector<std::vector<double>> weights_;
};

#endif // EDGRAPH_H