#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "MeshModel.h"
#include "BVH.h"
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "EDGraph.h"

class Optimizer {
public:
    Optimizer(double w_data,
              int maxStages,
              int maxIterations,
              double lambda_smooth = 1.0,
              double lambda_rot = 0.1);

    void setTemporalWeight(double lambda_temporal) { lambda_temporal_ = lambda_temporal; }

    // Frame 0 is treated as the fixed template (no data term, no variables).
    // Data term starts from frame 1, matching to per‑vertex template intensity sampled from frame 0.
    void optimize(
        const std::vector<MeshModel::Vertex>& mesh_vertices,
        const std::vector<MeshModel::Triangle>& mesh_triangles,
        const Eigen::Matrix3d& camera_intrinsics,
        const std::vector<cv::Mat>& observed_images,
        const std::vector<Eigen::Matrix4d>& camera_poses_gt,
        EDGraph& edGraph);

private:
    double w_data_;
    double lambda_smooth_;
    double lambda_rot_;
    double lambda_temporal_ = 1.0;   // temporal smoothness between adjacent frames (only if both frames have variables)
    int maxStages_;
    int maxIterations_;
};

#endif // OPTIMIZER_H