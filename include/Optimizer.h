#pragma once

#include <vector>
#include <functional>
#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "MeshModel.h"
#include "EDGraph.h"

class Optimizer {
public:
    // Constructor
    // w_photo   : weight for multi-frame monocular photometric residuals (per-frame vs template intensity)
    // w_stereo  : weight for Left–Right photometric residuals (per-frame binocular consistency)
    // maxStages : kept for compatibility (unused here)
    // maxIterations : outer Gauss–Newton iterations
    // lambda_smooth  : ED-graph smoothness (A,t differences on edges)
    // lambda_rot     : rotation orthogonality (A^T A ≈ I)
    // lambda_temporal: per-node temporal consistency across frames
    Optimizer(double w_photo,
              double w_stereo,
              int    maxStages,
              int    maxIterations,
              double lambda_smooth,
              double lambda_rot,
              double lambda_temporal = 0.0);

    using SaveCallback = std::function<void(int /*frameIdx*/, const EDGraph&)>;

    // Stereo Photometric optimizer (NO depth / NO ICP / NO scale variables)
    // - mesh_vertices / mesh_triangles: static template mesh (model space)
    // - K_left, K_right: 3x3 intrinsics for L/R
    // - rgb_left, rgb_right: per-frame L/R RGB images (CV_8UC3 or CV_32FC1 gray)
    // - poses_left_w2c, poses_right_w2c: per-frame world->camera for L/R
    // - edGraph: ED graph (frame 0 kept fixed)
    void optimize(
        const std::vector<struct MeshModel::Vertex>& mesh_vertices,
        const std::vector<struct MeshModel::Triangle>& mesh_triangles,
        const Eigen::Matrix3d& K_left,
        const Eigen::Matrix3d& K_right,
        const std::vector<cv::Mat>& rgb_left,
        const std::vector<cv::Mat>& rgb_right,
        const std::vector<Eigen::Matrix4d>& poses_left_w2c,
        const std::vector<Eigen::Matrix4d>& poses_right_w2c,
        EDGraph& edGraph,
        SaveCallback on_save = nullptr);

    // Optional setters/getters
    void setPhotoWeight(double w)       { w_photo_ = w; }
    void setStereoWeight(double w)      { w_stereo_ = w; }
    void setSmoothWeight(double w)      { lambda_smooth_ = w; }
    void setRotWeight(double w)         { lambda_rot_ = w; }
    void setTemporalWeight(double w)    { lambda_temporal_ = w; }

    double photoWeight()     const { return w_photo_; }
    double stereoWeight()    const { return w_stereo_; }
    double smoothWeight()    const { return lambda_smooth_; }
    double rotWeight()       const { return lambda_rot_; }
    double temporalWeight()  const { return lambda_temporal_; }

private:
    // weights / hyper-parameters
    double w_photo_         = 1.0;
    double w_stereo_        = 1.0;
    double lambda_smooth_   = 0.0;
    double lambda_rot_      = 0.0;
    double lambda_temporal_ = 0.0;

    int maxStages_     = 1;   // kept for API compatibility
    int maxIterations_ = 10;  // outer GN iterations
};