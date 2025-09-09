#pragma once

#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>

class MeshModel;   // forward decl
class EDGraph;     // forward decl

class Optimizer {
public:
    // Constructor
    // w_data          : data term weight (projective ICP)
    // maxStages       : kept for compatibility (unused in this variant)
    // maxIterations   : Gauss-Newton outer iterations
    // lambda_smooth   : ED graph smoothness (edge-wise A,t differences)
    // lambda_rot      : rotation orthogonality regularization
    // lambda_temporal : per-node temporal consistency between consecutive frames
    Optimizer(double w_data,
              int    maxStages,
              int    maxIterations,
              double lambda_smooth,
              double lambda_rot,
              double lambda_temporal = 0.0);

    // Main entry: FOV-only + ProjectiveICP residuals.
    // - mesh: will be updated each iteration (deformed vertices written back, normals recomputed)
    // - observed_images: depth frames (CV_32F, in mm; invalid as NaN)
    // - camera_intrinsics: 3x3 K
    // - camera_poses_gt: per-frame world-to-camera 4x4 (R|t) with our convention pc = R^T (pw - t)
    // - edGraph: ED graph whose per-frame node states are optimized (frame 0 fixed)
    void optimize(
        MeshModel& mesh,
        const std::vector<cv::Mat>& observed_images,
        const Eigen::Matrix3d& camera_intrinsics,
        const std::vector<Eigen::Matrix4d>& camera_poses_gt,
        EDGraph& edGraph);

    // Optional setters/getters
    void setDataWeight(double w)          { w_data_ = w; }
    void setSmoothWeight(double w)        { lambda_smooth_ = w; }
    void setRotWeight(double w)           { lambda_rot_ = w; }
    void setTemporalWeight(double w)      { lambda_temporal_ = w; }

    double dataWeight()        const { return w_data_; }
    double smoothWeight()      const { return lambda_smooth_; }
    double rotWeight()         const { return lambda_rot_; }
    double temporalWeight()    const { return lambda_temporal_; }

private:
    // weights / hyper-parameters
    double w_data_          = 1.0;
    double lambda_smooth_   = 0.0;
    double lambda_rot_      = 0.0;
    double lambda_temporal_ = 0.0;

    int maxStages_      = 1;   // kept for API compatibility
    int maxIterations_  = 10;  // outer GN iterations
};