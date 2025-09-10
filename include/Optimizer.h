#pragma once

#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>

class MeshModel;   // fwd
class EDGraph;     // fwd

class Optimizer {
public:
    // Constructor
    // w_photo        : weight for photometric residuals
    // w_icp          : weight for projective-ICP residuals (point-to-plane from depth)
    // maxStages      : kept for compatibility (unused here)
    // maxIterations  : outer Gauss-Newton iterations
    // lambda_smooth  : ED-graph smoothness (A,t differences on edges)
    // lambda_rot     : rotation orthogonality
    // lambda_temporal: per-node temporal consistency across frames
    Optimizer(double w_photo,
              double w_icp,
              int    maxStages,
              int    maxIterations,
              double lambda_smooth,
              double lambda_rot,
              double lambda_temporal = 0.0);

    // Combined optimizer: build one system that includes BOTH photometric and ICP terms.
    // - mesh_vertices / mesh_triangles: static template mesh in model space
    // - K: 3x3 intrinsics
    // - observed_rgb: per-frame BGR/RGB images (CV_8UC3 or CV_32FC1 gray); will be converted to gray [0,1]
    // - observed_depth: per-frame depth (CV_32F, in mm; invalid as NaN)
    // - camera_poses_gt: per-frame world->camera, used as pc = R^T (pw - t)
    // - edGraph: provides bindings/edges and deformVertex/Jacobians; frame 0 kept fixed
    void optimize(
        const std::vector<struct MeshModel::Vertex>& mesh_vertices,
        const std::vector<struct MeshModel::Triangle>& mesh_triangles,
        const Eigen::Matrix3d& K,
        const std::vector<cv::Mat>& observed_rgb,
        const std::vector<cv::Mat>& observed_depth,
        const std::vector<Eigen::Matrix4d>& camera_poses_gt,
        EDGraph& edGraph);

    // Optional setters/getters
    void setPhotoWeight(double w)       { w_photo_ = w; }
    void setICPWeight(double w)         { w_icp_ = w; }
    void setSmoothWeight(double w)      { lambda_smooth_ = w; }
    void setRotWeight(double w)         { lambda_rot_ = w; }
    void setTemporalWeight(double w)    { lambda_temporal_ = w; }

    double photoWeight()     const { return w_photo_; }
    double icpWeight()       const { return w_icp_; }
    double smoothWeight()    const { return lambda_smooth_; }
    double rotWeight()       const { return lambda_rot_; }
    double temporalWeight()  const { return lambda_temporal_; }

private:
    // weights / hyper-parameters
    double w_photo_        = 1.0;
    double w_icp_          = 1.0;
    double lambda_smooth_  = 0.0;
    double lambda_rot_     = 0.0;
    double lambda_temporal_= 0.0;

    int maxStages_     = 1;   // kept for API compatibility
    int maxIterations_ = 10;  // outer GN iterations
};