#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#include <Eigen/Core>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MeshModel.h"
#include "BVH.h"
#include "EDGraph.h"

// -----------------------------------------------------------------------------
// PhotometricError (existing)
// -----------------------------------------------------------------------------
// r = sqrt(w) * sqrt(w_huber) * ( I(u,v) - I_v )
// NOTE: Uses the T_wc convention: p_c = R^T * (p_w - t)
class PhotometricError {
public:
    PhotometricError(const MeshModel::Vertex& vertex,
                     int vertex_index,
                     const std::vector<MeshModel::Triangle>& mesh_triangles,
                     const Eigen::Matrix3d& K,
                     const cv::Mat& image_gray_float,   // CV_32F in [0,1]
                     const BVH& bvh,                    // kept for signature compatibility
                     double sqrt_w,
                     const EDGraph* edGraph);

    // Evaluate residual and Jacobians (pose is fixed, passed in)
    bool Evaluate(double intensity_i,
                  double& residual,
                  double* jacobian_intensity,     // (optional) d r / d I_v
                  Eigen::VectorXd* jacobian_ed,   // (optional) size = 12*G, only a few non-zeros filled
                  const Eigen::Matrix3d& R,
                  const Eigen::Vector3d& t) const;

    // Optional: adjust the Huber delta (in intensity units [0,1]).
    void setHuberDelta(double d) { huber_delta_ = d; }

private:
    // Bilinear fetch + gradient (du,dv) at (u,v). Clamped to image bounds.
    inline bool sampleBilinearAndGradient(float& u, float& v,
                                          float& I,
                                          float& dIdu,
                                          float& dIdv) const;

    static inline double huberWeight(double r, double delta) {
        const double ar = std::abs(r);
        if (ar <= delta) return 1.0;        // inside quadratic region
        return delta / (ar + 1e-12);        // outside: w = delta/|r|
    }

private:
    MeshModel::Vertex v_raw_;
    int vidx_;
    const std::vector<MeshModel::Triangle>& tris_;
    Eigen::Matrix3d K_;
    cv::Mat img_;           // CV_32F [0,1]
    const BVH& bvh_;
    double sqrt_w_;
    const EDGraph* ed_;

    // Robust kernel delta (default for [0,1] intensities)
    double huber_delta_ = 0.05;   // ~ 12/255
};

// -----------------------------------------------------------------------------
// ProjectiveICPError (NEW): point-to-plane residual using depth map
// -----------------------------------------------------------------------------
// r = sqrt(w) * sqrt(w_huber) * ((p_c - p_obs) · n_c)
// with p_c = R^T (p_w - t),
//      p_obs = backproject(u,v,z_obs),
//      n_c = R^T * n_w (unit length)
// Jacobian w.r.t. ED affine parameters mirrors PhotometricError, with Jimg replaced by
// Jgeo = n_c^T * R^T (1x3).
struct ProjectiveICPError {
    ProjectiveICPError(const MeshModel::Vertex& vertex,
                       int vertex_index,
                       const cv::Mat& depth_float,     // CV_32F, unit already unified (m or mm)
                       const EDGraph* edGraph,
                       const Eigen::Matrix3d& K,
                       const Eigen::Vector3d& normal_w, // current-iteration deformed world normal
                       double sqrt_w)
        : v_raw_(vertex), vidx_(vertex_index), depth_(depth_float), ed_(edGraph),
          K_(K), n_w_(normal_w), sqrt_w_(sqrt_w) {
        fx_ = K_(0,0); fy_ = K_(1,1); cx_ = K_(0,2); cy_ = K_(1,2);
    }

    // Evaluate residual and Jacobian wrt ED (pose is fixed and passed in)
    bool Evaluate(double& residual,
                                     Eigen::VectorXd* jacobian_ed,
                                     const Eigen::Matrix3d& R_wc,
                                     const Eigen::Vector3d& t_wc,
                                     double* jacobian_logscale /*=nullptr*/) const;


    void setHuberDelta(double d) { huber_delta_ = d; }
    void setDepthGate(double dz) { depth_gate_ = dz; } // gate on |pc.z - z_obs|

    // Expose a light bilinear sampler for depth (clamped)
    inline float sampleDepthBilinear(float& u, float& v) const {
        const int W = depth_.cols, H = depth_.rows;
        u = std::min(std::max(u, 1.0f), (float)W - 2.0f);
        v = std::min(std::max(v, 1.0f), (float)H - 2.0f);
        const int x = (int)std::floor(u), y = (int)std::floor(v);
        const float a = u - x, b = v - y;
        const float d00 = depth_.at<float>(y,   x  );
        const float d10 = depth_.at<float>(y,   x+1);
        const float d01 = depth_.at<float>(y+1, x  );
        const float d11 = depth_.at<float>(y+1, x+1);
        return (1-a)*(1-b)*d00 + a*(1-b)*d10 + (1-a)*b*d01 + a*b*d11;
    }

    static inline double huberWeight(double r, double delta) {
        const double ar = std::abs(r);
        if (ar <= delta) return 1.0;        // quadratic region
        return delta / (ar + 1e-12);        // outside: w = delta/|r|
    }

    // public members for convenience
    MeshModel::Vertex v_raw_;
    int vidx_ = -1;
    cv::Mat depth_;                 // CV_32F unified units
    const EDGraph* ed_ = nullptr;
    Eigen::Matrix3d K_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d n_w_ = Eigen::Vector3d(0,0,1);
    double sqrt_w_ = 1.0;

    // intrinsics cached
    double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;

    // robust & gating
    double huber_delta_ = 0.01;   // in geometry units along normal (same unit as depth)
    double depth_gate_  = -1.0;   // if >0, drop if |pc.z - z_obs| > gate
};

#endif // COSTFUNCTIONS_H