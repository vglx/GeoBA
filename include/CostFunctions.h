#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#include <Eigen/Core>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MeshModel.h"
#include "BVH.h"
#include "EDGraph.h"

// PhotometricError (data term):
// r = sqrt(w) * sqrt(w_huber) * ( I(u,v) - I_v )
// where (u,v) are the projection of the *deformed* vertex using fixed (R,t).
// Jacobians wrt intensity (scalar) and ED affine parameters (12 per node).
// NOTE: Uses the T_wc convention consistently: p_c = R^T * (p_w - t)
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

    // Evaluate residual and Jacobians.
    // Pose has been removed; we pass fixed R,t here.
    bool Evaluate(double intensity_i,
                  double& residual,
                  double* jacobian_intensity,     // (optional) d r / d I_v
                  Eigen::VectorXd* jacobian_ed,   // (optional) size = 12*G, only a few non-zeros filled
                  const Eigen::Matrix3d& R,
                  const Eigen::Vector3d& t) const;

    // Optional: adjust the Huber delta (in intensity units [0,1]).
    void setHuberDelta(double d) { huber_delta_ = d; }

private:
    // Bilinear fetch + gradient (du,dv) at (u,v). This version clamps to image bounds
    // so it never fails at borders; always returns true.
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

#endif // COSTFUNCTIONS_H