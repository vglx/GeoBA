#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#include <Eigen/Core>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MeshModel.h"
#include "EDGraph.h"

// -----------------------------------------------------------------------------
// PhotometricError (unchanged): single-view photometric residual
// r = sqrt(w) * sqrt(w_huber) * ( I(u,v) - I_v )
// NOTE: Uses the T_wc convention: p_c = R^T * (p_w - t)
// -----------------------------------------------------------------------------
class PhotometricError {
public:
    PhotometricError(const MeshModel::Vertex& vertex,
                     int vertex_index,
                     const std::vector<MeshModel::Triangle>& mesh_triangles,
                     const Eigen::Matrix3d& K,
                     const cv::Mat& image_gray_float,   // CV_32F in [0,1]
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
    double sqrt_w_;
    const EDGraph* ed_;

    // Robust kernel delta (default for [0,1] intensities)
    double huber_delta_ = 0.05;   // ~ 12/255
};

// -----------------------------------------------------------------------------
// StereoPhotometricError (NEW): Left–Right photometric consistency
// r = sqrt(w) * sqrt(w_huber) * ( I_L(u_L,v_L) - I_R(u_R,v_R) )
// Only ED parameters are optimized here (no per-vertex intensity variable).
// This drops ProjectiveICPError entirely.
// -----------------------------------------------------------------------------
class StereoPhotometricError {
public:
    StereoPhotometricError(const MeshModel::Vertex& vertex,
                           int vertex_index,
                           const EDGraph* edGraph,
                           // intrinsics
                           const Eigen::Matrix3d& K_left,
                           const Eigen::Matrix3d& K_right,
                           // grayscale images (CV_32F in [0,1])
                           const cv::Mat& img_left,
                           const cv::Mat& img_right,
                           // residual global weight (will be multiplied by robust sqrt weight)
                           double sqrt_w)
    : v_raw_(vertex), vidx_(vertex_index), ed_(edGraph),
      K_L_(K_left), K_R_(K_right),
      imgL_(img_left), imgR_(img_right),
      sqrt_w_(sqrt_w) {}

    // Evaluate residual & J wrt ED. Poses are provided per call.
    // T_wc convention for each camera: p_c = R^T (p_w - t)
    bool Evaluate(double& residual,
                  Eigen::VectorXd* jacobian_ed,   // (optional) size = 12*G
                  const Eigen::Matrix3d& R_wcl,   // world->cam-left rotation (actually cam->world given as W? we use transpose internally)
                  const Eigen::Vector3d& t_wcl,
                  const Eigen::Matrix3d& R_wcr,   // world->cam-right
                  const Eigen::Vector3d& t_wcr) const;

    void setHuberDelta(double d) { huber_delta_ = d; }

private:
    // Bilinear with gradient on a provided image; clamps to [1..W-2]/[1..H-2]
    static inline void sampleBilinearAndGradient(const cv::Mat& img,
                                                 float& u, float& v,
                                                 float& I, float& dIdu, float& dIdv) {
        const int W = img.cols, H = img.rows;
        u = std::min(std::max(u, 1.0f), (float)W - 2.0f);
        v = std::min(std::max(v, 1.0f), (float)H - 2.0f);
        const int x = (int)std::floor(u);
        const int y = (int)std::floor(v);
        const float a = u - x, b = v - y;
        const float I00 = img.at<float>(y,   x  );
        const float I10 = img.at<float>(y,   x+1);
        const float I01 = img.at<float>(y+1, x  );
        const float I11 = img.at<float>(y+1, x+1);
        I = (1-a)*(1-b)*I00 + a*(1-b)*I10 + (1-a)*b*I01 + a*b*I11;
        const float Gx00 = I10 - I00;
        const float Gx01 = I11 - I01;
        const float Gy00 = I01 - I00;
        const float Gy10 = I11 - I10;
        // du, dv gradients (forward diff blended)
        dIdu = (1-b)*Gx00 + b*Gx01;
        dIdv = (1-a)*Gy00 + a*Gy10;
    }

    static inline double huberWeight(double r, double delta) {
        const double ar = std::abs(r);
        if (ar <= delta) return 1.0;        // quadratic region
        return delta / (ar + 1e-12);        // outside: w = delta/|r|
    }

private:
    MeshModel::Vertex v_raw_;
    int vidx_ = -1;
    const EDGraph* ed_ = nullptr;

    Eigen::Matrix3d K_L_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d K_R_ = Eigen::Matrix3d::Identity();
    cv::Mat imgL_;    // CV_32F [0,1]
    cv::Mat imgR_;    // CV_32F [0,1]

    double sqrt_w_ = 1.0;
    double huber_delta_ = 0.05;   // intensity units
};

#endif // COSTFUNCTIONS_H