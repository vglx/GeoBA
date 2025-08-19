#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#include <Eigen/Core>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MeshModel.h"
#include "BVH.h"
#include "EDGraph.h"

// PhotometricError (data term):
// r = sqrt(w) * ( I(u,v) - I_v )
// where (u,v) are the projection of the *deformed* vertex using fixed (R,t).
// Jacobians wrt intensity (scalar) and ED affine parameters (12 per node).
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

private:
    // Bilinear fetch + gradient (du,dv) at (u,v)
    inline bool sampleBilinearAndGradient(float u, float v,
                                          float& I,
                                          float& dIdu,
                                          float& dIdv) const;

    // camera projection of a 3D point in world (p_w) with fixed R,t
    inline bool projectPoint(const Eigen::Vector3d& p_w,
                             float& u, float& v, float& Zc) const;

private:
    MeshModel::Vertex v_raw_;
    int vidx_;
    const std::vector<MeshModel::Triangle>& tris_;
    Eigen::Matrix3d K_;
    cv::Mat img_;           // CV_32F [0,1]
    const BVH& bvh_;
    double sqrt_w_;
    const EDGraph* ed_;
};

#endif // COSTFUNCTIONS_H