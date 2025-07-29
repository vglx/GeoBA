#include "CostFunctions.h"
#include "Projection.h"
#include "ImageProcessor.h"
#include <sophus/se3.hpp>
#include <cmath>

PhotometricError::PhotometricError(
    const MeshModel::Vertex& vertex,
    int vertex_index,
    const std::vector<MeshModel::Triangle>& triangles,
    const Eigen::Matrix3d& intrinsics,
    const cv::Mat& current_image,
    const BVH& bvh,
    double weight,
    const EDGraph* ed
) : vertex_(vertex), vidx_(vertex_index), triangles_(triangles),
    intrinsics_(intrinsics), current_image_(current_image),
    bvh_(bvh), weight_(weight), ed_(ed) {}

bool PhotometricError::Evaluate(
    const Eigen::Matrix<double,6,1>& se3,
    double intensity,
    double& residual,
    Eigen::Matrix<double,1,6>* jacobian_pose,
    double* jacobian_intensity,
    Eigen::VectorXd* jacobian_ed
) const {
    // SE3 -> R, t
    Sophus::SE3d T = Sophus::SE3d::exp(se3);
    Eigen::Matrix3d R = T.rotationMatrix();
    Eigen::Vector3d t = T.translation();

    int W = current_image_.cols;
    int H = current_image_.rows;

    // Compute deformed vertex
    Eigen::Vector3d v0(vertex_.x, vertex_.y, vertex_.z);
    Eigen::Vector3d v_def = ed_ ? ed_->deformVertex(vertex_, vidx_) : v0;

    // Visibility check
    if (!Projection::isVertexVisible(
            vertex_, intrinsics_, R, t,
            bvh_, W, H,
            vidx_, ed_)) {
        residual = 0.0;
        if (jacobian_pose) jacobian_pose->setZero();
        if (jacobian_intensity) *jacobian_intensity = 0.0;
        if (jacobian_ed) jacobian_ed->setZero();
        return true;
    }

    // Projection
    Eigen::Vector2d proj = Projection::projectPoint(
        vertex_, intrinsics_, R, t,
        vidx_, ed_);
    double u = proj.x();
    double v = proj.y();
    if (u < 0 || u >= W || v < 0 || v >= H) {
        residual = 0.0;
        if (jacobian_pose) jacobian_pose->setZero();
        if (jacobian_intensity) *jacobian_intensity = 0.0;
        if (jacobian_ed) jacobian_ed->setZero();
        return true;
    }

    // Photometric residual
    float Iuv = ImageProcessor::getBilinearInterpolatedIntensity(
        current_image_, u, v);
    double sw = std::sqrt(weight_);
    residual = sw * (Iuv - intensity);

    // Image gradient and J_proj
    auto g = ImageProcessor::computeGradient(current_image_, u, v);
    Eigen::Matrix<double,1,2> J_grad; J_grad << g.first, g.second;

    Eigen::Vector3d pc = R.transpose() * (v_def - t);
    double X = pc.x(), Y = pc.y(), Z = std::max(pc.z(), 1e-6);
    double fx = intrinsics_(0,0), fy = intrinsics_(1,1);
    Eigen::Matrix<double,2,3> J_proj;
    J_proj << fx/Z, 0, -fx*X/(Z*Z),
             0, fy/Z, -fy*Y/(Z*Z);

    // Pose Jacobian
    if (jacobian_pose) {
        *jacobian_pose = sw * computePoseJacobian(v_def, R, t, u, v);
    }

    // Intensity Jacobian
    if (jacobian_intensity) {
        *jacobian_intensity = -sw;
    }

    // EDGraph Jacobian
    if (jacobian_ed && ed_) {
        int G = ed_->numNodes();
        jacobian_ed->resize(G*6);
        jacobian_ed->setZero();
        const auto& bindings = ed_->getBindings();
        const auto& weights = ed_->getWeights();
        for (size_t k = 0; k < bindings[vidx_].size(); ++k) {
            int nid = bindings[vidx_][k];
            double wk = weights[vidx_][k];
            const DeformationNode& node = ed_->getGraphNodes()[nid];
            Eigen::Matrix<double,1,6> J_ed = sw *
                computeEDJacobianAtNode(J_grad, J_proj, node, v0, wk);
            jacobian_ed->segment<6>(nid*6) = J_ed;
        }
    }

    return true;
}

Eigen::Matrix<double,1,6> PhotometricError::computePoseJacobian(
    const Eigen::Vector3d& v_def,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    double u,
    double v
) const {
    // Recompute gradients
    auto g = ImageProcessor::computeGradient(current_image_, u, v);
    Eigen::Matrix<double,1,2> J_grad; J_grad << g.first, g.second;

    Eigen::Vector3d pc = R.transpose() * (v_def - t);
    double X = pc.x(), Y = pc.y(), Z = std::max(pc.z(),1e-6);
    double fx = intrinsics_(0,0), fy = intrinsics_(1,1);
    Eigen::Matrix<double,2,3> J_proj;
    J_proj << fx/Z, 0, -fx*X/(Z*Z),
             0, fy/Z, -fy*Y/(Z*Z);

    // Jacobian of v_def w.r.t pose: [ -R [v_def - t]_× , I3 ]
    Eigen::Vector3d diff = v_def - t;
    Sophus::Matrix3d hat = Sophus::SO3d::hat(diff);
    Eigen::Matrix<double,3,6> J_vp;
    J_vp.block<3,3>(0,0) = -R * hat;
    J_vp.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

    Eigen::Matrix<double,1,6> J0 = J_grad * J_proj * J_vp;

    const double eps = 1e-6;

    // 为每个分量做可见性＋越界门控
    for (int i = 0; i < 6; ++i) {
        // 构造 +eps，-eps 的扰动
        Eigen::Matrix<double,6,1> d = Eigen::Matrix<double,6,1>::Zero();
        d(i) = eps;
        Sophus::SE3d Tp = Sophus::SE3d::exp(d) * Sophus::SE3d(R, t);
        Sophus::SE3d Tm = Sophus::SE3d::exp(-d) * Sophus::SE3d(R, t);

        Eigen::Matrix3d Rp = Tp.rotationMatrix();
        Eigen::Vector3d tp = Tp.translation();
        Eigen::Matrix3d Rm = Tm.rotationMatrix();
        Eigen::Vector3d tm = Tm.translation();

        // 检查 +eps 后是否可见
        bool vis_p = Projection::isVertexVisible(
            vertex_, intrinsics_, Rp, tp,
            bvh_, current_image_.cols, current_image_.rows,
            vidx_, ed_);
        // 检查 -eps 后是否可见
        bool vis_m = Projection::isVertexVisible(
            vertex_, intrinsics_, Rm, tm,
            bvh_, current_image_.cols, current_image_.rows,
            vidx_, ed_);

        // 越界检查也隐含在 isVertexVisible 里
        if (!(vis_p && vis_m)) {
            J0(i) = 0.0;
        }
    }

    return J0;
}

Eigen::Matrix<double,1,6> PhotometricError::computeEDJacobianAtNode(
    const Eigen::Matrix<double,1,2>& J_grad,
    const Eigen::Matrix<double,2,3>& J_proj,
    const DeformationNode& node,
    const Eigen::Vector3d& v0,
    double weight_k
) const {
    // p_k = v0 - node.position
    Eigen::Vector3d p_k = v0 - node.position;
    Eigen::Matrix3d Rk = node.transform.rotationMatrix();

    // J_vp_k = weight_k * [ -Rk * hat(p_k), I3 ]
    Sophus::Matrix3d hat_pk = Sophus::SO3d::hat(p_k);
    Eigen::Matrix<double,3,6> J_vp_k;
    J_vp_k.block<3,3>(0,0) = -Rk * hat_pk;
    J_vp_k.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
    J_vp_k *= weight_k;

    // chain rule
    return J_grad * J_proj * J_vp_k;
}