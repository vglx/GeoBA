#include "CostFunctions.h"
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
    const EDGraph* ed_topology,
    const EDState* ed_state_for_frame
) : vertex_(vertex), vidx_(vertex_index), triangles_(triangles),
    intrinsics_(intrinsics), current_image_(current_image),
    bvh_(bvh), weight_(weight), ed_(ed_topology), ed_state_(ed_state_for_frame) {}

inline bool PhotometricError::projectPinhole(
    const Eigen::Matrix3d& K,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    const Eigen::Vector3d& Pw,
    Eigen::Vector2d& uv,
    Eigen::Vector3d& Pc
) {
    Pc = R.transpose() * (Pw - t); // world -> camera
    const double Z = Pc.z();
    if (Z <= 1e-6) return false;
    const double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    uv.x() = fx * Pc.x() / Z + cx;
    uv.y() = fy * Pc.y() / Z + cy;
    return true;
}

bool PhotometricError::Evaluate(
    const Eigen::Matrix<double,6,1>& se3,
    double intensity,
    double& residual,
    Eigen::Matrix<double,1,6>* jacobian_pose,
    double* jacobian_intensity,
    Eigen::VectorXd* jacobian_ed
) const {
    // 解出 R,t（位姿不一定求导，但用于投影）
    Sophus::SE3d T = Sophus::SE3d::exp(se3);
    Eigen::Matrix3d R = T.rotationMatrix();
    Eigen::Vector3d t = T.translation();

    const int W = current_image_.cols;
    const int H = current_image_.rows;

    // 变形后的顶点（仿射 ED）
    const Eigen::Vector3d x(vertex_.x, vertex_.y, vertex_.z);
    Eigen::Vector3d v_def = x;
    if (ed_ && ed_state_) {
        v_def = ed_->deformVertex(vertex_, vidx_, *ed_state_);
    }

    // 投影 + 视场检查（此处未做 BVH 遮挡射线测试；若需要可在调用端先做可见性过滤）
    Eigen::Vector2d uv; Eigen::Vector3d Pc;
    if (!projectPinhole(intrinsics_, R, t, v_def, uv, Pc)) {
        residual = 0.0;
        if (jacobian_pose) jacobian_pose->setZero();
        if (jacobian_intensity) *jacobian_intensity = 0.0;
        if (jacobian_ed) jacobian_ed->setZero();
        return true;
    }
    const double u = uv.x(), v = uv.y();
    if (u < 0 || u >= W || v < 0 || v >= H) {
        residual = 0.0;
        if (jacobian_pose) jacobian_pose->setZero();
        if (jacobian_intensity) *jacobian_intensity = 0.0;
        if (jacobian_ed) jacobian_ed->setZero();
        return true;
    }

    // 光度残差 r = sqrt(w) * (I(u,v) - I_v)
    const float Iuv = ImageProcessor::getBilinearInterpolatedIntensity(current_image_, u, v);
    const double sw = std::sqrt(std::max(0.0, weight_));
    residual = sw * (static_cast<double>(Iuv) - intensity);

    // 图像梯度 1x2 与投影雅可比 2x3
    auto g = ImageProcessor::computeGradient(current_image_, u, v);
    Eigen::Matrix<double,1,2> J_grad; J_grad << g.first, g.second;

    const double X = Pc.x(), Y = Pc.y(), Z = std::max(Pc.z(), 1e-6);
    const double fx = intrinsics_(0,0), fy = intrinsics_(1,1);
    Eigen::Matrix<double,2,3> J_proj;
    J_proj << fx/Z, 0,   -fx*X/(Z*Z),
              0,   fy/Z, -fy*Y/(Z*Z);

    // 可选：位姿雅可比（如果指针非空）
    if (jacobian_pose) {
        *jacobian_pose = sw * computePoseJacobian(v_def, R, t, u, v);
    }

    // 顶点强度雅可比
    if (jacobian_intensity) {
        *jacobian_intensity = -sw;
    }

    // ED (Affine) 雅可比：长度 = 12 * G，每个节点一段 [vec(A)=9, b=3]
    if (jacobian_ed && ed_ && ed_state_) {
        const int G = ed_->numNodes();
        jacobian_ed->resize(G * 12);
        jacobian_ed->setZero();

        const auto& bindings = ed_->getBindings();
        const auto& weights  = ed_->getWeights();
        const auto& nodes    = ed_->getGraphNodes();

        const auto& idxs = bindings[vidx_];
        const auto& wts  = weights[vidx_];

        for (size_t k = 0; k < idxs.size(); ++k) {
            const int nid = idxs[k];
            const double wk = wts[k];
            const Eigen::Vector3d& gi = nodes[nid].position;

            Eigen::Matrix<double,1,12> J_ed_k = sw * computeAffineJacobianAtNode(
                J_grad, J_proj, gi, x, wk);

            jacobian_ed->segment<12>(nid * 12) += J_ed_k;
        }
    }

    return true;
}

Eigen::Matrix<double,1,12> PhotometricError::computeAffineJacobianAtNode(
    const Eigen::Matrix<double,1,2>& J_grad,
    const Eigen::Matrix<double,2,3>& J_proj,
    const Eigen::Vector3d& g_i,
    const Eigen::Vector3d& x,
    double w_k
) const {
    // x' 对 b_i 的导数：w_k * I3
    // x' 对 A_i 的导数：w_k * (x - g_i) —— 按元素填充到 9 维 vec(A)；
    // 组合后通过链式乘以 J_proj 与 J_grad，得到标量残差对参数的导数。

    const Eigen::Vector3d p = x - g_i; // (x - g_i)

    // 先构造对 x' 的导数矩阵：dres/dx' = J_grad * J_proj  => 1x3
    const Eigen::RowVector3d J_img = (J_grad * J_proj);

    Eigen::Matrix<double,1,12> J; J.setZero();

    // 对 vec(A)（列主序 a00 a10 a20 a01 a11 a21 a02 a12 a22）
    // d x'_k / d A_{k\ell} = w_k * p_ell
    // 因此 d r / d A_{k\ell} = J_img_k * (w_k * p_ell)
    // 依照 Eigen 列主序的 vec(A) 展开：
    //  col 0: a00 a10 a20  (ell=0, k=0..2)
    //  col 1: a01 a11 a21  (ell=1, k=0..2)
    //  col 2: a02 a12 a22  (ell=2, k=0..2)
    int idx = 0;
    for (int ell = 0; ell < 3; ++ell) {
        for (int k = 0; k < 3; ++k) {
            J(0, idx++) = J_img(k) * (w_k * p(ell));
        }
    }

    // 对 b：d x'/d b = w_k * I3  =>  d r / d b = J_img * (w_k * I3)
    J.segment<3>(9) = w_k * J_img;

    return J;
}

Eigen::Matrix<double,1,6> PhotometricError::computePoseJacobian(
    const Eigen::Vector3d& v_def,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    double u,
    double v
) const {
    // 与你原实现一致：J_img (1x3) 乘以 [ -R [v_def - t]_x, I ]
    auto g = ImageProcessor::computeGradient(current_image_, u, v);
    Eigen::Matrix<double,1,2> J_grad; J_grad << g.first, g.second;

    const Eigen::Vector3d pc = R.transpose() * (v_def - t);
    const double X = pc.x(), Y = pc.y(), Z = std::max(pc.z(), 1e-6);
    const double fx = intrinsics_(0,0), fy = intrinsics_(1,1);
    Eigen::Matrix<double,2,3> J_proj;
    J_proj << fx/Z, 0,   -fx*X/(Z*Z),
              0,   fy/Z, -fy*Y/(Z*Z);

    const Eigen::RowVector3d J_img = (J_grad * J_proj);

    Eigen::Matrix<double,3,3> hat;
    hat <<     0, -(v_def - t).z(),  (v_def - t).y(),
           (v_def - t).z(),     0, -(v_def - t).x(),
          -(v_def - t).y(), (v_def - t).x(),     0;

    Eigen::Matrix<double,3,6> J_vp;
    J_vp.block<3,3>(0,0) = -R * hat;
    J_vp.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

    return (J_img * J_vp);
}