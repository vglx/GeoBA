#include "CostFunctions.h"
#include <algorithm>
#include <cmath>

// ---------------- PhotometricError (existing) ----------------
PhotometricError::PhotometricError(const MeshModel::Vertex& vertex,
                                   int vertex_index,
                                   const std::vector<MeshModel::Triangle>& mesh_triangles,
                                   const Eigen::Matrix3d& K,
                                   const cv::Mat& image_gray_float,
                                   const BVH& bvh,
                                   double sqrt_w,
                                   const EDGraph* edGraph)
    : v_raw_(vertex), vidx_(vertex_index), tris_(mesh_triangles),
      K_(K), img_(image_gray_float), bvh_(bvh), sqrt_w_(sqrt_w), ed_(edGraph) {}

bool PhotometricError::sampleBilinearAndGradient(float& u, float& v,
                                                 float& I,
                                                 float& dIdu,
                                                 float& dIdv) const {
    const int W = img_.cols;
    const int H = img_.rows;

    // Clamp to borders (keep 1px margin)
    u = std::min(std::max(u, 1.0f), (float)W - 2.0f);
    v = std::min(std::max(v, 1.0f), (float)H - 2.0f);

    const int x = (int)std::floor(u);
    const int y = (int)std::floor(v);
    const float a = u - x;
    const float b = v - y;

    const float I00 = img_.at<float>(y,   x  );
    const float I10 = img_.at<float>(y,   x+1);
    const float I01 = img_.at<float>(y+1, x  );
    const float I11 = img_.at<float>(y+1, x+1);

    // bilinear value
    I = (1-a)*(1-b)*I00 + a*(1-b)*I10 + (1-a)*b*I01 + a*b*I11;

    // image gradients (forward differences bilinearly blended)
    const float Gx00 = I10 - I00;
    const float Gx01 = I11 - I01;
    const float Gx = (1-b)*Gx00 + b*Gx01;
    const float Gy00 = I01 - I00;
    const float Gy10 = I11 - I10;
    const float Gy = (1-a)*Gy00 + a*Gy10;

    dIdu = Gx; dIdv = Gy;
    return true;
}

bool PhotometricError::Evaluate(double intensity_i,
                                double& residual,
                                double* jacobian_intensity,
                                Eigen::VectorXd* jacobian_ed,
                                const Eigen::Matrix3d& R,
                                const Eigen::Vector3d& t) const {
    // 1) deform vertex by affine ED
    const Eigen::Vector3d pw = ed_->deformVertex(v_raw_, vidx_);

    // 2) project with T_wc convention: p_c = R^T (p_w - t)
    const Eigen::Vector3d pc = R.transpose() * (pw - t);
    const double Z = pc.z();
    if (Z <= 1e-8) return false;

    const float fx = (float)K_(0,0), fy = (float)K_(1,1);
    const float cx = (float)K_(0,2), cy = (float)K_(1,2);

    float uf = fx * (float)(pc.x()/Z) + cx;
    float vf = fy * (float)(pc.y()/Z) + cy;

    // 3) sample intensity and gradient (clamped)
    float I, dIdu, dIdv;
    sampleBilinearAndGradient(uf, vf, I, dIdu, dIdv);

    // 4) raw residual and robust weight (Huber)
    const double r_raw = (double)I - intensity_i;           // intensity in [0,1]
    const double w_rob = huberWeight(r_raw, huber_delta_);  // in [0,1]
    const double sqrt_wr = std::sqrt(w_rob);

    // final residual
    residual = sqrt_w_ * sqrt_wr * r_raw;

    // 5) jacobian wrt intensity
    if (jacobian_intensity) *jacobian_intensity = -sqrt_w_ * sqrt_wr;

    // 6) jacobian wrt ED affine parameters (only for nodes bound to this vertex)
    if (jacobian_ed) {
        jacobian_ed->setZero();

        // du/dp and dv/dp (2x3)
        const double X = pc.x(), Y = pc.y();
        const double invZ = 1.0 / Z;
        const double invZ2 = invZ * invZ;
        Eigen::Matrix<double,2,3> Jproj;
        Jproj(0,0) = fx * invZ;         Jproj(0,1) = 0.0;            Jproj(0,2) = -fx * X * invZ2;
        Jproj(1,0) = 0.0;               Jproj(1,1) = fy * invZ;       Jproj(1,2) = -fy * Y * invZ2;

        // dI/d[u,v] (1x2)
        Eigen::RowVector2d JimgPix; JimgPix << dIdu, dIdv; // in image units

        // dI/dp (1x3) with T_wc: chain p_w -> p_c -> [u,v] -> I, and p_c = R^T(p_w - t)
        const Eigen::RowVector3d Jimg = JimgPix * Jproj * R.transpose();

        // accumulate per bound node
        const auto& binds   = ed_->getBindings()[vidx_];
        const auto& weights = ed_->getWeights()[vidx_];
        const Eigen::Vector3d vraw(v_raw_.x, v_raw_.y, v_raw_.z);

        for (size_t k = 0; k < binds.size(); ++k) {
            const int nid = binds[k];
            const double w = weights[k];
            const Eigen::Vector3d g = ed_->getGraphNodes()[nid].position;
            const Eigen::Vector3d q = vraw - g; // (v-g)

            // d p_w / d vec(A) = (q \otimes I_3)  -> 3x9;  d p_w / d t = I_3
            double dA[9];
            dA[0] = Jimg(0) * q(0);
            dA[1] = Jimg(0) * q(1);
            dA[2] = Jimg(0) * q(2);
            dA[3] = Jimg(1) * q(0);
            dA[4] = Jimg(1) * q(1);
            dA[5] = Jimg(1) * q(2);
            dA[6] = Jimg(2) * q(0);
            dA[7] = Jimg(2) * q(1);
            dA[8] = Jimg(2) * q(2);

            const int base = 12 * nid;
            for (int c = 0; c < 9; ++c) (*jacobian_ed)(base + c) += sqrt_w_ * sqrt_wr * w * dA[c];
            // translation t
            (*jacobian_ed)(base + 9)  += sqrt_w_ * sqrt_wr * w * Jimg(0);
            (*jacobian_ed)(base + 10) += sqrt_w_ * sqrt_wr * w * Jimg(1);
            (*jacobian_ed)(base + 11) += sqrt_w_ * sqrt_wr * w * Jimg(2);
        }
    }

    return true;
}

// ---------------- ProjectiveICPError (new) ----------------
bool ProjectiveICPError::Evaluate(double& residual,
                                  Eigen::VectorXd* jacobian_ed,
                                  const Eigen::Matrix3d& R_wc,
                                  const Eigen::Vector3d& t_wc,
                                  double* jacobian_logscale) const {
    // 1) deform vertex (world)
    const Eigen::Vector3d pw = ed_->deformVertex(v_raw_, vidx_);

    // 2) world -> camera: p_c = R^T (p_w - t)
    const Eigen::Vector3d pc = R_wc.transpose() * (pw - t_wc);
    const double Z = pc.z();
    if (Z <= 1e-8) return false; // behind camera or degenerate

    // 3) project to pixel
    float uf = (float)(fx_ * (pc.x()/Z) + cx_);
    float vf = (float)(fy_ * (pc.y()/Z) + cy_);
    if (uf < 1.0f || vf < 1.0f || uf > (float)depth_.cols - 2.0f || vf > (float)depth_.rows - 2.0f)
        return false; // out of bounds

    // 4) sample observed depth (this is ALREADY the per-iteration scaled depth)
    const float z_obs_f = sampleDepthBilinear(uf, vf);
    if (!(z_obs_f > 0.f) || !std::isfinite(z_obs_f)) return false; // invalid depth
    const double z_obs = (double)z_obs_f;

    // optional gate on depth disagreement along z (kept here for local pruning)
    if (depth_gate_ > 3) {
        if (std::abs(Z - z_obs) > depth_gate_) return false;
    }

    // 5) backproject observed to camera coordinates
    Eigen::Vector3d pobs;
    pobs.z() = z_obs;
    pobs.x() = ( (double)uf - cx_ ) * z_obs / fx_;
    pobs.y() = ( (double)vf - cy_ ) * z_obs / fy_;

    // 6) normal: world -> camera; ensure unit length and orientation stability
    Eigen::Vector3d nc = R_wc.transpose() * n_w_;
    const double nrm = nc.norm();
    if (nrm < 1e-12) return false;
    nc /= nrm;
    if (nc.z() < 0) nc = -nc;

    // 7) residual (raw and weighted)
    const double r_raw = (pc - pobs).dot(nc);
    const double w_rob = huberWeight(r_raw, huber_delta_);
    const double sqrt_wr = std::sqrt(w_rob);
    residual = sqrt_w_ * sqrt_wr * r_raw;

    // 8) Jacobian wrt ED affine parameters (kept as in your version, numeric FD)
    if (jacobian_ed) {
        jacobian_ed->setZero();

        const auto& binds   = ed_->getBindings()[vidx_];
        const auto& weights = ed_->getWeights()[vidx_];
        const auto& nodes   = ed_->getGraphNodes();
        const Eigen::Vector3d vraw(v_raw_.x, v_raw_.y, v_raw_.z);

        // Build baseline pw0 consistent with deformVertex
        Eigen::Vector3d pw0 = Eigen::Vector3d::Zero();
        std::vector<Eigen::Vector3d> p_node0(binds.size());
        for (size_t k = 0; k < binds.size(); ++k) {
            const int nid = binds[k];
            const double w = weights[k];
            const auto &node = nodes[nid];
            const Eigen::Vector3d q = vraw - node.position; // (v - g)
            const Eigen::Vector3d p_k = node.A * q + node.position + node.t;
            p_node0[k] = p_k;
            pw0 += w * p_k;
        }

        const double epsA = 1e-5;  // for A entries
        const double epsT = 1e-3;  // for t entries (mm)

        for (size_t kb = 0; kb < binds.size(); ++kb) {
            const int nid = binds[kb];
            const double w_bind = weights[kb];
            const auto &node = nodes[nid];
            const Eigen::Vector3d q = vraw - node.position;

            // A(3x3) entries
            for (int a = 0; a < 9; ++a) {
                Eigen::Matrix3d A_pert = node.A;
                A_pert(a/3, a%3) += epsA;
                const Eigen::Vector3d p_k_pert = A_pert * q + node.position + node.t;
                const Eigen::Vector3d dpw = w_bind * (p_k_pert - p_node0[kb]);
                const Eigen::Vector3d pc_pert = R_wc.transpose() * ((pw0 + dpw) - t_wc);
                const double r_raw_pert = (pc_pert - pobs).dot(nc);
                const double dr = (r_raw_pert - r_raw) / epsA;
                const int base = 12 * nid;
                (*jacobian_ed)(base + a) += sqrt_w_ * sqrt_wr * dr;
            }
            // t(3)
            for (int j = 0; j < 3; ++j) {
                Eigen::Vector3d t_pert = node.t; t_pert(j) += epsT;
                const Eigen::Vector3d p_k_pert = node.A * q + node.position + t_pert;
                const Eigen::Vector3d dpw = w_bind * (p_k_pert - p_node0[kb]);
                const Eigen::Vector3d pc_pert = R_wc.transpose() * ((pw0 + dpw) - t_wc);
                const double r_raw_pert = (pc_pert - pobs).dot(nc);
                const double dr = (r_raw_pert - r_raw) / epsT;
                const int base = 12 * nid;
                (*jacobian_ed)(base + 9 + j) += sqrt_w_ * sqrt_wr * dr;
            }
        }
    }

    // 9) Jacobian wrt log-scale (global or per-frame micro): d res / d log s_eff
    if (jacobian_logscale) {
        // Since p_obs ∝ z_obs and z_obs ∝ s_eff, we have dp_obs/d log s = p_obs.
        // Thus d r_raw / d log s = - p_obs · n_c, and with weights:
        *jacobian_logscale = sqrt_w_ * sqrt_wr * ( - pobs.dot(nc) );
    }

    return true;
}