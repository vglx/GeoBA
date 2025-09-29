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

// ---------------- StereoPhotometricError (new) ----------------
bool StereoPhotometricError::Evaluate(double& residual,
                                      Eigen::VectorXd* jacobian_ed,
                                      const Eigen::Matrix3d& R_wcl,
                                      const Eigen::Vector3d& t_wcl,
                                      const Eigen::Matrix3d& R_wcr,
                                      const Eigen::Vector3d& t_wcr) const {
    // 1) deform to world
    const Eigen::Vector3d pw = ed_->deformVertex(v_raw_, vidx_);

    // 2) world->camera projections (T_wc convention): p_c = R^T (p_w - t)
    const Eigen::Vector3d pcL = R_wcl.transpose() * (pw - t_wcl);
    const Eigen::Vector3d pcR = R_wcr.transpose() * (pw - t_wcr);
    if (pcL.z() <= 1e-8 || pcR.z() <= 1e-8) return false;

    const float fxL = (float)K_L_(0,0), fyL = (float)K_L_(1,1);
    const float cxL = (float)K_L_(0,2), cyL = (float)K_L_(1,2);
    const float fxR = (float)K_R_(0,0), fyR = (float)K_R_(1,1);
    const float cxR = (float)K_R_(0,2), cyR = (float)K_R_(1,2);

    float uL = fxL * (float)(pcL.x()/pcL.z()) + cxL;
    float vL = fyL * (float)(pcL.y()/pcL.z()) + cyL;
    float uR = fxR * (float)(pcR.x()/pcR.z()) + cxR;
    float vR = fyR * (float)(pcR.y()/pcR.z()) + cyR;

    // 3) sample intensities & gradients at both views (clamped)
    float IL, dILdu, dILdv;
    float IR, dIRdu, dIRdv;
    sampleBilinearAndGradient(imgL_, uL, vL, IL, dILdu, dILdv);
    sampleBilinearAndGradient(imgR_, uR, vR, IR, dIRdu, dIRdv);

    // 4) residual with robust weight
    const double r_raw = (double)IL - (double)IR;
    const double w_rob = huberWeight(r_raw, huber_delta_);
    const double sqrt_wr = std::sqrt(w_rob);
    residual = sqrt_w_ * sqrt_wr * r_raw;

    // 5) Jacobian wrt ED affine parameters (Left term - Right term)
    if (jacobian_ed) {
        jacobian_ed->setZero();

        // Left projection jacobian du/dp, dv/dp
        const double XL = pcL.x(), YL = pcL.y(), ZL = pcL.z();
        const double invZL = 1.0 / ZL, invZL2 = invZL * invZL;
        Eigen::Matrix<double,2,3> JprojL;
        JprojL(0,0) = fxL * invZL;         JprojL(0,1) = 0.0;            JprojL(0,2) = -fxL * XL * invZL2;
        JprojL(1,0) = 0.0;                  JprojL(1,1) = fyL * invZL;     JprojL(1,2) = -fyL * YL * invZL2;
        Eigen::RowVector2d JimgPixL; JimgPixL << dILdu, dILdv;    // [dI/du, dI/dv]
        const Eigen::RowVector3d JimgL = JimgPixL * JprojL * R_wcl.transpose();

        // Right projection
        const double XR = pcR.x(), YR = pcR.y(), ZR = pcR.z();
        const double invZR = 1.0 / ZR, invZR2 = invZR * invZR;
        Eigen::Matrix<double,2,3> JprojR;
        JprojR(0,0) = fxR * invZR;         JprojR(0,1) = 0.0;            JprojR(0,2) = -fxR * XR * invZR2;
        JprojR(1,0) = 0.0;                  JprojR(1,1) = fyR * invZR;     JprojR(1,2) = -fyR * YR * invZR2;
        Eigen::RowVector2d JimgPixR; JimgPixR << dIRdu, dIRdv;
        const Eigen::RowVector3d JimgR = JimgPixR * JprojR * R_wcr.transpose();

        // Chain rule: d(IL - IR)/dp_w = JimgL - JimgR
        const Eigen::RowVector3d Jimg = JimgL - JimgR;

        // accumulate over bound nodes
        const auto& binds   = ed_->getBindings()[vidx_];
        const auto& weights = ed_->getWeights()[vidx_];
        const Eigen::Vector3d vraw(v_raw_.x, v_raw_.y, v_raw_.z);

        for (size_t k = 0; k < binds.size(); ++k) {
            const int nid = binds[k];
            const double w = weights[k];
            const Eigen::Vector3d g = ed_->getGraphNodes()[nid].position;
            const Eigen::Vector3d q = vraw - g; // (v-g)

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
            const double wscale = sqrt_w_ * sqrt_wr * w;
            for (int c = 0; c < 9; ++c) (*jacobian_ed)(base + c) += wscale * dA[c];
            (*jacobian_ed)(base + 9)  += wscale * Jimg(0);
            (*jacobian_ed)(base + 10) += wscale * Jimg(1);
            (*jacobian_ed)(base + 11) += wscale * Jimg(2);
        }
    }

    return true;
}