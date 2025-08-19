#include "CostFunctions.h"
#include <cmath>

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

bool PhotometricError::projectPoint(const Eigen::Vector3d& p_w,
                                    float& u, float& v, float& Zc) const {
    // Camera: p_c = R * p_w + t;  pinhole: [u v 1]^T = K * (p_c / Z)
    // R,t are passed via Evaluate (fixed GT pose); here we only do K * (..)
    // This helper actually needs R,t — but to avoid storing them in the class,
    // we keep projection in Evaluate, not here. So this function remains unused.
    (void)p_w; (void)u; (void)v; (void)Zc; return false;
}

bool PhotometricError::sampleBilinearAndGradient(float u, float v,
                                                 float& I,
                                                 float& dIdu,
                                                 float& dIdv) const {
    const int W = img_.cols;
    const int H = img_.rows;

    if (u < 1.0f || v < 1.0f || u > W-2.0f || v > H-2.0f) return false;

    int x = (int)std::floor(u);
    int y = (int)std::floor(v);
    float a = u - x;
    float b = v - y;

    float I00 = img_.at<float>(y,   x  );
    float I10 = img_.at<float>(y,   x+1);
    float I01 = img_.at<float>(y+1, x  );
    float I11 = img_.at<float>(y+1, x+1);

    // bilinear value
    I = (1-a)*(1-b)*I00 + a*(1-b)*I10 + (1-a)*b*I01 + a*b*I11;

    // spatial gradients via bilinear of finite differences
    // du-direction (x):
    float Gx00 = I10 - I00;
    float Gx01 = I11 - I01;
    float Gx = (1-b)*Gx00 + b*Gx01;
    // dv-direction (y):
    float Gy00 = I01 - I00;
    float Gy10 = I11 - I10;
    float Gy = (1-a)*Gy00 + a*Gy10;

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
    Eigen::Vector3d pw = ed_->deformVertex(v_raw_, vidx_);

    // 2) project
    Eigen::Vector3d pc = R * pw + t;
    const double Z = pc.z();
    if (Z <= 1e-8) return false;

    const float fx = (float)K_(0,0), fy = (float)K_(1,1);
    const float cx = (float)K_(0,2), cy = (float)K_(1,2);

    float uf = fx * (float)(pc.x()/Z) + cx;
    float vf = fy * (float)(pc.y()/Z) + cy;

    // 3) sample intensity and gradient
    float I, dIdu, dIdv;
    if (!sampleBilinearAndGradient(uf, vf, I, dIdu, dIdv)) return false;

    // 4) residual
    residual = (float)sqrt_w_ * (I - (float)intensity_i);

    // 5) jacobian wrt intensity
    if (jacobian_intensity) *jacobian_intensity = -sqrt_w_;

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
        Eigen::RowVector2d JimgPix; JimgPix << dIdu, dIdv; // already in image units

        // dI/dp (1x3)
        Eigen::RowVector3d Jimg = JimgPix * Jproj * R; // chain: p_w -> p_c -> [u,v] -> I

        // accumulate per bound node
        const auto& binds = ed_->getBindings()[vidx_];
        const auto& weights = ed_->getWeights()[vidx_];

        for (size_t k = 0; k < binds.size(); ++k) {
            const int nid = binds[k];
            const double w = weights[k];
            const Eigen::Vector3d g = ed_->getGraphNodes()[nid].position;
            const Eigen::Vector3d q = Eigen::Vector3d(v_raw_.x, v_raw_.y, v_raw_.z) - g; // (v-g)

            // d p_w / d vec(A) = (q \otimes I_3)  -> 3x9;  d p_w / d t = I_3
            // We need (1x9) and (1x3) rows: Jimg (1x3) * above
            // (q \otimes I_3) in row-major blocks
            // columns order: A(0,0),A(0,1),A(0,2), A(1,0)...A(2,2)
            double dA[9];
            dA[0] = Jimg(0) * q(0); // d/d A00
            dA[1] = Jimg(0) * q(1); // d/d A01
            dA[2] = Jimg(0) * q(2); // d/d A02
            dA[3] = Jimg(1) * q(0); // d/d A10
            dA[4] = Jimg(1) * q(1); // d/d A11
            dA[5] = Jimg(1) * q(2); // d/d A12
            dA[6] = Jimg(2) * q(0); // d/d A20
            dA[7] = Jimg(2) * q(1); // d/d A21
            dA[8] = Jimg(2) * q(2); // d/d A22

            const int base = 12 * nid;
            for (int c = 0; c < 9; ++c) (*jacobian_ed)(base + c) += sqrt_w_ * w * dA[c];
            // translation t
            (*jacobian_ed)(base + 9)  += sqrt_w_ * w * Jimg(0);
            (*jacobian_ed)(base + 10) += sqrt_w_ * w * Jimg(1);
            (*jacobian_ed)(base + 11) += sqrt_w_ * w * Jimg(2);
        }
    }

    return true;
}