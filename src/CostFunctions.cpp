#include "CostFunctions.h"

// 实现中需要用到的依赖
#include "Projection.h"
#include "ImageProcessor.h"
#include <sophus/se3.hpp>
#include <cmath>

PhotometricError::PhotometricError(const MeshModel::Vertex& vertex,
                                   const std::vector<MeshModel::Triangle>& triangles,
                                   const Eigen::Matrix3d& intrinsics,
                                   const cv::Mat& current_image,
                                   const BVH& bvh,
                                   double weight)
    : vertex_(vertex),
      triangles_(triangles),
      intrinsics_(intrinsics),
      current_image_(current_image),
      bvh_(bvh),
      weight_(weight) {
    // 可在此进行一些预处理
}

bool PhotometricError::Evaluate(const Eigen::Matrix<double, 6, 1>& se3,
                                double intensity,
                                double& residual,
                                Eigen::Matrix<double, 1, 6>* jacobian_pose,
                                double* jacobian_intensity) const {
    // 将 se3 转换为 SE3 变换
    Sophus::SE3d transform = Sophus::SE3d::exp(se3);
    Eigen::Matrix3d R = transform.rotationMatrix();
    Eigen::Vector3d t = transform.translation();

    // 判断顶点是否可见：直接调用 Projection::isVertexVisible
    if (!Projection::isVertexVisible(vertex_, intrinsics_, R, t, bvh_, current_image_.cols, current_image_.rows)) {
        residual = 0.0;
        if (jacobian_pose) jacobian_pose->setZero();
        if (jacobian_intensity) *jacobian_intensity = 0.0;
        return true;
    }

    // 投影计算：直接调用 Projection::projectPoint
    Eigen::Vector2d proj = Projection::projectPoint(vertex_, intrinsics_, R, t);
    if (proj(0) < 0 || proj(0) >= current_image_.cols || proj(1) < 0 || proj(1) >= current_image_.rows) {
        residual = 0.0;
        if (jacobian_pose) jacobian_pose->setZero();
        if (jacobian_intensity) *jacobian_intensity = 0.0;
        return true;
    }

    // 获取图像像素值（假设图像为 CV_32F 类型）
    float pixel_value = ImageProcessor::getBilinearInterpolatedIntensity(current_image_, proj(0), proj(1));
    double sqrt_weight = std::sqrt(weight_);
    residual = sqrt_weight * (pixel_value - intensity);

    // 计算雅可比：调用 computeJacobian 封装函数
    if (jacobian_pose || jacobian_intensity) {
        Eigen::Matrix<double, 1, 6> J_total = computeAnalyticalJacobian(vertex_, intrinsics_, R, t, current_image_, proj(0), proj(1));
        // Eigen::Matrix<double, 1, 6> J_total = computeNumericalJacobian(se3, intensity);
        auto J_chain = computeChainNumericalJacobian(se3, intrinsics_, vertex_, current_image_, intensity, proj(0), proj(1));

        Eigen::Matrix<double,1,6> J_1 = computeNumericalJacobian(se3, intensity);
        std::cout << "Analytic: " << J_total << std::endl;
        std::cout << "Numerical: " << J_1 << std::endl;
        std::cout<<"Chain‐Num J_total: "<<J_chain<<std::endl;

        // ----- 新增：提取中间量 -----
        // 世界→相机
        Sophus::SE3d Tcw = Sophus::SE3d::exp(se3);
        Sophus::SE3d Twc = Tcw.inverse();
        Eigen::Matrix3d Rwc = Twc.rotationMatrix();
        Eigen::Vector3d twc = Twc.translation();
        // 世界点
        Eigen::Vector3d pw(vertex_.x, vertex_.y, vertex_.z);
        // 相机坐标
        Eigen::Vector3d pc = Rwc * pw + twc;
        double X = pc.x(), Y = pc.y(), Z = std::max(pc.z(), 1e-6);
        // 投影后的 uv
        Eigen::Vector2d uv;
        uv(0) = intrinsics_(0,0)*X/Z + intrinsics_(0,2);
        uv(1) = intrinsics_(1,1)*Y/Z + intrinsics_(1,2);

        // ----- 1) 数值 J_proj（2×3）：对 (X,Y,Z) 做有限差分 -----
        const double eps = 1e-6;
        Eigen::Matrix<double,2,3> J_proj_num;
        for (int i = 0; i < 3; ++i) {
            Eigen::Vector3d pc_p = pc;
            pc_p[i] += eps;
            // 重新投影
            double Xp = pc_p.x(), Yp = pc_p.y(), Zp = std::max(pc_p.z(), 1e-6);
            Eigen::Vector2d uv_p;
            uv_p(0) = intrinsics_(0,0)*Xp/Zp + intrinsics_(0,2);
            uv_p(1) = intrinsics_(1,1)*Yp/Zp + intrinsics_(1,2);
            J_proj_num.col(i) = (uv_p - uv) / eps;
        }

        // ----- 2) 数值 J_se3（3×6）：对 se3 的 6 个自由度做有限差分，监控 pc -----
        Eigen::Matrix<double,3,6> J_se3_num = Eigen::Matrix<double,3,6>::Zero();
        for (int i = 0; i < 6; ++i) {
            Eigen::Matrix<double,6,1> delta = Eigen::Matrix<double,6,1>::Zero();
            delta[i] = eps;
            // 扰动
            Sophus::SE3d Tcw_p = Tcw * Sophus::SE3d::exp(delta);
            // Sophus::SE3d Tcw_p = Sophus::SE3d::exp(delta) * Tcw;
            Eigen::Matrix3d Rwc_p = Tcw_p.inverse().rotationMatrix();
            Eigen::Vector3d twc_p = Tcw_p.inverse().translation();
            Eigen::Vector3d pc_p = Rwc_p * pw + twc_p;
            J_se3_num.col(i) = (pc_p - pc) / eps;
        }

        // ----- 3) 打印对比 -----
        // std::cout << "Numeric  J_proj:\n" << J_proj_num << "\n\n";

        // std::cout << "Numeric  J_se3:\n" << J_se3_num << "\n\n";

        if (jacobian_pose) {
            *jacobian_pose = sqrt_weight * J_total;
        }
        if (jacobian_intensity) {
            *jacobian_intensity = -sqrt_weight;
        }
    }

    return true;
}

Eigen::Matrix<double, 1, 6> PhotometricError::computeAnalyticalJacobian(const MeshModel::Vertex& vertex,
                                                                        const Eigen::Matrix3d& intrinsics,
                                                                        const Eigen::Matrix3d& R,
                                                                        const Eigen::Vector3d& t,
                                                                        const cv::Mat& image,
                                                                        double u, double v) const {
    Eigen::Matrix<double, 1, 6> J;
    J.setZero();

    int W = image.cols, H = image.rows;
    if (u < 1.0 || u > W - 2.0 || v < 1.0 || v > H - 2.0) {
        return Eigen::Matrix<double, 1, 6>::Zero();
    }                                                                            

    // 计算图像梯度
    auto grad = ImageProcessor::computeGradient(image, u, v);
    double grad_u = grad.first, grad_v = grad.second;
    Eigen::Matrix<double, 1, 2> J_grad;
    J_grad << grad_u, grad_v;

    // 计算点在相机坐标系下的坐标
    Eigen::Vector3d point_world(vertex.x, vertex.y, vertex.z);
    Eigen::Vector3d point_cam = R.transpose() * (point_world - t);
    double X = point_cam(0), Y = point_cam(1), Z = point_cam(2);
    Z = std::max(Z, 1e-6);
    double fx = intrinsics(0, 0), fy = intrinsics(1, 1);

    // 计算投影雅可比，参考相机模型导数公式
    Eigen::Matrix<double, 2, 3> J_proj;
    J_proj << fx / Z, 0, -fx * X / (Z * Z),
              0, fy / Z, -fy * Y / (Z * Z);

    // 计算 SE3 对投影点的影响（基于李代数求导）
    // 此处采用与之前 MultiViewPhotometricError::computeJacobian 类似的实现
    Eigen::Matrix<double, 3, 6> J_se3;
    // Eigen::Vector3d p_diff = point_world - t;  // p_w - t
    Eigen::Vector3d p_diff = R.transpose() * (point_world - t);
    Eigen::Matrix3d skew;
    skew << 0,           -p_diff(2),  p_diff(1),
            p_diff(2),    0,         -p_diff(0),
            -p_diff(1),   p_diff(0),   0;
    // J_se3 << -R.transpose(), R.transpose() * skew;
    J_se3 << -Eigen::Matrix<double, 3, 3>::Identity(), skew;

    // std::cout << "Analytic J_proj:\n" << J_proj << "\n";
    // std::cout << "Analytic J_se3:\n" << J_se3 << "\n";

    // 最终雅可比为链式法则相乘
    J = J_grad * J_proj * J_se3;
    return J;
}

// Eigen::Matrix<double, 1, 6> PhotometricError::computeNumericalJacobian(const Eigen::Matrix<double, 6, 1>& se3,
//                                                                         double intensity) const {
//     double epsilon = 1e-6;
//     double sqrt_weight = std::sqrt(weight_);
    
//     // 先计算当前 se3 参数下的光度误差 error0
//     Sophus::SE3d transform = Sophus::SE3d::exp(se3);
//     Eigen::Matrix3d R = transform.rotationMatrix();
//     Eigen::Vector3d t = transform.translation();

//     // 检查顶点是否可见
//     if (!Projection::isVertexVisible(vertex_, intrinsics_, R, t, bvh_, current_image_.cols, current_image_.rows)) {
//         return Eigen::Matrix<double, 1, 6>::Zero();
//     }

//     // 投影计算
//     Eigen::Vector2d proj = Projection::projectPoint(vertex_, intrinsics_, R, t);
//     if (proj(0) < 0 || proj(0) >= current_image_.cols || proj(1) < 0 || proj(1) >= current_image_.rows) {
//         return Eigen::Matrix<double, 1, 6>::Zero();
//     }

//     // 获取当前像素值（假设图像为 CV_32F 类型）
//     float pixel_value = pixel_value = ImageProcessor::getBilinearInterpolatedIntensity(current_image_, proj(0), proj(1));;
//     double error0 = sqrt_weight * (pixel_value - intensity);

//     // 数值雅可比
//     Eigen::Matrix<double, 1, 6> J_num;
//     J_num.setZero();

//     // 对 se3 中的每个自由度施加微小扰动，计算有限差分
//     for (int i = 0; i < 6; ++i) {
//         // Eigen::Matrix<double, 6, 1> se3_perturbed = se3;
//         // se3_perturbed(i) += epsilon;
//         // Sophus::SE3d transform_perturbed = Sophus::SE3d::exp(se3_perturbed);

//         Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
//         delta(i) = epsilon;
//         Sophus::SE3d transform_perturbed = transform * Sophus::SE3d::exp(delta);
//         // Sophus::SE3d transform_perturbed = Sophus::SE3d::exp(delta) * transform;

//         Eigen::Matrix3d R_perturbed = transform_perturbed.rotationMatrix();
//         Eigen::Vector3d t_perturbed = transform_perturbed.translation();

//         // 检查扰动后的顶点是否可见
//         if (!Projection::isVertexVisible(vertex_, intrinsics_, R_perturbed, t_perturbed, bvh_, current_image_.cols, current_image_.rows)) {
//             continue;
//         }
//         Eigen::Vector2d proj_perturbed = Projection::projectPoint(vertex_, intrinsics_, R_perturbed, t_perturbed);
//         if (proj_perturbed(0) < 0 || proj_perturbed(0) >= current_image_.cols || proj_perturbed(1) < 0 || proj_perturbed(1) >= current_image_.rows) {
//             continue;
//         }

//         float pixel_value_perturbed = ImageProcessor::getBilinearInterpolatedIntensity(current_image_, proj_perturbed(0), proj_perturbed(1));
//         double error_perturbed = sqrt_weight * (pixel_value_perturbed - intensity);

//         // 计算有限差分
//         J_num(i) = (error_perturbed - error0) / epsilon;
//     }
//     return J_num;
// }

Eigen::Matrix<double, 1, 6> PhotometricError::computeNumericalJacobian(
    const Eigen::Matrix<double, 6, 1>& se3,
    double intensity) const {
    const double epsilon     = 1e-6;
    const double sqrt_weight = std::sqrt(weight_);

    // 基准变换 T
    Sophus::SE3d transform = Sophus::SE3d::exp(se3);

    Eigen::Matrix<double, 1, 6> J_num = Eigen::Matrix<double, 1, 6>::Zero();
    for (int i = 0; i < 6; ++i) {
        // +ε 扰动
        Eigen::Matrix<double, 6, 1> d_plus = Eigen::Matrix<double, 6, 1>::Zero();
        d_plus(i) =  epsilon;
        // Sophus::SE3d T_plus = transform * Sophus::SE3d::exp(d_plus);
        Sophus::SE3d T_plus = Sophus::SE3d::exp(d_plus) * transform;

        // –ε 扰动
        Eigen::Matrix<double, 6, 1> d_minus = Eigen::Matrix<double, 6, 1>::Zero();
        d_minus(i) = -epsilon;
        // Sophus::SE3d T_minus = transform * Sophus::SE3d::exp(d_minus);
        Sophus::SE3d T_minus = Sophus::SE3d::exp(d_minus) * transform;

        // 计算两端扰动后的残差
        auto computeResidual = [&](const Sophus::SE3d& T)->double {
            Eigen::Matrix3d R = T.rotationMatrix();
            Eigen::Vector3d t = T.translation();

            // 可见性及边界检查
            if (!Projection::isVertexVisible(vertex_, intrinsics_, R, t,
                                            bvh_, current_image_.cols, current_image_.rows))
                return NAN;
            Eigen::Vector2d p = Projection::projectPoint(vertex_, intrinsics_, R, t);
            if (p.x() < 0 || p.x() >= current_image_.cols ||
                p.y() < 0 || p.y() >= current_image_.rows)
                return NAN;

            float pv = ImageProcessor::getBilinearInterpolatedIntensity(
                           current_image_, p.x(), p.y());
            return sqrt_weight * (pv - intensity);
        };

        double r_plus  = computeResidual(T_plus);
        double r_minus = computeResidual(T_minus);
        if (std::isnan(r_plus) || std::isnan(r_minus)) {
            // 如果任一端不可见／越界，就跳过该分量，保持为 0
            continue;
        }

        // 中心差分
        J_num(i) = (r_plus - r_minus) / (2.0 * epsilon);
    }

    return J_num;
}

Eigen::Matrix<double,1,6> PhotometricError::computeChainNumericalJacobian(
    const Eigen::Matrix<double,6,1>& se3,
    const Eigen::Matrix3d& intrinsics,
    const MeshModel::Vertex& vertex,
    const cv::Mat& image,
    double intensity,
    double u, double v) const
{
    // 1) compute camera→world then invert to world→camera
    Sophus::SE3d Tcw = Sophus::SE3d::exp(se3);
    Sophus::SE3d Twc = Tcw.inverse();              // <-- invert here
    Eigen::Matrix3d Rwc = Twc.rotationMatrix();
    Eigen::Vector3d twc = Twc.translation();

    // 2) project world→camera exactly as analytic does
    Eigen::Vector3d pw(vertex.x, vertex.y, vertex.z);
    Eigen::Vector3d pc = Rwc * pw + twc;           // = R^T*(p_w - t)
    double X = pc.x(), Y = pc.y(), Z = std::max(pc.z(),1e-6);

    // 3) J_grad same as before
    const double eps_g = 1e-4;
    float Ipu = ImageProcessor::getBilinearInterpolatedIntensity(image, u+eps_g, v);
    float Imu = ImageProcessor::getBilinearInterpolatedIntensity(image, u-eps_g, v);
    float Ipv = ImageProcessor::getBilinearInterpolatedIntensity(image, u, v+eps_g);
    float Imv = ImageProcessor::getBilinearInterpolatedIntensity(image, u, v-eps_g);
    Eigen::Matrix<double,1,2> Jg_num;
    Jg_num << (Ipu-Imu)/(2*eps_g), (Ipv-Imv)/(2*eps_g);

    // 4) J_proj numerically on (X,Y,Z)->(u,v)
    const double eps_p = 1e-6;
    Eigen::Matrix<double,2,3> Jp_num;
    Eigen::Vector2d uv0{intrinsics(0,0)*X/Z + intrinsics(0,2),
                       intrinsics(1,1)*Y/Z + intrinsics(1,2)};
    for(int i=0;i<3;i++){
      Eigen::Vector3d d = pc;
      d[i] += eps_p;
      double x=d.x(), y=d.y(), z=std::max(d.z(),1e-6);
      Eigen::Vector2d uv_p{intrinsics(0,0)*x/z + intrinsics(0,2),
                          intrinsics(1,1)*y/z + intrinsics(1,2)};
      Jp_num.col(i) = (uv_p - uv0)/eps_p;
    }

    // 5) J_se3 numerically on se3->pc using the same world→camera
    const double eps_s = 1e-6;
    Eigen::Matrix<double,3,6> Js_num;
    for(int i=0;i<6;i++){
      Eigen::Matrix<double,6,1> d = Eigen::Matrix<double,6,1>::Zero();
      d(i)=eps_s;
      //   Sophus::SE3d Tcw_p = Tcw * Sophus::SE3d::exp(d);
      Sophus::SE3d Tcw_p = Sophus::SE3d::exp(d) * Tcw;

      Sophus::SE3d Twc_p = Tcw_p.inverse();        // <-- invert here too
      Eigen::Vector3d pc_p = Twc_p.rotationMatrix()*pw
                          + Twc_p.translation();
      Js_num.col(i) = (pc_p - pc)/eps_s;
    }

    // 6) chain them
    return Jg_num * Jp_num * Js_num;
}