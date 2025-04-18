#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

// -------------------- 全局变量 --------------------
cv::Mat global_image;

// -------------------- 双线性插值 --------------------
float getBilinearInterpolatedValue(const cv::Mat& image, double u, double v) {
    int u0 = static_cast<int>(std::floor(u));
    int v0 = static_cast<int>(std::floor(v));
    int u1 = u0 + 1;
    int v1 = v0 + 1;

    if (u0 < 0 || u1 >= image.cols || v0 < 0 || v1 >= image.rows)
        return 0.f;

    double du = u - u0;
    double dv = v - v0;

    float I00 = image.at<float>(v0, u0);
    float I01 = image.at<float>(v0, u1);
    float I10 = image.at<float>(v1, u0);
    float I11 = image.at<float>(v1, u1);

    return static_cast<float>(
        (1 - du) * (1 - dv) * I00 +
        du * (1 - dv) * I01 +
        (1 - du) * dv * I10 +
        du * dv * I11
    );
}

// -------------------- 图像梯度 --------------------
std::pair<double, double> getImageGradient(double u, double v) {
    const double eps = 1e-4;
    float Ix1 = getBilinearInterpolatedValue(global_image, u + eps, v);
    float Ix2 = getBilinearInterpolatedValue(global_image, u - eps, v);
    float Iy1 = getBilinearInterpolatedValue(global_image, u, v + eps);
    float Iy2 = getBilinearInterpolatedValue(global_image, u, v - eps);
    return {(Ix1 - Ix2) / (2 * eps), (Iy1 - Iy2) / (2 * eps)};
}

double getInterpolatedIntensity(double u, double v) {
    return getBilinearInterpolatedValue(global_image, u, v);
}

// -------------------- 解析雅可比 --------------------
Eigen::Matrix<double, 1, 6> computeAnalyticJacobian(
    const Sophus::SE3d& T,
    const Eigen::Matrix3d& K,
    const Eigen::Vector3d& p_w,
    double I_mean,
    double& residual_out
) {
    double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
    Eigen::Matrix3d R = T.rotationMatrix();
    Eigen::Vector3d t = T.translation();
    Eigen::Vector3d p_cam = R.transpose() * (p_w - t);
    double X = p_cam(0), Y = p_cam(1), Z = p_cam(2);
    double u = fx * X / Z + cx;
    double v = fy * Y / Z + cy;

    auto [grad_u, grad_v] = getImageGradient(u, v);
    Eigen::Matrix<double, 1, 2> J_grad;
    J_grad << grad_u, grad_v;

    Eigen::Matrix<double, 2, 3> J_proj;
    J_proj << fx / Z, 0, -fx * X / (Z * Z),
              0, fy / Z, -fy * Y / (Z * Z);

    // Eigen::Vector3d p_diff = p_w - t; // 左扰动
    Eigen::Vector3d p_diff = R.transpose() * (p_w - t); // 右扰动
    Eigen::Matrix3d skew;
    skew <<     0, -p_diff(2),  p_diff(1),
           p_diff(2),        0, -p_diff(0),
          -p_diff(1),  p_diff(0),        0;

    Eigen::Matrix<double, 3, 6> J_se3;
    // J_se3 << R.transpose() * skew, R.transpose();
    // J_se3 << -R.transpose(), R.transpose() * skew;
    J_se3 << -Eigen::Matrix<double, 3, 3>::Identity(), skew;

    Eigen::Matrix<double, 1, 6> J = J_grad * J_proj * J_se3;
    double I_interp = getInterpolatedIntensity(u, v);
    residual_out = I_interp - I_mean;
    return J;
}

// -------------------- 数值雅可比（使用 Sophus） --------------------
Eigen::Matrix<double, 1, 6> computeNumericalJacobian(
    const Sophus::SE3d& T,
    const Eigen::Matrix3d& K,
    const Eigen::Vector3d& p_w,
    double I_mean
) {
    const double eps = 1e-6;
    Eigen::Matrix<double, 1, 6> J;
    J.setZero();

    double residual0;
    computeAnalyticJacobian(T, K, p_w, I_mean, residual0);

    for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
        delta(i) = eps;

        // Sophus::SE3d T_perturbed = Sophus::SE3d::exp(delta) * T;
        Sophus::SE3d T_perturbed = T * Sophus::SE3d::exp(delta);
        double residual_perturbed;
        computeAnalyticJacobian(T_perturbed, K, p_w, I_mean, residual_perturbed);

        J(0, i) = (residual_perturbed - residual0) / eps;
    }

    return J;
}

// -------------------- 主程序 --------------------
int main() {
    // 相机内参
    double fx = 155.0030, fy = 155.0030, cx = 160.0, cy = 120.0;
    Eigen::Matrix3d K;
    K << fx, 0, cx,
         0, fy, cy,
         0,  0,  1;

    // SE(3)
    double theta = M_PI / 6.0;
    Eigen::Matrix3d R;
    R << cos(theta), 0, sin(theta),
         0,          1, 0,
        -sin(theta), 0, cos(theta);
    Eigen::Vector3d t(1.5, -2.0, 3.0);
    Sophus::SE3d T(R, t);

    // 世界点
    Eigen::Vector3d p_w(23, 18, 7);

    // 生成图像
    int rows = 480, cols = 640;
    cv::Mat I(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float dx = x - cols / 2.0f;
            float dy = y - rows / 2.0f;
            float sigma = 50.0f;
            I.at<float>(y, x) = std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
        }
    }
    cv::normalize(I, I, 0.0f, 1.0f, cv::NORM_MINMAX);
    global_image = I;

    // 图像均值
    double I_mean = cv::mean(I)[0];

    // 计算雅可比和残差
    double residual;
    Eigen::Matrix<double, 1, 6> J_ana = computeAnalyticJacobian(T, K, p_w, I_mean, residual);
    Eigen::Matrix<double, 1, 6> J_num = computeNumericalJacobian(T, K, p_w, I_mean);

    std::cout << "Analytic Jacobian:\n" << J_ana << "\n\n";
    std::cout << "Numerical Jacobian:\n" << J_num << "\n\n";
    std::cout << "Difference:\n" << J_ana - J_num << "\n\n";
    std::cout << "Residual: " << residual << "\n";

    return 0;
}
