#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#include "MeshModel.h"
#include "BVH.h"
#include "EDGraph.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

// Photometric error for fixed camera pose, optimizing ED (affine) + vertex intensity.
// 支持可选的位姿雅可比（若传入指针非空），否则仅用于投影。
class PhotometricError {
public:
    PhotometricError(
        const MeshModel::Vertex& vertex,
        int vertex_index,
        const std::vector<MeshModel::Triangle>& triangles,
        const Eigen::Matrix3d& intrinsics,
        const cv::Mat& current_image,
        const BVH& bvh,
        double weight,
        const EDGraph* ed_topology,      // 全局拓扑（节点位置/绑定/权重）
        const EDState* ed_state_for_frame // 当帧 (A,b)
    );

    // Compute residual and Jacobians for pose(6), intensity(1), and ED params (12*G)
    bool Evaluate(
        const Eigen::Matrix<double,6,1>& se3, // 固定/传入的位姿（可不对其求导）
        double intensity,
        double& residual,
        Eigen::Matrix<double,1,6>* jacobian_pose,     // 可为 nullptr
        double* jacobian_intensity,                   // 可为 nullptr
        Eigen::VectorXd* jacobian_ed                  // 可为 nullptr（长度=12*G）
    ) const;

private:
    // inputs
    MeshModel::Vertex vertex_;
    int vidx_ = -1;
    std::vector<MeshModel::Triangle> triangles_;
    Eigen::Matrix3d intrinsics_;
    cv::Mat current_image_;
    BVH bvh_;
    double weight_ = 1.0;
    const EDGraph* ed_ = nullptr;           // 拓扑
    const EDState* ed_state_ = nullptr;     // 当帧状态

    // helpers
    static inline bool projectPinhole(
        const Eigen::Matrix3d& K,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t,
        const Eigen::Vector3d& Pw,
        Eigen::Vector2d& uv,
        Eigen::Vector3d& Pc
    );

    // 计算仿射节点对残差的一行雅可比（长度 12: vec(A)=9, b=3）
    Eigen::Matrix<double,1,12> computeAffineJacobianAtNode(
        const Eigen::Matrix<double,1,2>& J_grad,
        const Eigen::Matrix<double,2,3>& J_proj,
        const Eigen::Vector3d& g_i,     // 节点中心
        const Eigen::Vector3d& x,       // 原始顶点位置
        double w_k                       // 该节点权重
    ) const;

    // 可选：位姿雅可比（若需要对 se3 求导）
    Eigen::Matrix<double,1,6> computePoseJacobian(
        const Eigen::Vector3d& v_def,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t,
        double u,
        double v
    ) const;
};

#endif // COSTFUNCTIONS_H