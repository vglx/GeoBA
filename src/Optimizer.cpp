#include "Optimizer.h"
#include "CostFunctions.h"
#include <iostream>
#include <sophus/se3.hpp>

Optimizer::Optimizer(double weight)
    : weight_(weight) {
    // 配置 Ceres Solver
    options_.linear_solver_type = ceres::SPARSE_SCHUR;
    options_.minimizer_progress_to_stdout = true;
    options_.max_num_iterations = 100;
}

void Optimizer::optimize(const std::vector<MeshModel::Vertex>& mesh_vertices,
    const std::vector<MeshModel::Triangle>& mesh_triangles,
    const Eigen::Matrix3d& camera_intrinsics,
    const std::vector<cv::Mat>& observed_images,
    const std::vector<cv::Mat>& observed_depth_maps,
    const std::vector<Eigen::MatrixXf>& depth_normals,
    std::vector<Eigen::Matrix4d>& camera_poses) {
    size_t frame_count = observed_images.size();
    ceres::Problem problem;

    // 预先将 RGB 图像转换为灰度图，并转换为 CV_32F 类型
    std::vector<cv::Mat> observed_images_gray;
    for (size_t i = 0; i < frame_count; ++i) {
        cv::Mat gray;
        if (observed_images[i].channels() == 3) {
            cv::cvtColor(observed_images[i], gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = observed_images[i];
        }
        // 转换为浮点型，并归一化到 [0,1]（根据需要调整比例）
        gray.convertTo(gray, CV_32F, 1.0/255.0);
        // std::cout << "greycol: " << gray.cols << "  greyrow: " << gray.rows << std::endl;
        observed_images_gray.push_back(gray);
    }

    std::vector<double> poses(frame_count * 6);
    for (size_t i = 0; i < frame_count; ++i) {
        Sophus::SE3d pose_SE3(camera_poses[i].block<3,3>(0,0),
                              camera_poses[i].block<3,1>(0,3));
        Eigen::Matrix<double,6,1> se3_vec = pose_SE3.log();
        for (int j = 0; j < 6; ++j) {
            poses[i * 6 + j] = se3_vec[j];
        }
    }

    // 对每个帧 i，构造一个 MultiViewPhotometricError 残差块
    // 其他参考帧位姿直接从 poses 数组中取（跟随更新）
    for (size_t i = 0; i < frame_count; ++i) {
        // 构造其他帧图像集合与其他帧位姿指针（排除当前帧 i）
        std::vector<cv::Mat> other_imgs;
        std::vector<const double*> other_pose_ptrs;
        for (size_t j = 0; j < frame_count; ++j) {
            if (j == i) continue;
            other_imgs.push_back(observed_images_gray[j]);
            other_pose_ptrs.push_back(&poses[j * 6]);
        }
        
        ceres::CostFunction* photometric_cf = MultiViewPhotometricError::Create(
            mesh_vertices, mesh_triangles, camera_intrinsics,
            observed_images_gray[i],   // 当前帧图像
            other_imgs,           // 其他帧图像集合
            other_pose_ptrs,      // 其他帧位姿指针
            weight_);
        // 仅当前帧的参数块传入
        problem.AddResidualBlock(photometric_cf, nullptr, &poses[i * 6]);
    }


        // **3. 配置 Ceres Solver**
    ceres::Solver::Summary summary;
    ceres::Solve(options_, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

        // **4. 更新优化后的相机位姿**
    for (size_t i = 0; i < frame_count; ++i) {
        Eigen::Matrix<double,6,1> se3_vec;
        for (int j = 0; j < 6; ++j) {
            se3_vec[j] = poses[i * 6 + j];
        }

        // **转换回 SE(3)**
        Sophus::SE3d updated_SE3 = Sophus::SE3d::exp(se3_vec);

        camera_poses[i].block<3, 3>(0, 0) = updated_SE3.rotationMatrix();
        camera_poses[i].block<3, 1>(0, 3) = updated_SE3.translation();
        std::cout << "Updated Pose [" << i << "] :\n" << camera_poses[i] << std::endl;
    }
}