#include "Optimizer.h"
#include "CostFunctions.h"
#include "Projection.h"
#include "BVH.h"
#include <iostream>
#include <sophus/se3.hpp>
#include <omp.h>
#include <ceres/numeric_diff_cost_function.h>
#include "ImageProcessor.h"

Optimizer::Optimizer(double weight)
    : weight_(weight) {
    options_.linear_solver_type = ceres::SPARSE_SCHUR;
    options_.minimizer_progress_to_stdout = true;
    options_.trust_region_strategy_type = ceres::DOGLEG;
    options_.max_num_iterations = 20;
    options_.num_threads = 4;
}

void Optimizer::optimize(
    const std::vector<MeshModel::Vertex>& mesh_vertices,
    const std::vector<MeshModel::Triangle>& mesh_triangles,
    const Eigen::Matrix3d& camera_intrinsics,
    const std::vector<cv::Mat>& depth_images,
    std::vector<Eigen::Matrix4d>& camera_poses) {

    size_t frame_count = depth_images.size();
    size_t vertex_count = mesh_vertices.size();
    ceres::Problem problem;

    // **构建 BVH 结构**
    BVH bvh(mesh_triangles, mesh_vertices);

    // **初始化优化变量** (frame_count 个相机位姿，每个 6 维)
    std::vector<double> poses(frame_count * 6);
    for (size_t i = 0; i < frame_count; ++i) {
        Sophus::SE3d pose_SE3(camera_poses[i].block<3,3>(0,0), camera_poses[i].block<3,1>(0,3));
        Eigen::Matrix<double,6,1> se3_vec = pose_SE3.log();
        for (int j = 0; j < 6; ++j) {
            poses[i * 6 + j] = se3_vec[j];
        }
    }

    // **添加残差项**
    for (size_t i = 0; i < vertex_count; ++i) {
        for (size_t j = 0; j < frame_count; ++j) {
            if (Projection::isVertexVisible(mesh_vertices[i], camera_intrinsics,
                camera_poses[j].block<3,3>(0,0), camera_poses[j].block<3,1>(0,3),
                bvh, depth_images[j].cols, depth_images[j].rows)) {

                // **创建残差项**
                ceres::CostFunction* photometric_cf = MultiViewPhotometricError::Create(
                    mesh_vertices[i], mesh_triangles, camera_intrinsics, depth_images[j], bvh, weight_
                );

                Eigen::Vector2d proj = Projection::projectPoint(mesh_vertices[i], camera_intrinsics,
                    camera_poses[j].block<3,3>(0,0), camera_poses[j].block<3,1>(0,3));
                
                float depth_value = ImageProcessor::getBilinearInterpolatedValue(depth_images[j], proj(0), proj(1));

                // **添加到 Ceres 优化问题**
                problem.AddResidualBlock(photometric_cf, nullptr, &poses[j * 6], &depth_value);
                // ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
                // problem.AddResidualBlock(photometric_cf, loss, &poses[j * 6], &x2_values[i]);

                // ceres::CostFunction* numeric_cf =
                // new ceres::NumericDiffCostFunction<
                //     MultiViewPhotometricError, 
                //     ceres::CENTRAL, 
                //     1,    // residual dimension
                //     6,    // pose
                //     1     // intensity
                //     >( new MultiViewPhotometricError(
                //             mesh_vertices[i],
                //             mesh_triangles,
                //             camera_intrinsics,
                //             depth_images[j],
                //             bvh,
                //             weight_
                //         ) );

                // problem.AddResidualBlock(
                //     numeric_cf,
                //     nullptr,
                //     &poses[j * 6],
                //     &x2_values[i]
                // );
            }
        }
    }

    // **运行 Ceres 优化**
    ceres::Solver::Summary summary;
    ceres::Solve(options_, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    // **将优化后的 poses 传回 camera_poses**
    for (size_t i = 0; i < frame_count; ++i) {
        // 读取优化后的 SE(3) 6D 变量
        Eigen::Matrix<double,6,1> se3_vec;
        for (int j = 0; j < 6; ++j) {
            se3_vec[j] = poses[i * 6 + j];
        }

        // **从李代数转换回 SE(3) 变换矩阵**
        Sophus::SE3d pose_SE3 = Sophus::SE3d::exp(se3_vec);
        Eigen::Matrix4d pose_mat = Eigen::Matrix4d::Identity();
        pose_mat.block<3,3>(0,0) = pose_SE3.rotationMatrix();
        pose_mat.block<3,1>(0,3) = pose_SE3.translation();

        // **更新 camera_poses**
        camera_poses[i] = pose_mat;
    }
}