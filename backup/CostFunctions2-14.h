#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <ceres/ceres.h>
#include "Projection.h"
#include "ImageProcessor.h"
#include "MeshModel.h"

class LocalGeometricError {
public:
    LocalGeometricError(const std::vector<MeshModel::Vertex>& vertices,
                        const std::vector<MeshModel::Triangle>& triangles,
                        const Eigen::Matrix3f& camera_intrinsics,
                        const cv::Mat& depth_map,
                        const Eigen::MatrixXf& depth_normals,
                        double weight_local_depth,
                        double weight_local_normal);

    template <typename T>
    bool operator()(const T* const se3, T* residual) const;

    static ceres::CostFunction* Create(const std::vector<MeshModel::Vertex>& vertices,
                                       const std::vector<MeshModel::Triangle>& triangles,
                                       const Eigen::Matrix3f& camera_intrinsics,
                                       const cv::Mat& depth_map,
                                       const Eigen::MatrixXf& depth_normals,
                                       double weight_local_depth,
                                       double weight_local_normal);

private:
    const std::vector<MeshModel::Vertex>& vertices_;
    const std::vector<MeshModel::Triangle>& triangles_;
    const Eigen::Matrix3f& camera_intrinsics_;
    const cv::Mat& depth_map_;
    const Eigen::MatrixXf& depth_normals_;
    double weight_local_depth_;
    double weight_local_normal_;
};

// class TripletGlobalError {
//     public:
//         TripletGlobalError(const std::vector<MeshModel::Vertex>& vertices,
//                            const std::vector<MeshModel::Triangle>& triangles,
//                            const Eigen::Matrix3f& camera_intrinsics,
//                            const cv::Mat& depth_map_i,
//                            const cv::Mat& depth_map_j,
//                            const cv::Mat& depth_map_k,
//                            const cv::Mat& image_i,
//                            const cv::Mat& image_j,
//                            const cv::Mat& image_k,
//                            double weight_global_depth,
//                            double weight_global_gradient);
    
//         template <typename T>
//         bool operator()(const T* const pose_i, const T* const pose_j, const T* const pose_k, T* residual) const;
    
//         static ceres::CostFunction* Create(const std::vector<MeshModel::Vertex>& vertices,
//                                            const std::vector<MeshModel::Triangle>& triangles,
//                                            const Eigen::Matrix3f& camera_intrinsics,
//                                            const cv::Mat& depth_map_i,
//                                            const cv::Mat& depth_map_j,
//                                            const cv::Mat& depth_map_k,
//                                            const cv::Mat& image_i,
//                                            const cv::Mat& image_j,
//                                            const cv::Mat& image_k,
//                                            double weight_global_depth,
//                                            double weight_global_gradient);
    
//     private:
//         const std::vector<MeshModel::Vertex>& vertices_;
//         const std::vector<MeshModel::Triangle>& triangles_;
//         const Eigen::Matrix3f& camera_intrinsics_;
//         const cv::Mat& depth_map_i_;
//         const cv::Mat& depth_map_j_;
//         const cv::Mat& depth_map_k_;
//         const cv::Mat& image_i_;
//         const cv::Mat& image_j_;
//         const cv::Mat& image_k_;
//         double weight_global_depth_;
//         double weight_global_gradient_;
// };

#endif // COST_FUNCTIONS_H
