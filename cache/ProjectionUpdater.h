#ifndef PROJECTION_UPDATER_H
#define PROJECTION_UPDATER_H

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "SharedData.h"
#include "MeshModel.h"

class ProjectionUpdater : public ceres::IterationCallback {
public:
    ProjectionUpdater(SharedProjectionData* shared_data,
                      const std::vector<MeshModel::Vertex>& mesh_vertices, // **更改这里**
                      const std::vector<MeshModel::Triangle>& mesh_triangles,
                      const Eigen::Matrix3f& camera_intrinsics,
                      const std::vector<cv::Mat>& observed_images,
                      std::vector<Eigen::Matrix4f>& camera_poses);
                      
    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override;

private:
    SharedProjectionData* shared_data_;
    const std::vector<MeshModel::Vertex>& mesh_vertices_; // **更改这里**
    const std::vector<MeshModel::Triangle>& mesh_triangles_;
    const Eigen::Matrix3f& camera_intrinsics_;
    const std::vector<cv::Mat>& observed_images_;
    std::vector<Eigen::Matrix4f>& camera_poses_;
};

#endif // PROJECTION_UPDATER_H
