#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "MeshModel.h"
#include "BVH.h"
#include "EDGraph.h"
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

class Optimizer {
public:
    Optimizer(double weight, int maxStages, int maxIterations,
              double lambda_smooth = 5.0,
              double lambda_temp   = 1.0,
              double lambda_rigid  = 0.1,
              int    edge_knn      = 4);

    void optimize(
        const std::vector<MeshModel::Vertex>& mesh_vertices,
        const std::vector<MeshModel::Triangle>& mesh_triangles,
        const Eigen::Matrix3d& camera_intrinsics,
        const std::vector<cv::Mat>& observed_images,
        const std::vector<Eigen::Matrix4d>& camera_poses,
        EDGraph& edGraph,
        std::vector<EDState>& ed_states_per_frame,
        std::vector<double>& vertex_intensity
    );

private:
    double weight_;
    int maxStages_;
    int maxIterations_;

    double lambda_smooth_;
    double lambda_temp_;
    double lambda_rigid_;
    int    edge_knn_;

    static Sophus::SE3d mat4ToSE3(const Eigen::Matrix4d& T);
    static cv::Mat ensureGrayFloat(const cv::Mat& img);

    void initializeIntensityByAverage(
        const std::vector<MeshModel::Vertex>& vertices,
        const Eigen::Matrix3d& K,
        const std::vector<cv::Mat>& images_gray,
        const std::vector<Sophus::SE3d>& poses,
        std::vector<double>& I
    ) const;

    static std::vector<std::pair<int,int>> buildNodeEdges(
        const std::vector<DeformationNode>& nodes, int k);
};

#endif