#ifndef DATASET_MANAGER_H
#define DATASET_MANAGER_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "MeshModel.h"

struct FrameData {
    cv::Mat rgb_image;
    cv::Mat depth_image;
};

class DatasetManager {
public:
    DatasetManager(const std::string& dataset_path);

    // 加载所有 RGB 图像
    bool loadAllRGBImages(std::vector<cv::Mat>& rgb_images);

    // 加载所有深度图像
    bool loadAllDepthImages(std::vector<cv::Mat>& depth_images);

    // 加载网格模型
    bool loadMeshModel(MeshModel& mesh);

    // 加载相机内参
    bool loadCameraIntrinsics(Eigen::Matrix3d& intrinsics);

    // 加载 Ground Truth 位姿
    bool loadPoses(std::vector<Eigen::Matrix4d>& poses, const std::string& fileName);

private:
    std::string dataset_path_;

    // 私有辅助方法：解析 OBJ 文件
    bool parseOBJ(const std::string& filePath, std::vector<MeshModel::Vertex>& vertices, std::vector<MeshModel::Triangle>& triangles);
};

#endif // DATASET_MANAGER_H
