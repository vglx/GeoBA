#include "DatasetManager.h"
#include "Optimizer.h"
#include "MeshModel.h"
#include "ImageProcessor.h"
#include "EDGraph.h"
#include <iostream>
#include "Evaluation.h"

int main() {
    std::cout << "GeoBA System Starting with Dataset...\n";

    int sampling_interval = 1;  // 取样间隔，可调节

    // **1. 初始化数据集管理器**
    DatasetManager dataset_manager("../data/sim_rectum/");

    // **2. 准备 Optimizer**
    Optimizer optimizer(1, 10, 1);

    std::vector<cv::Mat> rgb_images;
    std::vector<Eigen::Matrix4d> gt_camera_poses;
    std::vector<Eigen::Matrix4d> camera_poses;
    std::vector<Eigen::Matrix4d> opt_camera_poses;
    Eigen::Matrix3d camera_intrinsics;
    MeshModel mesh_model;

    // **3. 加载网格模型**
    if (!dataset_manager.loadMeshModel(mesh_model)) {
        std::cerr << "Failed to load global mesh model.\n";
        return -1;
    }
    std::cout << "Loaded mesh with " << mesh_model.getVertices().size() << " vertices and " 
              << mesh_model.getTriangles().size() << " triangles.\n";

    // **4. 构建 EDGraph 控制节点并绑定顶点**
    EDGraph edGraph(4);  // 每顶点绑定 4 个最近节点
    int sampling_step = 10;  // 控制稀疏程度
    edGraph.initializeGraph(mesh_model.getVertices(), sampling_step);


    // **5. 加载 RGB 图像**
    if (!dataset_manager.loadAllRGBImages(rgb_images)) {
        std::cerr << "Failed to load RGB images.\n";
        return -1;
    }

    // **6. 加载相机内参**
    if (!dataset_manager.loadCameraIntrinsics(camera_intrinsics)) {
        std::cerr << "Failed to load camera intrinsics.\n";
        return -1;
    }

    // **7. 加载相机位姿**
    if (!dataset_manager.loadPoses(gt_camera_poses, "poses_gt")) {
        std::cerr << "Failed to load ground truth poses.\n";
        return -1;
    }
    if (!dataset_manager.loadPoses(camera_poses, "poses_init")) {
        std::cerr << "Failed to load initialized poses.\n";
        return -1;
    }

    if (rgb_images.size() != camera_poses.size() || rgb_images.size() != gt_camera_poses.size()) {
        std::cerr << "Mismatch in dataset sizes (RGB, Poses, GT Poses).\n";
        return -1;
    }

    std::cout << "Loaded " << rgb_images.size() << " frames from dataset.\n";

    // **8. 采样帧**
    std::vector<cv::Mat> sampled_rgb_images;
    std::vector<Eigen::Matrix4d> sampled_gt_camera_poses;
    std::vector<Eigen::Matrix4d> sampled_camera_poses;
    for (size_t i = 0; i < rgb_images.size(); i += sampling_interval) {
        sampled_rgb_images.push_back(rgb_images[i]);
        sampled_camera_poses.push_back(camera_poses[i]);
        sampled_gt_camera_poses.push_back(gt_camera_poses[i]);
    }
    std::cout << "Sampled " << sampled_rgb_images.size() << " frames with interval " 
              << sampling_interval << ".\n";

    opt_camera_poses = sampled_camera_poses;

    // **9. 运行优化（联合 pose + EDGraph + intensity）**
    std::cout << "Start optimization...\n";
    optimizer.optimize(
        mesh_model.getVertices(),
        mesh_model.getTriangles(),
        camera_intrinsics,
        sampled_rgb_images,
        opt_camera_poses,
        edGraph      // ✅ 传入 EDGraph
    );
    std::cout << "Optimization complete.\n";

    // **10. 评估结果**
    Evaluation::ComputeRMSE(sampled_gt_camera_poses, sampled_camera_poses, opt_camera_poses);

    return 0;
}