#include "DatasetManager.h"
#include "Optimizer.h"
#include "MeshModel.h"
#include "ImageProcessor.h"
#include <iostream>
#include "Evaluation.h"

int main() {
    std::cout << "GeoBA System Starting with Dataset...\n";

    // **1. 初始化数据集管理器**
    // DatasetManager dataset_manager("../data/Triplettest/");
    DatasetManager dataset_manager("../data/sim_rectum/");

    Optimizer optimizer(1, 10);

    std::vector<cv::Mat> rgb_images;
    std::vector<Eigen::Matrix4d> gt_camera_poses;
    std::vector<Eigen::Matrix4d> camera_poses;
    std::vector<Eigen::Matrix4d> opt_camera_poses;
    Eigen::Matrix3d camera_intrinsics;
    cv::Mat LR_imgs, MR_imgs, HR_imgs;
    MeshModel mesh_model;

    // **2. 加载网格模型**
    if (!dataset_manager.loadMeshModel(mesh_model)) {
        std::cerr << "Failed to load global mesh model.\n";
        return -1;
    }
    std::cout << "Loaded mesh with " << mesh_model.getVertices().size() << " vertices and " 
              << mesh_model.getTriangles().size() << " triangles.\n";

    
    if (!dataset_manager.loadAllRGBImages(rgb_images)) {
        std::cerr << "Failed to load RGB images.\n";
        return -1;
    }

    // **4. 加载相机内参**
    if (!dataset_manager.loadCameraIntrinsics(camera_intrinsics)) {
        std::cerr << "Failed to load camera intrinsics.\n";
        return -1;
    }

    // **5. 加载相机位姿**
    if (!dataset_manager.loadPoses(gt_camera_poses, "poses_gt")) {
        std::cerr << "Failed to load ground truth poses.\n";
        return -1;
    }

    // **5. 加载相机位姿**
    if (!dataset_manager.loadPoses(camera_poses, "poses_init")) {
        std::cerr << "Failed to load initialized poses.\n";
        return -1;
    }

    if (rgb_images.size() != camera_poses.size() || rgb_images.size() != gt_camera_poses.size()) {
        std::cerr << "Mismatch in dataset sizes (RGB, Poses, GT Poses).\n";
        return -1;
    }

    std::cout << "Loaded " << rgb_images.size() << " frames from dataset.\n";

    opt_camera_poses = camera_poses;

    LR_imgs = ImageProcessor::applyGaussianBlur(rgb_images, 3, 1.5);
    MR_imgs = ImageProcessor::applyGaussianBlur(rgb_images, 3, 1.0);
    HR_imgs = ImageProcessor::applyGaussianBlur(rgb_images, 3, 0.5);

    // **9. 运行优化**
    std::cout << "Start optimization.\n";

    std::cout << "Optimization with Low Resolution Images ----->\n";

    optimizer.optimize(mesh_model.getVertices(), mesh_model.getTriangles(), camera_intrinsics, LR_imgs, opt_camera_poses);

    Evaluation::ComputeRMSE(gt_camera_poses, camera_poses, opt_camera_poses);

    std::cout << "Optimization with Medium Resolution Images ----->\n";

    optimizer.optimize(mesh_model.getVertices(), mesh_model.getTriangles(), camera_intrinsics, MR_imgs, opt_camera_poses);

    Evaluation::ComputeRMSE(gt_camera_poses, camera_poses, opt_camera_poses);

    std::cout << "Optimization with Hign Resolution Images ----->\n";

    optimizer.optimize(mesh_model.getVertices(), mesh_model.getTriangles(), camera_intrinsics, HR_imgs, opt_camera_poses);

    std::cout << "Optimization complete.\n";

    Evaluation::ComputeRMSE(gt_camera_poses, camera_poses, opt_camera_poses);

    return 0;
}
