#include "DatasetManager.h"
#include "Optimizer.h"
#include "MeshModel.h"
#include "ImageProcessor.h"
#include <iostream>
#include "Evaluation.h"
#include <fstream>

void saveIntensityValues(const std::vector<double>& x2_values, const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing x2 values.\n";
        return;
    }
    for (double val : x2_values) {
        ofs << val << "\n";
    }
    ofs.close();
    std::cout << "Saved " << x2_values.size() << " x2 values to " << filename << "\n";
}

std::vector<double> loadIntensityValues(const std::string& filename) {
    std::vector<double> values;
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open " << filename << " for reading x2 values.\n";
        return values;
    }
    double val;
    while (ifs >> val) {
        values.push_back(val);
    }
    std::cout << "Loaded " << values.size() << " photometric x2 values from " << filename << "\n";
    return values;
}

int main() {
    std::cout << "GeoBA System Starting with Dataset...\n";

    int sampling_interval = 1;  // 取样间隔，可调节

    // **1. 初始化数据集管理器**
    // DatasetManager dataset_manager("../data/test/");
    DatasetManager dataset_manager("../data/sim_rectum/");

    Optimizer optimizer(1);  // 传入误差权重（可调节）

    std::vector<cv::Mat> rgb_images;
    std::vector<Eigen::Matrix4d> gt_camera_poses;
    std::vector<Eigen::Matrix4d> camera_poses;
    std::vector<Eigen::Matrix4d> opt_camera_poses;
    Eigen::Matrix3d camera_intrinsics;
    std::vector<cv::Mat> LR_imgs, MR_imgs, HR_imgs;
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

    // 采样数据
    std::vector<cv::Mat> sampled_rgb_images;
    std::vector<Eigen::Matrix4d> sampled_gt_camera_poses;
    std::vector<Eigen::Matrix4d> sampled_camera_poses;

    for (size_t i = 0; i < rgb_images.size(); i += sampling_interval) {
        sampled_rgb_images.push_back(rgb_images[i]);
        sampled_camera_poses.push_back(camera_poses[i]);
        sampled_gt_camera_poses.push_back(gt_camera_poses[i]);
    }

    std::cout << "Sampled " << sampled_rgb_images.size() << " frames with interval " << sampling_interval << ".\n";

    opt_camera_poses = sampled_camera_poses;

    std::vector<double> x2_values;

    std::cout << "Stage 1: Optimize Photometry Only\n";
    optimizer.optimizePhotometryOnly(mesh_model.getVertices(), mesh_model.getTriangles(), camera_intrinsics, LR_imgs, sampled_gt_camera_poses, x2_values);
    saveIntensityValues(x2_values, "x2_photometry_only.txt");

    std::cout << "Stage 2: Optimize With Initial Photometry\n";
    optimizer.optimizeWithInitialPhotometry(mesh_model.getVertices(), mesh_model.getTriangles(), camera_intrinsics, LR_imgs, opt_camera_poses, x2_values);
    saveIntensityValues(x2_values, "x2_final.txt");

    // 图像预处理
    // LR_imgs = ImageProcessor::applyGammaCorrection(sampled_rgb_images, 0.8);
    // LR_imgs = ImageProcessor::applyGaussianBlur(LR_imgs, 3, 1.5);
    // MR_imgs = ImageProcessor::applyCLAHE(sampled_rgb_images);
    // MR_imgs = ImageProcessor::applyGaussianBlur(MR_imgs, 3, 1.0);
    // HR_imgs = ImageProcessor::applyGammaCorrection(sampled_rgb_images, 0.9);
    // HR_imgs = ImageProcessor::applyGaussianBlur(HR_imgs, 3, 0.5);

    // LR_imgs = ImageProcessor::applyGaussianBlur(sampled_rgb_images, 3, 1.5);
    // MR_imgs = ImageProcessor::applyGaussianBlur(sampled_rgb_images, 3, 1.0);
    // HR_imgs = ImageProcessor::applyGaussianBlur(sampled_rgb_images, 3, 0.5);

    // **9. 运行优化**
    // std::cout << "Start optimization.\n";
    
    // optimizer.optimize(mesh_model.getVertices(), mesh_model.getTriangles(), camera_intrinsics, sampled_rgb_images, opt_camera_poses);

    // std::cout << "Optimization completed.\n";

    // Evaluation::ComputeRMSE(sampled_gt_camera_poses, sampled_camera_poses, opt_camera_poses);

    // std::cout << "Start optimization with Low Resolution Images.\n";
    // optimizer.optimize(mesh_model.getVertices(), mesh_model.getTriangles(), camera_intrinsics, LR_imgs, opt_camera_poses);
    // Evaluation::ComputeRMSE(sampled_gt_camera_poses, sampled_camera_poses, opt_camera_poses);

    // std::cout << "Start optimization with Medium Resolution Images.\n";
    // optimizer.optimize(mesh_model.getVertices(), mesh_model.getTriangles(), camera_intrinsics, MR_imgs, opt_camera_poses);
    // Evaluation::ComputeRMSE(sampled_gt_camera_poses, sampled_camera_poses, opt_camera_poses);

    // std::cout << "Start optimization with High Resolution Images.\n";
    // optimizer.optimize(mesh_model.getVertices(), mesh_model.getTriangles(), camera_intrinsics, HR_imgs, opt_camera_poses);
    // Evaluation::ComputeRMSE(sampled_gt_camera_poses, sampled_camera_poses, opt_camera_poses);

    std::cout << "Optimization Complete.\n";

    return 0;
}
