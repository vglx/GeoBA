#include "DatasetManager.h"
#include "Optimizer.h"
#include "MeshModel.h"
#include "ImageProcessor.h"
#include <iostream>
#include "Evaluation.h"

int main() {
    std::cout << "GeoBA System Starting with Dataset...\n";

    // **1. 初始化数据集管理器**
    DatasetManager dataset_manager("../data/Triplettest/");

    std::vector<cv::Mat> rgb_images;
    std::vector<cv::Mat> depth_images;
    std::vector<Eigen::Matrix4d> gt_camera_poses;
    std::vector<Eigen::Matrix4d> camera_poses;
    std::vector<Eigen::Matrix4d> opt_camera_poses;
    Eigen::Matrix3d camera_intrinsics;
    MeshModel mesh_model;
    ImageProcessor image_processor;

    // **2. 加载网格模型**
    if (!dataset_manager.loadMeshModel(mesh_model)) {
        std::cerr << "Failed to load global mesh model.\n";
        return -1;
    }
    std::cout << "Loaded mesh with " << mesh_model.getVertices().size() << " vertices and " 
              << mesh_model.getTriangles().size() << " triangles.\n";

    // **3. 加载 RGB 和深度图像**
    if (!dataset_manager.loadAllRGBImages(rgb_images) || !dataset_manager.loadAllDepthImages(depth_images)) {
        std::cerr << "Failed to load RGB or depth images.\n";
        return -1;
    }

    // double minVal1, maxVal1;
    // cv::minMaxLoc(depth_images[2], &minVal1, &maxVal1);
    // std::cout << "depth 2: min = " << minVal1 << ", max = " << maxVal1 << std::endl;

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

    // **6. 确保数据一致**
    if (rgb_images.size() != depth_images.size() || rgb_images.size() != camera_poses.size() || rgb_images.size() != gt_camera_poses.size()) {
        std::cerr << "Mismatch in dataset sizes (RGB, Depth, Poses, GT Poses).\n";
        return -1;
    }

    std::cout << "Loaded " << rgb_images.size() << " frames from dataset.\n";

    // **7. 预处理深度图**
    std::vector<Eigen::MatrixXf> depth_normals;
    for (size_t i = 0; i < depth_images.size(); ++i) {
        // 过滤无效深度值
        image_processor.filterInvalidDepth(depth_images[i], 0.1f, 100.0f);
        // 计算深度图法向量
        Eigen::MatrixXf normals = image_processor.computeDepthNormals(
            depth_images[i], camera_intrinsics(0, 0), camera_intrinsics(1, 1), camera_intrinsics(0, 2), camera_intrinsics(1, 2));
        depth_normals.push_back(normals);
        double minVal1, maxVal1;
        cv::minMaxLoc(depth_images[i], &minVal1, &maxVal1);
    }
    std::cout << "Computed depth map normals for " << depth_normals.size() << " frames.\n";

    // **8. 计算网格顶点法向量**
    mesh_model.computeVertexNormals();
    std::cout << "Computed vertex normals for mesh model.\n";

    opt_camera_poses = camera_poses;

    // **9. 运行优化**
    std::cout << "Start optimization.\n";
    Optimizer optimizer(1);  // 传入误差权重（可调节）
    optimizer.optimize(mesh_model.getVertices(), mesh_model.getTriangles(), camera_intrinsics, rgb_images, opt_camera_poses);

    std::cout << "Optimization completed.\n";

    Evaluation::ComputeRMSE(gt_camera_poses, camera_poses, opt_camera_poses);

    return 0;
}
