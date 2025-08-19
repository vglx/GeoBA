#include "DatasetManager.h"
#include "Optimizer.h"
#include "MeshModel.h"
#include "ImageProcessor.h"
#include "EDGraph.h"
#include <iostream>
// #include "Evaluation.h"  // 不再评估位姿RMSE，若有其它评估可自行保留

int main() {
    std::cout << "GeoBA System Starting with Dataset...\n";

    int sampling_interval = 1;  // 可调

    // 1) 数据集
    DatasetManager dataset_manager("../data/sim_rectum/");

    // 2) Optimizer（固定位姿 + 仿射ED）
    //    这里的 lambda_smooth / lambda_rot 可按需调
    Optimizer optimizer(/*w_data=*/1.0, /*maxStages=*/10, /*maxIterations=*/1,
                        /*lambda_smooth=*/1.0, /*lambda_rot=*/0.1);

    std::vector<cv::Mat> rgb_images;
    std::vector<Eigen::Matrix4d> gt_camera_poses;
    Eigen::Matrix3d camera_intrinsics;
    MeshModel mesh_model;

    // 3) 网格
    if (!dataset_manager.loadMeshModel(mesh_model)) {
        std::cerr << "Failed to load global mesh model.\n";
        return -1;
    }
    std::cout << "Loaded mesh with " << mesh_model.getVertices().size() << " vertices and "
              << mesh_model.getTriangles().size() << " triangles.\n";

    // 4) EDGraph（仿射）初始化与绑定
    EDGraph edGraph(/*K=*/4);
    int sampling_step = 10;
    edGraph.initializeGraph(mesh_model.getVertices(), sampling_step);

    // 5) 图像
    if (!dataset_manager.loadAllRGBImages(rgb_images)) {
        std::cerr << "Failed to load RGB images.\n";
        return -1;
    }

    // 6) 相机内参
    if (!dataset_manager.loadCameraIntrinsics(camera_intrinsics)) {
        std::cerr << "Failed to load camera intrinsics.\n";
        return -1;
    }

    // 7) 仅加载 GT 位姿（优化中固定使用）
    if (!dataset_manager.loadPoses(gt_camera_poses, "poses_gt")) {
        std::cerr << "Failed to load ground truth poses.\n";
        return -1;
    }

    if (rgb_images.size() != gt_camera_poses.size()) {
        std::cerr << "Mismatch in dataset sizes (RGB vs GT Poses).\n";
        return -1;
    }
    std::cout << "Loaded " << rgb_images.size() << " frames from dataset.\n";

    // 8) 采样
    std::vector<cv::Mat> sampled_rgb_images;
    std::vector<Eigen::Matrix4d> sampled_gt_camera_poses;
    for (size_t i = 0; i < rgb_images.size(); i += sampling_interval) {
        sampled_rgb_images.push_back(rgb_images[i]);
        sampled_gt_camera_poses.push_back(gt_camera_poses[i]);
    }
    std::cout << "Sampled " << sampled_rgb_images.size() << " frames with interval "
              << sampling_interval << ".\n";

    // 9) 仅以 GT 位姿运行优化（不再传/维护 init/opt poses）
    std::cout << "Start optimization...\n";
    optimizer.optimize(
        mesh_model.getVertices(),
        mesh_model.getTriangles(),
        camera_intrinsics,
        sampled_rgb_images,
        sampled_gt_camera_poses,  // ✅ 固定的 GT 位姿
        edGraph
    );
    std::cout << "Optimization complete.\n";

    // 10) 若需要，可在此导出变形后的网格或统计光度误差等
    // Evaluation::... (此处不再做位姿 RMSE，因未优化位姿)

    return 0;
}