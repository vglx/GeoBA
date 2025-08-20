#include "DatasetManager.h"
#include "Optimizer.h"
#include "MeshModel.h"
#include "EDGraph.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    std::cout << "==== GeoBA (Fixed Poses + Temporal Affine ED + Active Subgraph) ====\n";

    // ---- dataset root (optional argv[1])
    std::string dataset_root = "../data/sim_rectum/";
    if (argc > 1) dataset_root = argv[1];

    // ---- controls
    const int sampling_interval   = 1;   // sample every k frames
    const int neighborK           = 8;   // graph smoothness neighborhood size
    const int K_bind              = 4;   // KNN bindings per vertex
    const int sampling_step_nodes = 10;  // node sampling step when building ED graph

    // ---- dataset manager
    DatasetManager dataset_manager(dataset_root);

    // ---- mesh
    MeshModel mesh_model;
    if (!dataset_manager.loadMeshModel(mesh_model)) {
        std::cerr << "[main] Failed to load mesh model from: " << dataset_root << std::endl;
        return -1;
    }
    const auto& V = mesh_model.getVertices();
    const auto& F = mesh_model.getTriangles();
    std::cout << "[main] Mesh: " << V.size() << " vertices, " << F.size() << " triangles" << std::endl;

    // ---- build affine ED graph (nodes+bindings+neighbors)
    EDGraph edGraph(/*K=*/K_bind);
    edGraph.initializeGraph(V, sampling_step_nodes, /*build_neighbors=*/true);
    edGraph.setNeighborsForSmoothing(neighborK);

    // ---- images (RGB) & intrinsics
    std::vector<cv::Mat> rgb_images;
    if (!dataset_manager.loadAllRGBImages(rgb_images)) {
        std::cerr << "[main] Failed to load RGB images" << std::endl;
        return -1;
    }
    std::cout << "[main] Loaded " << rgb_images.size() << " RGB frames" << std::endl;

    Eigen::Matrix3d K;
    if (!dataset_manager.loadCameraIntrinsics(K)) {
        std::cerr << "[main] Failed to load camera intrinsics" << std::endl;
        return -1;
    }

    // ---- poses: use ONLY ground-truth (fixed during optimization)
    std::vector<Eigen::Matrix4d> gt_camera_poses;
    if (!dataset_manager.loadPoses(gt_camera_poses, "poses_gt")) {
        std::cerr << "[main] Failed to load ground-truth poses" << std::endl;
        return -1;
    }

    if (rgb_images.size() != gt_camera_poses.size()) {
        std::cerr << "[main] Mismatch: RGB frames (" << rgb_images.size()
                  << ") vs GT poses (" << gt_camera_poses.size() << ")" << std::endl;
        return -1;
    }

    // ---- frame sampling
    std::vector<cv::Mat> sampled_images;
    std::vector<Eigen::Matrix4d> sampled_gt_poses;
    sampled_images.reserve((rgb_images.size() + sampling_interval - 1) / sampling_interval);
    sampled_gt_poses.reserve(sampled_images.capacity());

    for (size_t i = 0; i < rgb_images.size(); i += sampling_interval) {
        sampled_images.push_back(rgb_images[i]);
        sampled_gt_poses.push_back(gt_camera_poses[i]);
    }

    std::cout << "[main] Sampled " << sampled_images.size()
              << " frames (interval=" << sampling_interval << ")" << std::endl;

    // ---- optimizer (data + smooth + rotation + anchor + temporal)
    const double w_data        = 1.0;   // photometric weight
    const int    maxStages     = 10;    // outer stages (refit BVH, update anchors)
    const int    maxIterations = 3;     // inner GN iters per stage
    const double lambda_smooth = 1.0;   // spatial smoothness between neighbor nodes
    const double lambda_rot    = 0.1;   // rotation (A^T A - I)

    Optimizer optimizer(w_data, maxStages, maxIterations, lambda_smooth, lambda_rot);
    // new knobs from rewritten optimizer
    optimizer.setTemporalWeight(1.0);   // temporal consistency between adjacent frames
    // optimizer.setWindowSize(W);      // reserved: not used in this all-frames version

    std::cout << "[main] Start optimization..." << std::endl;
    optimizer.optimize(
        V,
        F,
        K,
        sampled_images,
        sampled_gt_poses,   // fixed GT poses (not optimized)
        edGraph
    );
    std::cout << "[main] Optimization complete." << std::endl;

    // (Optional) TODO: export deformed mesh / per-frame results
    return 0;
}