#include "DatasetManager.h"
#include "Optimizer.h"
#include "MeshModel.h"
#include "EDGraph.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>

// -----------------------------------------------------------------------------
// This main wires the pipeline for the "frame 0 as template intensity" variant.
// - Frame 0 is used ONLY to sample per‑vertex template intensity (fixed).
// - Data terms start from frame 1 (>=1) and optimize only ED for those frames.
// - Poses are fixed from GT; no pose optimization.
// -----------------------------------------------------------------------------

struct Args {
    std::string dataset_root = "../data/halfDef/1/";
    int sampling_interval = 1;      // sample every k frames
    int max_frames = 0;             // 0 = use all after sampling; >0 = cap after sampling

    // EDGraph params
    EDGraph::SamplingMode mode = EDGraph::SamplingMode::FPS;
    int    stride      = 30;        // Stride only
    double voxel_size  = 3.0;       // Voxel only (model units)
    int    fps_target  = 1500;      // FPS only
    int    neighborK   = 3;         // graph smoothness neighborhood size
    int    K_bind      = 3;         // KNN bindings per vertex
} args;

static void parse_cli(int argc, char** argv) {
    // Usage examples:
    //   ./GeoBA /path/to/dataset --mode voxel --voxel 0.02 --interval 3 --Kbind 3 --neighborK 6
    //   ./GeoBA /path/to/dataset --mode fps   --fps 1200
    //   ./GeoBA /path/to/dataset --mode stride --stride 20
    //   ./GeoBA /path/to/dataset --limit 2   (use only 2 frames after sampling)
    if (argc > 1 && argv[1][0] != '-') {
        args.dataset_root = argv[1];
    }
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto next  = [&](double def)->double{ return (i+1<argc? std::atof(argv[++i]) : def); };
        auto nexti = [&](int def)->int{ return (i+1<argc? std::atoi(argv[++i]) : def); };
        if      (!std::strcmp(a, "--interval"))   args.sampling_interval = nexti(args.sampling_interval);
        else if (!std::strcmp(a, "--limit"))      args.max_frames        = nexti(args.max_frames);
        else if (!std::strcmp(a, "--mode")) {
            if (i+1<argc) {
                const char* m = argv[++i];
                if      (!std::strcmp(m, "voxel"))  args.mode = EDGraph::SamplingMode::Voxel;
                else if (!std::strcmp(m, "fps"))    args.mode = EDGraph::SamplingMode::FPS;
                else if (!std::strcmp(m, "stride")) args.mode = EDGraph::SamplingMode::Stride;
            }
        }
        else if (!std::strcmp(a, "--voxel"))     args.voxel_size = next(args.voxel_size);
        else if (!std::strcmp(a, "--fps"))       args.fps_target = nexti(args.fps_target);
        else if (!std::strcmp(a, "--stride"))    args.stride     = nexti(args.stride);
        else if (!std::strcmp(a, "--neighborK")) args.neighborK  = nexti(args.neighborK);
        else if (!std::strcmp(a, "--Kbind"))     args.K_bind     = nexti(args.K_bind);
    }
}

int main(int argc, char** argv) {
    std::cout << "==== GeoBA (Frame0 Template Intensities + Affine ED, Poses Fixed) ====\n";
    parse_cli(argc, argv);

    // ---- dataset manager
    DatasetManager dataset_manager(args.dataset_root);

    // ---- mesh
    MeshModel mesh_model;
    if (!dataset_manager.loadMeshModel(mesh_model)) {
        std::cerr << "[main] Failed to load mesh model from: " << args.dataset_root << std::endl;
        return -1;
    }
    const auto& V = mesh_model.getVertices();
    const auto& F = mesh_model.getTriangles();
    std::cout << "[main] Mesh: " << V.size() << " vertices, " << F.size() << " triangles" << std::endl;

    // ---- build affine ED graph (nodes+bindings+neighbors)
    EDGraph edGraph(/*K=*/args.K_bind, /*neighborK=*/args.neighborK);
    EDGraph::BuildParams p;
    p.mode       = args.mode;
    p.stride     = args.stride;
    p.voxel_size = args.voxel_size;
    p.fps_target = args.fps_target;
    p.K_bind     = args.K_bind;
    p.neighborK  = args.neighborK;

    if (!edGraph.initializeGraph(V, p, /*build_neighbors=*/true)) {
        std::cerr << "[main] Failed to initialize EDGraph" << std::endl;
        return -1;
    }
    std::cout << "[main] EDGraph: nodes=" << edGraph.numNodes()
              << ", K_bind=" << args.K_bind
              << ", neighborK=" << args.neighborK << std::endl;

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

    // ---- poses: use ONLY ground‑truth (fixed during optimization)
    std::vector<Eigen::Matrix4d> gt_camera_poses;
    if (!dataset_manager.loadPoses(gt_camera_poses, "poses_gt")) {
        std::cerr << "[main] Failed to load ground‑truth poses" << std::endl;
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
    sampled_images.reserve((rgb_images.size() + args.sampling_interval - 1) / args.sampling_interval);
    sampled_gt_poses.reserve(sampled_images.capacity());

    for (size_t i = 0; i < rgb_images.size(); i += args.sampling_interval) {
        sampled_images.push_back(rgb_images[i]);
        sampled_gt_poses.push_back(gt_camera_poses[i]);
    }

    if (args.max_frames > 0 && (int)sampled_images.size() > args.max_frames) {
        sampled_images.resize(args.max_frames);
        sampled_gt_poses.resize(args.max_frames);
    }

    std::cout << "[main] Sampled " << sampled_images.size()
              << " frames (interval=" << args.sampling_interval
              << (args.max_frames>0? ", limit="+std::to_string(args.max_frames):"")
              << ")" << std::endl;

    if (sampled_images.size() < 2) {
        std::cerr << "[main] Need at least 2 frames (frame 0 = template, frame 1 = optimized)." << std::endl;
        return -1;
    }

    // ---- optimizer (data + smooth + rotation + optional temporal)
    const double w_data        = 1.0;   // photometric weight
    const int    maxStages     = 6;     // outer stages (kept for compatibility)
    const int    maxIterations = 20;     // inner GN iters per stage
    const double lambda_smooth = 0.23;  // spatial smoothness between neighbor nodes
    const double lambda_rot    = 0.52;  // rotation (A^T A - I)

    Optimizer optimizer(w_data, maxStages, maxIterations, lambda_smooth, lambda_rot);
    optimizer.setTemporalWeight(0.0);   // set >0 to enable temporal consistency between (f-1,f) when both have variables

    std::cout << "[main] Start optimization...\n"
                 "[main] Note: frame 0 is used ONLY to sample template intensities;\n"
                 "              no data term / variables are created for frame 0.\n";

    optimizer.optimize(
        V,
        F,
        K,
        sampled_images,
        sampled_gt_poses,   // fixed GT poses (not optimized)
        edGraph
    );

    std::cout << "[main] Optimization complete." << std::endl;

    dataset_manager.saveDeformedMeshAsPLY(args.dataset_root + "/deformed_mesh_f1.ply",
                                          mesh_model,
                                          edGraph /* already updated by optimizer */);

    return 0;
}