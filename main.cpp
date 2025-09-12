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
// Updated main for the combined Photometric (RGB) + Projective-ICP (Depth)
// optimizer API. Frame 0 is the template frame (no ED variables, no data term).
// Poses are fixed (GT). Intensities are fixed to template sampling in optimizer.
// -----------------------------------------------------------------------------

struct Args {
    std::string dataset_root = "../data/halfDef/5/";
    int sampling_interval = 1;      // sample every k frames
    int max_frames = 0;             // 0 = use all after sampling; >0 = cap after sampling

    // EDGraph params
    EDGraph::SamplingMode mode = EDGraph::SamplingMode::FPS;
    int    stride      = 30;        // Stride only
    double voxel_size  = 3.0;       // Voxel only (model units)
    int    fps_target  = 1500;      // FPS only
    int    neighborK   = 3;         // graph smoothness neighborhood size
    int    K_bind      = 3;         // KNN bindings per vertex

    // Optimizer weights
    double w_photo = 1.0;           // photometric term
    double w_icp   = 1.0;           // depth ICP term
    double lambda_smooth = 0.01;
    double lambda_rot    = 0.01;
    double lambda_temp   = 0.0;     // start disabled
    int    maxStages     = 1;       // kept for compatibility
    int    maxIterations = 50;      // GN iterations
} args;

static void parse_cli(int argc, char** argv) {
    if (argc > 1 && argv[1][0] != '-') {
        args.dataset_root = argv[1];
    }
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto nextd = [&](double def)->double{ return (i+1<argc? std::atof(argv[++i]) : def); };
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
        else if (!std::strcmp(a, "--voxel"))       args.voxel_size   = nextd(args.voxel_size);
        else if (!std::strcmp(a, "--fps"))         args.fps_target   = nexti(args.fps_target);
        else if (!std::strcmp(a, "--stride"))      args.stride       = nexti(args.stride);
        else if (!std::strcmp(a, "--neighborK"))   args.neighborK    = nexti(args.neighborK);
        else if (!std::strcmp(a, "--Kbind"))       args.K_bind       = nexti(args.K_bind);
        else if (!std::strcmp(a, "--wphoto"))      args.w_photo      = nextd(args.w_photo);
        else if (!std::strcmp(a, "--wicp"))        args.w_icp        = nextd(args.w_icp);
        else if (!std::strcmp(a, "--lsmooth"))     args.lambda_smooth= nextd(args.lambda_smooth);
        else if (!std::strcmp(a, "--lrot"))        args.lambda_rot   = nextd(args.lambda_rot);
        else if (!std::strcmp(a, "--ltemp"))       args.lambda_temp  = nextd(args.lambda_temp);
        else if (!std::strcmp(a, "--iters"))       args.maxIterations= nexti(args.maxIterations);
    }
}

int main(int argc, char** argv) {
    std::cout << "==== GeoBA (Combined Photometric RGB + Projective ICP Depth, Poses Fixed) ====\n";
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

    // ---- RGB images (for photometric term)
    std::vector<cv::Mat> rgb_images;
    if (!dataset_manager.loadAllRGBImages(rgb_images)) {
        std::cerr << "[main] Failed to load RGB images" << std::endl;
        return -1;
    }
    std::cout << "[main] Loaded " << rgb_images.size() << " RGB frames" << std::endl;

    // ---- depth images (CV_32F) & intrinsics
    std::vector<cv::Mat> depth_images;
    if (!dataset_manager.loadAllDepthImages(depth_images)) {
        std::cerr << "[main] Failed to load depth images" << std::endl;
        return -1;
    }
    std::cout << "[main] Loaded " << depth_images.size() << " depth frames" << std::endl;

    if (rgb_images.size() != depth_images.size()) {
        std::cerr << "[main] Mismatch: RGB frames (" << rgb_images.size()
                  << ") vs Depth frames (" << depth_images.size() << ")" << std::endl;
        return -1;
    }

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
        std::cerr << "[main] Mismatch: frames (" << rgb_images.size()
                  << ") vs GT poses (" << gt_camera_poses.size() << ")" << std::endl;
        return -1;
    }

    // ---- frame sampling (apply the same indices to RGB/Depth/Poses)
    std::vector<cv::Mat> sampled_rgbs, sampled_depths;
    std::vector<Eigen::Matrix4d> sampled_gt_poses;
    for (size_t i = 0; i < rgb_images.size(); i += args.sampling_interval) {
        sampled_rgbs.push_back(rgb_images[i]);
        sampled_depths.push_back(depth_images[i]);
        sampled_gt_poses.push_back(gt_camera_poses[i]);
        if (args.max_frames>0 && (int)sampled_rgbs.size() >= args.max_frames) break;
    }

    std::cout << "[main] Sampled " << sampled_rgbs.size()
              << " frames (interval=" << args.sampling_interval
              << (args.max_frames>0? ", limit="+std::to_string(args.max_frames):"")
              << ")" << std::endl;

    if (sampled_rgbs.size() < 2) {
        std::cerr << "[main] Need at least 2 frames (frame 0 = template, frame 1 = optimized)." << std::endl;
        return -1;
    }

    // ---- optimizer (data + smooth + rotation + optional temporal)
    Optimizer optimizer(args.w_photo, args.w_icp,
                        args.maxStages, args.maxIterations,
                        args.lambda_smooth, args.lambda_rot, args.lambda_temp);

    std::cout << "[main] Start optimization...\n";

    optimizer.optimize(
        V, F,                 // mesh vertices & triangles
        K,                    // intrinsics
        sampled_rgbs,         // RGB frames
        sampled_depths,       // depth frames (CV_32F, mm)
        sampled_gt_poses,     // GT poses (frame0 fixed)
        edGraph,
        [&](int f, const EDGraph& g){
            dataset_manager.saveDeformedMeshAsPLY(
                args.dataset_root + "results/PLYs/deformed_mesh_f" + std::to_string(f) + ".ply",
                mesh_model, g);
        }
    );

    std::cout << "[main] Optimization complete." << std::endl;

    return 0;
}

