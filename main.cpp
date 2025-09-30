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
// Updated main: Stereo Photometric (Left+Right RGB only, no depth)
// -----------------------------------------------------------------------------

struct Args {
    std::string dataset_root = "../data/halfDef/6/";
    int sampling_interval = 1;
    int max_frames = 0;

    // EDGraph params
    EDGraph::SamplingMode mode = EDGraph::SamplingMode::FPS;
    int    stride      = 30;
    double voxel_size  = 3.0;
    int    fps_target  = 1500;
    int    neighborK   = 3;
    int    K_bind      = 3;

    // Optimizer weights
    double w_photo = 0.3;           // monocular photometric
    double w_stereo= 0.3;           // stereo photometric
    double lambda_smooth = 0.01;
    double lambda_rot    = 0.01;
    double lambda_temp   = 0.0;
    int    maxStages     = 1;
    int    maxIterations = 5;
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
        else if (!std::strcmp(a, "--wstereo"))     args.w_stereo     = nextd(args.w_stereo);
        else if (!std::strcmp(a, "--lsmooth"))     args.lambda_smooth= nextd(args.lambda_smooth);
        else if (!std::strcmp(a, "--lrot"))        args.lambda_rot   = nextd(args.lambda_rot);
        else if (!std::strcmp(a, "--ltemp"))       args.lambda_temp  = nextd(args.lambda_temp);
        else if (!std::strcmp(a, "--iters"))       args.maxIterations= nexti(args.maxIterations);
    }
}

int main(int argc, char** argv) {
    std::cout << "==== GeoBA (Stereo Photometric Only, Poses Fixed) ====\n";
    parse_cli(argc, argv);

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

    // ---- build ED graph
    EDGraph edGraph(args.K_bind, args.neighborK);
    EDGraph::BuildParams p;
    p.mode       = args.mode;
    p.stride     = args.stride;
    p.voxel_size = args.voxel_size;
    p.fps_target = args.fps_target;
    p.K_bind     = args.K_bind;
    p.neighborK  = args.neighborK;

    if (!edGraph.initializeGraph(V, p, true)) {
        std::cerr << "[main] Failed to initialize EDGraph" << std::endl;
        return -1;
    }
    std::cout << "[main] EDGraph: nodes=" << edGraph.numNodes()
              << ", K_bind=" << args.K_bind
              << ", neighborK=" << args.neighborK << std::endl;

    // ---- load left/right RGB sequences
    std::vector<cv::Mat> rgb_left, rgb_right;
    if (!dataset_manager.loadAllRGB(args.dataset_root + "/rgb_left", rgb_left)) {
        std::cerr << "[main] Failed to load left RGB images" << std::endl;
        return -1;
    }
    if (!dataset_manager.loadAllRGB(args.dataset_root + "/rgb_right", rgb_right)) {
        std::cerr << "[main] Failed to load right RGB images" << std::endl;
        return -1;
    }
    if (rgb_left.size() != rgb_right.size()) {
        std::cerr << "[main] Mismatch: left RGB (" << rgb_left.size()
                  << ") vs right RGB (" << rgb_right.size() << ")" << std::endl;
        return -1;
    }
    std::cout << "[main] Loaded " << rgb_left.size() << " stereo RGB frame pairs" << std::endl;

    // ---- intrinsics (assume separate files for left/right)
    Eigen::Matrix3d K_left, K_right;
    if (!dataset_manager.loadCameraIntrinsics(K_left)) { // adjust to left.json if separated
        std::cerr << "[main] Failed to load left intrinsics" << std::endl;
        return -1;
    }
    K_right = K_left; // TODO: load separately if available

    // ---- poses (assume only left provided, right derived via extrinsics)
    std::vector<Eigen::Matrix4d> poses_left;
    if (!dataset_manager.loadPoses(poses_left, "poses_left")) {
        std::cerr << "[main] Failed to load left poses" << std::endl;
        return -1;
    }
    std::vector<Eigen::Matrix4d> poses_right;
    if (!dataset_manager.loadPoses(poses_right, "poses_right")) {
        std::cerr << "[main] Failed to load right poses" << std::endl;
        return -1;
    }
    if (poses_left.size() != poses_right.size() || rgb_left.size()!=rgb_right.size()) {
        std::cerr << "[main] Mismatch between frames and poses" << std::endl;
        return -1;
    }

    // ---- sampling
    std::vector<cv::Mat> Ls, Rs;
    std::vector<Eigen::Matrix4d> posesL, posesR;
    for (size_t i=0;i<rgb_left.size(); i+=args.sampling_interval) {
        Ls.push_back(rgb_left[i]);
        Rs.push_back(rgb_right[i]);
        posesL.push_back(poses_left[i]);
        posesR.push_back(poses_right[i]);
        if (args.max_frames>0 && (int)Ls.size()>=args.max_frames) break;
    }
    if (Ls.size()<2) {
        std::cerr << "[main] Need at least 2 stereo frames" << std::endl;
        return -1;
    }

    // ---- optimizer
    Optimizer optimizer(args.w_photo, args.w_stereo,
                        args.maxStages, args.maxIterations,
                        args.lambda_smooth, args.lambda_rot, args.lambda_temp);

    std::cout << "[main] Start optimization...\n";
    optimizer.optimize(
        V, F,
        K_left, K_right,
        Ls, Rs,
        posesL, posesR,
        edGraph,
        [&](int f,const EDGraph& g){
            dataset_manager.saveMeshAsPLY(
                args.dataset_root+"/results/PLYs/deformed_mesh_f"+std::to_string(f)+".ply",
                V,F);
        });

    std::cout << "[main] Optimization complete." << std::endl;
    return 0;
}