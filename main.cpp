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
    std::string dataset_root = ".";
    int    sampling_interval = 1;
    int    max_frames   = -1;

    // EDGraph sampling params
    EDGraph::SamplingMode mode = EDGraph::SamplingMode::Voxel;
    double voxel_size = 2.0; // mm
    int    fps_target = 1500;
    int    stride     = 8;
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

    DatasetManager dataset_manager;

    // ---- load mesh
    std::vector<MeshModel::Vertex> V;
    std::vector<MeshModel::Triangle> F;
    if (!dataset_manager.loadMesh(args.dataset_root+"/mesh.obj", V, F)) {
        std::cerr << "[main] Failed to load mesh" << std::endl;
        return -1;
    }
    std::cout << "[main] Mesh: V=" << V.size() << ", F=" << F.size() << std::endl;

    // ---- EDGraph build
    EDGraph edGraph;
    bool ok = false;
    if (args.mode == EDGraph::SamplingMode::Voxel) {
        ok = edGraph.initializeFromMeshVoxel(V, F, args.voxel_size, args.neighborK, args.K_bind);
    } else if (args.mode == EDGraph::SamplingMode::FPS) {
        ok = edGraph.initializeFromMeshFPS(V, F, args.fps_target, args.neighborK, args.K_bind);
    } else {
        ok = edGraph.initializeFromMeshStride(V, F, args.stride, args.neighborK, args.K_bind);
    }
    if (!ok) {
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

    // ---- poses: load once (center camera T_cw) and derive left/right using baseline
    std::vector<Eigen::Matrix4d> poses_center;
    bool poses_ok = dataset_manager.loadPoses(poses_center, "poses_center");
    if (!poses_ok) {
        // fallback: try default (no key) if your DatasetManager supports it
        poses_ok = dataset_manager.loadPoses(poses_center);
    }
    if (!poses_ok) {
        std::cerr << "[main] Failed to load center poses" << std::endl;
        return -1;
    }
    if (poses_center.size() != rgb_left.size()) {
        std::cerr << "[main] Mismatch: center poses (" << poses_center.size()
                  << ") vs stereo RGB frames (" << rgb_left.size() << ")" << std::endl;
        return -1;
    }

    // Baseline in mm (already unified to mm). Center-based rig: left at +b/2 on x, right at -b/2 on x in camera frame.
    const double b_mm = 1.5;                 // full baseline
    const double half_b = b_mm * 0.5;        // 0.75 mm
    const Eigen::Vector3d tC_L(+half_b, 0.0, 0.0); // translation from Left to Center, in Center frame
    const Eigen::Vector3d tC_R(-half_b, 0.0, 0.0); // translation from Right to Center, in Center frame

    // Precompute left/right T_cw from center T_cw:
    std::vector<Eigen::Matrix4d> poses_left, poses_right;
    poses_left.resize(poses_center.size());
    poses_right.resize(poses_center.size());
    for (size_t i = 0; i < poses_center.size(); ++i) {
        const Eigen::Matrix4d& Tcw = poses_center[i];
        Eigen::Matrix3d R = Tcw.block<3,3>(0,0);
        Eigen::Vector3d t = Tcw.block<3,1>(0,3);

        // T_lw = [R | t + R * tC_L];  T_rw = [R | t + R * tC_R]
        Eigen::Matrix4d Tlw = Eigen::Matrix4d::Identity();
        Tlw.block<3,3>(0,0) = R;
        Tlw.block<3,1>(0,3) = t + R * tC_L;

        Eigen::Matrix4d Trw = Eigen::Matrix4d::Identity();
        Trw.block<3,3>(0,0) = R;
        Trw.block<3,1>(0,3) = t + R * tC_R;

        poses_left[i]  = Tlw;
        poses_right[i] = Trw;
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