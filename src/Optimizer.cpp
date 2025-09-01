#include "Optimizer.h"
#include "CostFunctions.h"
#include "Projection.h"
#include "BVH.h"  // kept only for PhotometricError ctor compatibility

#include <iostream>
#include <limits>
#include <cmath>
#include <omp.h>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

namespace {
inline float bilinearSample(const cv::Mat& img, float u, float v){
    int x = (int)std::floor(u), y = (int)std::floor(v);
    int x1 = x + 1, y1 = y + 1;
    if (x < 0 || y < 0 || x1 >= img.cols || y1 >= img.rows) return 0.f;
    float a = u - x, b = v - y;
    float I00 = img.at<float>(y, x);
    float I10 = img.at<float>(y, x1);
    float I01 = img.at<float>(y1, x);
    float I11 = img.at<float>(y1, x1);
    return (1-a)*(1-b)*I00 + a*(1-b)*I10 + (1-a)*b*I01 + a*b*I11;
}
}

Optimizer::Optimizer(double w_data, int maxStages, int maxIterations,
                     double lambda_smooth, double lambda_rot)
    : w_data_(w_data), lambda_smooth_(lambda_smooth), lambda_rot_(lambda_rot),
      maxStages_(maxStages), maxIterations_(maxIterations) {}

void Optimizer::optimize(const std::vector<MeshModel::Vertex>& mesh_vertices,
                         const std::vector<MeshModel::Triangle>& mesh_triangles,
                         const Eigen::Matrix3d& K,
                         const std::vector<cv::Mat>& observed_images,
                         const std::vector<Eigen::Matrix4d>& camera_poses_gt,
                         EDGraph& edGraph) {
    const int F = (int)observed_images.size();
    const int N = (int)mesh_vertices.size();
    const int G = edGraph.numNodes();
    if (F <= 1 || N == 0 || G == 0) return;

    // grayscale images
    std::vector<cv::Mat> imgs_gray; imgs_gray.reserve(F);
    for (const auto& img : observed_images) {
        cv::Mat gray;
        if (img.channels()==3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); else gray = img;
        gray.convertTo(gray, CV_32F, 1.0/255.0);
        imgs_gray.push_back(gray);
    }

    // ED state per frame (full 12*G per frame). Frame 0 will remain fixed (no columns)
    const int edDimPerFrameFull = 12 * G;
    std::vector<Eigen::VectorXd> Xfull(F, Eigen::VectorXd::Zero(edDimPerFrameFull));
    for (int f=0; f<F; ++f) edGraph.writeToStateVector(Xfull[f]);

    const auto& bindings = edGraph.getBindings();   // [N] -> node ids
    const auto& edges    = edGraph.getEdges();      // graph edges

    // ----------------------------------------------------------------------------
    // Helper: boundary-only visibility (Z>0 and inside image). No occlusion test.
    // ----------------------------------------------------------------------------
    auto compute_visibility_boundary = [&](const Eigen::VectorXd& Xf,
                                           const cv::Mat& img,
                                           const Eigen::Matrix3d& R,
                                           const Eigen::Vector3d& t,
                                           std::vector<int>& vis_out){
        vis_out.clear(); vis_out.reserve(N/2);
        edGraph.updateFromStateVector(Xf);
        #pragma omp parallel
        {
            std::vector<int> vis_local; vis_local.reserve(256);
            #pragma omp for nowait
            for (int i = 0; i < N; ++i) {
                Eigen::Vector3d pw = edGraph.deformVertex(mesh_vertices[i], i);
                Eigen::Vector3d pc = R.transpose() * (pw - t);
                if (pc.z() > 1e-8) {
                    float u = (float)(K(0,0)*(pc.x()/pc.z()) + K(0,2));
                    float v = (float)(K(1,1)*(pc.y()/pc.z()) + K(1,2));
                    if (u >= 0 && u < img.cols && v >= 0 && v < img.rows)
                        vis_local.push_back(i);
                }
            }
            #pragma omp critical
            vis_out.insert(vis_out.end(), vis_local.begin(), vis_local.end());
        }
    };

    // ----------------------------------------------------------------------------
    // (1) One-time: build dummy BVHs (Evaluate ignores them) using persistent vertex buffers
    //     We avoid assignment (BVH is non-assignable) by emplacing with reserved capacity.
    // ----------------------------------------------------------------------------
    std::vector<std::vector<MeshModel::Vertex>> Vdef_init(F, std::vector<MeshModel::Vertex>(N));
    for (int f=0; f<F; ++f) {
        edGraph.updateFromStateVector(Xfull[f]);
        auto& Vd = Vdef_init[f];
        #pragma omp parallel for
        for (int i=0; i<N; ++i) {
            Eigen::Vector3d p = edGraph.deformVertex(mesh_vertices[i], i);
            Vd[i].x = (float)p.x(); Vd[i].y = (float)p.y(); Vd[i].z = (float)p.z();
        }
    }
    std::vector<BVH> bvhs; bvhs.reserve(F);
    for (int f=0; f<F; ++f) { bvhs.emplace_back(mesh_triangles, Vdef_init[f]); }

    // ----------------------------------------------------------------------------
    // (2) Initialize visibility for f=0 and f=1, then map intensities (I_init/I_prior)
    // ----------------------------------------------------------------------------
    std::vector<int> vis0, vis1;
    compute_visibility_boundary(Xfull[0], imgs_gray[0],
                                camera_poses_gt[0].block<3,3>(0,0),
                                camera_poses_gt[0].block<3,1>(0,3), vis0);
    compute_visibility_boundary(Xfull[1], imgs_gray[1],
                                camera_poses_gt[1].block<3,3>(0,0),
                                camera_poses_gt[1].block<3,1>(0,3), vis1);

    std::vector<double> I_init(N, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> I_prior(N, std::numeric_limits<double>::quiet_NaN());
    {
        const double w0 = 0.5, w1 = 0.5;
        std::vector<char> m0(N,0), m1(N,0);
        for (int id: vis0) m0[id]=1; for (int id: vis1) m1[id]=1;
        const Eigen::Matrix3d R0 = camera_poses_gt[0].block<3,3>(0,0);
        const Eigen::Vector3d t0 = camera_poses_gt[0].block<3,1>(0,3);
        const Eigen::Matrix3d R1 = camera_poses_gt[1].block<3,3>(0,0);
        const Eigen::Vector3d t1 = camera_poses_gt[1].block<3,1>(0,3);
        #pragma omp parallel for
        for (int i=0; i<N; ++i){
            double v0 = std::numeric_limits<double>::quiet_NaN();
            double v1 = std::numeric_limits<double>::quiet_NaN();
            if (m0[i]){
                Eigen::Vector3d pw = edGraph.deformVertex(mesh_vertices[i], i);
                Eigen::Vector3d pc = R0.transpose() * (pw - t0);
                if (pc.z()>1e-8){ float u=(float)(K(0,0)*(pc.x()/pc.z())+K(0,2)); float v=(float)(K(1,1)*(pc.y()/pc.z())+K(1,2)); v0 = bilinearSample(imgs_gray[0],u,v); }
            }
            if (m1[i]){
                Eigen::Vector3d pw = edGraph.deformVertex(mesh_vertices[i], i);
                Eigen::Vector3d pc = R1.transpose() * (pw - t1);
                if (pc.z()>1e-8){ float u=(float)(K(0,0)*(pc.x()/pc.z())+K(0,2)); float v=(float)(K(1,1)*(pc.y()/pc.z())+K(1,2)); v1 = bilinearSample(imgs_gray[1],u,v); }
            }
            bool ok0 = std::isfinite(v0), ok1 = std::isfinite(v1);
            if (ok0 && ok1) I_init[i] = I_prior[i] = 0.5*(v0+v1);
            else if (ok0)   I_init[i] = I_prior[i] = v0;
            else if (ok1)   I_init[i] = I_prior[i] = v1;
        }
    }

    // intensity variable mapping (one scalar per vertex having prior)
    std::vector<int> colI(N, -1); int Icount=0;
    for (int i=0;i<N;++i) if (std::isfinite(I_prior[i])) colI[i] = Icount++;
    std::vector<double> I_var(Icount, 0.0);
    for (int i=0;i<N;++i) if (colI[i]>=0) I_var[colI[i]] = std::isfinite(I_init[i]) ? I_init[i] : 0.0;

    const double lambda_I = 0.0; // set >0 for soft prior if desired
    double prev_cost = std::numeric_limits<double>::max();

    for (int it=0; it<maxIterations_; ++it){
        // (3) Recompute visibility for ALL frames under current Xfull
        std::vector<std::vector<int>> visible_vertices(F);
        for (int f=0; f<F; ++f){
            compute_visibility_boundary(Xfull[f], imgs_gray[f],
                                        camera_poses_gt[f].block<3,3>(0,0),
                                        camera_poses_gt[f].block<3,1>(0,3),
                                        visible_vertices[f]);
        }

        // (4) Build active nodes/edges and compact index per frame (f>=1)
        std::vector<std::vector<char>> active_node(F, std::vector<char>(G,0));
        std::vector<std::vector<std::pair<int,int>>> active_edges(F);
        std::vector<std::vector<int>> compact_idx(F, std::vector<int>(G,-1));
        std::vector<int> Sf(F,0);
        for (int f=1; f<F; ++f){
            for (int vid : visible_vertices[f]){
                if (colI[vid] < 0) continue;
                const auto& b = bindings[vid];
                for (int nid : b) active_node[f][nid]=1;
            }
            int acc=0; for (int j=0;j<G;++j) if (active_node[f][j]) compact_idx[f][j]=acc++;
            Sf[f]=acc;
            std::vector<std::pair<int,int>> Ef; Ef.reserve(edges.size());
            for (const auto& e: edges) if (active_node[f][e.first] && active_node[f][e.second]) Ef.push_back(e);
            active_edges[f].swap(Ef);
        }
        auto offsEDc = [&](int f){ int ofs=0; for (int k=0;k<f;++k) ofs += 12 * Sf[k]; return ofs; };
        auto colA_c  = [&](int f,int node,int k){ int ci=compact_idx[f][node]; if (ci<0) return -1; return offsEDc(f)+12*ci+k; };
        auto colt_c  = [&](int f,int node,int k){ int ci=compact_idx[f][node]; if (ci<0) return -1; return offsEDc(f)+12*ci+9+k; };

        // (5) Row layout with current visibility
        std::vector<int> data_row_ofs(F,0);
        int total_data_rows=0; for (int f=0; f<F; ++f){ data_row_ofs[f]=total_data_rows; for (int vid: visible_vertices[f]) if (colI[vid]>=0) ++total_data_rows; }
        int smooth_rows=0, rot_rows=0; for (int f=1; f<F; ++f){ smooth_rows += (int)active_edges[f].size()*(9+3); rot_rows += Sf[f]*9; }
        int temporal_rows=0; for (int f=2; f<F; ++f) for (int j=0;j<G;++j) if (active_node[f-1][j] && active_node[f][j]) temporal_rows += 12;
        int Iprior_rows=0; for (int i=0;i<N;++i) if (colI[i]>=0 && std::isfinite(I_prior[i])) ++Iprior_rows;
        const int row_smooth_begin   = total_data_rows;
        const int row_rot_begin      = row_smooth_begin + smooth_rows;
        const int row_temporal_begin = row_rot_begin    + rot_rows;
        const int row_Iprior_begin   = row_temporal_begin + temporal_rows;
        const int total_rows         = row_Iprior_begin + Iprior_rows;

        int edDimCompact=0; for (int f=0; f<F; ++f) edDimCompact += 12 * Sf[f];
        const int intensDim = Icount;
        const int stateDimCompact = edDimCompact + intensDim;

        std::cout << "[Layout] residual counts  data=" << total_data_rows
                  << ", smooth=" << smooth_rows
                  << ", rot=" << rot_rows
                  << ", temporal=" << temporal_rows
                  << ", Iprior=" << Iprior_rows
                  << ", total=" << total_rows << std::endl;
        std::cout << "[Layout] state dims  edCompact=" << edDimCompact
                  << ", intens=" << intensDim
                  << ", total=" << stateDimCompact << std::endl;

        // (6) Assemble normal equations
        const double sqrt_w   = std::sqrt(std::max(0.0, w_data_));
        const double sqrt_ls  = std::sqrt(std::max(0.0, lambda_smooth_));
        const double sqrt_lr  = std::sqrt(std::max(0.0, lambda_rot_));
        const double sqrt_ltp = std::sqrt(std::max(0.0, lambda_temporal_));
        const double sqrt_lI  = std::sqrt(std::max(0.0, lambda_I));

        std::vector<double> Fvec(total_rows, 0.0);
        int num_threads = omp_get_max_threads();
        std::vector<std::vector<Eigen::Triplet<double>>> triplets_thr(num_threads);
        for (auto& v: triplets_thr) v.reserve((size_t)std::max(1,total_data_rows/std::max(1,num_threads))*48);

        // DATA term (f>=0; f=0 has NO ED columns)
        #pragma omp parallel for schedule(static)
        for (int f=0; f<F; ++f){
            int tid = omp_get_thread_num(); auto& Tlocal = triplets_thr[tid];
            edGraph.updateFromStateVector(Xfull[f]);
            const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
            const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);
            const cv::Mat& img = imgs_gray[f];
            const BVH& bvh = bvhs[f]; // not used inside Evaluate currently
            int r = data_row_ofs[f];
            for (int idx=0; idx<(int)visible_vertices[f].size(); ++idx){
                const int i = visible_vertices[f][idx];
                const int ci = colI[i]; if (ci < 0) { ++r; continue; }
                const double Icurr = I_var[ci];
                PhotometricError cost(mesh_vertices[i], i, mesh_triangles, K, img, bvh, sqrt_w, &edGraph);
                double residual=0.0, J_I=0.0; Eigen::VectorXd J_ed(12*G); J_ed.setZero();
                cost.Evaluate(Icurr, residual, &J_I, (f==0? nullptr : &J_ed), R, t);
                Fvec[r] = residual;
                if (J_I != 0.0){ int gc = edDimCompact + ci; Tlocal.emplace_back(r, gc, J_I); }
                if (f>=1){
                    const auto& b = bindings[i];
                    for (int nid : b){
                        int base = 12*nid;
                        for (int c=0;c<9;++c){ double v = J_ed[base+c]; if (v==0.0) continue; int col = colA_c(f,nid,c); if (col>=0) Tlocal.emplace_back(r,col,v); }
                        for (int c=0;c<3;++c){ double v = J_ed[base+9+c]; if (v==0.0) continue; int col = colt_c(f,nid,c); if (col>=0) Tlocal.emplace_back(r,col,v); }
                    }
                }
                ++r;
            }
        }

        // SMOOTH
        int row_ptr = row_smooth_begin;
        for (int f=1; f<F; ++f){
            for (const auto& e: active_edges[f]){
                const int i=e.first, j=e.second;
                for (int m=0;m<9;++m){ int ci=colA_c(f,i,m), cj=colA_c(f,j,m); Fvec[row_ptr]=sqrt_ls*(Xfull[f][12*i+m]-Xfull[f][12*j+m]); if (ci>=0) triplets_thr[0].emplace_back(row_ptr,ci,sqrt_ls); if (cj>=0) triplets_thr[0].emplace_back(row_ptr,cj,-sqrt_ls); ++row_ptr; }
                for (int m=0;m<3;++m){ int ci=colt_c(f,i,m), cj=colt_c(f,j,m); Fvec[row_ptr]=sqrt_ls*(Xfull[f][12*i+9+m]-Xfull[f][12*j+9+m]); if (ci>=0) triplets_thr[0].emplace_back(row_ptr,ci,sqrt_ls); if (cj>=0) triplets_thr[0].emplace_back(row_ptr,cj,-sqrt_ls); ++row_ptr; }
            }
        }

        // ROT
        for (int f=1; f<F; ++f){
            for (int j=0;j<G;++j) if (active_node[f][j]){
                for (int k=0;k<9;++k){ int col = colA_c(f,j,k); double target = (k==0||k==4||k==8)?1.0:0.0; Fvec[row_ptr]=sqrt_lr*(Xfull[f][12*j+k]-target); if (col>=0) triplets_thr[0].emplace_back(row_ptr,col,sqrt_lr); ++row_ptr; }
            }
        }

        // TEMPORAL
        for (int f=2; f<F; ++f){
            for (int j=0;j<G;++j) if (active_node[f-1][j] && active_node[f][j]){
                for (int m=0;m<12;++m){ int c1=(m<9)?colA_c(f-1,j,m):colt_c(f-1,j,m-9); int c2=(m<9)?colA_c(f,j,m):colt_c(f,j,m-9); Fvec[row_ptr]=sqrt_ltp*(Xfull[f][12*j+m]-Xfull[f-1][12*j+m]); if (c2>=0) triplets_thr[0].emplace_back(row_ptr,c2,sqrt_ltp); if (c1>=0) triplets_thr[0].emplace_back(row_ptr,c1,-sqrt_ltp); ++row_ptr; }
            }
        }

        // INTENSITY PRIOR (optional; lambda_I==0 disables it)
        for (int i=0, rI=row_Iprior_begin; i<N; ++i) if (colI[i]>=0 && std::isfinite(I_prior[i])){
            int ci = colI[i]; Fvec[rI] = sqrt_lI * (I_var[ci] - I_prior[i]); int gc = edDimCompact + ci; triplets_thr[0].emplace_back(rI,gc,sqrt_lI); ++rI; }

        // Solve normal equations
        std::vector<Eigen::Triplet<double>> triplets; size_t tot=0; for (auto& t: triplets_thr) tot += t.size(); triplets.reserve(tot); for (auto& t: triplets_thr) { triplets.insert(triplets.end(), t.begin(), t.end()); }
        Eigen::SparseMatrix<double> J(total_rows, stateDimCompact); J.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::VectorXd Fv(total_rows); for (int r=0;r<total_rows;++r) Fv[r]=Fvec[r];
        double cost = 0.5 * Fv.squaredNorm();
        std::cout << "[GN it=" << it << "] cost(raw)=" << cost << std::endl;
        if (cost > prev_cost * (1.0 - 1e-9)) break; prev_cost = cost;
        Eigen::SparseMatrix<double> At = J.transpose();
        Eigen::SparseMatrix<double> AtA = At * J; Eigen::VectorXd Atb = -At * Fv;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver; solver.compute(AtA);
        if (solver.info()!=Eigen::Success){ std::cerr << "[Optimizer] LDLT factorization failed.\n"; break; }
        Eigen::VectorXd dx = solver.solve(Atb); if (solver.info()!=Eigen::Success){ std::cerr << "[Optimizer] Linear solve failed.\n"; break; }

        // Apply update: intensities + ED (f>=1)
        for (int i=0;i<N;++i) if (colI[i]>=0){ int gc = edDimCompact + colI[i]; I_var[colI[i]] += dx[gc]; }
        for (int f=1; f<F; ++f){
            for (int j=0;j<G;++j) if (active_node[f][j]){
                int base = 12*j;
                for (int k=0;k<9;++k){ int col = colA_c(f,j,k); if (col>=0) Xfull[f][base+k] += dx[col]; }
                for (int k=0;k<3;++k){ int col = colt_c(f,j,k); if (col>=0) Xfull[f][base+9+k] += dx[col]; }
            }
        }
        // loop, next iteration will recompute visibility
    }
}