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
    for (int f=0; f<F; ++f) edGraph.writeToStateVector(Xfull[f], /*offset=*/0);

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
        edGraph.updateFromStateVector(Xf, /*offset=*/0);
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
    // ----------------------------------------------------------------------------
    std::vector<std::vector<MeshModel::Vertex>> Vdef_init(F, std::vector<MeshModel::Vertex>(N));
    for (int f=0; f<F; ++f) {
        edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);
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
    // (2) Initialize visibility for f=0 (template only), then map intensities from frame 0
    // ----------------------------------------------------------------------------
    std::vector<int> vis0;
    compute_visibility_boundary(Xfull[0], imgs_gray[0],
                                camera_poses_gt[0].block<3,3>(0,0),
                                camera_poses_gt[0].block<3,1>(0,3), vis0);

    std::vector<double> I_init(N, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> I_prior(N, std::numeric_limits<double>::quiet_NaN());
    {
        // 仅使用 frame 0 采样作为 intensity 初值和（可选）先验
        std::vector<char> m0(N, 0);
        for (int id : vis0) m0[id] = 1;

        const Eigen::Matrix3d R0 = camera_poses_gt[0].block<3,3>(0,0);
        const Eigen::Vector3d t0 = camera_poses_gt[0].block<3,1>(0,3);

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double v0 = std::numeric_limits<double>::quiet_NaN();
            if (m0[i]) {
                Eigen::Vector3d pw = edGraph.deformVertex(mesh_vertices[i], i);
                Eigen::Vector3d pc = R0.transpose() * (pw - t0);
                if (pc.z() > 1e-8) {
                    float u = (float)(K(0,0) * (pc.x()/pc.z()) + K(0,2));
                    float v = (float)(K(1,1) * (pc.y()/pc.z()) + K(1,2));
                    v0 = bilinearSample(imgs_gray[0], u, v);
                }
            }
            if (std::isfinite(v0)) {
                I_init[i]  = v0;
                I_prior[i] = v0;    // 若后面启用 lambda_I，可作为软先验
            }
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
            edGraph.updateFromStateVector(Xfull[f], /*offset=*/0);
            const Eigen::Matrix3d R = camera_poses_gt[f].block<3,3>(0,0);
            const Eigen::Vector3d t = camera_poses_gt[f].block<3,1>(0,3);
            const cv::Mat& img = imgs_gray[f];
            const BVH& bvh = bvhs[f]; // not used inside Evaluate currently
            int r = data_row_ofs[f];
            for (int idx=0; idx<(int)visible_vertices[f].size(); ++idx){
                const int i = visible_vertices[f][idx];
                const int ci = colI[i]; if (ci < 0) { ++r; continue; }
                const double Icurr = I_var[ci];
                PhotometricError cost(mesh_vertices[i], i, mesh_triangles,
                                      K, img, bvh, sqrt_w, &edGraph);
                double res; double dI; Eigen::VectorXd Jed;
                const bool ok = cost.Evaluate(Icurr, res, &dI, (f==0? nullptr : &Jed), R, t);
                if (!ok) { ++r; continue; }
                // RHS
                Fvec[r] = res;
                // J for intensity
                const int colIglob = edDimCompact + ci;
                Tlocal.emplace_back(r, colIglob, dI);
                // J for ED (only from f>=1 using compact mapping)
                if (f>=1 && Jed.size()==12*(int)bindings[i].size()){
                    const auto& b = bindings[i];
                    for (int kk=0; kk<(int)b.size(); ++kk){
                        const int nid = b[kk];
                        const int ciN = compact_idx[f][nid];
                        if (ciN < 0) continue;
                        // A(3x3) part
                        for (int kA=0; kA<9; ++kA){
                            const double v = Jed[kk*12 + kA];
                            const int cg = colA_c(f, nid, kA);
                            if (cg>=0 && std::abs(v)>0) Tlocal.emplace_back(r, cg, v);
                        }
                        // t(3) part
                        for (int kt=0; kt<3; ++kt){
                            const double v = Jed[kk*12 + 9 + kt];
                            const int cg = colt_c(f, nid, kt);
                            if (cg>=0 && std::abs(v)>0) Tlocal.emplace_back(r, cg, v);
                        }
                    }
                }
                ++r;
            }
        }

        // SMOOTH + ROT regularization over active edges/nodes (f>=1)
        #pragma omp parallel for schedule(static)
        for (int f=1; f<F; ++f){
            int tid = omp_get_thread_num(); auto& Tlocal = triplets_thr[tid];
            int r = row_smooth_begin; for (int k=1; k<f; ++k) r += (int)active_edges[k].size()*(9+3);
            // smoothness: (A_i - A_j) and (t_i - t_j)
            for (const auto& e : active_edges[f]){
                int i = e.first, j = e.second; int ci=compact_idx[f][i], cj=compact_idx[f][j];
                if (ci<0 || cj<0) continue;
                // A diff (9 rows)
                for (int k=0;k<9;++k){
                    int rr = r++;
                    Fvec[rr] = 0.0; // residual = sqrt_ls*(A_i(k)-A_j(k))
                    Tlocal.emplace_back(rr, colA_c(f,i,k),  sqrt_ls);
                    Tlocal.emplace_back(rr, colA_c(f,j,k), -sqrt_ls);
                }
                // t diff (3 rows)
                for (int k=0;k<3;++k){
                    int rr = r++;
                    Fvec[rr] = 0.0;
                    Tlocal.emplace_back(rr, colt_c(f,i,k),  sqrt_ls);
                    Tlocal.emplace_back(rr, colt_c(f,j,k), -sqrt_ls);
                }
            }
            // rotation-like regularization: keep A close to orthonormal rows (simple Tikhonov on A off-diagonals)
            int rrot = row_rot_begin; for (int k=1; k<f; ++k) rrot += Sf[k]*9;
            for (int i=0;i<G;++i){
                int ci=compact_idx[f][i]; if (ci<0) continue;
                // penalize deviation of A from identity (9 rows)
                for (int k=0;k<9;++k){
                    int rr = rrot++;
                    double target = (k==0||k==4||k==8)? 1.0 : 0.0; // I3
                    Fvec[rr] = sqrt_lr * target;
                    Tlocal.emplace_back(rr, colA_c(f,i,k), sqrt_lr);
                }
            }
        }

        // TEMPORAL consistency between f-1 and f over overlapping active nodes
        #pragma omp parallel for schedule(static)
        for (int f=2; f<F; ++f){
            int tid = omp_get_thread_num(); auto& Tlocal = triplets_thr[tid];
            int r = row_temporal_begin; for (int k=2; k<f; ++k){
                for (int j=0;j<G;++j) if (active_node[k-1][j] && active_node[k][j]) r += 12;
            }
            for (int j=0;j<G;++j){
                if (!(active_node[f-1][j] && active_node[f][j])) continue;
                for (int k=0;k<12;++k){
                    int rr = r++;
                    Fvec[rr] = 0.0; // sqrt_ltp * (state_f(j,k) - state_fm1(j,k))
                    if (k<9){
                        Tlocal.emplace_back(rr, colA_c(f,  j, k),  sqrt_ltp);
                        Tlocal.emplace_back(rr, colA_c(f-1,j, k), -sqrt_ltp);
                    } else {
                        int kt = k-9;
                        Tlocal.emplace_back(rr, colt_c(f,  j, kt),  sqrt_ltp);
                        Tlocal.emplace_back(rr, colt_c(f-1,j, kt), -sqrt_ltp);
                    }
                }
            }
        }

        // Intensity prior (optional)
        if (lambda_I > 0.0){
            int r = row_Iprior_begin;
            auto &T0 = triplets_thr[0]; // use thread 0's bucket outside parallel regions
            for (int i=0;i<N;++i){
                const int ci = colI[i]; if (ci<0 || !std::isfinite(I_prior[i])) continue;
                Fvec[r] = sqrt_lI * I_prior[i];
                T0.emplace_back(r, edDimCompact + ci, sqrt_lI);
                ++r;
            }
        }

        // Stitch triplets
        std::vector<Eigen::Triplet<double>> triplets; triplets.reserve(1<<20);
        for (auto& v: triplets_thr) triplets.insert(triplets.end(), v.begin(), v.end());

        // Solve normal equations
        Eigen::SparseMatrix<double> J(total_rows, stateDimCompact);
        J.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::VectorXd Fv = Eigen::Map<Eigen::VectorXd>(Fvec.data(), (int)Fvec.size());

        const double cost = 0.5 * Fv.squaredNorm();
        std::cout << "[GN it=" << it << "] cost(raw)=" << (cost/(double)std::max(1,total_rows)) << std::endl;

        Eigen::SparseMatrix<double> H(stateDimCompact, stateDimCompact);
        H = J.transpose() * J;
        Eigen::VectorXd b = - J.transpose() * Fv;

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
        ldlt.compute(H);
        if (ldlt.info() != Eigen::Success){
            std::cerr << "[Optimizer] LDLT failed to factorize H.\n";
            break;
        }
        Eigen::VectorXd dx = ldlt.solve(b);
        if (ldlt.info() != Eigen::Success){
            std::cerr << "[Optimizer] LDLT failed to solve.\n";
            break;
        }

        // Update state: split dx into ED(compact per-frame) and I
        // 1) ED compact updates → scatter back into per‑frame full state Xfull
        int ofs = 0;
        for (int f=0; f<F; ++f){
            if (Sf[f]==0) continue;
            Eigen::VectorXd dXc = dx.segment(ofs, 12*Sf[f]); ofs += 12*Sf[f];
            // apply to Xfull[f] via compact map
            for (int j=0;j<G;++j){
                const int cj = compact_idx[f][j]; if (cj<0) continue;
                for (int k=0;k<9;++k)  Xfull[f][12*j + k]     += dXc[12*cj + k];
                for (int k=0;k<3;++k)  Xfull[f][12*j + 9 + k] += dXc[12*cj + 9 + k];
            }
        }
        // 2) Intensity
        if (intensDim>0){
            Eigen::VectorXd dI = dx.segment(edDimCompact, intensDim);
            for (int i=0;i<intensDim;++i) I_var[i] += dI[i];
        }

        // Early stop
        const double rel = std::abs(prev_cost - cost) / std::max(1.0, prev_cost);
        prev_cost = cost;
        if (rel < 1e-6){
            std::cout << "[EarlyStop] |Δcost|/cost < 1e-6\n";
            break;
        }
    }

    // write last Xfull back into edGraph for the final frame if desired by caller
    // (caller may choose a specific frame to export deformed mesh)
}