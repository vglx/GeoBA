#ifndef EVALUATION_H
#define EVALUATION_H

#include <vector>
#include <Eigen/Dense>

class Evaluation {
public:
       static void ComputeRMSE(const std::vector<Eigen::Matrix4d>& ground_truth,
                               const std::vector<Eigen::Matrix4d>& initial_poses,
                               const std::vector<Eigen::Matrix4d>& optimized_poses);

       static void ComputeEulerRMSE(const std::vector<Eigen::Matrix4d>& ground_truth,
                                    const std::vector<Eigen::Matrix4d>& initial_poses,
                                    const std::vector<Eigen::Matrix4d>& optimized_poses);

private:
       static void ComputeRMSE(const std::vector<Eigen::Matrix4d>& gt, 
                               const std::vector<Eigen::Matrix4d>& poses);

       static Eigen::Vector3d RotationMatrixToEulerZYX(const Eigen::Matrix3d& R);

       static void ComputeEulerRMSE(const std::vector<Eigen::Matrix4d>& gt,
                                    const std::vector<Eigen::Matrix4d>& poses,
                                    const std::string& label);
};

#endif // EVALUATION_H