#ifndef EVALUATION_H
#define EVALUATION_H

#include <vector>
#include <Eigen/Dense>

class Evaluation {
public:
       static void ComputeRMSE(const std::vector<Eigen::Matrix4d>& ground_truth,
                               const std::vector<Eigen::Matrix4d>& initial_poses,
                               const std::vector<Eigen::Matrix4d>& optimized_poses);

private:
       static double ComputeRMSE(const std::vector<Eigen::Matrix4d>& gt, 
                                 const std::vector<Eigen::Matrix4d>& poses);
};

#endif // EVALUATION_H