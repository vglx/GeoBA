#ifndef SHARED_DATA_H
#define SHARED_DATA_H

#include <vector>
#include <Eigen/Core>

// 用于共享投影相关数据的结构体
struct SharedProjectionData {
    std::vector<std::vector<Eigen::Vector2f>> projected_points; // 投影到相机平面的像素坐标
    std::vector<std::vector<float>> projected_depths;           // 顶点的投影深度
    std::vector<std::vector<bool>> visible_mask;                // 可见性布尔掩码
};

#endif // SHARED_DATA_H