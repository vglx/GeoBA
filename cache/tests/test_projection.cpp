#include <iostream>
#include <Eigen/Core>
#include <vector>
#include "Projection.h" // 假设 Projection 模块已实现

void testSinglePointProjection() {
    Eigen::Vector3f point3D(1.0, 2.0, 5.0);

    Eigen::Matrix3f intrinsics;
    intrinsics << 525.0, 0.0, 320.0,
                  0.0, 525.0, 240.0,
                  0.0, 0.0, 1.0;

    Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
    Eigen::Vector3f translation(0.0, 0.0, 0.0);

    Eigen::Vector2f point2D = Projection::projectPoint(point3D, intrinsics, rotation, translation);
    std::cout << "Test Single Point Projection:" << std::endl;
    std::cout << "Projected 2D point: (" << point2D.x() << ", " << point2D.y() << ")" << std::endl;
}

void testMultiplePointProjection() {
    std::vector<Eigen::Vector3f> points3D = {
        {1.0, 2.0, 5.0},
        {2.0, 3.0, 10.0},
        {-1.0, -1.0, 3.0}
    };

    Eigen::Matrix3f intrinsics;
    intrinsics << 525.0, 0.0, 320.0,
                  0.0, 525.0, 240.0,
                  0.0, 0.0, 1.0;

    Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
    Eigen::Vector3f translation(0.0, 0.0, 0.0);

    std::vector<Eigen::Vector2f> projectedPoints = Projection::projectPoints(points3D, intrinsics, rotation, translation);

    std::cout << "Test Multiple Point Projection:" << std::endl;
    for (size_t i = 0; i < projectedPoints.size(); ++i) {
        std::cout << "3D Point: (" << points3D[i].x() << ", " << points3D[i].y() << ", " << points3D[i].z() << ")"
                  << " -> 2D Point: (" << projectedPoints[i].x() << ", " << projectedPoints[i].y() << ")" << std::endl;
    }
}

void testOcclusionHandling() {
    std::vector<Eigen::Vector2f> projectedPoints = {
        {100, 200}, {100, 200}, {150, 250}, {100, 200}
    };
    std::vector<float> depths = {1.0, 0.5, 2.0, 0.8};

    std::vector<Eigen::Vector2f> visiblePoints = Projection::handleOcclusion(projectedPoints, depths);

    std::cout << "Test Occlusion Handling:" << std::endl;
    for (const auto& point : visiblePoints) {
        std::cout << "Visible Point: (" << point.x() << ", " << point.y() << ")" << std::endl;
    }
}

int main() {
    std::cout << "Starting Projection Tests..." << std::endl;

    testSinglePointProjection();
    std::cout << std::endl;

    testMultiplePointProjection();
    std::cout << std::endl;

    testOcclusionHandling();
    std::cout << std::endl;

    std::cout << "Projection Tests Completed." << std::endl;
    return 0;
}
