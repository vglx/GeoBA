#include "Projection.h"
#include <algorithm>


bool Projection::isPointInImage(
    const Eigen::Vector2f& point, 
    int imageWidth, 
    int imageHeight) {
    return point.x() >= 0 && point.x() < imageWidth && 
           point.y() >= 0 && point.y() < imageHeight;
}

void Projection::rasterizeTriangle(
    const Eigen::Vector2f& p0, const Eigen::Vector2f& p1, const Eigen::Vector2f& p2,
    float z0, float z1, float z2,
    std::vector<std::vector<float>>& depthBuffer,
    int imageWidth, int imageHeight) {
    // 获取三角形的边界框
    int minX = std::max(0, static_cast<int>(std::floor(std::min({p0.x(), p1.x(), p2.x()}))));
    int maxX = std::min(imageWidth - 1, static_cast<int>(std::ceil(std::max({p0.x(), p1.x(), p2.x()}))));
    int minY = std::max(0, static_cast<int>(std::floor(std::min({p0.y(), p1.y(), p2.y()}))));
    int maxY = std::min(imageHeight - 1, static_cast<int>(std::ceil(std::max({p0.y(), p1.y(), p2.y()}))));

    // 遍历边界框内的像素
    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            Eigen::Vector2f pixel(x + 0.5f, y + 0.5f);

            // 使用重心坐标法判断像素是否在三角形内部
            float w0 = (p1.x() - p0.x()) * (pixel.y() - p0.y()) - (p1.y() - p0.y()) * (pixel.x() - p0.x());
            float w1 = (p2.x() - p1.x()) * (pixel.y() - p1.y()) - (p2.y() - p1.y()) * (pixel.x() - p1.x());
            float w2 = (p0.x() - p2.x()) * (pixel.y() - p2.y()) - (p0.y() - p2.y()) * (pixel.x() - p2.x());

            if ((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0)) {
                // 计算深度值（插值）
                float area = std::abs(w0 + w1 + w2) + 1e-6f;
                float alpha = w1 / area, beta = w2 / area, gamma = w0 / area;
                float z = alpha * z0 + beta * z1 + gamma * z2;

                // 更新深度缓冲区
                if (z < depthBuffer[y][x]) {
                    depthBuffer[y][x] = z;
                }
            }
        }
    }
}