#include "DatasetManager.h"
#include <filesystem>
#include <nlohmann/json.hpp> // 用于加载 JSON
#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Geometry>

namespace fs = std::filesystem;

DatasetManager::DatasetManager(const std::string& dataset_path) : dataset_path_(dataset_path) {}

// 加载所有 RGB 图像
bool DatasetManager::loadAllRGBImages(std::vector<cv::Mat>& rgb_images) {
    rgb_images.clear();
    std::string rgb_path = dataset_path_ + "rgb/";

    // 1. 收集所有符合扩展名的文件路径
    std::vector<std::string> image_filenames;
    for (const auto& entry : fs::directory_iterator(rgb_path)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            image_filenames.push_back(entry.path().string());
        }
    }

    // 2. 按照文件名进行排序（字典序）
    std::sort(image_filenames.begin(), image_filenames.end());

    // 3. 依次读取排序后的图像
    for (const auto& filename : image_filenames) {
        cv::Mat image = cv::imread(filename);
        if (image.empty()) {
            std::cerr << "Failed to load RGB image: " << filename << std::endl;
            return false;
        }
        rgb_images.push_back(image);
    }

    std::cout << "Loaded " << rgb_images.size() << " RGB images.\n";
    return true;
}

bool DatasetManager::loadAllDepthImages(std::vector<cv::Mat>& depth_images) {
    depth_images.clear();
    std::string depth_path = dataset_path_ + "depth/";

    // 1. 收集文件路径
    std::vector<std::string> depth_filenames;
    for (const auto& entry : fs::directory_iterator(depth_path)) {
        if (entry.path().extension() == ".png" || entry.path().extension() == ".tiff") {
            depth_filenames.push_back(entry.path().string());
        }
    }

    // 2. 对文件名进行排序（字典序）
    std::sort(depth_filenames.begin(), depth_filenames.end());

    // 3. 依次读取并处理深度图
    for (const auto& filename : depth_filenames) {
        cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);
        if (image.empty()) {
            std::cerr << "Failed to load depth image: " << filename << std::endl;
            return false;
        }

        if (image.type() != CV_16U) {  // 确保是 16-bit 深度图
            std::cerr << "Unexpected depth image format (expected CV_16U): " 
                      << filename << std::endl;
            return false;
        }
        cv::Mat depth_in_mm;
        // 假设这里你想把 0~65535 的深度值线性缩放到 0~100mm
        image.convertTo(depth_in_mm, CV_32F, 100.0 / 65535.0);

        depth_images.push_back(depth_in_mm);
    }

    std::cout << "Loaded " << depth_images.size() << " depth images.\n";
    return true;
}

bool DatasetManager::parseOBJ(const std::string& filePath, std::vector<MeshModel::Vertex>& vertices, std::vector<MeshModel::Triangle>& triangles) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open OBJ file: " << filePath << std::endl;
        return false;
    }

    std::string line;
    std::vector<Eigen::Vector3f> temp_vertices;
    std::vector<Eigen::Vector3f> temp_normals;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            temp_vertices.emplace_back(x, y, z);
        } else if (prefix == "vn") {
            float nx, ny, nz;
            iss >> nx >> ny >> nz;
            temp_normals.emplace_back(nx, ny, nz);
        } else if (prefix == "f") {
            std::vector<int> vertex_indices;
            std::string token;
            while (iss >> token) {
                std::replace(token.begin(), token.end(), '/', ' '); // `/` 替换为空格
                std::istringstream token_stream(token);
                int v;
                token_stream >> v;
                vertex_indices.push_back(v);
            }

            if (vertex_indices.size() < 3) {
                std::cerr << "Warning: Invalid triangle line: " << line << std::endl;
                continue;
            }

            // 处理负数索引
            for (int& v : vertex_indices) {
                if (v < 0) v += temp_vertices.size();
            }

            if (vertex_indices[0] >= 0 && vertex_indices[1] >= 0 && vertex_indices[2] >= 0) {
                triangles.push_back({vertex_indices[0] - 1, vertex_indices[1] - 1, vertex_indices[2] - 1});
            } else {
                std::cerr << "Error: Triangle index out of range in line: " << line << std::endl;
            }
        }
    }

    // 创建 MeshModel 顶点
    for (const auto& v : temp_vertices) {
        vertices.push_back({v.x(), v.y(), v.z(), 0, 0, 0});
    }

    std::cout << "Loaded " << vertices.size() << " vertices and " << triangles.size() << " triangles from OBJ file.\n";
    return true;
}

// 加载网格模型
bool DatasetManager::loadMeshModel(MeshModel& mesh) {
    std::string mesh_path = dataset_path_ + "mesh.obj";

    // 解析 OBJ 文件
    std::vector<MeshModel::Vertex> vertices;
    std::vector<MeshModel::Triangle> triangles;
    if (!parseOBJ(mesh_path, vertices, triangles)) {
        std::cerr << "Failed to parse mesh model: " << mesh_path << std::endl;
        return false;
    }

    // 将解析结果存入 MeshModel 实例
    mesh.setVertices(vertices);
    mesh.setTriangles(triangles);
    return true;
}

// 加载相机内参
bool DatasetManager::loadCameraIntrinsics(Eigen::Matrix3d& intrinsics) {
    std::string intrinsics_path = dataset_path_ + "camera_intrinsics.json";
    std::ifstream file(intrinsics_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open camera intrinsics file: " << intrinsics_path << std::endl;
        return false;
    }

    nlohmann::json json_data;
    file >> json_data;

    // 读取 double 类型的相机内参
    intrinsics << static_cast<double>(json_data["fx"]), 0.0, static_cast<double>(json_data["cx"]),
                  0.0, static_cast<double>(json_data["fy"]), static_cast<double>(json_data["cy"]),
                  0.0, 0.0, 1.0;

    return true;
}

bool DatasetManager::loadPoses(std::vector<Eigen::Matrix4d>& poses, const std::string& fileName) {
    std::string pose_file_path = dataset_path_ + fileName + ".txt";
    std::ifstream pose_file(pose_file_path);

    if (!pose_file.is_open()) {
        std::cerr << "Failed to open poses file: " << pose_file_path << std::endl;
        return false;
    }

    poses.clear();

    std::string line;
    while (std::getline(pose_file, line)) {
        std::vector<double> values;
        std::stringstream ss(line);
        std::string value_str;

        // **改为手动解析逗号**
        while (std::getline(ss, value_str, ',')) {
            try {
                values.push_back(std::stod(value_str));  // 确保是 float
            } catch (const std::exception& e) {
                std::cerr << "Error parsing double: " << value_str << " in line: " << line << "\n";
                return false;
            }
        }

        if (values.size() != 16) {
            std::cerr << "Error: Each line must contain exactly 16 values.\n";
            return false;
        }

        // **使用 Eigen::Map 直接映射 std::vector<float> 为 row-major 矩阵**
        Eigen::Matrix4d pose = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::ColMajor>>(values.data());

        // 检查矩阵的齐次特性（最后一行应该是 [0, 0, 0, 1]）
        if (!pose.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-6)) {
            std::cerr << "Error: Invalid homogeneous transformation matrix.\n";
            return false;
        }

        Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d R_fixed = svd.matrixU() * svd.matrixV().transpose();

        // **确保 det(R) = 1**
        if (R_fixed.determinant() < 0) {
            R_fixed.col(0) *= -1.0f;  // 反转一个轴，确保是正交的 SO(3) 旋转矩阵
        }

        // **对 R 进行归一化，确保正交**
        Eigen::Quaterniond q(R_fixed);
        q.normalize();
        R_fixed = q.toRotationMatrix();

        pose.block<3, 3>(0, 0) = R_fixed;

        poses.push_back(pose);
    }

    std::cout << "Loaded " << poses.size() << " poses.\n";
    return true;
}