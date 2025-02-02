#include "DatasetManager.h"
#include <filesystem>
#include <nlohmann/json.hpp> // 用于加载 JSON
#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Core>


namespace fs = std::filesystem;

DatasetManager::DatasetManager(const std::string& dataset_path) : dataset_path_(dataset_path) {}

// 加载所有 RGB 图像
bool DatasetManager::loadAllRGBImages(std::vector<cv::Mat>& rgb_images) {
    rgb_images.clear();
    std::string rgb_path = dataset_path_ + "/rgb/";

    for (const auto& entry : fs::directory_iterator(rgb_path)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            cv::Mat image = cv::imread(entry.path().string());
            if (image.empty()) {
                std::cerr << "Failed to load RGB image: " << entry.path().string() << std::endl;
                return false;
            }
            rgb_images.push_back(image);
        }
    }

    std::cout << "Loaded " << rgb_images.size() << " RGB images.\n";
    return true;
}

bool DatasetManager::loadAllDepthImages(std::vector<cv::Mat>& depth_images) {
    depth_images.clear();
    std::string depth_path = dataset_path_ + "/depth/";

    for (const auto& entry : fs::directory_iterator(depth_path)) {
        if (entry.path().extension() == ".png") { // 假设深度图为 PNG 格式
            cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
            if (image.empty()) {
                std::cerr << "Failed to load depth image: " << entry.path().string() << std::endl;
                return false;
            }

            if (image.type() != CV_16U) {  // 确保是 16-bit 深度图
                std::cerr << "Unexpected depth image format (expected CV_16U): " 
                          << entry.path().string() << std::endl;
                return false;
            }

            // **将 16-bit 深度图转换回 0-100mm 真实深度值**
            cv::Mat depth_in_mm;
            image.convertTo(depth_in_mm, CV_32F, 100.0 / 65535.0); // 0-100mm 线性缩放

            depth_images.push_back(depth_in_mm);
        }
    }

    std::cout << "Loaded " << depth_images.size() << " depth images (converted to meters).\n";
    return true;
}

bool DatasetManager::parseOBJ(const std::string& filePath, std::vector<MeshModel::Vertex>& vertices, std::vector<MeshModel::Triangle>& triangles) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open OBJ file: " << filePath << std::endl;
        return false;
    }

    std::string line;
    std::vector<Eigen::Vector3f> temp_vertices;  // 存储顶点坐标
    std::vector<Eigen::Vector3f> temp_normals;   // 存储法向量

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") { // 顶点
            float x, y, z;
            iss >> x >> y >> z;
            temp_vertices.emplace_back(x, y, z);
        } else if (prefix == "vn") { // 法向量
            float nx, ny, nz;
            iss >> nx >> ny >> nz;
            temp_normals.emplace_back(nx, ny, nz);
        } else if (prefix == "f") { // 面
            int v0, v1, v2, vn0, vn1, vn2;
            char slash;
            iss >> v0 >> slash >> slash >> vn0
                >> v1 >> slash >> slash >> vn1
                >> v2 >> slash >> slash >> vn2;

            triangles.push_back({v0 - 1, v1 - 1, v2 - 1});
        }
    }

    // 将顶点和法向量存入 MeshModel 的 Vertex 结构
    for (size_t i = 0; i < temp_vertices.size(); ++i) {
        MeshModel::Vertex vertex;
        vertex.x = temp_vertices[i].x();
        vertex.y = temp_vertices[i].y();
        vertex.z = temp_vertices[i].z();

        if (i < temp_normals.size()) {
            vertex.nx = temp_normals[i].x();
            vertex.ny = temp_normals[i].y();
            vertex.nz = temp_normals[i].z();
        } else {
            vertex.nx = vertex.ny = vertex.nz = 0.0f; // 如果没有法向量，则置为零
        }

        vertices.push_back(vertex);
    }

    return true;
}

// 加载网格模型
bool DatasetManager::loadMeshModel(MeshModel& mesh) {
    std::string mesh_path = dataset_path_ + "/mesh.obj";

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
bool DatasetManager::loadCameraIntrinsics(Eigen::Matrix3f& intrinsics) {
    std::string intrinsics_path = dataset_path_ + "/camera_intrinsics.json";
    std::ifstream file(intrinsics_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open camera intrinsics file: " << intrinsics_path << std::endl;
        return false;
    }

    nlohmann::json json_data;
    file >> json_data;
    intrinsics << json_data["fx"], 0, json_data["cx"],
                  0, json_data["fy"], json_data["cy"],
                  0, 0, 1;

    return true;
}

bool DatasetManager::loadPoses(std::vector<Eigen::Matrix4f>& poses, const std::string& fileName) {
    std::string pose_file_path = dataset_path_ + "/" + fileName + ".txt";
    std::ifstream pose_file(pose_file_path);

    if (!pose_file.is_open()) {
        std::cerr << "Failed to open poses file: " << pose_file_path << std::endl;
        return false;
    }

    poses.clear();

    std::string line;
    while (std::getline(pose_file, line)) {
        std::istringstream iss(line);
        std::vector<float> values;
        float value;

        // 读取每行的 16 个值
        while (iss >> value) {
            values.push_back(value);
        }

        if (values.size() != 16) {
            std::cerr << "Error: Each line must contain exactly 16 values.\n";
            return false;
        }

        // 构造 4x4 齐次变换矩阵
        Eigen::Matrix4f pose;
        pose << values[0], values[1], values[2], values[3],
                values[4], values[5], values[6], values[7],
                values[8], values[9], values[10], values[11],
                values[12], values[13], values[14], values[15];

        // 检查矩阵的齐次特性（最后一行应该是 [0, 0, 0, 1]）
        if (!pose.block<1, 4>(3, 0).isApprox(Eigen::RowVector4f(0, 0, 0, 1), 1e-6)) {
            std::cerr << "Error: Invalid homogeneous transformation matrix.\n";
            return false;
        }

        poses.push_back(pose);
    }

    std::cout << "Loaded " << poses.size() << " poses.\n";
    return true;
}