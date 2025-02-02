#include "Initializer.h"
#include <iostream>
#include <fstream>
#include <sstream>

bool Initializer::estimateWithDSO(const std::vector<cv::Mat>& rgb_images, 
                                  const Eigen::Matrix3f& camera_intrinsics, 
                                  std::vector<Eigen::Matrix4f>& estimated_poses, 
                                  const std::string& config_file) {
    if (rgb_images.empty()) {
        std::cerr << "Error: No images provided for DSO initialization." << std::endl;
        return false;
    }

    std::cout << "Running DSO initialization on " << rgb_images.size() << " images..." << std::endl;

    // **保存临时图像**
    std::vector<std::string> temp_image_paths;
    for (size_t i = 0; i < rgb_images.size(); ++i) {
        std::string img_path = "temp_frame_" + std::to_string(i) + ".png";
        cv::imwrite(img_path, rgb_images[i]);
        temp_image_paths.push_back(img_path);
    }

    // **执行 DSO**
    std::string dso_command = "./dso " + config_file;
    for (const auto& path : temp_image_paths) {
        dso_command += " " + path;
    }

    int ret = system(dso_command.c_str());
    if (ret != 0) {
        std::cerr << "Error: DSO execution failed." << std::endl;
        return false;
    }

    // **读取 DSO 结果**
    std::ifstream pose_file("dso_poses.txt");
    if (!pose_file.is_open()) {
        std::cerr << "Error: Failed to open DSO output file." << std::endl;
        return false;
    }

    estimated_poses.clear();
    std::string line;
    while (std::getline(pose_file, line)) {
        std::stringstream ss(line);
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        for (int i = 0; i < 16; ++i) {
            ss >> pose(i / 4, i % 4);
        }
        estimated_poses.push_back(pose);
    }

    std::cout << "DSO initialization completed: " << estimated_poses.size() << " poses estimated." << std::endl;
    return true;
}
