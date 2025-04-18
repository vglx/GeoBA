cmake_minimum_required(VERSION 3.10)
project(TestJ)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找 Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# 查找 Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# 添加你的 cpp 文件
add_executable(testJ testJ.cpp)

# 链接库
target_link_libraries(testJ
    ${OpenCV_LIBS}
    Sophus::Sophus
)
