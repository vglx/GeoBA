cmake_minimum_required(VERSION 3.10)

# 项目名称
project(GeoBA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译模式（优化 Release 版）
set(CMAKE_BUILD_TYPE Release)

# 方便调试：生成 compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 包含头文件路径
include_directories(
    include
    /opt/eigen
    /opt/tinyobjloader
)

# 添加源文件
set(SOURCE_FILES
    src/CostFunctions.cpp
    src/DatasetManager.cpp
    src/ImageProcessor.cpp
    src/MeshModel.cpp
    src/Optimizer.cpp
    src/Projection.cpp
    main.cpp
)

# 添加可执行文件
add_executable(GeoBA ${SOURCE_FILES})

# 查找 OpenCV, Eigen, Ceres
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

# 链接必要的库
target_link_libraries(GeoBA
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ceres
    tinyobjloader
)