cmake_minimum_required(VERSION 3.10)

project(GeoBA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(
    include
    /opt/eigen
    /opt/tinyobjloader
)

set(SOURCE_FILES
    src/CostFunctions.cpp
    src/DatasetManager.cpp
    src/ImageProcessor.cpp
    src/MeshModel.cpp
    src/Optimizer.cpp
    src/Projection.cpp
    main.cpp
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(GeoBA ${SOURCE_FILES})

# 将 OpenMP 库变量设定为空字符串（如果没有找到）
if(OpenMP_CXX_FOUND)
    set(OPENMP_LIB OpenMP::OpenMP_CXX)
else()
    set(OPENMP_LIB "")
endif()

target_link_libraries(GeoBA
    PUBLIC
        ${OpenCV_LIBS}
        Eigen3::Eigen
        ceres
        tinyobjloader
        ${OPENMP_LIB}
)
