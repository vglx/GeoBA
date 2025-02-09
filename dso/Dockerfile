# 使用 Ubuntu 20.04 作为基础镜像
FROM ubuntu:20.04

# 设置非交互模式，避免 tzdata 等包安装时交互卡住
ENV DEBIAN_FRONTEND=noninteractive

# 更新软件包列表并安装基本依赖，同时添加一些 Pangolin/DSO 可能需要的依赖
RUN apt-get update && apt-get install -y \
    git wget build-essential pkg-config \
    g++ gcc \
    libeigen3-dev libboost-all-dev \
    libtbb-dev \
    libglew-dev libglfw3-dev \
    libopenblas-dev libatlas-base-dev libsuitesparse-dev \
    libepoxy-dev \
    ffmpeg libavcodec-dev libavutil-dev libavformat-dev \
    libx11-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev \
    libpng-dev libjpeg-dev && \
    rm -rf /var/lib/apt/lists/*

# 移除系统自带旧版本的 CMake（如果存在）
RUN apt-get remove -y cmake && apt-get autoremove -y

# 下载并安装 CMake 3.22
RUN wget https://cmake.org/files/v3.22/cmake-3.22.0-linux-x86_64.tar.gz && \
    tar -xzvf cmake-3.22.0-linux-x86_64.tar.gz -C /usr/local --strip-components=1 && \
    ln -sf /usr/local/bin/cmake /usr/bin/cmake && \
    cmake --version

# 安装 OpenCV 3.4.16
WORKDIR /opt
RUN git clone --branch 3.4.16 https://github.com/opencv/opencv.git && \
    git clone --branch 3.4.16 https://github.com/opencv/opencv_contrib.git && \
    cd opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D WITH_TBB=ON \
          -D WITH_V4L=ON \
          -D WITH_OPENGL=ON \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          .. && \
    make -j$(nproc) && make install && ldconfig

# 编译并安装 Pangolin
RUN git clone https://github.com/stevenlovegrove/Pangolin.git /opt/Pangolin && \
    cd /opt/Pangolin && \
    mkdir build && cd build && \
    cmake .. -DBUILD_PANGOLIN_GUI=ON -DBUILD_PANGOLIN_VIDEO=ON -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig

# 设置 pkg-config 搜索路径（用于查找 Pangolin、OpenCV 等 pkg-config 文件）
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# 复制 DSO 项目代码
WORKDIR /app
COPY . .

# 编译 DSO（假设 DSO 工程根目录中已有合适的 CMakeLists.txt）
RUN mkdir -p build
WORKDIR /app/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/usr/local && \
    make -j$(nproc)