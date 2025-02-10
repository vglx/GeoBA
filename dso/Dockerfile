# 使用 Ubuntu 16.04 作为基础镜像（较低版本依赖）
FROM ubuntu:16.04
ENV DEBIAN_FRONTEND=noninteractive

# 安装编译工具和依赖（系统自带的版本较旧）
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libeigen3-dev \
    libopencv-dev \
    libglew-dev \
    libboost-all-dev \
    libpython2.7-dev \
    libsuitesparse-dev \
    wget \
 && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /workspace

#############################
# 编译安装 Pangolin
#############################
# 为了兼容较低版本依赖，这里选取 Pangolin 的 v0.4 版本（你也可以根据需要调整版本）
RUN git clone https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    git checkout v0.4

# 编译并安装 Pangolin
WORKDIR /workspace/Pangolin/build
RUN cmake .. -DPYTHON_INCLUDE_DIR=/usr/include/python2.7 && \
    make -j$(nproc) && \
    make install && \
    ldconfig

#############################
# 编译安装 Direct Sparse Odometry (DSO)
#############################
WORKDIR /workspace
RUN git clone https://github.com/JakobEngel/dso.git

WORKDIR /workspace/dso/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

#############################
# 默认进入交互式 shell，或者你也可以直接运行编译好的 dso 可执行文件
#############################
CMD ["/bin/bash"]
