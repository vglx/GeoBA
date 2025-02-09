# 使用 Ubuntu 22.04 作为基础镜像
FROM ubuntu:22.04

# 设置时区，防止 tzdata 交互式安装卡住
ENV DEBIAN_FRONTEND=noninteractive

# 更新软件包索引并安装基本工具
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    git \
    wget \
    unzip \
    pkg-config \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libopencv-dev \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 CMake 3.27.6
RUN wget -O cmake.sh https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6-linux-x86_64.sh && \
    chmod +x cmake.sh && \
    ./cmake.sh --prefix=/usr/local --skip-license && \
    rm cmake.sh && \
    cmake --version

# 手动编译 Eigen 3.4.0（避免 Eigen 版本不兼容）
RUN git clone --branch 3.4.0 https://gitlab.com/libeigen/eigen.git /opt/eigen && \
    cd /opt/eigen && mkdir build && cd build && \
    cmake .. && make install

# 手动编译 Ceres Solver 2.1.0
WORKDIR /opt
RUN git clone --branch 2.1.0 https://ceres-solver.googlesource.com/ceres-solver && \
    cd ceres-solver && mkdir build && cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j$(nproc) && make install && \
    ldconfig

# 安装 TinyObjLoader（用于解析 OBJ 文件）
RUN git clone https://github.com/tinyobjloader/tinyobjloader.git /opt/tinyobjloader && \
    cd /opt/tinyobjloader && mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install

RUN git clone https://github.com/strasdat/Sophus.git /opt/sophus && \
    cd /opt/sophus && mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install

# 创建工作目录
WORKDIR /app

# 复制项目文件（使用 .dockerignore 忽略无关文件）
COPY . .

# 创建构建目录
RUN mkdir -p /app/build
WORKDIR /app/build

# 运行 CMake 配置并编译
RUN cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# 运行容器时的默认命令
CMD ["./GeoBA"]