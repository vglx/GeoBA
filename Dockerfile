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
    libeigen3-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Ceres Solver（Ubuntu 22.04 可能已经内置 `libceres-dev`）
RUN apt-get update && apt-get install -y \
    libceres-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# 复制项目文件
COPY . .

# 创建构建目录
RUN mkdir -p build
WORKDIR /app/build

# 运行 CMake 配置并编译
RUN cmake .. && make -j$(nproc)

# 运行容器时的默认命令
CMD ["./GeoBA"]
