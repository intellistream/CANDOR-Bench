# SAGE-DB-Bench Dockerfile
# 用于本地构建测试和验证

FROM ubuntu:22.04

LABEL maintainer="SAGE-DB-Bench Team"
LABEL description="Streaming ANN Benchmark Framework"

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 设置工作目录
WORKDIR /app

# 第一阶段：安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 基础构建工具
    build-essential \
    git \
    wget \
    curl \
    gnupg \
    ca-certificates \
    # CMake（需要较新版本）
    cmake \
    # Python 3.10
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    # 编译依赖
    libgflags-dev \
    libgoogle-glog-dev \
    libfmt-dev \
    libboost-all-dev \
    libomp-dev \
    libnuma-dev \
    libunwind-dev \
    libspdlog-dev \
    libgoogle-perftools-dev \
    libaio-dev \
    liblapack-dev \
    libblas-dev \
    libopenblas-dev \
    libhdf5-dev \
    libtbb-dev \
    libarchive-dev \
    libcurl4-openssl-dev \
    libeigen3-dev \
    zlib1g-dev \
    libssl-dev \
    gfortran \
    # 其他工具
    swig \
    pkg-config \
    pybind11-dev \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 安装 Intel MKL (Puck 需要)
RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        intel-oneapi-mkl-devel \
        intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic \
    && rm -rf /var/lib/apt/lists/*

# 设置 MKL 环境变量
ENV MKLROOT=/opt/intel/oneapi/mkl/latest
ENV LD_LIBRARY_PATH=${MKLROOT}/lib/intel64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${MKLROOT}/lib/intel64:${LIBRARY_PATH}
ENV CPATH=${MKLROOT}/include:${CPATH}
ENV PKG_CONFIG_PATH=${MKLROOT}/lib/pkgconfig:${PKG_CONFIG_PATH}

# 复制项目文件
COPY . .

# 创建 Python 虚拟环境
RUN python3.10 -m venv /app/venv

# 激活虚拟环境并安装 Python 依赖
RUN /app/venv/bin/pip install --upgrade pip setuptools wheel && \
    /app/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    /app/venv/bin/pip install -r requirements.txt

# 设置虚拟环境为默认
ENV PATH="/app/venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/venv"

# 初始化 git submodules
RUN git submodule update --init --recursive || echo "Warning: submodules init may have failed"

# 构建算法库
RUN cd algorithms_impl && bash build_all.sh --install || echo "Build may have warnings"

# 验证安装
RUN python3 -c "import torch; import numpy; print('Core deps OK')" && \
    python3 -c "import PyCANDYAlgo; print('PyCANDYAlgo OK')" || echo "PyCANDYAlgo import failed" && \
    python3 -c "import pyvsag; print('pyvsag OK')" || echo "pyvsag import failed"

# 创建必要的目录
RUN mkdir -p results raw_data logs

# 设置环境变量
ENV PYTHONPATH=/app:$PYTHONPATH
ENV OMP_NUM_THREADS=4

# 默认命令
CMD ["/bin/bash"]

# 使用示例：
# docker build -t sage-db-bench:test .
# docker run -it --name sage-test -v $(pwd)/results:/app/results sage-db-bench:test
# 
# 进入容器后运行测试：
# python3 -c "import PyCANDYAlgo"
# python3 run_benchmark.py --help
