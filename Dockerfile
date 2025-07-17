FROM ubuntu:22.04

WORKDIR /app

COPY . /app

# Install core system dependencies, including gnupg for reliability
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip git build-essential \
    liblapack-dev libblas-dev libopenblas-dev \
    libboost-all-dev \
    libnuma-dev \
    libgflags-dev libgoogle-glog-dev \
    swig \
    libhdf5-dev \
    libaio-dev \
    libgoogle-perftools-dev \
    libomp-dev \
    libtbb-dev \
    libarchive-dev \
    libcurl4-openssl-dev \
    wget \
    curl \
    gnupg \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* && \
    ldconfig

RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.2/cmake-3.30.2-linux-x86_64.sh -O cmake.sh && \
    chmod +x cmake.sh && \
    ./cmake.sh --skip-license --prefix=/usr/local && \
    rm cmake.sh && \
    ln -sf /usr/local/bin/cmake /usr/bin/cmake

# Install Intel oneAPI MKL via APT repository
# This method is more reliable than direct wget from a general download page.
# gnupg is now ensured to be installed in the first RUN command.
RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    intel-oneapi-mkl-devel \
    && rm -rf /var/lib/apt/lists/*

ENV MKLROOT="/opt/intel/oneapi/mkl/latest"
ENV LD_LIBRARY_PATH="${MKLROOT}/lib/intel64:${LD_LIBRARY_PATH}"

# Installing CPU-only version of PyTorch
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    torchaudio==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# *** 新增：设置 Torch_DIR 环境变量 ***
# 这是最常见的 pip 安装路径，假设 Python 3.10
ENV Torch_DIR="/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch"
# 如果你的 Python 版本不是 3.10，或者 find 命令结果不是 dist-packages，请修改此行。
# 验证方法：
# 1. 临时将 CMD ["bash"] 放在这里（即 ENV Torch_DIR 下面）。
# 2. docker build 到此步。
# 3. docker run -it your_image_name bash
# 4. 在容器内执行 find /usr/local -name "TorchConfig.cmake" 确认准确路径。

# *** 恢复并修复 CMake 和 Make 构建步骤 ***
RUN mkdir -p /app/build_temp

WORKDIR /app/build_temp

RUN /usr/local/bin/cmake /app -DCMAKE_BUILD_TYPE=Release || { echo "CMake configuration failed." ; exit 1; }

# 移除了 VERBOSE=1，因为 -Werror 已经足够清晰了
RUN gmake -j$(nproc) 2>&1 | tee /app/build_error.log || { \
    echo "----------------------------------------------------" ; \
    echo "C++ COMPILATION FAILED. CHECK /app/build_error.log FOR DETAILS." ; \
    cat /app/build_error.log ; \
    echo "----------------------------------------------------" ; \
    exit 1; \
}

RUN cat /app/build_error.log

WORKDIR /app

# RUN pip install .

CMD ["bash"]