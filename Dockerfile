FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

COPY . /app

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
    python3-dev \
    && rm -rf /var/lib/apt/lists/* && \
    ldconfig

RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.2/cmake-3.30.2-linux-x86_64.sh -O cmake.sh && \
    chmod +x cmake.sh && \
    ./cmake.sh --skip-license --prefix=/usr/local && \
    rm cmake.sh && \
    ln -sf /usr/local/bin/cmake /usr/bin/cmake

RUN wget https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-downloads.html -O intel_oneapi.sh # 示例链接
RUN bash intel_oneapi.sh -y --install-dir=/opt/intel --silent
ENV MKLROOT="/opt/intel/oneapi/mkl/latest" # 根据实际安装路径调整
ENV LD_LIBRARY_PATH="${MKLROOT}/lib/intel64:${LD_LIBRARY_PATH}"

RUN pip install --no-cache-dir \
    torch==2.3.0+cu121 \
    torchvision==0.18.0+cu121 \
    torchaudio==2.3.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    torchaudio==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p /app/build_temp

WORKDIR /app/build_temp

RUN /usr/local/bin/cmake /app -DCMAKE_BUILD_TYPE=Release || { echo "CMake configuration failed." ; exit 1; }

RUN gmake -j$(nproc) VERBOSE=1 2>&1 | tee /app/build_error.log || { \
    echo "----------------------------------------------------" ; \
    echo "C++ COMPILATION FAILED. CHECK /app/build_error.log FOR DETAILS." ; \
    echo "----------------------------------------------------" ; \
    exit 1; \
}

WORKDIR /app

RUN pip install .

CMD ["bash"]