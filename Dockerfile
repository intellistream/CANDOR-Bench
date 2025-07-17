FROM ubuntu:22.04

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

# Install Intel oneAPI MKL via APT repository
# This method is more reliable than direct wget from a general download page.
RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg && \
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
