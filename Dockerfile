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
    gnupg \
    libfmt-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* && \
    ldconfig

RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.2/cmake-3.30.2-linux-x86_64.sh -O cmake.sh && \
    chmod +x cmake.sh && \
    ./cmake.sh --skip-license --prefix=/usr/local && \
    rm cmake.sh && \
    ln -sf /usr/local/bin/cmake /usr/bin/cmake

RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    intel-oneapi-mkl-devel \
    && rm -rf /var/lib/apt/lists/*

ENV MKLROOT="/opt/intel/oneapi/mkl/latest"
ENV LD_LIBRARY_PATH="${MKLROOT}/lib/intel64:${LD_LIBRARY_PATH}"

RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    torchaudio==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

ENV Torch_DIR="/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch"

WORKDIR /app
RUN pip install .

WORKDIR /app/GTI/GTI/extern_libraries/n2
RUN mkdir -p build && make shared_lib

WORKDIR /app/GTI/GTI
RUN mkdir -p bin build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j && make install

WORKDIR /app/DiskANN
RUN mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j && make install

WORKDIR /app/IP-DiskANN
RUN mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j && make install

CMD ["bash"]