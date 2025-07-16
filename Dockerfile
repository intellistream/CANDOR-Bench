FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt ./

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
    wget \
    curl && \
    rm -rf /var/lib/apt/lists/* # 清理 apt 缓存以减小镜像大小

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda
RUN rm miniconda.sh

RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN /opt/conda/bin/conda install -y mkl mkl-devel numpy scipy intel-openmp -c conda-forge -c defaults
RUN /opt/conda/bin/conda clean --all -y 

ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"
ENV MKLROOT="/opt/conda"
ENV MKL_SERVICE_FORCE_INTEL=1

RUN pip install --upgrade pip && pip install "cmake<4.0" -r requirements.txt

COPY . /app

RUN pip install .
RUN cd DiskANN && mkdir build && cd build && cmake .. -DMKL_PATH=/opt/conda -DMKL_INCLUDE_PATH=/opt/conda/include && make

RUN pip install .

CMD ["bash"]