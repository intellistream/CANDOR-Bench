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
    wget \
    curl \
    cmake \
    && rm -rf /var/lib/apt/lists/* && \
    ldconfig

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
ENV MKLROOT="/opt/conda"
ENV MKL_SERVICE_FORCE_INTEL=1

RUN rm -f /root/.condarc && \
    /opt/conda/bin/conda config --set always_yes true

RUN /opt/conda/bin/conda init

RUN /opt/conda/bin/conda update -n base conda -y -c conda-forge && \
    /opt/conda/bin/conda install -n base conda-libmamba-solver -y -c conda-forge

RUN /opt/conda/bin/conda config --add channels conda-forge && \
    /opt/conda/bin/conda config --set channel_priority strict && \
    /opt/conda/bin/conda config --set solver libmamba

RUN /opt/conda/bin/conda clean --index-cache -y

RUN /opt/conda/bin/conda --version && \
    /opt/conda/bin/conda config --show channels && \
    /opt/conda/bin/conda config --show solver && \
    /opt/conda/bin/conda info --all && \
    ls -l /usr/lib/x86_64-linux-gnu/libarchive*

RUN /opt/conda/bin/conda install -n base python=3.10 -y -c conda-forge

RUN /opt/conda/bin/conda install -y -c conda-forge \
    mkl=2024.1 \
    mkl-devel=2024.1 \
    numpy=1.26.4 \
    scipy=1.14.0 \
    intel-openmp=2024.1 && \
    /opt/conda/bin/conda clean --all -y

RUN pip install --upgrade pip && \
    pip install -r big-ann-benchmarks/requirements_py3.10.txt

RUN pip install .

RUN cd DiskANN && \
    mkdir -p build && cd build && \
    cmake .. -DMKL_PATH=/opt/conda -DMKL_INCLUDE_PATH=/opt/conda/include && \
    make

RUN cd GTI/GTI/extern_libraries/n2 && \
    mkdir -p build && \
    make shared_lib

RUN cd GTI/GTI && \
    mkdir -p bin build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && \
    make install

CMD ["bash"]