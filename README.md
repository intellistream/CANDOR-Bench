# CANDOR-Bench: Benchmarking In-Memory Continuous ANNS under Dynamic Open-World Streams

CANDOR-Bench (Continuous Approximate Nearest neighbor search under Dynamic Open-woRld Streams) is a benchmarking framework designed to evaluate in-memory ANNS algorithms under realistic, dynamic data stream conditions. 

## Table of Contents

- [Project Structure](#Project-Structure)
- [Quick Start Guide](#quick-start-guide)
  - [Build Without Docker](Build-Without-Docker)
  - [Build With Docker](Build-With-Docker)
- [Usage](#Usage)
<!--   - [Docker Support](#docker-support)
  - [Build Without Docker](#build-without-docker)
    - [Build with CUDA Support](#build-with-cuda-support)
    - [Build without CUDA (CPU-Only Version)](#build-without-cuda-cpu-only-version)
  - [Installing PyCANDY](#installing-pycandy)
  - [CLion Configuration](#clion-configuration)
- [Evaluation Scripts](#evaluation-scripts) -->
- [Additional Information](#additional-information) 
---

## Project Structure
<!--
- **[`big-ann-benchmarks/`]**  
  The core benchmarking framework of CANDOR-Bench, responsible for evaluation logic and stream orchestration.

- **[`GTI/`]**  
  External project integrated to support the GTI algorithm.

- **[`DiskANN/`]**  
  External project including FreshDiskANN, Pyanns, and Cufe, adapted for streaming evaluation.

- **[`src/`](./src/)**  
  Source directory containing the majority of the ANNS algorithms evaluated in the benchmark.

- **[`Dockerfile`](./Dockerfile)**  
  Provides a fully reproducible Docker environment for deploying and running CANDOR-Bench.
-->
```
CANDY-Benchmark/
├── benchmark/             
├── big-ann-benchmarks/             # Core benchmarking framework (Dynamic Open-World conditions)
│   ├── README.md
│   ├── algos-2021.yaml
│   ├── benchmark/
│   │   ├── algorithms/             # Concurrent Track
│   │   ├── concurrent/             # Congestion Track
│   │   ├── congestion/
│   │   ├── amin.py
│   │   ├── runner.py
│   │   └── ……
│   ├── create_dataset.py
│   ├── dataset_preparation/
│   ├── eval/
│   ├── install/
│   ├── install.py
│   ├── logging.conf
│   ├── neurips21/
│   ├── neurips23/                  # NeurIPS'23 benchmark configurations and scripts
│   │   ├── concurrent/             # Concurrent Track
│   │   ├── congestion/             # Congestion Track
│   │   ├── filter/
│   │   ├── ood/
│   │   ├── runbooks/               # Dynamic benchmark scenario definitions (e.g., T1, T3, etc.)
│   │   ├── sparse/
│   │   ├── streaming/              
│   │   └── ……
│   └──……
├── DiskANN/                        # Integrated DiskANN-based algorithms
├── GTI/                            # Integrated GTI algorithm source
├── src/                            # Main algorithm implementations
├── test/
├── include/                        # C++ header files
├── doc/
├── docker/
├── figures/
├── cmake/
├── thirdparty/                     # External dependencies
├── Dockerfile                      # Docker build recipe
├── buildCPUOnly.sh
├── buildWithCuda.sh
├── requirements.txt
├── setup.py                        # Python package setup
├── CMakeLists.txt
├── README.md
└── ……
```
## 1. Quick Start Guide

### Build With Docker

---
# 🚨🚨🚨 Strong Recommendation: Use Docker! 🚨🚨🚨

> **We strongly recommend using Docker to build and run this project.**
>
> There are many algorithm libraries with complex dependencies. Setting up the environment locally can be difficult and error-prone.
> **Docker provides a consistent and reproducible environment, saving you time and avoiding compatibility issues.**
>
> **Note:** Building the Docker image may take **10–20 minutes** depending on your network and hardware.

---

```bash
git submodule update --init --recursive
```
This pulls in all third-party dependencies, including:
- DiskANN/ (with FreshDiskANN, Pyanns, Cufe, etc.)
- GTI/
- IP-DiskANN/ 
- big-ann-benchmarks/

#### 2. Build the Docker image

```bash
docker build -t candor .
```
This will build the Docker image named `candor`.

#### 3. Enter the container

```bash
docker run -it --rm candor
```
This command will start an interactive shell inside the container (default path: `/app`).

#### 4. Scripts for Paper Sections

The `big-ann-benchmarks/scripts/` directory provides ready-to-use scripts for reproducing the experiments in different sections of the paper.  
Each script corresponds to a specific benchmark or experiment described in the paper. For example:

- `run_general.sh` — Main benchmark for Section 4.1: General ANNS evaluation
- `run_congestion.sh` — Section 4.2: Congestion Track experiments
- `run_concurrent.sh` — Section 4.3: Concurrent Track experiments
- `run_ood.sh` — Section 4.4: Out-of-Distribution (OOD) evaluation
- `run_sparse.sh` — Section 4.5: Sparse data benchmark
- `run_streaming.sh` — Section 4.6: Streaming scenario evaluation

> **Tip:**  
> You can edit the scripts in `big-ann-benchmarks/scripts/` to specify the algorithms and datasets you want to test.  
> The available algorithm and dataset names can be found in the next section of this README.

#### 5. Run benchmark scripts

Navigate to the scripts directory and run the desired script. For example:

```bash
cd big-ann-benchmarks
bash scripts/run_general.sh
```

> **Tip:**  
> You can freely modify the scripts (e.g., in `big-ann-benchmarks/scripts/`) on your local machine at any time.  
> For development and debugging, it is recommended to **edit your scripts after building the Docker image**.  
> 
> If you want your changes to take effect inside the container immediately, you can mount your local scripts directory into the container using the `-v` option:
> 
> ```bash
> docker run -it --rm -v /absolute/path/to/your/scripts:/app/big-ann-benchmarks/scripts candor
> ```
> 
> This way, any changes you make to the scripts on your host will be instantly reflected inside the container, and you do **not** need to rebuild the Docker image for every modification.

---

## 2. Algorithm and Datasets
 