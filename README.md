# CANDOR-Bench: Benchmarking In-Memory Continuous ANNS under Dynamic Open-World Streams

CANDOR-Bench (Continuous Approximate Nearest neighbor search under Dynamic Open-woRld Streams) is a benchmarking framework designed to evaluate in-memory ANNS algorithms under realistic, dynamic data stream conditions. 

## Table of Contents

- [Project Structure](#Project-Structure)
- [Quick Start Guide](#quick-start-guide)
  - [Build With Docker](#Build-With-Docker)
  - [Build Without Docker](#Build-Without-Docker)
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
│   ├── benchmark/
│   │   ├── algorithms/             # Concurrent Track
│   │   ├── concurrent/             # Congestion Track
│   │   ├── congestion/
│   │   ├── main.py
│   │   ├── runner.py
│   │   └── ……
│   ├── create_dataset.py
│   ├── requirements_py3.10.txt
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
├── IP-DiskANN/                     # Integrated IP-DiskANN algorithm source
├── src/                            # Main algorithm implementations
├── include/                        # C++ header files
├── thirdparty/                     # External dependencies
├── Dockerfile                      # Docker build recipe
├── requirements.txt
├── setup.py                        # Python package setup
└── ……
```
## 1. Quick Start Guide

### Build With Docker
=======
Support for building and running CANDOR-Bench via Docker is currently under development. Please stay tuned for updates.

### Build Without Docker

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
=======
- IP-DiskANN/
  
#### 2. Install System Dependencies

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
=======
#### 7. Install Python Interface

```bash
pip install .
```

#### 8. Install Python dependencies for big-ann-benchmarks

```bash
pip install -r requirements_py3.10.txt
```
#### 9. Build GTI

```bash
cd GTI/GTI/extern_libraries/n2
mkdir build
make shared_lib

cd ../../
mkdir bin
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

#### 10. Build DiskANN

```bash
cd DiskANN
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

#### 11. Build DiskANN

```bash
cd IP-DiskANN
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

<!-- 
## Quick Start Guide old

### Docker Support

We provide Docker support to simplify the setup process.

1. **Navigate to the `./docker` directory:**

   ```shell
   cd ./docker
   ```

2. **Build and start the Docker container:**

   ```shell
   ./start.sh
   ```

   This script will build the Docker container and start it.

3. **Inside the Docker container, run the build script to install dependencies and build the project:**

  - **With CUDA support:**

    ```shell
    ./buildWithCuda.sh
    ```

  - **Without CUDA (CPU-only version):**

    ```shell
    ./buildCPUOnly.sh
    ```

### Build Without Docker

If you prefer to build without Docker, follow these steps.

#### Build with CUDA Support

To build CANDY and PyCANDY with CUDA support:

```shell
./buildWithCuda.sh
```

#### Build without CUDA (CPU-Only Version)

For a CPU-only version:

```shell
./buildCPUOnly.sh
```

These scripts will install dependencies and build the project.

### Installing PyCANDY

After building, you can install PyCANDY to your default Python environment:

```shell
python3 setup.py install --user
```

### CLion Configuration

When developing in CLion, you must manually configure:

1. **CMake Prefix Path:**


### Requires BLAS, LAPACK, boost and swig

```shell
sudo apt install liblapack-dev libblas-dev libboost-all-dev swig
```

  - Run the following command in your terminal to get the CMake prefix path:

    ```shell
    python3 -c 'import torch; print(torch.utils.cmake_prefix_path)'
    ```


  - Copy the output path and set it in CLion's CMake settings as:

    ```
    -DCMAKE_PREFIX_PATH=<output_path>
    ```

2. **Environment Variable `CUDACXX`:**

  - Manually set the environment variable `CUDACXX` to:

    ```
    /usr/local/cuda/bin/nvcc
    ```

## Evaluation Scripts

Evaluation scripts are located under `benchmark/scripts`.

To run an evaluation (e.g., scanning the dimensions):

```shell
cd build/benchmark/scripts/scanIPDimensions
sudo ls  # Required for perf events
python3 drawTogether.py 2
cd ../figures
```

Figures will be generated in the `figures` directory.

---
-->
## Usage

### 1. Preparing dataset
Create a small, sample dataset.  For example, to create a dataset with 10000 20-dimensional random floating point vectors, run:
```
python create_dataset.py --dataset random-xs
```
To see a complete list of datasets, run the following:
```
python create_dataset.py --help
```

### 2. Running Algorithms on the **congestion** Track

To evaluate an algorithm under the `congestion` track, use the following command:
```bash
python3 run.py \
  --neurips23track congestion \
  --algorithm "$ALGO" \
  --nodocker \
  --rebuild \
  --runbook_path "$PATH" \
  --dataset "$DS"
```
- algorithm "$ALGO": Name of the algorithm to evaluate.
- dataset "$DS": Name of the dataset to use.
- runbook_path "$PATH": Path to the runbook file describing the test scenario.
- rebuild: Rebuild the target before running.

### 3. Computing Ground Truth for Runbooks

To compute ground truth for an runbook:
1. **Clone and build the [DiskANN repository](https://github.com/Microsoft/DiskANN)**
2. Use the provided script to compute ground truth at various checkpoints:
```
python3 benchmark/congestion/compute_gt.py \
  --runbook "$PATH_TO_RUNBOOK" \
  --dataset "$DATASET_NAME" \
  --gt_cmdline_tool ~/DiskANN/build/apps/utils/compute_groundtruth
```

### 4. Exporting Results
1. To make the results available for post-processing, change permissions of the results folder
```
sudo chmod 777 -R results/
```
2. The following command will summarize all results files into a single csv file
```
python data_export.py --out "$OUT" --track congestion
```
The `--out` path "$OUT" should be adjusted according to the testing scenario. Common values include:
- `gen`
- `batch`
- `event`
- `conceptDrift`
- `randomContamination`
- `randomDrop`
- `wordContamination`
- `bulkDeletion`
- `batchDeletion`
- `multiModal`
- ……

## Additional Information

<details>
<summary><strong>Click to Expand</strong></summary>

### Table of Contents

- [Extra CMake Options](#extra-cmake-options)
- [Manual Build Instructions](#manual-build-instructions)
  - [Requirements](#requirements)
  - [Build Steps](#build-steps)
  - [CLion Build Tips](#clion-build-tips)
- [CUDA Installation (Optional)](#cuda-installation-optional)
  - [Install CUDA (if using CUDA-based Torch)](#install-cuda-if-using-cuda-based-torch)
  - [CUDA on Jetson Devices](#cuda-on-jetson-devices)
- [Torch Installation](#torch-installation)
  - [Install Python and Pip](#install-python-and-pip)
  - [Install PyTorch](#install-pytorch)
- [PAPI Support (Optional)](#papi-support-optional)
  - [Build PAPI](#build-papi)
  - [Verify PAPI Installation](#verify-papi-installation)
  - [Enable PAPI in CANDY](#enable-papi-in-candy)
- [Distributed CANDY with Ray (Optional)](#distributed-candy-with-ray-optional)
  - [Build with Ray Support](#build-with-ray-support)
  - [Running with Ray](#running-with-ray)
  - [Ray Dashboard (Optional)](#ray-dashboard-optional)
- [Local Documentation Generation (Optional)](#local-documentation-generation-optional)
  - [Install Required Packages](#install-required-packages)
  - [Generate Documentation](#generate-documentation)
    - [Accessing Documentation](#accessing-documentation)
- [Known Issues](#known-issues)
>>>>>>> dd068b958060f24e4d0c2cdf899f229efccc0b2b

---

## 2. Algorithm and Datasets
