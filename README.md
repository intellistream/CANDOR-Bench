# CANDOR-Bench: Benchmarking In-Memory Continuous ANNS under Dynamic Open-World Streams

CANDOR-Bench (Continuous Approximate Nearest neighbor search under Dynamic Open-woRld Streams) is a benchmarking framework designed to evaluate in-memory ANNS algorithms under realistic, dynamic data stream conditions. 

## Table of Contents

- [Project Structure](#Project-Structure)
- [Datasets and Algorithms](#Datasets-and-Algorithms)
  - [Summary of Datasets](#Summary-of-Datasets)
  - [Summary of Algorithms](#Summary-of-Algorithms)
- [Quick Start Guide](#quick-start-guide)
  - [Build With Docker](#Build-With-Docker)
  - [Example](#Example)
  - [Usage](#Usage)
<!--   - [Docker Support](#docker-support)
  - [Build Without Docker](#build-without-docker)
    - [Build with CUDA Support](#build-with-cuda-support)
    - [Build without CUDA (CPU-Only Version)](#build-without-cuda-cpu-only-version)
  - [Installing PyCANDY](#installing-pycandy)
  - [CLion Configuration](#clion-configuration)
- [Evaluation Scripts](#evaluation-scripts) -->
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
## Datasets and Algorithms

Our evaluation involves the following datasets and algorithms.

### Summary of Datasets

<table>
<thead>
  <tr>
    <th align="center">Category</th>
    <th align="center">Name</th>
    <th align="center">Description</th>
    <th align="center">Dimension</th>
    <th align="center">Data Size</th>
    <th align="center">Query Size</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="9" align="center"><b>Real-world</b></td>
    <td align="center">SIFT</td><td align="center">Image</td><td align="center">128</td><td align="center">1M</td><td align="center">10K</td>
  </tr>
  <tr><td align="center">OpenImagesStreaming</td><td align="center">Image</td><td align="center">512</td><td align="center">1M</td><td align="center">10K</td></tr>
  <tr><td align="center">Sun</td><td align="center">Image</td><td align="center">512</td><td align="center">79K</td><td align="center">200</td></tr>
  <tr><td align="center">SIFT100M</td><td align="center">Image</td><td align="center">128</td><td align="center">100M</td><td align="center">10K</td></tr>
  <tr><td align="center">Trevi</td><td align="center">Image</td><td align="center">4096</td><td align="center">100K</td><td align="center">200</td></tr>
  <tr><td align="center">Msong</td><td align="center">Audio</td><td align="center">420</td><td align="center">990K</td><td align="center">200</td></tr>
  <tr><td align="center">COCO</td><td align="center">Multi-Modal</td><td align="center">768</td><td align="center">100K</td><td align="center">500</td></tr>
  <tr><td align="center">Glove</td><td align="center">Text</td><td align="center">100</td><td align="center">1.192M</td><td align="center">200</td></tr>
  <tr><td align="center">MSTuring</td><td align="center">Text</td><td align="center">100</td><td align="center">30M</td><td align="center">10K</td></tr>
  <tr>
    <td rowspan="4" align="center"><b>Synthetic</b></td>
    <td align="center">Gaussian</td><td align="center">i.i.d values</td><td align="center">Adjustable</td><td align="center">500K</td><td align="center">1000</td>
  </tr>
  <tr><td align="center">Blob</td><td align="center">Gaussian Blobs</td><td align="center">768</td><td align="center">500K</td><td align="center">1000</td></tr>
  <tr><td align="center">WTE</td><td align="center">Text</td><td align="center">768</td><td align="center">100K</td><td align="center">100</td></tr>
  <tr><td align="center">FreewayML</td><td align="center">Constructed</td><td align="center">128</td><td align="center">100K</td><td align="center">1K</td></tr>
</tbody>
</table>

### Summary of Algorithms

<table>
<thead>
  <tr>
    <th style="text-align: center;">Category</th>
    <th style="text-align: center;">Algorithm Name</th>
    <th style="text-align: left;">Description</th>
    <th style="text-align: center;">Code Identifier</th>
  </tr>
</thead>
<tbody>
  <!-- Tree-based -->
  <tr>
    <td rowspan="1" align="center" style="background-color: #f0f0f0;">
      <b>Tree-based</b>
    </td>
    <td align="center">SPTAG</td>
    <td style="text-align: left;">Space-partitioning tree structure for efficient data segmentation.</td>
    <td align="center">candy_sptag</td>
  </tr>

  <!-- LSH-based -->
  <tr>
    <td rowspan="2" align="center" style="background-color: #f8f8f8;">
      <b>LSH-based</b>
    </td>
    <td align="center">LSH</td>
    <td style="text-align: left;">Data-independent hashing to reduce dimensionality and approximate nearest neighbors.</td>
    <td align="center">faiss_lsh</td>
  </tr>
  <tr>
    <td align="center">LSHAPG</td>
    <td style="text-align: left;">LSH-driven optimization using LSB-Tree to differentiate graph regions.</td>
    <td align="center">candy_lshapg</td>
  </tr>

  <!-- Clustering-based -->
  <tr>
    <td rowspan="5" align="center" style="background-color: #f0f0f0;">
      <b>Clustering-based</b>
    </td>
    <td align="center">PQ</td>
    <td style="text-align: left;">Product quantization for efficient clustering into compact subspaces.</td>
    <td align="center">faiss_pq</td>
  </tr>
  <tr>
    <td align="center">IVFPQ</td>
    <td style="text-align: left;">Inverted index with product quantization for hierarchical clustering.</td>
    <td align="center">faiss_IVFPQ</td>
  </tr>
  <tr>
    <td align="center">OnlinePQ</td>
    <td style="text-align: left;">Incremental updates of centroids in product quantization for streaming data.</td>
    <td align="center">faiss_onlinepq</td>
  </tr>
  <tr>
    <td align="center">Puck</td>
    <td style="text-align: left;">Non-orthogonal inverted indexes with multiple quantization optimized for large-scale datasets.</td>
    <td align="center">puck</td>
  </tr>
  <tr>
    <td align="center">SCANN</td>
    <td style="text-align: left;">Small-bit quantization to improve register utilization.</td>
    <td align="center">faiss_fast_scan</td>
  </tr>

  <!-- Graph-based -->
  <tr>
    <td rowspan="10" align="center" style="background-color: #f8f8f8;">
      <b>Graph-based</b>
    </td>
    <td align="center">NSW</td>
    <td style="text-align: left;">Navigable Small World graph for fast nearest neighbor search.</td>
    <td align="center">faiss_NSW</td>
  </tr>
  <tr>
    <td align="center">HNSW</td>
    <td style="text-align: left;">Hierarchical Navigable Small World for scalable search.</td>
    <td align="center">faiss_HNSW</td>
  </tr>
<!--   <tr>
    <td align="center">FreshDiskANN</td>
    <td style="text-align: left;">Streaming graph construction for large-scale proximity-based search with refined robust edge pruning.</td>
    <td align="center">FreshDiskANN</td>
  </tr> -->
  <tr>
    <td align="center">MNRU</td>
    <td style="text-align: left;">Enhances HNSW with efficient updates to prevent unreachable points in dynamic environments.</td>
    <td align="center">candy_mnru</td>
  </tr>
  <tr>
    <td align="center">Cufe</td>
    <td style="text-align: left;">Enhances FreshDiskANN with batched neighbor expansion.</td>
    <td align="center">cufe</td>
  </tr>
  <tr>
    <td align="center">Pyanns</td>
    <td style="text-align: left;">Enhances FreshDiskANN with fix-sized huge pages for optimized memory access.</td>
    <td align="center">pyanns</td>
  </tr>
  <tr>
    <td align="center">IPDiskANN</td>
    <td style="text-align: left;">Enables efficient in-place deletions for FreshDiskANN, improving update performance without reconstructions.</td>
    <td align="center">ipdiskann</td>
  </tr>
  <tr>
    <td align="center">GTI</td>
    <td style="text-align: left;">Hybrid tree-graph indexing for efficient, dynamic high-dimensional search, with optimized updates and construction.</td>
    <td align="center">gti</td>
  </tr>
</tbody>
</table>

## Quick Start Guide

---
# 🚨🚨 Strong Recommendation: Use Docker! 🚨🚨

> **We strongly recommend using Docker to build and run this project.**
>
> There are many algorithm libraries with complex dependencies. Setting up the environment locally can be difficult and error-prone.
> **Docker provides a consistent and reproducible environment, saving you time and avoiding compatibility issues.**
>
> **Note:** Building the Docker image may take **15–30 minutes** depending on your network and hardware, please be patient.

---

### Build With Docker
To build the project using Docker, simply use the provided Dockerfile located in the root directory. This ensures a consistent and reproducible environment for all dependencies and build steps.

1. To initialize and update all submodules in the project, you can run:
```
git submodule update --init --recursive
```
2. You can build the Docker image with:
```
docker build -t <your-image-name> .
```
3. Once the image is built, you can run a container from it using the following command.
```
docker run -it <your-image-name>
```
4. After entering the container, navigate to the project directory:
```
cd /app/big-ann-benchmarks
```
<!--
### Build Without Docker

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
-->
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

### Example
Prepare dataset and compute groundtruth
```
cd big-ann-benchmarks
bash scripts/compute_general.sh
```

Run general experiments 
```
bash scripts/run_general.sh
```

Wait experiments completed, and generate results, will be as gen-congestion.csv
```
python3 data_exporter.py --output gen --track congestion
```


### More Usage

All the following operations are performed in the root directory of big-ann-benchmarks.

#### 2.1 Preparing dataset
Create a small, sample dataset.  For example, to create a dataset with 10000 20-dimensional random floating point vectors, run:
```
python3 create_dataset.py --dataset random-xs
```
To see a complete list of datasets, run the following:
```
python3 create_dataset.py --help
```

#### 2.2 Running Algorithms on the **congestion** Track

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
- algorithm "$ALGO": Name of the algorithm to evaluate.Detailed names of the algorithms can be found in the "Code Identifier" column (the last column) of the "summary of algorithms" table.
- dataset "$DS": Name of the dataset to use.
- runbook_path "$PATH": Path to the runbook file describing the test scenario.For example, the runbook path for the general experiment is `neurips23/runbooks/congestion/general_experiment/general_experiment.yaml.`
- rebuild: Rebuild the target before running. 

#### 2.3 Computing Ground Truth for Runbooks

To compute ground truth for an runbook, Use the provided script to compute ground truth at various checkpoints:
```
python3 benchmark/congestion/compute_gt.py \
  --runbook "$PATH" \
  --dataset "$DS" \
  --gt_cmdline_tool ./DiskANN/build/apps/utils/compute_groundtruth
```

#### 2.4 Exporting Results
1. To make the results available for post-processing, change permissions of the results folder
```
chmod 777 -R results/
```
2. The following command will summarize all results files into a single csv file
```
python3 data_export.py --out "$OUT" --track congestion
```
The `--out` parameter "$OUT" should be adjusted according to the testing scenario. For example, the value corresponding to the general experiment is `gen`.
Common values include:
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
