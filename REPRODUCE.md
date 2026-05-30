# Reproducibility Package: ANN-CC-Bench

## Overview

This document provides comprehensive instructions for reproducing all experimental results presented in the SIGMOD paper "Rethinking Concurrent Vector Search: From Segment Isolation to Graph-Native MVCC with Computation Sharing".

**Paper Claims Reproduced:**
- S3 (Piggyback) computation sharing effectiveness 
- MVCC vs Segment vs Lock concurrent performance comparison
- Workload generalization analysis across noise levels
- P99 tail latency analysis under concurrent load
- Memory overhead comparison across architectures

## System Requirements

### Hardware Requirements

**Minimum Requirements:**
- CPU: 16 cores, 2.4GHz+ (tested on Intel Xeon Gold 6128)
- RAM: 32GB DDR4 (experiments use up to 16GB)  
- Storage: 100GB available SSD space
- Network: Not required (all experiments are local)

**Recommended Configuration:**
- CPU: 32+ cores for full concurrent benchmarks (96 threads tested)
- RAM: 64GB for large-scale experiments (10M+ vectors)
- Storage: 500GB NVMe SSD for fast dataset I/O
- OS: Ubuntu 20.04+ or similar Linux distribution

**Hardware Notes:**
- Experiments are CPU and memory intensive
- Thread scaling tests require many physical cores
- Memory bandwidth affects concurrent performance significantly
- Results may vary on different CPU architectures (ARM, AMD)

### Software Dependencies

#### Core Build Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    libomp-dev \
    libopenblas-dev \
    pkg-config

# CentOS/RHEL  
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake git python3 python3-pip openblas-devel
```

#### Specific Version Requirements
- **GCC**: 9.0+ (C++17 support required)
- **CMake**: 3.16+ 
- **Python**: 3.8+
- **OpenMP**: 4.5+ (for parallel HNSW construction)

#### Python Dependencies
```bash
pip3 install -r requirements.txt
```

Contents of `requirements.txt`:
```
numpy>=1.21.0
matplotlib>=3.5.0  
pandas>=1.3.0
scipy>=1.7.0
h5py>=3.1.0
```

#### Optional Dependencies
```bash
# For advanced profiling (optional)
sudo apt install -y linux-tools-common linux-tools-generic perf

# For memory analysis (optional)  
sudo apt install -y valgrind massif-visualizer

# For plotting (optional)
pip3 install seaborn plotly
```

### Verified Platforms

| Platform | Architecture | Status | Notes |
|----------|--------------|---------|-------|
| Ubuntu 20.04 | x86_64 | ✅ Fully tested | Primary development platform |
| Ubuntu 22.04 | x86_64 | ✅ Fully tested | CI/CD platform |
| CentOS 8 | x86_64 | ⚠️ Partial | Some package name differences |
| macOS 12+ | arm64 | ⚠️ Partial | OpenMP configuration required |
| Windows 11 | x86_64 | ❌ Not supported | Use WSL2 instead |

## Dataset Requirements

### Required Datasets

All datasets must be downloaded before running experiments:

#### 1. SIFT Dataset (Required)
```bash
# Download SIFT 1M dataset
wget http://corpus-texmex.irisa.fr/sift.tar.gz
tar -xzf sift.tar.gz
mv sift/ data/sift/

# Expected files:
# data/sift/sift_base.fvecs (1M base vectors, 128D)
# data/sift/sift_query.fvecs (10K query vectors, 128D)
# data/sift/sift_groundtruth.ivecs (ground truth neighbors)
```

#### 2. GIST Dataset (Required) 
```bash
wget http://corpus-texmex.irisa.fr/gist.tar.gz  
tar -xzf gist.tar.gz
mv gist/ data/gist/
```

#### 3. Deep1M Dataset (Optional)
```bash
wget http://sites.skoltech.ru/compvision/noimi/deep1m.tar.gz
tar -xzf deep1m.tar.gz  
mv deep1m/ data/deep1m/
```

#### 4. MSong Dataset (Optional)
```bash
# Convert from Million Song Dataset subset
python3 utils/convert_msong.py --input msong_raw --output data/msong/
```

### Dataset Storage Structure
```
data/
├── sift/
│   ├── sift_base.fvecs          # 1M x 128D base vectors
│   ├── sift_query.fvecs         # 10K x 128D query vectors  
│   └── sift_groundtruth.ivecs   # Ground truth k-NN
├── gist/
│   ├── gist_base.fvecs          # 1M x 960D base vectors
│   ├── gist_query.fvecs         # 1K x 960D query vectors
│   └── gist_groundtruth.ivecs   # Ground truth k-NN  
└── [other datasets...]
```

### Dataset Size Requirements

| Dataset | Base Size | Query Size | Disk Space | Memory Usage |
|---------|-----------|------------|------------|--------------|
| SIFT 1M | 512MB | 5MB | 600MB | 2GB peak |
| GIST 1M | 3.8GB | 4MB | 4GB | 8GB peak |
| Deep1M | 1GB | 5MB | 1.2GB | 4GB peak |
| All datasets | - | - | **6GB** | **14GB peak** |

## Build Instructions

### Quick Start (5 minutes)
```bash
git clone --recursive https://github.com/your-org/ANN-CC-bench.git
cd ANN-CC-bench

# Download datasets (run in background)
./scripts/download_datasets.sh &

# Build all experiments 
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Wait for dataset download to complete
wait

# Run basic validation
./validate_setup
```

### Detailed Build Process

#### 1. Clone Repository
```bash
git clone --recursive https://github.com/your-org/ANN-CC-bench.git
cd ANN-CC-bench

# Ensure submodules are updated
git submodule update --init --recursive
```

#### 2. Configure Build
```bash
mkdir build && cd build

# Release build (for benchmarking)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DBUILD_PROFILING=ON \
    -DUSE_OPENMP=ON

# Debug build (for development)
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_TESTS=ON \
    -DSANITIZE_ADDRESS=ON
```

#### 3. Build All Targets
```bash
# Build everything (10-15 minutes on 16 cores)
make -j$(nproc)

# Or build specific targets
make mvcc_benchmark        # Core MVCC benchmarks
make s3_experiments       # S3 deep-dive experiments  
make concurrent_benchmark # Concurrent RW benchmarks
make memory_analysis      # Memory overhead tools
```

#### 4. Validate Installation
```bash
# Run validation suite (5 minutes)
make test

# Validate datasets are properly loaded
./validate_datasets

# Quick smoke test of all architectures
./smoke_test
```

### Build Configuration Options

| Option | Description | Default | Impact |
|--------|-------------|---------|--------|
| `CMAKE_BUILD_TYPE` | Release/Debug | Release | Performance |
| `BUILD_TESTS` | Unit tests | ON | Build time |
| `BUILD_PROFILING` | Profiling tools | OFF | Build time |
| `USE_OPENMP` | Parallel construction | ON | Performance |
| `SANITIZE_ADDRESS` | Memory debugging | OFF | Runtime speed |
| `OPTIMIZE_NATIVE` | CPU-specific optimizations | OFF | Portability |

## Experiment Execution

### Core Experiments (Paper Results)

#### 1. S3 Deep-Dive Analysis (~2 hours)
```bash
cd experiments/helping/build

# Run S3 effectiveness analysis  
./exp_s3_deepdive --dataset sift --output results/s3/
./exp_s3_deepdive --dataset gist --output results/s3/  
./exp_s3_deepdive --dataset deep1m --output results/s3/

# Generate analysis reports
python3 ../../../scripts/analyze_s3_results.py results/s3/
```

**Expected Output:**
- `results/s3/deepdive_sift_1M.log` - S3 performance across noise levels
- `results/s3/s3_deepdive_results.csv` - Raw numerical results
- `WORKLOAD_GENERALIZATION.md` - Analysis summary

#### 2. Concurrent Performance Comparison (~1 hour)
```bash
# Run concurrent read-write benchmarks
./concurrent_benchmark \
    --architectures mvcc,segment,lock \
    --datasets sift,gist,deep1m \
    --threads "1+1,1+4,4+4,1+8,4+8" \
    --output results/concurrent/

# Generate latency analysis
python3 ../../../scripts/analyze_latency.py results/concurrent/
```

**Expected Output:**  
- `results/concurrent/sift_concurrent_rw.txt` - Per-dataset results
- `LATENCY_ANALYSIS.md` - P99 tail latency analysis

#### 3. Memory Overhead Analysis (~30 minutes)
```bash
# Run memory profiling
python3 memory_analysis.py > results/MEMORY_ANALYSIS.md

# Validate with runtime measurements  
./memory_profiler --scales 1M,10M,100M --architectures all
```

#### 4. Theory-Practice Gap Analysis (~1 hour)
```bash  
# Compare theoretical predictions with experimental results
python3 ../../../scripts/theory_practice_gap.py \
    --theory_file ../../../sigmod/theoretical_predictions.json \
    --results_dir results/ \
    --output results/THEORY_PRACTICE_GAP.md
```

### Extended Experiments (Additional Analysis)

#### Large-Scale Evaluation (~6 hours, requires 64GB RAM)
```bash
# Test scalability to larger datasets
./scale_test \
    --datasets sift10m,gist10m \
    --max_points 10000000 \
    --architectures mvcc,segment \
    --output results/scaling/
```

#### Profiling and Optimization Analysis (~2 hours)
```bash
# Detailed CPU profiling  
perf record -g ./mvcc_benchmark --dataset sift --duration 60s
perf report > results/profiling/cpu_profile.txt

# Memory profiling
valgrind --tool=massif ./mvcc_benchmark --dataset sift
ms_print massif.out.* > results/profiling/memory_profile.txt
```

## Expected Results

### Performance Baselines

Reference results from paper's evaluation platform (96-core Xeon Gold 6128):

#### S3 Computation Sharing Effectiveness
| Dataset | Distance Savings | Throughput Improvement | Recall Impact |
|---------|------------------|------------------------|---------------|
| SIFT 1M | 40.8% ± 0.5% | +31% ± 2% | +3.4% ± 0.3% |
| GIST 1M | 39.0% ± 0.8% | +40% ± 3% | +0.7% ± 0.2% |

#### Concurrent Performance (4W+8R configuration)
| Architecture | InsertQPS | SearchQPS | P50 Latency | P99 Latency |
|--------------|-----------|-----------|-------------|-------------|
| **MVCC** | 36,563 | 317,830 | 21μs | 50μs |
| **Segment** | 2,446 | 34 | 2.6ms | 4,216ms |
| **Lock** | 942 | 28,833 | 254μs | 579μs |

### Result Validation

#### Automatic Validation
```bash
# Compare your results against reference
python3 scripts/validate_results.py \
    --your_results build/results/ \
    --reference_results reference_data/ \
    --tolerance 10%  # Allow 10% variance
```

#### Manual Validation Checklist
- [ ] S3 distance savings: 35-45% for chasing workloads
- [ ] MVCC P99 latency: < 100μs for all configurations
- [ ] Segment P99 latency: > 1000ms (catastrophically bad)  
- [ ] Workload degradation: >50% recall drop for random queries
- [ ] Memory overhead: MVCC 2-6x higher than segment model

### Platform-Specific Variations

**Expected variance by platform:**
- **High-end servers**: ±5% throughput, ±10% latency
- **Desktop workstations**: ±10% throughput, ±15% latency  
- **Cloud instances**: ±15% throughput, ±25% latency (due to noisy neighbors)
- **ARM platforms**: ±20% throughput (different CPU characteristics)

**Red flags (investigate if you see):**
- MVCC P99 > 200μs (suggests system issues)
- S3 distance savings < 30% (suggests implementation problems)
- Segment P99 < 100ms (suggests benchmark not stressing system)

## Docker Support

### Quick Start with Docker
```bash
# Build Docker container (20 minutes)
docker build -t ann-cc-bench .

# Run basic experiments (2 hours)  
docker run --rm -v $(pwd)/results:/app/results ann-cc-bench

# Run specific experiment
docker run --rm ann-cc-bench ./exp_s3_deepdive --dataset sift
```

### Docker Environment Details
```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git python3 python3-pip \
    libomp-dev libopenblas-dev && \
    pip3 install numpy pandas matplotlib

# Copy source and build
COPY . /app/
WORKDIR /app
RUN mkdir build && cd build && cmake .. && make -j$(nproc)

# Download datasets during build (cached layer)
RUN ./scripts/download_datasets.sh

ENTRYPOINT ["./scripts/run_experiments.sh"]
```

### Multi-Stage Docker Builds
```bash
# Build development container (includes debugging tools)
docker build --target development -t ann-cc-bench:dev .

# Build production container (optimized, smaller size)  
docker build --target production -t ann-cc-bench:prod .
```

## Troubleshooting

### Common Build Issues

#### CMake Configuration Fails
```
Error: Could not find OpenMP
Solution: sudo apt install libomp-dev
```

```
Error: BLAS not found
Solution: sudo apt install libopenblas-dev
```

#### Compilation Errors
```
Error: C++17 features not supported  
Solution: Use GCC 9+ or Clang 10+
```

```
Error: Undefined reference to omp_*
Solution: Add -fopenmp to CXXFLAGS
```

### Runtime Issues

#### Dataset Loading Fails
```
Error: Cannot open sift_base.fvecs
Solution: Run ./scripts/download_datasets.sh
```

```
Error: Insufficient memory for dataset
Solution: Reduce dataset size or increase RAM/swap
```

#### Performance Issues
```
Issue: Much slower than expected results
Cause: Debug build instead of Release
Solution: cmake -DCMAKE_BUILD_TYPE=Release
```

```
Issue: Thread scaling doesn't work  
Cause: OpenMP not available or limited
Solution: export OMP_NUM_THREADS=$(nproc)
```

### System-Specific Issues

#### macOS
```bash
# Install OpenMP via Homebrew
brew install libomp

# Set compiler flags
export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
export LDFLAGS="$LDFLAGS -lomp"
```

#### CentOS/RHEL
```bash
# Enable PowerTools repository for cmake3
sudo dnf config-manager --enable powertools
sudo dnf install cmake3
alias cmake=cmake3
```

### Getting Help

If you encounter issues not covered here:

1. **Check system requirements** - Ensure all dependencies are installed
2. **Review build logs** - Look for specific error messages
3. **Test with Docker** - Docker environment is guaranteed to work
4. **Create GitHub issue** - Include full error logs and system details

**Required information for bug reports:**
- Operating system and version
- Compiler version (`gcc --version`)
- CMake version (`cmake --version`)
- Full error logs
- System hardware specs

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{ann-cc-bench-2026,
    title={Rethinking Concurrent Vector Search: From Segment Isolation to Graph-Native MVCC with Computation Sharing},
    author={[Authors]},
    booktitle={Proceedings of the 2026 ACM SIGMOD International Conference on Management of Data},
    year={2026},
    publisher={ACM}
}
```

---

**Total estimated reproduction time**: 6-8 hours (including dataset downloads and compilation)
**Minimum viable reproduction**: 2 hours (core S3 and concurrent experiments only)
**Full reproduction with analysis**: 12-16 hours (including large-scale and profiling experiments)