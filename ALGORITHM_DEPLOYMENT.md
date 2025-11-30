# SAGE-DB-Bench 算法部署指南

本文档说明如何构建和部署 SAGE-DB-Bench 中的所有算法实现。

## 概述

SAGE-DB-Bench 的算法实现位于 `algorithms_impl/` 目录，包含三类算法：

1. **PyCANDY 算法集**：通过 CMake + pybind11 构建，生成 `PyCANDYAlgo.so`
   - 包含：CANDY 系列、Faiss、DiskANN、SPTAG、Puck

2. **第三方 C++ 库**：独立 CMake 构建
   - GTI (Graph-based Tree Index)
   - IP-DiskANN (Insertion-Prioritized DiskANN)
   - PLSH (Parallel LSH)

3. **VSAG**：Makefile + Python wheel 构建
   - 生成：`pyvsag-*.whl`

## 快速部署

### 前置条件

#### 系统依赖

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libgflags-dev \
    libboost-all-dev \
    libomp-dev \
    pkg-config
```

**macOS:**
```bash
brew install cmake gflags boost libomp git
```

#### Python 依赖

```bash
pip install -r requirements.txt
```

或手动安装核心依赖：
```bash
pip install torch numpy pybind11 PyYAML pandas
```

### 一键部署（推荐）

```bash
# 1. 克隆仓库并初始化 submodules
git clone https://github.com/intellistream/SAGE-DB-Bench.git
cd SAGE-DB-Bench
git submodule update --init --recursive

# 2. 构建并安装所有算法
cd algorithms_impl
./build_all.sh --install
```

这将自动完成：
- ✓ 构建 PyCANDY 算法
- ✓ 构建第三方库 (GTI, IP-DiskANN, PLSH)
- ✓ 构建 VSAG Python wheel
- ✓ 安装所有 Python 包

### 验证安装

```bash
# 验证 PyCANDY
python3 -c "import PyCANDYAlgo; print('✓ PyCANDYAlgo OK')"

# 验证 VSAG
python3 -c "import pyvsag; print('✓ pyvsag OK')"

# 运行测试
cd ..
python3 -m pytest tests/ -v
```

## 高级部署选项

### 选择性构建

```bash
cd algorithms_impl

# 仅构建 PyCANDY
./build_all.sh --skip-third-party --skip-vsag

# 仅构建第三方库
./build_all.sh --skip-pycandy --skip-vsag

# 仅构建 VSAG
./build_all.sh --skip-pycandy --skip-third-party

# 构建但不自动安装
./build_all.sh
./install_packages.sh  # 稍后手动安装
```

### 手动构建各个组件

#### PyCANDY 算法

```bash
cd algorithms_impl
./build.sh

# 手动安装
pip install -e . --no-build-isolation
```

#### 第三方库

```bash
cd algorithms_impl

# GTI
cd gti/GTI
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
make install

# IP-DiskANN
cd ../../ipdiskann
mkdir -p build && cd build
cmake ..
make -j$(nproc)
make install

# PLSH
cd ../../plsh
mkdir -p build && cd build
cmake ..
make -j$(nproc)
make install
```

#### VSAG

```bash
cd algorithms_impl/vsag

# 检测 Python 版本
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# 构建
make release COMPILE_JOBS=$(nproc)
make pyvsag PY_VERSION=$PYTHON_VERSION

# 安装
pip install wheelhouse/pyvsag*.whl
```

## 部署架构

```
SAGE-DB-Bench/
├── algorithms_impl/              # 算法源码和构建
│   ├── build_all.sh             # 一键构建脚本
│   ├── install_packages.sh      # 安装脚本
│   ├── build.sh                 # PyCANDY 构建脚本
│   ├── setup.py                 # PyCANDYAlgo 打包
│   ├── candy/                   # CANDY C++ 源码
│   ├── gti/                     # GTI submodule
│   ├── ipdiskann/              # IP-DiskANN submodule
│   ├── plsh/                    # PLSH submodule
│   ├── vsag/                    # VSAG submodule
│   └── [其他 submodules...]
│
├── bench/                       # Benchmark 框架
│   ├── algorithms/             # Python wrapper 层
│   │   ├── vsag_hnsw/         # VSAG wrapper
│   │   ├── candy_hnsw/        # CANDY wrapper
│   │   └── ...
│   ├── runner.py               # Benchmark 运行器
│   └── metrics.py              # 评估指标
│
├── datasets/                    # 数据集加载
├── runbooks/                    # 实验配置
└── results/                     # 实验结果
```

## 使用算法

### 在 Python 中直接使用

#### PyCANDYAlgo
```python
import PyCANDYAlgo

# 创建索引
index = PyCANDYAlgo.createIndex("HNSWNaive", dim=128)

# 配置参数
config = PyCANDYAlgo.newConfigMap()
config.edit("vecDim", 128)
config.edit("M", 16)

# 加载数据
db = PyCANDYAlgo.loadTensorFromFile("data.fvecs")
index.loadInitialTensor(db, config)

# 搜索
query = PyCANDYAlgo.loadTensorFromFile("query.fvecs")
results = index.searchTensor(query, 10)
```

#### VSAG
```python
import pyvsag
import numpy as np

# 创建索引
index = pyvsag.Index('hnsw', 'l2', 128)

# 构建参数
parameters = {
    'max_degree': 16,
    'ef_construction': 200
}
index.build(data, parameters)

# 搜索
k = 10
search_params = {'ef_search': 100}
ids, dists = index.search(queries, k, search_params)
```

### 在 Benchmark 中使用

```bash
# 运行单个算法
python run_benchmark.py \
    --dataset sift \
    --algorithm vsag_hnsw \
    --config runbooks/algo_optimizations/vsag_hnsw.yaml

# 运行完整 benchmark
python run_benchmark.py \
    --runbook runbooks/baseline.yaml
```

## 故障排除

### 构建失败

**问题**: CMake 找不到依赖
```bash
# 检查系统依赖
cmake --version
pkg-config --list-all | grep -E 'gflags|boost|omp'

# 重新安装依赖
sudo apt-get install --reinstall build-essential cmake libgflags-dev
```

**问题**: 编译内存不足
```bash
# 减少并行编译数
export COMPILE_JOBS=2
./build_all.sh
```

**问题**: Submodule 为空
```bash
git submodule update --init --recursive
```

### 导入失败

**问题**: `ImportError: No module named PyCANDYAlgo`
```bash
# 检查 .so 文件是否存在
ls algorithms_impl/PyCANDYAlgo*.so

# 检查 Python 路径
python3 -c "import sys; print('\n'.join(sys.path))"

# 重新安装
cd algorithms_impl
pip install -e . --no-build-isolation --force-reinstall
```

**问题**: `ImportError: No module named pyvsag`
```bash
# 检查 wheel 文件
ls algorithms_impl/vsag/wheelhouse/

# 重新安装
pip install algorithms_impl/vsag/wheelhouse/pyvsag*.whl --force-reinstall
```

**问题**: `undefined symbol` 错误
```bash
# 清理并重新构建
cd algorithms_impl
rm -rf build/ PyCANDYAlgo*.so
./build.sh
pip install -e . --force-reinstall
```

### 运行时问题

**问题**: `GLIBCXX` 版本错误
```bash
# 检查 GCC 版本
gcc --version

# Ubuntu: 升级到更新的 GCC
sudo apt-get install gcc-11 g++-11
export CXX=g++-11
export CC=gcc-11

# 重新构建
cd algorithms_impl && ./build_all.sh
```

**问题**: OpenMP 库找不到
```bash
# Ubuntu
sudo apt-get install libomp-dev

# macOS
brew install libomp
export LDFLAGS="-L/usr/local/opt/libomp/lib"
export CPPFLAGS="-I/usr/local/opt/libomp/include"
```

## 环境要求

| 组件 | 最低版本 | 推荐版本 |
|------|---------|---------|
| **OS** | Ubuntu 20.04 / macOS 11 | Ubuntu 22.04 / macOS 13 |
| **GCC** | 9.0 | 11.0+ |
| **CMake** | 3.18 | 3.24+ |
| **Python** | 3.8 | 3.10+ |
| **PyTorch** | 1.12 | 2.0+ |
| **NumPy** | 1.20 | 1.24+ |

## 持续集成

在 CI/CD 环境中部署：

```bash
#!/bin/bash
# ci_deploy.sh

set -e

# 安装系统依赖
apt-get update && apt-get install -y build-essential cmake libgflags-dev libboost-all-dev

# 克隆并初始化
git clone --recurse-submodules https://github.com/intellistream/SAGE-DB-Bench.git
cd SAGE-DB-Bench

# 安装 Python 依赖
pip install -r requirements.txt

# 构建算法
cd algorithms_impl
./build_all.sh --install

# 验证
cd ..
python3 -c "import PyCANDYAlgo, pyvsag"
pytest tests/ -v
```

## 更新算法

### 更新 submodule 到最新版本

```bash
cd algorithms_impl

# 更新单个 submodule
cd vsag
git pull origin main
cd ..

# 更新所有 submodules
git submodule update --remote --recursive

# 重新构建
./build_all.sh --install
```

### 添加新算法

1. 作为 submodule 添加：
```bash
cd algorithms_impl
git submodule add <repo_url> <folder_name>
```

2. 更新 `build_all.sh` 添加构建逻辑

3. 在 `bench/algorithms/` 创建 Python wrapper

4. 更新文档

## 支持

- **文档**: [algorithms_impl/README.md](algorithms_impl/README.md)
- **Issues**: https://github.com/intellistream/SAGE-DB-Bench/issues
- **Wiki**: https://github.com/intellistream/SAGE-DB-Bench/wiki
