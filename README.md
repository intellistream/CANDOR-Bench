# CANDOR-Bench: Benchmarking In-Memory Continuous ANNS under Dynamic Open-World Streams [SIGMOD'2026]

<div align="center">

# 🚀 SAGE-DB-Bench

**Benchmarking In-Memory Continuous ANNS under Dynamic Open-World Streams**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux-orange.svg)](https://www.linux.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

*一个用于评估开放世界场景下流式向量索引性能的综合基准测试框架*

[📖 文档](#3-目录结构) • [🚀 快速开始](#2-安装与环境) • [💡 功能特性](#1-功能概览) • [🤝 贡献指南](#9-常见问题)

</div>

---

## ✨ 功能概览

<table>
<tr>
<td width="50%">

### 🌊 流式场景
支持批量插入、删除、混合读写、概念漂移等多种负载模式

### 🔬 多算法对比
封装 Faiss、DiskANN、VSAG、CANDY 等多种 ANN/图索引实现

</td>
<td width="50%">

### 📊 多数据集
内置 `sift`、`glove`、多规模随机数据等，支持扩展自定义数据集

### ⚙️ 可配置实验
通过 YAML 描述实验流程（插入、删除、搜索、等待等）

</td>
</tr>
</table>

### 📈 丰富的性能指标

| 指标类型 | 描述 |
|---------|------|
| 🎯 **召回率 (Recall)** | 衡量搜索结果的准确性 |
| ⚡ **吞吐量 (QPS)** | 每秒查询处理能力 |
| ⏱️ **延迟 (Latency)** | 查询响应时间统计 |
| 💾 **缓存分析** | Cache Miss / CPU 性能指标（可选） |

---

## 🛠️ 安装与环境

>  **环境要求**：Linux 系统，Python 3.8+

### `uv` 工作流

仓库根目录使用 `uv` 管理 Python 依赖；`algorithms_impl/GammaFresh` 保持独立子项目环境。

```bash
uv sync
uv sync --project algorithms_impl/GammaFresh
uv run candor-build-gammafresh
uv run sage-bench --list-algorithms
uv run candor-test
```

### 📦 方式一：一键部署（推荐）

适用于需要完整算法集（Faiss、DiskANN、VSAG、PyCANDY 等）以及性能测试的场景。

```bash
# 克隆仓库（包含子模块）
git clone --recursive https://github.com/intellistream/SAGE-DB-Bench.git
cd SAGE-DB-Bench

# 一键部署
./deploy.sh

# 激活环境
source sage-db-bench/bin/activate
```

<details>
<summary>📋 <b>部署选项</b></summary>

```bash
./deploy.sh --skip-system-deps   # 已手动安装系统依赖时使用
./deploy.sh --skip-build         # 仅创建 Python 环境，不编译算法
./deploy.sh --help               # 查看所有参数
```

</details>

### 🔧 方式二：算法独立构建

所有底层 C++ / 第三方算法实现位于 `algorithms_impl/`：

| 组件 | 说明 |
|------|------|
| **PyCANDY** | CMake + pybind11 构建，包含 CANDY、Faiss、DiskANN、SPTAG、Puck 等 |
| **第三方库** | 独立子模块（GTI、IP-DiskANN、PLSH 等） |
| **VSAG** | 单独子模块，生成 `pyvsag` Python wheel |

```bash
cd algorithms_impl
./build_all.sh --install
```

> 更细粒度的控制参考 `algorithms_impl/README.md`

---

## 📁 目录结构

```
SAGE-DB-Bench/
│
├── bench/                     # 基准测试框架核心
│   └── algorithms/            # 各算法的 Python wrapper
│
├── algorithms_impl/           # C++/第三方算法源码与构建脚本
│
├── datasets/                  # 数据集描述与装载逻辑
│
├── runbooks/                  # 实验配置（YAML）
│
├── raw_data/                  # 原始数据集及 ground truth
│
├── results/                   # 实验结果（CSV/日志）
│
├── deploy.sh                  # 一键部署脚本
├── compute_gt.py              # 计算 Ground Truth
├── run_benchmark.py           # 主基准测试入口
└── export_results.py          # 结果导出与整理
```

---

## 📊 数据集

### 内置数据集

| 数据集 | 维度 | 规模 | 说明 |
|:------:|:----:|:----:|:-----|
| `sift` | 128 | 1M | SIFT 特征向量 |
| `glove` | 100 | 1.2M | GloVe 词向量 |
| `random-xs` | 32 | 10K | 小规模随机数据 |
| `random-s` | 64 | 100K | 中等规模随机数据 |
| `random-m` | 128 | 1M | 大规模随机数据 |
| `wte-0.05` | - | - | Word-to-Embed 概念漂移数据（漂移率 5%） |
| `wte-0.2` | - | - | Word-to-Embed 概念漂移数据（漂移率 20%） |
| `wte-0.4` | - | - | Word-to-Embed 概念漂移数据（漂移率 40%） |
| `wte-0.6` | - | - | Word-to-Embed 概念漂移数据（漂移率 60%） |
| `wte-0.8` | - | - | Word-to-Embed 概念漂移数据（漂移率 80%） |
| `coco-0.05` | 768 | 117K | COCO 概念漂移数据（漂移率 5%） |
| `coco-0.2` | 768 | 117K | COCO 概念漂移数据（漂移率 20%） |
| `coco-0.4` | 768 | 117K | COCO 概念漂移数据（漂移率 40%） |
| `coco-0.6` | 768 | 117K | COCO 概念漂移数据（漂移率 60%） |
| `coco-0.8` | 768 | 117K | COCO 概念漂移数据（漂移率 80%） |

### 下载数据集

```bash
# 下载 SIFT 数据集
python prepare_dataset.py --dataset sift

# 下载 GloVe 数据集
python prepare_dataset.py --dataset glove
```

> 数据将自动保存至 `raw_data/` 目录

<details>
<summary> <b>添加自定义数据集</b></summary>

在 `datasets/registry.py` 中注册新的 `Dataset` 子类，实现以下接口：

```python
class MyDataset(Dataset):
    def prepare(self):                          # 下载或生成数据
        ...
    def get_data_in_range(self, start, end):    # 返回数据块
        ...
    def get_queries(self):                      # 返回查询向量
        ...
    def distance(self):                         # 距离度量类型
        return "euclidean"  # 或 "ip"
```

</details>

---

## 🧠 算法接入

### 内置算法

| 算法 | 类型 | 说明 |
|:-----|:----:|:-----|
| `faiss_HNSW` | 图索引 | 基于 Faiss 的 HNSW 实现 |
| `faiss_HNSW_Optimized` | 图索引 | 支持 Gorder 等布局优化 |
| `faiss_IVFPQ` | 量化索引 | 倒排文件 + 乘积量化 |
| `diskann` | 磁盘索引 | DiskANN 磁盘友好图索引 |
| `vsag_hnsw` | 图索引 | 基于 VSAG 的 HNSW 实现 |

> 算法封装位于 `bench/algorithms/` 目录

<details>
<summary> <b>添加自定义算法</b></summary>

#### Step 1: 创建算法目录

```bash
mkdir -p bench/algorithms/my_algo
touch bench/algorithms/my_algo/__init__.py
```

#### Step 2: 实现算法类

```python
class MyAlgo(BaseStreamingANN):
    def setup(self, dtype, max_pts, ndim): ...
    def insert(self, X, ids): ...
    def delete(self, ids): ...
    def query(self, X, k): ...  # 返回 (ids, distances)
    def set_query_arguments(self, query_args): ...
```

#### Step 3: 编写 config.yaml

```yaml
module: bench.algorithms.my_algo
constructor: MyAlgo
base-args:
  - "@metric"
run-groups:
  default:
    args: [[16, 200]]  # 索引参数
    query-args: [[100]]  # 查询参数
```

</details>

---

## 📋 Runbook 与实验流程

实验流程通过 YAML runbook 描述：

```yaml
sift:
  max_pts: 1000000
  1:
    operation: "startHPC"
  2:
    operation: "initial"
    start: 0
    end: 50000
  3:
    operation: "batch_insert"
    start: 50000
    end: 1000000
    batchSize: 2500
    eventRate: 10000
  4:
    operation: "waitPending"
  5:
    operation: "search"
  6:
    operation: "endHPC"
  gt_url: "none"
```

### 支持的操作类型

| 操作 | 说明 |
|:-----|:-----|
| `startHPC` / `endHPC` | 启动 / 停止工作线程 |
| `initial` | 初始批量数据加载 |
| `batch_insert` | 批量插入（可伴随查询） |
| `batch_insert_delete` | 带删除的批量插入 |
| `search` | 纯查询阶段 |
| `waitPending` | 等待前序操作完成 |

> 更多示例参考 `runbooks/` 目录下的子文件夹

---

## ▶️ 快速开始

### Step 1️⃣ 计算 Ground Truth

```bash
python3 compute_gt.py \
  --dataset sift \
  --runbook_file runbooks/simple.yaml \
  --gt_cmdline_tool ./algorithms_impl/DiskANN/build/apps/utils/compute_groundtruth
```

### Step 2️⃣ 运行基准测试

```bash
python3 run_benchmark.py \
  --algorithm faiss_HNSW_Optimized \
  --dataset sift \
  --runbook runbooks/simple.yaml
```

<details>
<summary><b>启用缓存性能分析</b></summary>

```bash
python3 run_benchmark.py \
  --algorithm faiss_HNSW_Optimized \
  --dataset sift \
  --runbook runbooks/simple.yaml \
  --enable-cache-profiling
```

</details>

### Step 3️⃣ 导出结果

```bash
python3 export_results.py \
  --dataset sift \
  --algorithm faiss_HNSW_Optimized \
  --runbook simple
```

### 📊 输出指标

结果保存至 `results/{dataset}/{algorithm}/`：

| 文件 | 内容 |
|:-----|:-----|
| `recall` | 各阶段召回率 |
| `query_qps` | 查询吞吐量 |
| `query_latency_ms` | 延迟统计 |
| `cache_misses` | 缓存未命中（可选） |

---

## 🐳 部署与 CI/CD

<table>
<tr>
<td width="50%">

### ✅ 推荐做法

- 使用 `deploy.sh` 统一安装构建
- 脚本自动处理 MKL、tcmalloc 等依赖
- CI 中复用 `deploy.sh` + `pytest tests/ -v`

</td>
<td width="50%">

### ⚠️ Docker 注意事项

- 容器 CPU cache/IO 行为与裸机有差异
- 容器适合功能验证与回归测试
- 严谨性能评测请使用物理机

</td>
</tr>
</table>

---

## ❓ 常见问题

<details>
<summary><b>🔴 子模块为空或缺失代码</b></summary>

```bash
# 解决方案
git submodule update --init --recursive
```

或重新克隆时使用 `git clone --recursive`

</details>

<details>
<summary><b>🔴 编译失败（CMake 找不到依赖）</b></summary>

检查是否安装了必要依赖：

```bash
# Ubuntu/Debian
sudo apt install cmake g++ libomp-dev libgflags-dev libboost-all-dev
```

详见 `algorithms_impl/build_all.sh` 注释

</details>

<details>
<summary><b>🔴 ImportError: No module named PyCANDYAlgo/pyvsag</b></summary>

1. 确认 `deploy.sh` 已运行完毕
2. 确认虚拟环境已激活：`source sage-db-bench/bin/activate`
3. 检查 `algorithms_impl/` 下 `.so` 或 `.whl` 是否生成

</details>

<details>
<summary><b>🔴 性能测试偏差较大</b></summary>


关于性能结果的说明：不同硬件环境（CPU 型号、内存带宽、缓存大小、磁盘类型等）会影响绝对性能数值，但**算法之间的相对性能趋势通常保持一致**。因此：

- ✅ 可以在同一环境下进行算法对比，关注相对差异和趋势
- ✅ 可以用于验证优化效果、参数调优
- ⚠️ 跨环境对比绝对数值意义有限

如需获得稳定、可复现的测试结果，建议：

```bash
# 固定 CPU 频率，避免动态调频影响
sudo cpupower frequency-set -g performance

# 绑定 CPU 核心，减少调度抖动
taskset -c 0-7 python3 run_benchmark.py ...

# NUMA 感知，避免跨节点内存访问
numactl --cpunodebind=0 --membind=0 python3 run_benchmark.py ...
```

</details>

---

<div align="center">

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

[![GitHub Issues](https://img.shields.io/github/issues/intellistream/SAGE-DB-Bench?style=flat-square)](https://github.com/intellistream/SAGE-DB-Bench/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/intellistream/SAGE-DB-Bench?style=flat-square)](https://github.com/intellistream/SAGE-DB-Bench/pulls)

---

**Made with ❤️ by [IntelliStream](https://github.com/intellistream)**

</div>
