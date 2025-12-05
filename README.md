# SAGE-DB-Bench

流式向量索引基准测试框架，用于评估在持续插入、删除和查询负载下的向量索引性能（QPS、延迟、召回率以及缓存行为等）。

---

## 1. 功能概览

- **流式场景**：支持批量插入、删除、混合读写、概念漂移等多种负载模式。
- **多算法对比**：封装了 Faiss、DiskANN、VSAG、CANDY 等多种 ANN/图索引实现，并支持自定义算法接入。
- **多数据集**：内置 `sift`、`glove`、多规模随机数据等，支持扩展自定义数据集。
- **可配置实验**：通过 `runbooks/` 中的 YAML 描述实验流程（插入、删除、搜索、等待等）。
- **指标丰富**：支持召回率、QPS、延迟统计，可选 Cache Miss / CPU 性能指标分析。

---

## 2. 安装与环境

项目推荐在 Linux 环境下使用，Python 3.8+。

### 2.1 一键部署（包含算法构建）

适用于需要完整算法集（Faiss、DiskANN、VSAG、PyCANDY 等）以及性能测试的场景。

```bash
git clone --recursive https://github.com/intellistream/SAGE-DB-Bench.git
cd SAGE-DB-Bench

./deploy.sh

source sage-db-bench/bin/activate
```

常用选项：

```bash
./deploy.sh --skip-system-deps   # 已手动安装系统依赖时使用
./deploy.sh --skip-build         # 仅创建 Python 环境，不编译算法
./deploy.sh --help               # 查看所有参数
```

### 2.2 仅 Python 框架（快速上手）

如果只想快速体验框架逻辑或开发框架本身，可只安装 Python 依赖：

```bash
git clone https://github.com/intellistream/SAGE-DB-Bench.git
cd SAGE-DB-Bench

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2.3 算法构建与部署概览

所有底层 C++ / 第三方算法实现位于 `algorithms_impl/`：

- **PyCANDY 算法集**：通过 CMake + pybind11 构建，生成 `PyCANDYAlgo`，包含 CANDY 系列、Faiss、DiskANN、SPTAG、Puck 等。
- **第三方库**：独立子模块与 CMake 构建（GTI、IP-DiskANN、PLSH 等）。
- **VSAG**：单独的子模块，生成 `pyvsag` Python wheel。

统一构建与安装可通过：

```bash
cd algorithms_impl
./build_all.sh --install
```

更细粒度的控制（仅构建部分算法或延后安装），可参考 `algorithms_impl/build_all.sh` 与 `algorithms_impl/README.md` 中注释。

---

## 3. 目录结构

```text
SAGE-DB-Bench/
├── bench/                 # 基准测试框架核心：调度、worker、指标计算等
│   └── algorithms/        # 各算法的 Python wrapper
├── algorithms_impl/       # C++/第三方算法源码与构建脚本
├── datasets/              # 数据集描述与装载逻辑
├── runbooks/              # 各类实验配置（YAML）
├── raw_data/              # 原始数据集及 ground truth
├── results/               # 实验结果（CSV/日志）
├── deploy.sh              # 一键部署脚本
├── compute_gt.py          # 计算 Ground Truth
├── run_benchmark.py       # 主基准测试入口
└── export_results.py      # 结果导出与整理
```

---

## 4. 数据集

### 4.1 内置数据集

当前常用数据集示例：

| 数据集      | 维度 | 规模       | 说明                |
|-------------|------|------------|---------------------|
| `sift`      | 128  | 1M         | SIFT 特征向量       |
| `glove`     | 100  | 1.2M       | GloVe 词向量        |
| `random-xs` | 32   | 1 万级     | 小规模随机数据      |
| `random-s`  | 64   | 10 万级    | 中等规模随机数据    |
| `random-m`  | 128  | 百万级     | 大规模随机数据      |

### 4.2 下载与准备

```bash
python prepare_dataset.py --dataset sift
python prepare_dataset.py --dataset glove
```

脚本会自动在 `raw_data/` 下组织对应的数据与 ground truth 目录结构。

### 4.3 自定义数据集

要增加新数据集，可在 `datasets/registry.py` 中注册一个新的 `Dataset` 子类，并实现以下接口：

- `prepare()`：下载或生成数据，并写入 `raw_data/...`。
- `get_data_in_range(start, end)`：返回给定区间的数据块。
- `get_queries()`：返回查询向量集合。
- `distance()`：指定距离度量类型（如 `euclidean` 或 `ip`）。

完成后，将其加入 `DATASETS` 字典即可被 benchmark 使用。

---

## 5. 算法接入

### 5.1 内置算法示例

框架已内置多种典型索引算法，举例：

- `faiss_HNSW`：基于 Faiss 的 HNSW 图索引。
- `faiss_HNSW_Optimized`：在 HNSW 基础上支持 Gorder 等布局优化。
- `faiss_IVFPQ`：倒排文件 + 乘积量化索引。
- `diskann`：DiskANN 磁盘友好图索引。
- `vsag_hnsw`：基于 VSAG 的 HNSW 实现。

对应的 Python 封装位于 `bench/algorithms/` 各子目录中。

### 5.2 添加自定义算法

典型步骤：

1. 在 `bench/algorithms/` 下创建新目录，例如 `my_algo/`，包含 `__init__.py` 与实现文件。
2. 实现继承自基类（如 `BaseStreamingANN`）的算法类，至少实现：
   - `setup(dtype, max_pts, ndim)`
   - `insert(X, ids)` / `delete(ids)`
   - `query(X, k)`（返回 ids 与距离）
   - `set_query_arguments(query_args)`（例如 `ef` 等可调参数）。
3. 为该算法编写对应的 `config.yaml`，指定：
   - `module`：Python 模块路径
   - `constructor`：类名
   - `base-args`：基础参数（如 `@metric`）
   - `run-groups`：不同实验组下的索引参数与查询参数列表。
4. 在该目录的 `__init__.py` 中导出主类，使框架能够按名加载。

具体可参考现有算法目录（如 `bench/algorithms/faiss_hnsw/`、`vsag_hnsw/`）。

---

## 6. Runbook 与实验流程

实验流程通过 YAML runbook 描述，典型示例如下（来自 `runbooks/simple.yaml` 一类配置）：

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
    end: 100000
    batchSize: 2500
    eventRate: 10000
  4:
    operation: "waitPending"
  5:
    operation: "search"
  6:
    operation: "endHPC"
```

常见 `operation` 类型：

- `startHPC` / `endHPC`：启动 / 停止工作线程。
- `initial`：初始批量数据加载。
- `batch_insert`：在已有数据基础上继续批量插入（可伴随查询）。
- `batch_insert_delete`：带删除操作的批量插入场景。
- `search`：纯查询阶段。
- `waitPending`：等待前序操作全部完成。

更复杂的运行模式（概念漂移、删除模式、不同事件速率等）可参考 `runbooks/` 下的子目录示例。

---

## 7. 基本使用流程

### 7.1 计算 Ground Truth

在运行 benchmark 之前，先为目标数据集与 runbook 计算真值：

```bash
python3 compute_gt.py \
  --dataset sift \
  --runbook_file runbooks/simple.yaml \
  --gt_cmdline_tool ./DiskANN/build/apps/utils/compute_groundtruth
```

生成的真值会被保存在 `raw_data/{dataset}/...` 对应目录中。

### 7.2 运行基准测试

```bash
python3 run_benchmark.py \
  --algorithm faiss_HNSW_Optimized \
  --dataset sift \
  --runbook runbooks/simple.yaml
```

如果需要额外采集 Cache Miss 等硬件指标（依赖额外工具或编译选项）：

```bash
python3 run_benchmark.py \
  --algorithm faiss_HNSW_Optimized \
  --dataset sift \
  --runbook runbooks/simple.yaml \
  --enable-cache-profiling
```

### 7.3 导出与查看结果

```bash
python3 export_results.py \
  --dataset sift \
  --algorithm faiss_HNSW_Optimized \
  --runbook simple
```

导出后的 CSV 文件将保存在：

- `results/{dataset}/{algorithm}/...`

其中包含：

- `recall`：各阶段召回率。
- `query_qps`：查询吞吐量（QPS）。
- `query_latency_ms`：查询延迟统计。
- `cache_misses`：缓存未命中指标（如果开启 profiling）。

---

## 8. 部署与 CI/CD 提示

- 生产或 CI 环境中建议使用 `deploy.sh` 进行统一安装与构建，脚本内部已经考虑了 VSAG 依赖 MKL、GTI 依赖 tcmalloc 等问题，并在虚拟环境激活脚本中写入必要的库路径。
- 若在 Docker / 容器中运行，请注意：
  - 容器环境下的 CPU cache/IO 行为与裸机存在差异，不适合作为绝对性能对比结论。
  - 更推荐将容器用作功能验证与回归测试，性能论文或严谨评测应在受控物理机环境执行。
- 在 CI 中进行构建与测试时，可以复用 `deploy.sh` 或 `algorithms_impl/build_all.sh --install`，再运行 `pytest tests/ -v` 做回归。

---

## 9. 常见问题

- **子模块为空或缺失代码**：确认使用 `git clone --recursive`，或执行 `git submodule update --init --recursive`。
- **编译失败（CMake 找不到依赖）**：检查是否安装了 `cmake`、`g++/clang`、`libomp`、`gflags`、`boost` 等；必要时报错信息可参考 `algorithms_impl/build_all.sh` 内部注释进行排查。
- **导入算法失败（如 `ImportError: No module named PyCANDYAlgo/pyvsag`）**：优先确认 `deploy.sh` 是否运行完毕、虚拟环境是否正确激活，以及 `algorithms_impl` 目录下对应 `.so` 或 `.whl` 是否已生成并安装。
- **性能测试偏差较大**：确保在固定 CPU 频率、线程数和隔离程度（如 `taskset`、`numactl`、关闭省电模式等）下重复测试，避免其他进程干扰。

如果需要更细致的部署/CI 修复细节，可直接查阅脚本中的注释或阅读 `algorithms_impl/` 下各子模块的 README。
