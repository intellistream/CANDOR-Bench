# CANDOR-Bench

<details open>
<summary><strong>English</strong></summary>

Streaming vector index benchmarking framework.

## 1. One-click deployment

```bash
# Clone the repo (with submodules)
git clone --recursive https://github.com/intellistream/CANDOR-Bench.git
cd CANDOR-Bench

# Run the deployment script
./deploy.sh

# Activate the virtual environment
source sage-db-bench/bin/activate
```

**Deployment options:**
```bash
./deploy.sh --skip-system-deps  # Skip system dependency installation (when deps are already installed)
./deploy.sh --skip-build        # Skip build (setup environment only)
./deploy.sh --help              # Show help
```

## 2. Datasets

### Supported datasets

| Dataset | Dim | Size | Description |
|--------|-----|------|-------------|
| sift | 128 | 1M | SIFT feature vectors |
| glove | 100 | 1.2M | GloVe word vectors |
| random-xs | 32 | 10K | Random data (testing) |
| random-s | 64 | 100K | Random data (small scale) |
| random-m | 128 | 1M | Random data (mid scale) |

### Download datasets

```bash
python prepare_dataset.py --dataset sift
python prepare_dataset.py --dataset glove
```

### Add a new dataset

Add it in `datasets/registry.py`:

```python
class MyDataset(Dataset):
    def __init__(self):
        self.nb = 100000      # number of vectors
        self.nq = 10000       # number of queries
        self.d = 128          # vector dimension
        self.dtype = 'float32'
        self.basedir = 'raw_data/mydataset'

    def prepare(self):
        # Download or generate data
        pass

    def get_data_in_range(self, start, end):
        # Return data in [start, end)
        pass

    def get_queries(self):
        # Return query vectors
        pass

    def distance(self):
        return 'euclidean'  # or 'ip'

# Register dataset
DATASETS['mydataset'] = lambda: MyDataset()
```

## 3. Algorithms

### Supported algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| faiss_HNSW | Graph index | Faiss HNSW implementation |
| faiss_HNSW_Optimized | Graph index | HNSW with Gorder optimization |
| faiss_IVFPQ | Quantization | Inverted file + product quantization |
| diskann | Graph index | DiskANN |
| vsag_hnsw | Graph index | VSAG HNSW |

### Add a new algorithm

1. Create a directory under `bench/algorithms/`:

```
bench/algorithms/my_algo/
├── __init__.py
├── my_algo.py
└── config.yaml
```

2. Implement the algorithm interface (`my_algo.py`):

```python
from ..base import BaseStreamingANN

class MyAlgorithm(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.metric = metric
        self.name = "my_algo"
        # parse index_params

    def setup(self, dtype, max_pts, ndim):
        # Initialize index
        pass

    def insert(self, X, ids):
        # Insert vectors
        pass

    def delete(self, ids):
        # Delete vectors
        pass

    def query(self, X, k):
        # Query, return (ids, distances)
        pass

    def set_query_arguments(self, query_args):
        # Set query parameters (e.g., ef)
        pass
```

3. Create the config file (`config.yaml`):

```yaml
sift:
  my_algo:
    module: benchmark_anns.bench.algorithms.my_algo.my_algo
    constructor: MyAlgorithm
    base-args: ["@metric"]
    run-groups:
      base:
        args: |
          [{"param1": 32, "param2": 100}]
        query-args: |
          [{"ef": 40}]
```

4. Export it in `__init__.py`:

```python
from .my_algo import MyAlgorithm
__all__ = ['MyAlgorithm']
```

## 4. Benchmark workflow

### 4.1 Compute ground truth

```bash
python compute_gt.py \
    --dataset sift \
    --runbook_file runbooks/simple.yaml \
    --gt_cmdline_tool ./DiskANN/build/apps/utils/compute_groundtruth
```

Ground-truth files are saved in `raw_data/{dataset}/{size}/{runbook}.yaml/`.

### 4.2 Run benchmarks

```bash
# Basic usage
python run_benchmark.py \
    --algorithm faiss_HNSW_Optimized \
    --dataset sift \
    --runbook runbooks/simple.yaml

# Enable cache miss profiling
python run_benchmark.py \
    --algorithm faiss_HNSW_Optimized \
    --dataset sift \
    --runbook runbooks/simple.yaml \
    --enable-cache-profiling
```

### 4.3 Export results

```bash
python export_results.py \
    --dataset sift \
    --algorithm faiss_HNSW_Optimized \
    --runbook simple
```

Exported results include:
- **recall**: recall per batch
- **query_qps**: query throughput
- **query_latency_ms**: query latency
- **cache_misses**: cache miss counts (if enabled)

Result files are saved in `results/{dataset}/{algorithm}/`.

</details>

<details>
<summary><strong>中文</strong></summary>

流式向量索引基准测试框架

## 1. 一键部署

```bash
# 克隆仓库（包含 submodules）
git clone --recursive https://github.com/intellistream/CANDOR-Bench.git
cd CANDOR-Bench

# 运行部署脚本
./deploy.sh

# 激活虚拟环境
source sage-db-bench/bin/activate
```

**部署选项：**
```bash
./deploy.sh --skip-system-deps  # 跳过系统依赖安装（已有依赖时使用）
./deploy.sh --skip-build        # 跳过构建（仅设置环境）
./deploy.sh --help              # 查看帮助
```

## 2. 数据集

### 支持的数据集

| 数据集 | 维度 | 数据量 | 说明 |
|--------|------|--------|------|
| sift | 128 | 1M | SIFT 特征向量 |
| glove | 100 | 1.2M | GloVe 词向量 |
| random-xs | 32 | 10K | 随机数据（测试用） |
| random-s | 64 | 100K | 随机数据（小规模） |
| random-m | 128 | 1M | 随机数据（中规模） |

### 下载数据集

```bash
python prepare_dataset.py --dataset sift
python prepare_dataset.py --dataset glove
```

### 添加新数据集

在 `datasets/registry.py` 中添加：

```python
class MyDataset(Dataset):
    def __init__(self):
        self.nb = 100000      # 数据量
        self.nq = 10000       # 查询数量
        self.d = 128          # 向量维度
        self.dtype = 'float32'
        self.basedir = 'raw_data/mydataset'

    def prepare(self):
        # 下载或生成数据
        pass

    def get_data_in_range(self, start, end):
        # 返回 [start, end) 范围的数据
        pass

    def get_queries(self):
        # 返回查询向量
        pass

    def distance(self):
        return 'euclidean'  # 或 'ip'

# 注册数据集
DATASETS['mydataset'] = lambda: MyDataset()
```

## 3. 算法

### 支持的算法

| 算法 | 类型 | 说明 |
|------|------|------|
| faiss_HNSW | 图索引 | Faiss HNSW 实现 |
| faiss_HNSW_Optimized | 图索引 | 支持 Gorder 优化的 HNSW |
| faiss_IVFPQ | 量化 | 倒排文件 + 乘积量化 |
| diskann | 图索引 | DiskANN |
| vsag_hnsw | 图索引 | VSAG HNSW |

### 添加新算法

1. 在 `bench/algorithms/` 下创建目录：

```
bench/algorithms/my_algo/
├── __init__.py
├── my_algo.py
└── config.yaml
```

2. 实现算法接口 (`my_algo.py`)：

```python
from ..base import BaseStreamingANN

class MyAlgorithm(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.metric = metric
        self.name = "my_algo"
        # 解析 index_params

    def setup(self, dtype, max_pts, ndim):
        # 初始化索引
        pass

    def insert(self, X, ids):
        # 插入向量
        pass

    def delete(self, ids):
        # 删除向量
        pass

    def query(self, X, k):
        # 查询，返回 (ids, distances)
        pass

    def set_query_arguments(self, query_args):
        # 设置查询参数（如 ef）
        pass
```

3. 创建配置文件 (`config.yaml`)：

```yaml
sift:
  my_algo:
    module: benchmark_anns.bench.algorithms.my_algo.my_algo
    constructor: MyAlgorithm
    base-args: ["@metric"]
    run-groups:
      base:
        args: |
          [{"param1": 32, "param2": 100}]
        query-args: |
          [{"ef": 40}]
```

4. 在 `__init__.py` 中导出：

```python
from .my_algo import MyAlgorithm
__all__ = ['MyAlgorithm']
```

## 4. 测试流程

### 4.1 计算 Ground Truth

```bash
python compute_gt.py \
    --dataset sift \
    --runbook_file runbooks/simple.yaml \
    --gt_cmdline_tool ./DiskANN/build/apps/utils/compute_groundtruth
```

生成的真值文件保存在 `raw_data/{dataset}/{size}/{runbook}.yaml/` 目录。

### 4.2 运行测试

```bash
# 基本用法
python run_benchmark.py \
    --algorithm faiss_HNSW_Optimized \
    --dataset sift \
    --runbook runbooks/simple.yaml

# 启用 Cache Miss 测量
python run_benchmark.py \
    --algorithm faiss_HNSW_Optimized \
    --dataset sift \
    --runbook runbooks/simple.yaml \
    --enable-cache-profiling
```

### 4.3 导出结果

```bash
python export_results.py \
    --dataset sift \
    --algorithm faiss_HNSW_Optimized \
    --runbook simple
```

导出的结果包含：
- **recall**: 每个批次的召回率
- **query_qps**: 查询吞吐量
- **query_latency_ms**: 查询延迟
- **cache_misses**: Cache Miss 数量（如果启用）

结果文件保存在 `results/{dataset}/{algorithm}/` 目录。

</details>
