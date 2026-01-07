# CANDOR-Bench

[English](README.md) | [中文](README.zh-CN.md)

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
