# 算法实现（Algorithm Implementations）

本目录包含所有 ANN 算法的 Python 封装实现，提供统一的流式索引接口。

## 📁 目录结构

每个算法独立一个文件夹，包含实现文件和配置：

```
bench/algorithms/
├── base.py              # 基类接口定义
├── registry.py          # 自动注册机制
├── candy_lshapg/        # CANDY LSH+APG
│   ├── candy_lshapg.py
│   └── config.yaml
├── faiss_HNSW/          # Faiss HNSW
│   ├── faiss_HNSW.py
│   └── config.yaml
└── ... (17+ algorithms)
```

## 🔧 已实现算法

### CANDY 系列
- **candy_lshapg** - LSH + Approximate Proximity Graph
- **candy_mnru** - Most Nearly Recently Used
- **candy_sptag** - Space Partition Tree And Graph

### Faiss 系列
- **faiss_HNSW** - Hierarchical NSW
- **faiss_IVFPQ** - IVF + Product Quantization
- **faiss_lsh** - Locality Sensitive Hashing
- **faiss_NSW** - Navigable Small World
- **faiss_pq** - Product Quantization
- **faiss_fast_scan** - Fast Scan variant
- **faiss_onlinepq** - Online PQ with buffering

### 其他算法
- **diskann** / **ipdiskann** - DiskANN 系列
- **puck** - Puck 索引
- **gti** - Graph-based Tree Index
- **plsh** - Partition-based LSH
- **cufe** / **pyanns** - 其他实现

## 🚀 使用方法

### 通过注册表获取算法

```python
from bench.algorithms import get_algorithm

# 获取算法实例
algo = get_algorithm('faiss_HNSW', metric='euclidean')

# 初始化
algo.setup(dtype='float32', max_pts=100000, ndim=128)

# 插入数据
algo.insert(vectors, ids)

# 查询
algo.set_query_arguments({'efSearch': 100})
results = algo.query(query_vectors, k=10)
```

### 查看可用算法

```python
from bench.algorithms import ALGORITHMS

# 列出所有已注册算法
print(list(ALGORITHMS.keys()))
```

## ➕ 添加新算法

### 1. 创建算法目录和文件

```bash
mkdir bench/algorithms/my_algorithm/
touch bench/algorithms/my_algorithm/my_algorithm.py
touch bench/algorithms/my_algorithm/config.yaml
```

### 2. 实现算法类

```python
# my_algorithm.py
from bench.algorithms.base import BaseStreamingANN
import numpy as np

class MyAlgorithm(BaseStreamingANN):
    def __init__(self, **params):
        super().__init__()
        self.params = params
    
    def setup(self, dtype: str, max_pts: int, ndim: int) -> None:
        """初始化索引"""
        self.ndim = ndim
        self.max_pts = max_pts
    
    def insert(self, X: np.ndarray, ids: np.ndarray) -> None:
        """插入向量"""
        pass
    
    def delete(self, ids: np.ndarray) -> None:
        """删除向量"""
        pass
    
    def query(self, X: np.ndarray, k: int):
        """查询 k 近邻，返回 (indices, distances)"""
        return np.array([]), np.array([])
    
    def set_query_arguments(self, query_args):
        """设置查询参数"""
        pass
```

### 3. 创建配置文件

```yaml
# config.yaml
random-xs:
  my_algorithm:
    module: bench.algorithms.my_algorithm.my_algorithm
    constructor: MyAlgorithm
    base-args: ["@metric"]
    run-groups:
      base:
        args: |
          [{"param1": 10, "param2": 100}]
        query-args: |
          [{"query_param": 50}]
```

### 4. 自动注册

算法会在模块加载时自动注册，无需手动修改 `registry.py`。

## 🏗️ 架构关系

```
benchmark_anns/
├── algorithms_impl/         # C++ 源码和编译（可选）
│   ├── candy/              # CANDY C++ 实现
│   ├── diskann/            # DiskANN C++ 实现
│   ├── PyCANDY.cpp         # Python 绑定
│   └── build/              # 编译输出 PyCANDYAlgo.so
│
└── bench/algorithms/       # Python 封装层（本目录）
    ├── base.py            # 统一接口
    ├── registry.py        # 自动注册
    └── */                 # 各算法实现

职责分离：
- algorithms_impl/ 负责 C++ 编译
- bench/algorithms/ 负责 Python 接口封装
```

## 📋 接口规范

所有算法必须继承 `BaseStreamingANN` 并实现：

| 方法 | 说明 | 必需 |
|------|------|------|
| `setup(dtype, max_pts, ndim)` | 初始化索引 | ✅ |
| `insert(X, ids)` | 插入向量 | ✅ |
| `delete(ids)` | 删除向量 | ✅ |
| `query(X, k)` | 查询 k 近邻 | ✅ |
| `set_query_arguments(args)` | 设置查询参数 | ✅ |
| `fit(X)` | 批量建索引 | ⚪ |
| `get_memory_usage()` | 获取内存使用 | ⚪ |
