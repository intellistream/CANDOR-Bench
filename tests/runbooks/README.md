# 测试 Runbooks

本目录包含用于测试的 runbook 配置文件。

## 文件说明

- **test_simple.yaml**: 简单的基础功能测试
- **test_continuous_query.yaml**: 连续查询功能测试
- **test_congestion_drop.yaml**: 拥塞丢弃场景测试

## 使用方法

```bash
# 运行简单测试
python run_benchmark.py \
    --algorithm faiss_HNSW \
    --dataset sift \
    --runbook tests/runbooks/test_simple.yaml

# 运行拥塞测试
python -m benchmark_anns run-streaming \
    --runbook tests/runbooks/test_congestion_drop.yaml \
    --algorithm faiss_hnsw \
    --dataset sift
```

## 注意事项

这些测试配置使用较小的数据量和参数，适合快速验证功能。
生产环境实验请使用 `runbooks/` 目录下的标准配置文件。
