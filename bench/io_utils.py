"""
IO Utils - 结果保存工具

提供将测评结果保存为 HDF5 和 CSV 文件的工具函数
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from .metrics import BenchmarkMetrics


def save_run_results(
    metrics: BenchmarkMetrics,
    all_results_continuous: List[np.ndarray],
    output_dir: str,
    algorithm_name: str,
    dataset_name: str,
    runbook_name: Optional[str] = None,
    query_timestamps: Optional[List[Dict]] = None
):
    """
    保存测评结果到 HDF5 和 CSV 文件
    
    Args:
        metrics: 测评指标对象
        all_results_continuous: 连续查询结果列表 (每个元素是一个查询的k个邻居)
        output_dir: 输出目录
        algorithm_name: 算法名称
        dataset_name: 数据集名称
        runbook_name: Runbook名称（可选）
        query_timestamps: 查询时间戳列表，包含 total_time, lock_wait_time, query_time
    
    生成的文件:
        - {algorithm}_{dataset}_{runbook}.hdf5: HDF5文件，包含查询结果
        - {algorithm}_{dataset}_{runbook}.csv: CSV文件，包含摘要指标
        - {algorithm}_batch_insert_qps.csv: 每批次插入QPS
        - {algorithm}_batch_query_qps.csv: 每批次查询QPS
        - {algorithm}_batch_query_latency.csv: 每批次查询延迟
        - {algorithm}_query_timing_breakdown.csv: 查询时间拆分（锁等待/纯查询）
        - {algorithm}_insert_cache_miss.csv: 插入操作的 cache miss 统计
        - {algorithm}_query_cache_miss.csv: 查询操作的 cache miss 统计（包含连续查询和search）
    """
    # 创建输出目录
    output_path = Path(output_dir) / dataset_name / algorithm_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 构建文件名前缀
    if runbook_name:
        file_prefix = f"{algorithm_name}_{dataset_name}_{runbook_name}"
    else:
        file_prefix = f"{algorithm_name}_{dataset_name}"
    
    # 1. 保存 HDF5 文件（查询结果）
    hdf5_file = output_path / f"{file_prefix}.hdf5"
    save_hdf5_results(hdf5_file, all_results_continuous)
    print(f"  ✓ HDF5 results saved to: {hdf5_file}")
    
    # 2. 保存主CSV文件（摘要指标）
    csv_file = output_path / f"{file_prefix}.csv"
    save_summary_csv(csv_file, metrics)
    print(f"  ✓ Summary CSV saved to: {csv_file}")
    
    # 3. 保存批次级别的指标
    # 3.1 插入QPS
    if metrics.insert_throughput and len(metrics.insert_throughput) > 0:
        insert_qps_file = output_path / f"{algorithm_name}_batch_insert_qps.csv"
        save_batch_metric_csv(
            insert_qps_file,
            metrics.insert_throughput,
            'insert_qps'
        )
        print(f"  ✓ Batch insert QPS saved to: {insert_qps_file}")
    
    # 3.2 查询QPS和延迟
    # 优先使用 query_timestamps 中的纯查询时间（排除锁等待），更准确地反映真实查询性能
    if query_timestamps and len(query_timestamps) > 0:
        # 从 worker 的 query_timestamps 中提取纯查询时间（微秒）
        pure_query_times_us = [ts['query_time'] for ts in query_timestamps]
        # 转换为秒
        pure_query_times_sec = [t / 1e6 for t in pure_query_times_us]
        
        # 计算查询QPS（使用纯查询时间）
        queries_per_batch = getattr(metrics, 'queries_per_continuous_query', 10000)
        query_qps = [queries_per_batch / lat if lat > 0 else 0 for lat in pure_query_times_sec]
        
        # 保存查询QPS（基于纯查询时间）
        query_qps_file = output_path / f"{algorithm_name}_batch_query_qps.csv"
        save_batch_metric_csv(query_qps_file, query_qps, 'query_qps')
        print(f"  ✓ Batch query QPS saved to: {query_qps_file} (using pure query time, excluding lock wait)")
        
        # 保存查询延迟（纯查询时间，微秒转毫秒）
        pure_query_times_ms = [t / 1000 for t in pure_query_times_us]
        query_latency_file = output_path / f"{algorithm_name}_batch_query_latency.csv"
        save_batch_metric_csv(query_latency_file, pure_query_times_ms, 'query_latency_ms')
        print(f"  ✓ Batch query latency saved to: {query_latency_file} (pure query time)")
        
    elif hasattr(metrics, 'continuous_query_latencies') and metrics.continuous_query_latencies:
        # 回退：使用 runner 层面的端到端延迟（包含锁等待）
        # 提取第一个元素（因为是嵌套列表）
        query_latencies = metrics.continuous_query_latencies[0] if metrics.continuous_query_latencies else []
        
        if query_latencies:
            # 计算查询QPS
            queries_per_batch = getattr(metrics, 'queries_per_continuous_query', 10000)
            query_qps = [queries_per_batch / lat if lat > 0 else 0 for lat in query_latencies]
            
            # 保存查询QPS
            query_qps_file = output_path / f"{algorithm_name}_batch_query_qps.csv"
            save_batch_metric_csv(query_qps_file, query_qps, 'query_qps')
            print(f"  ✓ Batch query QPS saved to: {query_qps_file} (end-to-end, includes lock wait)")
            
            # 保存查询延迟（转换为毫秒）
            query_latency_ms = [lat * 1000 for lat in query_latencies]
            query_latency_file = output_path / f"{algorithm_name}_batch_query_latency.csv"
            save_batch_metric_csv(query_latency_file, query_latency_ms, 'query_latency_ms')
            print(f"  ✓ Batch query latency saved to: {query_latency_file} (end-to-end)")
    
    # 3.3 Cache Miss 统计
    # 保存插入的cache miss统计
    if metrics.cache_miss_per_batch and len(metrics.cache_miss_per_batch) > 0:
        cache_miss_file = output_path / f"{algorithm_name}_insert_cache_miss.csv"
        save_cache_miss_csv(
            cache_miss_file,
            metrics.cache_miss_per_batch,
            metrics.cache_references_per_batch,
            metrics.cache_miss_rate_per_batch
        )
        print(f"  ✓ Insert cache miss saved to: {cache_miss_file}")
    
    # 保存查询的cache miss统计（包含连续查询和search操作）
    if metrics.query_cache_miss_per_batch and len(metrics.query_cache_miss_per_batch) > 0:
        query_cache_miss_file = output_path / f"{algorithm_name}_query_cache_miss.csv"
        save_cache_miss_csv(
            query_cache_miss_file,
            metrics.query_cache_miss_per_batch,
            metrics.query_cache_references_per_batch,
            metrics.query_cache_miss_rate_per_batch
        )
        print(f"  ✓ Query cache miss saved to: {query_cache_miss_file}")
    
    # 3.4 查询时间拆分（锁等待时间 vs 纯查询时间）
    if query_timestamps and len(query_timestamps) > 0:
        timing_breakdown_file = output_path / f"{algorithm_name}_query_timing_breakdown.csv"
        save_query_timing_breakdown_csv(timing_breakdown_file, query_timestamps)
        print(f"  ✓ Query timing breakdown saved to: {timing_breakdown_file}")


def save_hdf5_results(hdf5_file: Path, all_results_continuous: List[np.ndarray]):
    """
    保存查询结果到 HDF5 文件
    
    Args:
        hdf5_file: HDF5文件路径
        all_results_continuous: 连续查询结果列表
    """
    with h5py.File(str(hdf5_file), 'w') as f:
        if all_results_continuous and len(all_results_continuous) > 0:
            # 将列表转换为 numpy 数组
            # 每个元素是 (k,) 形状的数组，拼接成 (nq, k) 形状
            neighbors_array = np.vstack(all_results_continuous)
            f.create_dataset('neighbors_continuous', data=neighbors_array, compression='gzip')
        else:
            # 创建空数据集
            f.create_dataset('neighbors_continuous', data=np.array([]).reshape(0, 10))


def save_summary_csv(csv_file: Path, metrics: BenchmarkMetrics):
    """
    保存摘要指标到 CSV 文件
    
    Args:
        csv_file: CSV文件路径
        metrics: 测评指标对象
    """
    summary = {
        'algorithm': [metrics.algorithm_name],
        'dataset': [metrics.dataset_name],
        'mean_query_latency_ms': [metrics.mean_latency()],
        'p50_query_latency_ms': [metrics.p50_latency()],
        'p95_query_latency_ms': [metrics.p95_latency()],
        'p99_query_latency_ms': [metrics.p99_latency()],
        'mean_query_throughput_qps': [metrics.mean_query_throughput()],
        'mean_insert_throughput_ops': [metrics.mean_insert_throughput()],
        'total_time_s': [metrics.total_time / 1e6] if metrics.total_time else [0],
        'num_searches': [metrics.num_searches],
        'distance': [metrics.distance],
        'search_type': [metrics.search_type],
        'k': [metrics.count],
    }
    
    # 添加 cache miss 统计（如果有）
    if metrics.cache_miss_per_batch and len(metrics.cache_miss_per_batch) > 0:
        summary['mean_cache_miss'] = [np.mean(metrics.cache_miss_per_batch)]
        summary['mean_cache_references'] = [np.mean(metrics.cache_references_per_batch)]
        summary['mean_cache_miss_rate'] = [np.mean(metrics.cache_miss_rate_per_batch)]
    
    df = pd.DataFrame(summary)
    df.to_csv(csv_file, index=False)


def save_batch_metric_csv(csv_file: Path, values: List[float], column_name: str):
    """
    保存批次级别的单个指标到 CSV 文件
    
    Args:
        csv_file: CSV文件路径
        values: 指标值列表
        column_name: 列名
    """
    df = pd.DataFrame({
        'batch_idx': list(range(len(values))),
        column_name: values
    })
    df.to_csv(csv_file, index=False)


def save_cache_miss_csv(
    csv_file: Path,
    cache_miss_per_batch: List[int],
    cache_references_per_batch: List[int],
    cache_miss_rate_per_batch: List[float]
):
    """
    保存 cache miss 统计数据到 CSV 文件
    
    Args:
        csv_file: CSV文件路径
        cache_miss_per_batch: 每个批次的 cache miss 数量列表
        cache_references_per_batch: 每个批次的 cache references 数量列表
        cache_miss_rate_per_batch: 每个批次的 cache miss 率列表
    """
    # 确保所有列表长度一致
    max_len = max(
        len(cache_miss_per_batch),
        len(cache_references_per_batch),
        len(cache_miss_rate_per_batch)
    )
    
    # 填充缺失值
    cache_miss = list(cache_miss_per_batch) + [0] * (max_len - len(cache_miss_per_batch))
    cache_refs = list(cache_references_per_batch) + [0] * (max_len - len(cache_references_per_batch))
    cache_rate = list(cache_miss_rate_per_batch) + [0.0] * (max_len - len(cache_miss_rate_per_batch))
    
    df = pd.DataFrame({
        'batch_idx': list(range(max_len)),
        'cache_misses': cache_miss,
        'cache_references': cache_refs,
        'cache_miss_rate': cache_rate
    })
    df.to_csv(csv_file, index=False)


def save_query_timing_breakdown_csv(
    csv_file: Path,
    query_timestamps: List[Dict]
):
    """
    保存查询时间拆分数据到 CSV 文件
    
    用于诊断锁争用问题：
    - total_time: 总查询时间（包括锁等待）
    - lock_wait_time: 锁等待时间
    - query_time: 纯查询执行时间
    - lock_wait_ratio: 锁等待时间占比
    
    Args:
        csv_file: CSV文件路径
        query_timestamps: 查询时间戳列表，每个元素包含 total_time, lock_wait_time, query_time
    """
    if not query_timestamps:
        return
    
    batch_indices = []
    total_times = []
    lock_wait_times = []
    query_times = []
    lock_wait_ratios = []
    
    for idx, ts in enumerate(query_timestamps):
        batch_indices.append(idx)
        total_time = ts.get('total_time', 0) / 1000  # 微秒转毫秒
        lock_wait = ts.get('lock_wait_time', 0) / 1000  # 微秒转毫秒
        query_time = ts.get('query_time', 0) / 1000  # 微秒转毫秒
        
        total_times.append(total_time)
        lock_wait_times.append(lock_wait)
        query_times.append(query_time)
        
        # 计算锁等待时间占比
        if total_time > 0:
            lock_wait_ratios.append(lock_wait / total_time)
        else:
            lock_wait_ratios.append(0.0)
    
    df = pd.DataFrame({
        'batch_idx': batch_indices,
        'total_time_ms': total_times,
        'lock_wait_time_ms': lock_wait_times,
        'query_time_ms': query_times,
        'lock_wait_ratio': lock_wait_ratios
    })
    df.to_csv(csv_file, index=False)
