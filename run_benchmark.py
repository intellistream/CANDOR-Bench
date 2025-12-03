#!/usr/bin/env python3
"""
benchmark_anns Main Runner

主运行程序，用于执行流式索引基准测试
设计参考 big-ann-benchmarks/run.py 和 benchmark/runner.py

用法示例：
    # 基础用法
    python run_benchmark.py --algorithm faiss_hnsw --dataset sift --runbook batch_sizes/batch_2500
    
    # 指定输出目录
    python run_benchmark.py --algorithm faiss_hnsw --dataset sift --runbook batch_sizes/batch_2500 --output results/exp1
    
    # 指定算法参数（JSON 格式）
    python run_benchmark.py --algorithm faiss_hnsw --dataset sift --runbook event_rates/rate_10000 \
        --algo-params '{"M": 32, "efConstruction": 200, "efSearch": 100}'
    
    # 多次运行取最佳结果
    python run_benchmark.py --algorithm faiss_hnsw --dataset sift --runbook batch_sizes/batch_2500 --runs 3
    
    # 列出可用选项
    python run_benchmark.py --list-algorithms
    python run_benchmark.py --list-datasets
    python run_benchmark.py --list-runbooks
"""

import os
import sys
import argparse
import json
import yaml
import time
import traceback
import re
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# benchmark_anns 是独立项目，使用相对导入
from bench.algorithms.registry import get_algorithm, auto_register_algorithms, ALGORITHMS, get_algorithm_params_from_config, get_all_algorithm_param_combinations
from datasets.registry import get_dataset, DATASETS
from bench.runner import BenchmarkRunner
from bench.metrics import BenchmarkMetrics


def _extract_key_params(params: Dict[str, Any], max_depth: int = 4) -> Dict[str, Any]:
    """
    从嵌套参数中提取关键参数用于生成文件夹名
    
    Args:
        params: 参数字典（可能嵌套）
        max_depth: 最大递归深度
        
    Returns:
        扁平化的关键参数字典
    """
    result = {}
    
    # 我们关心的关键参数
    key_params = {
        'max_degree', 'M', 'ef_construction', 'efConstruction', 
        'ef_search', 'efSearch', 'prefetch_mode', 'nlist', 'nprobe',
        'search_L', 'index_L', 'R', 'L', 'alpha'
    }
    
    def _flatten(d: Dict[str, Any], prefix: str = "", depth: int = 0):
        if depth >= max_depth:
            return
        for key, value in d.items():
            new_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                _flatten(value, f"{new_key}_", depth + 1)
            elif isinstance(value, (str, int, float, bool)):
                # 只保留关键参数，或 prefetch_mode 这类特殊参数
                if key in key_params or 'prefetch' in key.lower():
                    result[key] = value  # 使用简短的 key，不带前缀
    
    _flatten(params)
    return result


def _generate_params_folder_name(algorithm_params: Dict[str, Any]) -> str:
    """
    根据算法参数生成有意义的文件夹名
    
    Args:
        algorithm_params: 包含 build_params 和 query_params 的字典
        
    Returns:
        文件夹名（如 "M32_ef200_prefetch-hardcoded_efsearch40"）
    """
    if not algorithm_params:
        return "default"
    
    parts = []
    
    # 处理构建参数
    build_params = algorithm_params.get('build_params', {})
    if build_params:
        flat_build = _extract_key_params(build_params)
        for key, value in sorted(flat_build.items()):
            # 简化参数名
            short_key = key.replace('max_degree', 'M').replace('ef_construction', 'ef')
            short_key = short_key.replace('efConstruction', 'ef').replace('prefetch_mode', 'prefetch')
            
            # 格式化值
            if isinstance(value, bool):
                if value:
                    parts.append(short_key)
            elif isinstance(value, float):
                parts.append(f"{short_key}{value:.0f}")
            else:
                parts.append(f"{short_key}-{value}")
    
    # 处理查询参数
    query_params = algorithm_params.get('query_params', {})
    if query_params:
        flat_query = _extract_key_params(query_params)
        for key, value in sorted(flat_query.items()):
            short_key = key.replace('ef_search', 'efsearch').replace('efSearch', 'efsearch')
            
            if isinstance(value, bool):
                if value:
                    parts.append(short_key)
            elif isinstance(value, float):
                parts.append(f"{short_key}{value:.0f}")
            else:
                parts.append(f"{short_key}-{value}")
    
    if not parts:
        return "default"
    
    # 组合并限制长度
    folder_name = "_".join(parts)
    
    # 清理非法字符
    folder_name = re.sub(r'[^\w\-]', '_', folder_name)
    folder_name = re.sub(r'_+', '_', folder_name).strip('_')
    
    # 限制长度
    if len(folder_name) > 100:
        folder_name = folder_name[:100]
    
    return folder_name if folder_name else "default"


def list_algorithms():
    """列出所有可用的算法"""
    auto_register_algorithms()
    print("\n可用算法：")
    print("=" * 60)
    for name in sorted(ALGORITHMS.keys()):
        print(f"  - {name}")
    print()


def list_datasets():
    """列出所有可用的数据集"""
    print("\n可用数据集：")
    print("=" * 60)
    for name in sorted(DATASETS.keys()):
        print(f"  - {name}")
    print()


def list_runbooks():
    """列出所有可用的 runbooks"""
    runbook_dir = Path(__file__).parent / "runbooks"
    print("\n可用 Runbooks（按类别分类）：")
    print("=" * 60)
    
    categories = {}
    for category_dir in sorted(runbook_dir.iterdir()):
        if category_dir.is_dir() and not category_dir.name.startswith('_'):
            yaml_files = list(category_dir.glob("*.yaml"))
            if yaml_files:
                categories[category_dir.name] = [f.stem for f in sorted(yaml_files)]
    
    for category, files in sorted(categories.items()):
        print(f"\n[{category}] ({len(files)} 个)")
        for file in files:
            print(f"  - {file}")
    print()


def find_runbook_path(runbook_name: str) -> Optional[Path]:
    """
    查找 runbook 文件路径
    
    Args:
        runbook_name: runbook 名称（不带 .yaml 后缀）
        
    Returns:
        完整路径，如果找不到返回 None
    """
    runbook_dir = Path(__file__).parent / "runbooks"
    
    # 如果已经是完整路径
    if Path(runbook_name).exists():
        return Path(runbook_name)
    
    # 先在 runbooks 根目录查找
    yaml_path = runbook_dir / f"{runbook_name}.yaml"
    if yaml_path.exists():
        return yaml_path
    
    # 在所有类别目录中搜索
    for category_dir in runbook_dir.iterdir():
        if category_dir.is_dir():
            yaml_path = category_dir / f"{runbook_name}.yaml"
            if yaml_path.exists():
                return yaml_path
    
    return None




def load_runbook(runbook_path: Path, dataset_name: str = None) -> Tuple[Dict, str]:
    """
    加载 runbook 文件
    
    Args:
        runbook_path: runbook 文件路径
        dataset_name: 数据集名称（用于选择配置）
        
    Returns:
        (runbook字典, 数据集名称) 元组
        runbook 格式:
        {
            'dataset_name': {
                'max_pts': 1000000,
                1: {'operation': 'startHPC'},
                2: {'operation': 'initial', 'start': 0, 'end': 50000},
                ...
            }
        }
    """
    with open(runbook_path, 'r') as f:
        content = yaml.safe_load(f)
    
    # 如果没有指定数据集，使用 runbook 中的第一个数据集
    if dataset_name is None:
        # 找到第一个包含操作的数据集配置
        for key, value in content.items():
            if isinstance(value, dict) and any(isinstance(k, int) for k in value.keys()):
                dataset_name = key
                break
        
        if dataset_name is None:
            raise ValueError(f"无法从 runbook 中找到数据集配置: {runbook_path}")
    
    return content, dataset_name


def run_benchmark(
    algorithm,
    dataset,
    runbook: Dict,
    dataset_name: str,
    k: int = 10,
    run_count: int = 1,
    output_dir: str = "results",
    enable_cache_profiling: bool = False
) -> Tuple[BenchmarkMetrics, List]:
    """
    执行基准测试主逻辑
    适配新的 BenchmarkRunner API
    
    Args:
        algorithm: 算法实例
        dataset: 数据集实例
        runbook: runbook 操作字典
        k: kNN 的 k 值
        run_count: 运行次数（取最佳结果）
        output_dir: 输出目录
        
    Returns:
        (metrics, results): 性能指标和搜索结果
    """
    print(f"\n{'='*80}")
    print(f"运行算法: {getattr(algorithm, 'name', algorithm.__class__.__name__)}")
    print(f"数据集: {dataset.short_name()}")
    print(f"距离度量: {dataset.distance()}")
    print(f"搜索类型: {dataset.search_type()}")
    print(f"k 值: {k}")
    print(f"运行次数: {run_count}")
    print(f"{'='*80}\n")
    
    best_metrics = None
    best_results = None
    best_results_continuous = []
    best_attrs = {}
    best_time = float('inf')
    
    # 多次运行，取最佳结果
    for run_idx in range(run_count):
        if run_count > 1:
            print(f"\n--- 第 {run_idx + 1}/{run_count} 次运行 ---\n")
        
        try:
            # 创建 runner 实例
            runner = BenchmarkRunner(
                algorithm=algorithm,
                dataset=dataset,
                k=k,
                save_timestamps=True,
                output_dir=output_dir,
                enable_cache_profiling=enable_cache_profiling
            )
            
            # 执行 runbook
            metrics = runner.run_runbook(runbook, dataset_name=dataset_name)
            
            # 记录最佳结果
            total_time = metrics.total_time if hasattr(metrics, 'total_time') else runner.attrs['totalTime']
            if total_time < best_time:
                best_time = total_time
                best_metrics = metrics
                best_results = runner.all_results
                best_results_continuous = runner.all_results_continuous
                best_attrs = runner.attrs
            
            if run_count > 1:
                print(f"\n第 {run_idx + 1} 次运行完成，总时间: {total_time:.2f} 秒")
        
        except Exception as e:
            print(f"\n✗ 第 {run_idx + 1} 次运行失败: {e}")
            traceback.print_exc()
            if run_idx == run_count - 1:  # 最后一次尝试
                raise
            continue
    
    if run_count > 1:
        print(f"\n最佳运行时间: {best_time:.2f} 秒")
    
    return best_metrics, best_results, best_results_continuous, best_attrs


def get_result_filename(
    dataset: str,
    algorithm: str,
    algorithm_params: Dict[str, Any],
    runbook_name: str,
    output_dir: Optional[Path] = None
) -> str:
    """
    生成结果文件路径，按数据集/算法/参数组织
    
    格式: results/[dataset]/[algorithm]/[params_folder]
    
    Args:
        dataset: 数据集名称
        algorithm: 算法名称
        algorithm_params: 算法参数字典（包含 build_params 和 query_params）
        runbook_name: runbook 名称
        output_dir: 输出根目录
    
    Returns:
        结果文件路径（不含扩展名）
    """
    if output_dir is None:
        output_dir = Path('results')
    
    # 构建目录结构: results/dataset/algorithm/
    parts = [str(output_dir), dataset, algorithm]
    
    # 生成参数文件夹名
    params_folder = _generate_params_folder_name(algorithm_params)
    parts.append(params_folder)
    
    return os.path.join(*parts)


def store_results(
    metrics: BenchmarkMetrics,
    results: List,
    results_continuous: List,
    attrs: Dict[str, Any],
    output_dir: Path,
    metadata: Dict[str, Any]
):
    """
    保存测试结果到文件，兼容 big-ann-benchmarks 格式
    
    存储格式:
    - .hdf5: 存储 neighbor 结果（正式查询 + 周期性查询）
    - .csv: 存储归一化的性能指标（单行）
    - _batchLatency.csv: 存储批次级延迟数据
    - _batchThroughput.csv: 存储批次级吞吐量数据
    - _continuousQueryLatency.csv: 周期性查询延迟
    - _batch_insert_qps.csv: 批次插入QPS（每批次的插入吞吐量）
    - _batch_query_qps.csv: 批次查询QPS（每批次的查询吞吐量）
    - _batch_query_latency.csv: 批次查询延迟（每批次的查询延迟）
    - _summary.txt: 人类可读的测试摘要
    
    Args:
        metrics: 性能指标对象
        results: 正式查询结果列表 (search operation)
        results_continuous: 周期性查询结果列表 (batch_insert)
        attrs: 运行时属性字典（包含周期性查询数据）
        output_dir: 输出根目录
        metadata: 元数据信息
    """
    # 生成结果路径
    result_path = get_result_filename(
        dataset=metadata['dataset'],
        algorithm=metadata['algorithm'],
        algorithm_params=metadata.get('algorithm_params', {}),
        runbook_name=metadata['runbook'],
        output_dir=output_dir
    )
    
    result_dir = Path(result_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 基础文件名
    base_name = Path(result_path).name
    
    # ========== 1. HDF5 文件：存储 neighbors 结果 ==========
    hdf5_file = result_dir / f"{base_name}.hdf5"
    
    try:
        with h5py.File(hdf5_file, 'w') as f:
            # 1.1 存储正式查询结果 (search operation)
            if results and len(results) > 0:
                all_neighbors = np.vstack(results) if isinstance(results[0], np.ndarray) else np.array(results)
                f.create_dataset('neighbors', data=all_neighbors, compression='gzip')
                print(f"✓ HDF5 正式查询结果: {hdf5_file} (shape: {all_neighbors.shape})")
            
            # 1.2 存储周期性查询结果 (batch_insert continuous queries)
            if results_continuous and len(results_continuous) > 0:
                all_continuous = np.vstack(results_continuous) if isinstance(results_continuous[0], np.ndarray) else np.array(results_continuous)
                f.create_dataset('neighbors_continuous', data=all_continuous, compression='gzip')
                print(f"✓ HDF5 周期性查询结果: (shape: {all_continuous.shape})")
            
            if not results and not results_continuous:
                print(f"⚠ 无查询结果，创建空 HDF5 文件: {hdf5_file}")
    except Exception as e:
        print(f"⚠ HDF5 保存失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 2. CSV 文件：存储归一化的性能指标 ==========
    csv_file = result_dir / f"{base_name}.csv"
    
    # 构建指标字典（归一化为单行）
    summary_attrs = {
        'algorithm': metadata['algorithm'],
        'dataset': metadata['dataset'],
        'runbook': metadata['runbook'],
        'k': metadata.get('k', 10),
        'distance': metrics.distance,
        'count': metrics.count,
        'run_count': metadata.get('run_count', 1),
        'mean_query_throughput': float(metrics.mean_query_throughput()) if hasattr(metrics, 'mean_query_throughput') else 0.0,
        'mean_recall': float(metrics.mean_recall()) if hasattr(metrics, 'mean_recall') else 0.0,
        'mean_latency_ms': float(metrics.mean_latency()) if hasattr(metrics, 'mean_latency') else 0.0,
        'p50_latency_ms': float(metrics.p50_latency()) if hasattr(metrics, 'p50_latency') else 0.0,
        'p95_latency_ms': float(metrics.p95_latency()) if hasattr(metrics, 'p95_latency') else 0.0,
        'p99_latency_ms': float(metrics.p99_latency()) if hasattr(metrics, 'p99_latency') else 0.0,
        'mean_insert_throughput': float(metrics.mean_insert_throughput()) if hasattr(metrics, 'mean_insert_throughput') else 0.0,
        'total_time_seconds': float(getattr(metrics, 'total_time', 0.0)),
        'num_searches': getattr(metrics, 'num_searches', 0),
    }
    
    # 添加算法参数
    if metadata.get('algorithm_params'):
        for key, value in metadata['algorithm_params'].items():
            summary_attrs[f'param_{key}'] = value
    
    # 保存为单行 CSV
    df = pd.DataFrame([summary_attrs])
    df.to_csv(csv_file, index=False)
    print(f"✓ CSV 指标已保存: {csv_file}")
    
    # ========== 3. 批次级延迟 CSV ==========
    if hasattr(metrics, 'latencies') and len(metrics.latencies) > 0:
        latency_file = result_dir / f"{base_name}_batchLatency.csv"
        latency_df = pd.DataFrame({
            'batch_idx': range(len(metrics.latencies)),
            'latency_us': metrics.latencies
        })
        latency_df.to_csv(latency_file, index=False)
        print(f"✓ 批次延迟已保存: {latency_file}")
    
    # ========== 4. 批次级吞吐量 CSV ==========
    if hasattr(metrics, 'throughputs') and len(metrics.throughputs) > 0:
        throughput_file = result_dir / f"{base_name}_batchThroughput.csv"
        throughput_df = pd.DataFrame({
            'batch_idx': range(len(metrics.throughputs)),
            'throughput': metrics.throughputs
        })
        throughput_df.to_csv(throughput_file, index=False)
        print(f"✓ 批次吞吐量已保存: {throughput_file}")
    
    # ========== 5. 周期性查询延迟 CSV ==========
    # 注意：已合并到 batch_query_latency.csv 中，此处不再单独保存
    
    # ========== 6. 周期性查询结果（邻居ID数组）- 已在HDF5中保存 ==========
    # 注意：continuousQueryResults 存储的是邻居 ID 数组，已保存到 HDF5 的 neighbors_continuous dataset
    # 不再单独保存 CSV（与 big-ann-benchmarks 一致）
    
    # ========== 7. 批次级插入QPS (Throughput) CSV ==========
    if 'batchinsertThroughtput' in attrs and len(attrs['batchinsertThroughtput']) > 0:
        insert_qps_file = result_dir / f"{base_name}_batch_insert_qps.csv"
        insert_qps_df = pd.DataFrame({
            'batch_idx': range(len(attrs['batchinsertThroughtput'])),
            'insert_qps': attrs['batchinsertThroughtput']
        })
        insert_qps_df.to_csv(insert_qps_file, index=False)
        print(f"✓ 批次插入QPS已保存: {insert_qps_file}")
    
    # ========== 8. 批次级查询QPS CSV ==========
    # 查询QPS = 查询数量 / 查询延迟（秒）
    if 'continuousQueryLatencies' in attrs and len(attrs['continuousQueryLatencies']) > 0:
        query_qps_file = result_dir / f"{base_name}_batch_query_qps.csv"
        # 获取每批次的查询数量（默认使用 dataset 的查询集大小）
        queries_per_batch = attrs.get('querySize', 100)
        query_qps_list = []
        batch_indices = []
        
        for idx, latency_seconds in enumerate(attrs['continuousQueryLatencies']):
            # 过滤异常值：只保留正常的延迟（>0且合理范围）
            if latency_seconds > 0 and latency_seconds < 3600:  # 最多1小时
                qps = queries_per_batch / latency_seconds
                query_qps_list.append(qps)
                batch_indices.append(idx)
            else:
                # 异常延迟，跳过该批次
                print(f"  ⚠️  跳过异常查询延迟: batch_idx={idx}, latency={latency_seconds:.2f}s")
        
        if query_qps_list:
            query_qps_df = pd.DataFrame({
                'batch_idx': batch_indices,
                'query_qps': query_qps_list
            })
            query_qps_df.to_csv(query_qps_file, index=False)
            print(f"✓ 批次查询QPS已保存: {query_qps_file} ({len(query_qps_list)} 个有效批次)")
    
    # ========== 9. Cache Miss CSV ==========
    # 保存插入的cache miss统计
    if hasattr(metrics, 'cache_miss_per_batch') and len(metrics.cache_miss_per_batch) > 0:
        cache_miss_file = result_dir / f"{base_name}_insert_cache_miss.csv"
        cache_miss_df = pd.DataFrame({
            'batch_idx': range(len(metrics.cache_miss_per_batch)),
            'cache_misses': metrics.cache_miss_per_batch,
            'cache_references': metrics.cache_references_per_batch if hasattr(metrics, 'cache_references_per_batch') else [0] * len(metrics.cache_miss_per_batch),
            'cache_miss_rate': metrics.cache_miss_rate_per_batch if hasattr(metrics, 'cache_miss_rate_per_batch') else [0.0] * len(metrics.cache_miss_per_batch)
        })
        cache_miss_df.to_csv(cache_miss_file, index=False)
        print(f"✓ 插入 Cache Miss 已保存: {cache_miss_file}")
    
    # 保存查询的cache miss统计
    if hasattr(metrics, 'query_cache_miss_per_batch') and len(metrics.query_cache_miss_per_batch) > 0:
        query_cache_miss_file = result_dir / f"{base_name}_query_cache_miss.csv"
        query_cache_miss_df = pd.DataFrame({
            'query_idx': range(len(metrics.query_cache_miss_per_batch)),
            'cache_misses': metrics.query_cache_miss_per_batch,
            'cache_references': metrics.query_cache_references_per_batch if hasattr(metrics, 'query_cache_references_per_batch') else [0] * len(metrics.query_cache_miss_per_batch),
            'cache_miss_rate': metrics.query_cache_miss_rate_per_batch if hasattr(metrics, 'query_cache_miss_rate_per_batch') else [0.0] * len(metrics.query_cache_miss_per_batch)
        })
        query_cache_miss_df.to_csv(query_cache_miss_file, index=False)
        print(f"✓ 查询 Cache Miss 已保存: {query_cache_miss_file}")
    
    # ========== 10. 批次级查询延迟 CSV (毫秒) ==========
    if 'continuousQueryLatencies' in attrs and len(attrs['continuousQueryLatencies']) > 0:
        query_latency_file = result_dir / f"{base_name}_batch_query_latency.csv"
        # 转换为毫秒，并过滤异常值
        query_latency_ms = []
        batch_indices = []
        
        for idx, lat_seconds in enumerate(attrs['continuousQueryLatencies']):
            # 过滤异常值：只保留正常的延迟（>0且合理范围）
            if lat_seconds > 0 and lat_seconds < 3600:  # 最多1小时
                query_latency_ms.append(lat_seconds * 1000)  # 转换为毫秒
                batch_indices.append(idx)
            else:
                # 异常延迟，跳过
                print(f"  ⚠️  跳过异常查询延迟: batch_idx={idx}, latency={lat_seconds:.2f}s")
        
        if query_latency_ms:
            query_latency_df = pd.DataFrame({
                'batch_idx': batch_indices,
                'query_latency_ms': query_latency_ms
            })
            query_latency_df.to_csv(query_latency_file, index=False)
            print(f"✓ 批次查询延迟已保存: {query_latency_file} ({len(query_latency_ms)} 个有效批次)")
    
    # ========== 11. 生成人类可读的摘要 ==========
    summary_file = result_dir / f"{base_name}_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("benchmark_anns 测试结果摘要\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("[测试配置]\n")
        f.write(f"算法: {metadata['algorithm']}\n")
        f.write(f"数据集: {metadata['dataset']}\n")
        f.write(f"Runbook: {metadata['runbook']}\n")
        f.write(f"k 值: {metadata.get('k', 10)}\n")
        f.write(f"运行次数: {metadata.get('run_count', 1)}\n")
        f.write(f"测试时间: {metadata.get('timestamp', 'N/A')}\n")
        
        if metadata.get('algorithm_params'):
            f.write(f"\n[算法参数]\n")
            for key, value in metadata['algorithm_params'].items():
                f.write(f"  {key}: {value}\n")
        
        f.write("\n[性能指标]\n")
        f.write(f"平均查询吞吐量: {summary_attrs['mean_query_throughput']:.2f} queries/s\n")
        f.write(f"平均召回率: {summary_attrs['mean_recall']:.4f}\n")
        f.write(f"平均延迟: {summary_attrs['mean_latency_ms']:.2f} ms\n")
        f.write(f"P50 延迟: {summary_attrs['p50_latency_ms']:.2f} ms\n")
        f.write(f"P95 延迟: {summary_attrs['p95_latency_ms']:.2f} ms\n")
        f.write(f"P99 延迟: {summary_attrs['p99_latency_ms']:.2f} ms\n")
        f.write(f"平均插入吞吐量: {summary_attrs['mean_insert_throughput']:.2f} ops/s\n")
        f.write(f"总时间: {summary_attrs['total_time_seconds']:.2f} 秒\n")
        f.write(f"查询次数: {summary_attrs['num_searches']}\n")
        
        # 周期性查询统计
        if 'continuousQueryLatencies' in attrs and len(attrs['continuousQueryLatencies']) > 0:
            f.write(f"\n[周期性查询统计]\n")
            f.write(f"周期性查询次数: {len(attrs['continuousQueryLatencies'])}\n")
            cq_latencies_ms = [lat * 1000 for lat in attrs['continuousQueryLatencies']]
            f.write(f"平均延迟: {np.mean(cq_latencies_ms):.2f} ms\n")
            f.write(f"查询结果已保存到 HDF5: neighbors_continuous dataset\n")
        
        # 批次级数据统计
        if 'batchinsertThroughtput' in attrs and len(attrs['batchinsertThroughtput']) > 0:
            f.write(f"\n[批次级数据统计]\n")
            f.write(f"批次数量: {len(attrs['batchinsertThroughtput'])}\n")
            f.write(f"平均插入QPS: {np.mean(attrs['batchinsertThroughtput']):.2f} ops/s\n")
            f.write(f"最大插入QPS: {np.max(attrs['batchinsertThroughtput']):.2f} ops/s\n")
            f.write(f"最小插入QPS: {np.min(attrs['batchinsertThroughtput']):.2f} ops/s\n")
            f.write(f"批次插入QPS已保存到: {base_name}_batch_insert_qps.csv\n")
            
            if 'continuousQueryLatencies' in attrs and len(attrs['continuousQueryLatencies']) > 0:
                f.write(f"批次查询QPS已保存到: {base_name}_batch_query_qps.csv\n")
                f.write(f"批次查询延迟已保存到: {base_name}_batch_query_latency.csv\n")
        
        # Cache Miss 统计
        if hasattr(metrics, 'cache_miss_per_batch') and len(metrics.cache_miss_per_batch) > 0:
            f.write(f"\n[Cache Miss 统计]\n")
            f.write(f"插入批次数量: {len(metrics.cache_miss_per_batch)}\n")
            f.write(f"插入平均 Cache Misses: {np.mean(metrics.cache_miss_per_batch):,.0f}\n")
            if hasattr(metrics, 'cache_miss_rate_per_batch') and len(metrics.cache_miss_rate_per_batch) > 0:
                f.write(f"插入平均 Cache Miss Rate: {np.mean(metrics.cache_miss_rate_per_batch):.2%}\n")
            f.write(f"插入 Cache Miss 已保存到: {base_name}_insert_cache_miss.csv\n")
        
        if hasattr(metrics, 'query_cache_miss_per_batch') and len(metrics.query_cache_miss_per_batch) > 0:
            f.write(f"\n查询次数: {len(metrics.query_cache_miss_per_batch)}\n")
            f.write(f"查询平均 Cache Misses: {np.mean(metrics.query_cache_miss_per_batch):,.0f}\n")
            if hasattr(metrics, 'query_cache_miss_rate_per_batch') and len(metrics.query_cache_miss_rate_per_batch) > 0:
                f.write(f"查询平均 Cache Miss Rate: {np.mean(metrics.query_cache_miss_rate_per_batch):.2%}\n")
            f.write(f"查询 Cache Miss 已保存到: {base_name}_query_cache_miss.csv\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✓ 测试摘要已保存: {summary_file}")
    print(f"\n结果目录: {result_dir}")



def print_results_summary(metrics):
    """打印结果摘要到控制台"""
    print("\n" + "=" * 80)
    print("测试结果摘要")
    print("=" * 80)
    print(f"算法: {metrics.algorithm_name}")
    print(f"数据集: {metrics.dataset_name}")
    
    # 计算总时间（支持不同的格式）
    total_time = float(metrics.total_time) if hasattr(metrics, 'total_time') else 0
    if total_time < 1000:  # 如果小于 1000，可能是秒
        print(f"总时间: {total_time:.2f} 秒")
    else:
        print(f"总时间: {total_time/1e6:.2f} 秒")
    
    print(f"查询次数: {metrics.num_searches}")
    print(f"\n性能指标:")
    
    # if hasattr(metrics, 'mean_recall'):
    #     print(f"  平均召回率: {metrics.mean_recall():.4f}")
    if hasattr(metrics, 'mean_query_throughput'):
        print(f"  平均查询吞吐量: {metrics.mean_query_throughput():.2f} queries/s")
    if hasattr(metrics, 'mean_insert_throughput'):
        print(f"  平均插入吞吐量: {metrics.mean_insert_throughput():.2f} ops/s")
    if hasattr(metrics, 'mean_latency'):
        print(f"  平均延迟: {metrics.mean_latency():.2f} ms")
    if hasattr(metrics, 'p50_latency'):
        print(f"  P50 延迟: {metrics.p50_latency():.2f} ms")
    if hasattr(metrics, 'p95_latency'):
        print(f"  P95 延迟: {metrics.p95_latency():.2f} ms")
    if hasattr(metrics, 'p99_latency'):
        print(f"  P99 延迟: {metrics.p99_latency():.2f} ms")
    
    if hasattr(metrics, 'maintenance_budget_used'):
        print(f"  维护预算使用: {metrics.maintenance_budget_used/1e6:.2f} 秒")
    
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='benchmark_anns - 流式索引基准测试主程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 运行基准测试
  python run_benchmark.py --algorithm faiss_hnsw --dataset sift --runbook batch_2500
  
  # 指定算法参数
  python run_benchmark.py --algorithm faiss_hnsw --dataset sift --runbook rate_10000 \\
      --algo-params '{"M": 32, "efConstruction": 200, "efSearch": 100}'
  
  # 指定输出目录
  python run_benchmark.py --algorithm faiss_hnsw --dataset sift --runbook batch_2500 \\
      --output results/exp1
  
  # 启用 cache miss 测量
  python run_benchmark.py --algorithm faiss_hnsw --dataset sift --runbook general_experiment \\
      --enable-cache-miss
  
  # 列出可用选项
  python run_benchmark.py --list-algorithms
  python run_benchmark.py --list-datasets
  python run_benchmark.py --list-runbooks
        """
    )
    
    # 列表选项
    parser.add_argument('--list-algorithms', action='store_true', 
                       help='列出所有可用的算法')
    parser.add_argument('--list-datasets', action='store_true',
                       help='列出所有可用的数据集')
    parser.add_argument('--list-runbooks', action='store_true',
                       help='列出所有可用的 runbooks')
    
    # 主要参数
    parser.add_argument('--algorithm', type=str,
                       help='算法名称（如: faiss_hnsw, candy_lshapg）')
    parser.add_argument('--dataset', type=str,
                       help='数据集名称（如: sift, random-xs）')
    parser.add_argument('--runbook', type=str,
                       help='Runbook 名称（如: batch_2500, rate_10000）')
    
    # 可选参数
    parser.add_argument('--algo-params', type=str, default='{}',
                       help='算法参数（JSON 格式字符串）')
    parser.add_argument('--k', type=int, default=10,
                       help='kNN 查询的 k 值（默认: 10）')
    parser.add_argument('--runs', type=int, default=1,
                       help='运行次数，取最佳结果（默认: 1）')
    parser.add_argument('--output', type=str, default='results',
                       help='输出目录（默认: results）')
    parser.add_argument('--rebuild', action='store_true',
                       help='强制重建索引（即使索引文件存在）')
    parser.add_argument('--enable-cache-profiling', action='store_true',
                       help='启用 cache miss 性能监测（需要 perf 工具支持）')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存结果文件')
        
    args = parser.parse_args()
    
    # 处理列表请求
    if args.list_algorithms:
        list_algorithms()
        return
    
    if args.list_datasets:
        list_datasets()
        return
    
    if args.list_runbooks:
        list_runbooks()
        return
    
    # 验证必需参数
    if not all([args.algorithm, args.dataset, args.runbook]):
        parser.error("必须指定 --algorithm, --dataset 和 --runbook（或使用 --list-* 选项）")
    
    # 解析命令行算法参数
    try:
        cli_algo_params = json.loads(args.algo_params)
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析算法参数 JSON: {e}")
        sys.exit(1)
    
    # 自动注册算法
    auto_register_algorithms()
    
    # 获取所有参数组合
    if cli_algo_params:
        # 如果命令行指定了参数，只运行这一组
        param_combinations = [{'build_params': cli_algo_params, 'query_params': {}}]
    else:
        # 从配置文件获取所有参数组合
        param_combinations = get_all_algorithm_param_combinations(args.algorithm, args.dataset)
    
    total_combinations = len(param_combinations)
    
    print("\n" + "=" * 80)
    print("benchmark_anns 流式索引基准测试")
    print("=" * 80)
    print(f"算法: {args.algorithm}")
    print(f"数据集: {args.dataset}")
    print(f"Runbook: {args.runbook}")
    print(f"k 值: {args.k}")
    print(f"参数组合数: {total_combinations}")
    print("=" * 80 + "\n")
    
    # 1. 加载数据集（只加载一次）
    print("[1/4] 加载数据集...")
    try:
        dataset = get_dataset(args.dataset)
        print(f"✓ 数据集加载成功: {dataset.short_name()}")
        print(f"  - 距离度量: {dataset.distance()}")
        print(f"  - 向量维度: {dataset.d}")
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        sys.exit(1)
    
    # 2. 加载 runbook（只加载一次）
    print("\n[2/4] 加载 Runbook...")
    try:
        runbook_path = find_runbook_path(args.runbook)
        if not runbook_path:
            print(f"✗ 找不到 runbook: {args.runbook}")
            print("使用 --list-runbooks 查看可用的 runbooks")
            sys.exit(1)
        
        dataset_arg = args.dataset if hasattr(args, 'dataset') and args.dataset else None
        runbook, dataset_name = load_runbook(runbook_path, dataset_name=dataset_arg)
        
        if dataset_name in runbook:
            dataset_config = runbook[dataset_name]
            op_count = sum(1 for k in dataset_config.keys() if isinstance(k, int))
            print(f"✓ Runbook 加载成功: {runbook_path}")
            print(f"  - 数据集: {dataset_name}")
            print(f"  - 操作步骤: {op_count}")
            if 'max_pts' in dataset_config:
                print(f"  - 最大数据点: {dataset_config['max_pts']}")
        else:
            print(f"✓ Runbook 加载成功: {runbook_path}")
    except Exception as e:
        print(f"✗ Runbook 加载失败: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # 3. 遍历所有参数组合执行测试
    print(f"\n[3/4] 测试 ({total_combinations} 组参数)...")
    
    all_results = []
    for combo_idx, param_combo in enumerate(param_combinations, 1):
        build_params = param_combo.get('build_params', {})
        query_params = param_combo.get('query_params', {})
        
        # 生成简洁的参数描述
        param_desc = _generate_params_folder_name({'build_params': build_params, 'query_params': query_params})
        
        print(f"\n{'═' * 70}")
        print(f"  参数组合 [{combo_idx}/{total_combinations}]: {param_desc}")
        print(f"{'═' * 70}")
        
        # 打印关键构建参数
        if build_params:
            print(f"  📦 构建参数:")
            flat_build = _extract_key_params(build_params)
            for key, value in sorted(flat_build.items()):
                print(f"      • {key}: {value}")
        
        # 打印关键查询参数  
        if query_params:
            print(f"  🔍 查询参数:")
            flat_query = _extract_key_params(query_params)
            for key, value in sorted(flat_query.items()):
                print(f"      • {key}: {value}")
        
        print(f"{'─' * 70}")
        
        try:
            # 初始化算法（每次用不同参数）
            # 将 build_params 包装成 index_params 传给算法构造函数
            algo_kwargs = {'index_params': build_params} if build_params else {}
            algorithm = get_algorithm(args.algorithm, dataset=args.dataset, **algo_kwargs)
            
            # 如果算法支持设置查询参数
            if hasattr(algorithm, 'set_query_arguments') and query_params:
                algorithm.set_query_arguments(query_params)
            
            # 执行测试
            metrics, best_results, best_results_continuous, best_attrs = run_benchmark(
                algorithm=algorithm,
                dataset=dataset,
                runbook=runbook,
                dataset_name=dataset_name,
                k=args.k,
                run_count=args.runs,
                output_dir=args.output,
                enable_cache_profiling=args.enable_cache_profiling
            )
            print(f"✓ 参数组合 [{combo_idx}] 测试完成")
            
            # 保存结果
            if not args.no_save:
                actual_algo_params = {
                    'build_params': build_params,
                    'query_params': query_params
                }
                
                metadata = {
                    'algorithm': args.algorithm,
                    'algorithm_params': actual_algo_params,
                    'dataset': args.dataset,
                    'runbook': args.runbook,
                    'k': args.k,
                    'run_count': args.runs,
                    'timestamp': datetime.now().isoformat(),
                    'param_combo_index': combo_idx,
                    'total_param_combos': total_combinations,
                }
                
                output_dir = Path(args.output)
                store_results(metrics, best_results, best_results_continuous, best_attrs, output_dir, metadata)
                print(f"✓ 参数组合 [{combo_idx}] 结果已保存")
            
            all_results.append({
                'combo_idx': combo_idx,
                'build_params': build_params,
                'query_params': query_params,
                'metrics': metrics,
                'success': True
            })
            
            # 打印单次结果摘要
            print_results_summary(metrics)
            
        except Exception as e:
            print(f"✗ 参数组合 [{combo_idx}] 执行失败: {e}")
            traceback.print_exc()
            all_results.append({
                'combo_idx': combo_idx,
                'build_params': build_params,
                'query_params': query_params,
                'error': str(e),
                'success': False
            })
    
    # 4. 打印总体摘要
    print(f"\n[4/4] 测试完成摘要")
    print("=" * 80)
    success_count = sum(1 for r in all_results if r['success'])
    fail_count = len(all_results) - success_count
    print(f"总参数组合数: {total_combinations}")
    print(f"成功: {success_count}, 失败: {fail_count}")
    
    if success_count > 0:
        print("\n成功的测试结果:")
        for result in all_results:
            if result['success']:
                metrics = result['metrics']
                build_info = _generate_params_folder_name({'build_params': result['build_params'], 'query_params': result['query_params']})
                recall = metrics.mean_recall() if hasattr(metrics, 'mean_recall') else 'N/A'
                qps = metrics.mean_qps() if hasattr(metrics, 'mean_qps') else 'N/A'
                print(f"  [{result['combo_idx']}] {build_info}")
                if recall != 'N/A':
                    print(f"      Recall: {recall:.4f}, QPS: {qps:.2f}")
    
    print("=" * 80)
    print("\n测试完成！")


if __name__ == '__main__':
    main()
