"""
Main entry point for streaming benchmarks
"""

import argparse
import yaml
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmark_anns.datasets import load_dataset
from benchmark_anns.bench import BenchmarkRunner, MaintenancePolicy, get_algorithm


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark ANNS - Streaming Index Benchmark with Runbook'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to runbook YAML file'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None, 
        help='Output directory (overrides config)'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # 准备数据集
    dataset_config = config.get('dataset', {})
    dataset_name = dataset_config.get('name', 'random-xs')
    print(f"\nPreparing dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    print(f"Dataset: {dataset}")
    
    # 获取算法
    algo_config = config.get('algorithm', {})
    algo_name = algo_config.get('name', 'dummy')
    algo_params = algo_config.get('parameters', {})
    print(f"\nInitializing algorithm: {algo_name}")
    algorithm = get_algorithm(algo_name, **algo_params)
    print(f"Algorithm: {algorithm}")
    
    # 测试参数
    test_config = config.get('test', {})
    k = test_config.get('k', 10)
    num_workers = test_config.get('num_workers', 1)
    
    # 维护策略
    maintenance_config = config.get('maintenance', {})
    maintenance_thresholds = maintenance_config.get('thresholds', {})
    maintenance_policy = MaintenancePolicy(maintenance_thresholds)
    
    # 创建 runner
    print(f"\nCreating benchmark runner...")
    runner = BenchmarkRunner(
        algorithm=algorithm,
        dataset=dataset,
        k=k,
        num_workers=num_workers,
        maintenance_policy=maintenance_policy
    )
    
    # 获取 runbook
    runbook = config.get('runbook', [])
    if not runbook:
        print("Error: No runbook defined in config")
        return 1
    
    print(f"\nRunbook contains {len(runbook)} operations:")
    for i, entry in enumerate(runbook):
        op = entry.get('operation', 'unknown')
        print(f"  {i+1}. {op}")
    
    # 运行 runbook
    print(f"\n{'='*60}")
    print("Starting benchmark execution...")
    print(f"{'='*60}")
    
    try:
        metrics = runner.run_runbook(runbook)
    except Exception as e:
        print(f"\nError during benchmark execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 保存结果
    output_config = config.get('output', {})
    output_dir = args.output or output_config.get('output_dir', 'results')
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存时间戳
    if output_config.get('save_timestamps', True):
        timestamp_file = os.path.join(output_dir, 'timestamps.csv')
        runner.save_timestamps(timestamp_file)
    
    # 保存指标
    if output_config.get('save_results', True):
        metrics_file = os.path.join(output_dir, 'metrics.json')
        runner.save_metrics(metrics_file)
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("Benchmark Summary")
    print(f"{'='*60}")
    print(f"Algorithm: {metrics.algorithm_name}")
    print(f"Dataset: {metrics.dataset_name}")
    print(f"Total time: {metrics.total_time/1e6:.2f}s")
    print(f"Initial loads: {runner.counts['initial_load']}")
    print(f"Batch inserts: {runner.counts['batch_insert']}")
    print(f"Searches: {runner.counts['search']}")
    print(f"Maintenance rebuilds: {runner.counts['maintenance_rebuild']}")
    
    if metrics.insert_throughput:
        avg_throughput = sum(metrics.insert_throughput) / len(metrics.insert_throughput)
        print(f"Avg insert throughput: {avg_throughput:.2f} ops/s")
    
    if metrics.latency_query:
        avg_query_latency = sum(metrics.latency_query) / len(metrics.latency_query)
        print(f"Avg query latency: {avg_query_latency/1e3:.2f}ms")
    
    print(f"Update memory peak: {metrics.update_memory_footprint/1024/1024:.2f}MB")
    print(f"Search memory peak: {metrics.search_memory_footprint/1024/1024:.2f}MB")
    print(f"Maintenance budget used: {metrics.maintenance_budget_used/1e6:.2f}s")
    print(f"Final deletion ratio: {metrics.maintenance_deletion_ratio_final:.3f}")
    print(f"{'='*60}\n")
    
    print(f"Results saved to {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
