#!/usr/bin/env python3
"""
Export Results with Recall Calculation

计算批次级别的召回率并导出最终结果
参考 big-ann-benchmarks/data_export.py 和 benchmark/plotting/utils.py

用法:
    python export_results.py --dataset sift --algorithm faiss_HNSW --runbook general_experiment
    python export_results.py --dataset sift --algorithm faiss_HNSW --runbook general_experiment --output final_results.csv
"""

import argparse
import os
import sys
import h5py
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from datasets.registry import get_dataset, DATASETS


def load_runbook(dataset_name: str, nb: int, runbook_path: str) -> Tuple[int, Dict]:
    """
    加载 runbook 文件
    
    Args:
        dataset_name: 数据集名称
        nb: 数据集大小
        runbook_path: runbook 文件路径
        
    Returns:
        (max_pts, runbook) 元组
    """
    with open(runbook_path, 'r') as f:
        content = yaml.safe_load(f)
    
    if dataset_name not in content:
        raise ValueError(f"Dataset {dataset_name} not found in runbook: {runbook_path}")
    
    runbook = content[dataset_name]
    max_pts = runbook.get('max_pts', nb)
    
    return max_pts, runbook


def knn_result_read(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取真值文件（.gt100 格式）
    
    参考 big-ann-benchmarks/benchmark/dataset_io.py
    
    Returns:
        (ids, distances): 真值的ID和距离数组
    """
    with open(filepath, 'rb') as f:
        # 读取头部：nq(查询数), k(邻居数)
        nq = np.fromfile(f, dtype=np.uint32, count=1)[0]
        k = np.fromfile(f, dtype=np.uint32, count=1)[0]
        
        # 读取数据
        ids = np.fromfile(f, dtype=np.uint32, count=nq * k).reshape(nq, k)
        distances = np.fromfile(f, dtype=np.float32, count=nq * k).reshape(nq, k)
    
    return ids, distances


def compute_recall(true_ids: np.ndarray, true_dists: np.ndarray, 
                   run_ids: np.ndarray, k: int) -> Tuple[float, np.ndarray]:
    """
    计算召回率
    
    参考 big-ann-benchmarks/benchmark/plotting/metrics.py 的实现
    
    Args:
        true_ids: 真值ID数组 (nq, k_gt)
        true_dists: 真值距离数组 (nq, k_gt)
        run_ids: 查询结果ID数组 (nq, k)
        k: 需要的邻居数
        
    Returns:
        (mean_recall, recalls): 平均召回率和每个查询的召回率
    """
    nq = run_ids.shape[0]
    recalls = np.zeros(nq)
    
    for i in range(nq):
        # 简单版本：计算交集大小
        correct = len(set(true_ids[i, :k]) & set(run_ids[i]))
        recalls[i] = correct / k
    
    return np.mean(recalls), recalls


def load_groundtruth_for_batch_inserts(dataset, runbook: Dict, dataset_name: str, 
                                       runbook_path: str) -> List[List[Tuple]]:
    """
    加载所有 batch_insert 操作的真值
    
    参考 big-ann-benchmarks/benchmark/plotting/utils.py 的 compute_metrics_all_runs
    
    Returns:
        List of [List of (ids, distances)] for each batch_insert operation
    """
    # 获取真值目录
    runbook_filename = os.path.basename(runbook_path)
    gt_dir = os.path.join(dataset.basedir, str(dataset.nb), runbook_filename)
    
    if not os.path.exists(gt_dir):
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    
    # 解析 runbook 找到所有 batch_insert 操作
    operations = []
    if dataset_name in runbook:
        dataset_config = runbook[dataset_name]
        # 只处理整数键（步骤编号）
        for key in sorted([k for k in dataset_config.keys() if isinstance(k, int)]):
            operations.append(dataset_config[key])
    else:
        raise ValueError(f"Dataset {dataset_name} not found in runbook")
    
    # 加载每个 batch_insert 的真值
    true_nn_across_batches = []
    num_batch_insert = 0
    
    for step, entry in enumerate(operations):
        if entry['operation'] == 'batch_insert' or entry['operation'] == 'batch_insert_delete':
            batch_gts = []
            start = entry['start']
            end = entry['end']
            batch_size = entry.get('batchSize', 2500)
            
            # 计算批次数量
            num_batches = (end - start + batch_size - 1) // batch_size
            
            # continuous_query_interval: 每插入总数据量的 1/100 执行一次查询
            total_data = end - start
            continuous_query_interval = total_data // 100
            
            continuous_counter = 0
            for batch_idx in range(num_batches):
                continuous_counter += batch_size
                
                # 每插入 total_data/100 就查询一次
                if continuous_counter >= continuous_query_interval:
                    gt_filename = f'batch{num_batch_insert}_{batch_idx}.gt100'
                    gt_path = os.path.join(gt_dir, gt_filename)
                    
                    if os.path.exists(gt_path):
                        true_ids, true_dists = knn_result_read(gt_path)
                        batch_gts.append((true_ids, true_dists))
                        print(f"  加载真值: {gt_filename} (shape: {true_ids.shape})")
                    else:
                        print(f"  ⚠️  真值文件缺失: {gt_filename}")
                    
                    continuous_counter = 0
            
            true_nn_across_batches.append(batch_gts)
            num_batch_insert += 1
    
    return true_nn_across_batches


def compute_batch_recalls(result_hdf5: str, groundtruth_batches: List[List[Tuple]], 
                         k: int = 10) -> Tuple[List[float], List[List[float]]]:
    """
    计算每个批次的召回率
    
    Args:
        result_hdf5: 结果HDF5文件路径
        groundtruth_batches: 真值数据 (每个batch_insert包含多个查询批次)
        k: kNN的k值
        
    Returns:
        (mean_recalls, all_recalls): 每个查询批次的平均召回率和详细召回率
    """
    # 读取查询结果
    with h5py.File(result_hdf5, 'r') as f:
        if 'neighbors_continuous' not in f:
            raise ValueError("HDF5 file does not contain 'neighbors_continuous' dataset")
        
        neighbors_continuous = np.array(f['neighbors_continuous'])
        print(f"查询结果形状: {neighbors_continuous.shape}")
    
    # 展平所有真值（因为真值是按batch_insert分组的）
    all_groundtruth = []
    for batch_gts in groundtruth_batches:
        all_groundtruth.extend(batch_gts)
    
    print(f"真值批次数: {len(all_groundtruth)}")
    
    # 计算每个批次的召回率
    mean_recalls = []
    all_recalls = []
    
    query_idx = 0
    dataset_nq = None  # 每批次的查询数量
    
    for gt_idx, (true_ids, true_dists) in enumerate(all_groundtruth):
        nq = true_ids.shape[0]
        if dataset_nq is None:
            dataset_nq = nq
        
        # 提取对应的查询结果
        if query_idx + nq > neighbors_continuous.shape[0]:
            print(f"  ⚠️  查询结果不足: 需要 {query_idx + nq}, 实际 {neighbors_continuous.shape[0]}")
            break
        
        run_ids = neighbors_continuous[query_idx:query_idx + nq, :k]
        
        # 计算召回率
        mean_recall, recalls = compute_recall(true_ids, true_dists, run_ids, k)
        mean_recalls.append(mean_recall)
        all_recalls.append(recalls.tolist())
        
        query_idx += nq
        
        print(f"  批次 {gt_idx}: 召回率 = {mean_recall:.4f}")
    
    return mean_recalls, all_recalls


def export_results(dataset_name: str, algorithm: str, runbook_name: str,
                   output_dir: str = 'results', output_file: Optional[str] = None,
                   param_folder: Optional[str] = None):
    """
    导出带召回率的最终结果
    
    Args:
        dataset_name: 数据集名称
        algorithm: 算法名称
        runbook_name: Runbook名称
        output_dir: 结果目录
        output_file: 输出CSV文件名（可选）
        param_folder: 参数文件夹名（可选，用于新的参数化目录结构）
    """
    param_info = f" [{param_folder}]" if param_folder else ""
    print(f"\n{'='*80}")
    print(f"导出结果: {algorithm}{param_info} @ {dataset_name} / {runbook_name}")
    print(f"{'='*80}\n")
    
    # 1. 加载数据集和runbook
    print("[1/5] 加载数据集和Runbook...")
    dataset = get_dataset(dataset_name)
    
    runbook_path = f"runbooks/{runbook_name}/{runbook_name}.yaml"
    if not os.path.exists(runbook_path):
        runbook_path = f"runbooks/{runbook_name}.yaml"
    
    if not os.path.exists(runbook_path):
        raise FileNotFoundError(f"Runbook not found: {runbook_path}")
    
    # 加载 runbook（使用正确的签名）
    max_pts, runbook_config = load_runbook(dataset_name, dataset.nb, runbook_path)
    
    # 将 runbook_config 转换为字典格式（用于后续处理）
    with open(runbook_path) as f:
        runbook = yaml.safe_load(f)
    
    print(f"  ✓ 数据集: {dataset_name}")
    print(f"  ✓ Runbook: {runbook_path}")
    
    # 2. 定位结果文件
    print("\n[2/5] 定位结果文件...")
    result_dir = Path(output_dir) / dataset_name / algorithm
    
    # 如果指定了参数文件夹，使用参数化目录结构
    if param_folder:
        result_dir = result_dir / param_folder
        result_base = param_folder
    else:
        result_base = f"{algorithm}_sift_{runbook_name}" if dataset_name == "sift" else f"{algorithm}"
    
    hdf5_file = result_dir / f"{result_base}.hdf5"
    csv_file = result_dir / f"{result_base}.csv"
    batch_insert_qps_file = result_dir / f"{result_base}_batch_insert_qps.csv"
    batch_query_qps_file = result_dir / f"{result_base}_batch_query_qps.csv"
    batch_query_latency_file = result_dir / f"{result_base}_batch_query_latency.csv"
    
    if not hdf5_file.exists():
        # 尝试简化的文件名
        hdf5_file = result_dir / f"{algorithm}.hdf5"
        csv_file = result_dir / f"{algorithm}.csv"
        batch_insert_qps_file = result_dir / f"{algorithm}_batch_insert_qps.csv"
        batch_query_qps_file = result_dir / f"{algorithm}_batch_query_qps.csv"
        batch_query_latency_file = result_dir / f"{algorithm}_batch_query_latency.csv"
    
    if not hdf5_file.exists():
        # 尝试在参数目录中查找任意 hdf5 文件
        if param_folder:
            hdf5_files = list(result_dir.glob("*.hdf5"))
            if hdf5_files:
                hdf5_file = hdf5_files[0]
                base_name = hdf5_file.stem
                csv_file = result_dir / f"{base_name}.csv"
                batch_insert_qps_file = result_dir / f"{base_name}_batch_insert_qps.csv"
                batch_query_qps_file = result_dir / f"{base_name}_batch_query_qps.csv"
                batch_query_latency_file = result_dir / f"{base_name}_batch_query_latency.csv"
    
    if not hdf5_file.exists():
        raise FileNotFoundError(f"Result HDF5 file not found: {hdf5_file}")
    
    print(f"  ✓ HDF5: {hdf5_file}")
    if csv_file.exists():
        print(f"  ✓ CSV: {csv_file}")
    
    # 3. 加载真值
    print("\n[3/5] 加载真值...")
    groundtruth_batches = load_groundtruth_for_batch_inserts(
        dataset, runbook, dataset_name, runbook_path
    )
    print(f"  ✓ 加载了 {len(groundtruth_batches)} 个 batch_insert 操作的真值")
    
    # 4. 计算召回率
    print("\n[4/5] 计算召回率...")
    k = 10  # 默认 k=10
    mean_recalls, all_recalls = compute_batch_recalls(str(hdf5_file), groundtruth_batches, k)
    
    # 5. 导出结果
    print("\n[5/5] 导出结果...")
    
    # 5.1 读取现有的批次数据
    data = {
        'batch_idx': list(range(len(mean_recalls))),
        'recall': mean_recalls
    }
    
    # 5.2 合并插入QPS
    if batch_insert_qps_file.exists():
        insert_qps_df = pd.read_csv(batch_insert_qps_file)
        if len(insert_qps_df) == len(mean_recalls):
            data['insert_qps'] = insert_qps_df['insert_qps'].values
        else:
            print(f"  ⚠️  插入QPS数量不匹配: {len(insert_qps_df)} vs {len(mean_recalls)}")
    
    # 5.3 查询QPS和延迟
    if batch_query_qps_file.exists():
        query_qps_df = pd.read_csv(batch_query_qps_file)
        # 对齐batch_idx
        query_qps_dict = dict(zip(query_qps_df['batch_idx'], query_qps_df['query_qps']))
        data['query_qps'] = [query_qps_dict.get(i, np.nan) for i in data['batch_idx']]
    
    # 5.4 查询延迟
    if batch_query_latency_file.exists():
        query_latency_df = pd.read_csv(batch_query_latency_file)
        # 对齐batch_idx
        query_latency_dict = dict(zip(query_latency_df['batch_idx'], query_latency_df['query_latency_ms']))
        data['query_latency_ms'] = [query_latency_dict.get(i, np.nan) for i in data['batch_idx']]
    
    # 5.5 Cache Miss 统计
    batch_cache_miss_file = result_dir / f"{result_base}_batch_cache_miss.csv"
    if not batch_cache_miss_file.exists():
        batch_cache_miss_file = result_dir / f"{algorithm}_batch_cache_miss.csv"
    if batch_cache_miss_file.exists():
        cache_miss_df = pd.read_csv(batch_cache_miss_file)
        # 对齐batch_idx
        cache_miss_dict = dict(zip(cache_miss_df['batch_idx'], cache_miss_df['cache_misses']))
        cache_refs_dict = dict(zip(cache_miss_df['batch_idx'], cache_miss_df['cache_references']))
        cache_rate_dict = dict(zip(cache_miss_df['batch_idx'], cache_miss_df['cache_miss_rate']))
        
        data['cache_misses'] = [cache_miss_dict.get(i, np.nan) for i in data['batch_idx']]
        data['cache_references'] = [cache_refs_dict.get(i, np.nan) for i in data['batch_idx']]
        data['cache_miss_rate'] = [cache_rate_dict.get(i, np.nan) for i in data['batch_idx']]
    
    # 5.6 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存详细结果到参数目录
    detail_output_file = result_dir / f"{result_base}_final_results.csv"
    df.to_csv(detail_output_file, index=False)
    print(f"  ✓ 详细结果已保存: {detail_output_file}")
    
    # 计算汇总统计
    summary = {
        'params': param_folder if param_folder else 'default',
        'batch_count': len(mean_recalls),
        'mean_recall': np.mean(mean_recalls),
        'min_recall': np.min(mean_recalls),
        'max_recall': np.max(mean_recalls),
    }
    
    if 'insert_qps' in data:
        summary['mean_insert_qps'] = np.nanmean(data['insert_qps'])
    if 'query_qps' in data:
        summary['mean_query_qps'] = np.nanmean(data['query_qps'])
    if 'query_latency_ms' in data:
        summary['mean_query_latency_ms'] = np.nanmean(data['query_latency_ms'])
    if 'cache_misses' in data and not all(pd.isna(data['cache_misses'])):
        summary['mean_cache_misses'] = np.nanmean(data['cache_misses'])
    if 'cache_miss_rate' in data and not all(pd.isna(data['cache_miss_rate'])):
        summary['mean_cache_miss_rate'] = np.nanmean(data['cache_miss_rate'])
    
    # 打印统计信息
    print(f"\n{'─'*60}")
    print(f"统计摘要: {param_folder if param_folder else 'default'}")
    print(f"{'─'*60}")
    print(f"批次数量: {len(mean_recalls)}")
    print(f"平均召回率: {np.mean(mean_recalls):.4f}")
    
    if 'insert_qps' in data:
        print(f"平均插入QPS: {np.nanmean(data['insert_qps']):.2f} ops/s")
    if 'query_qps' in data:
        print(f"平均查询QPS: {np.nanmean(data['query_qps']):.2f} queries/s")
    if 'query_latency_ms' in data:
        print(f"平均查询延迟: {np.nanmean(data['query_latency_ms']):.2f} ms")
    if 'cache_misses' in data and not all(pd.isna(data['cache_misses'])):
        print(f"平均 Cache Misses: {np.nanmean(data['cache_misses']):,.0f}")
    
    print(f"{'─'*60}\n")
    
    return df, summary


def main():
    parser = argparse.ArgumentParser(
        description='Export results with recall calculation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., sift)')
    parser.add_argument('--algorithm', required=True, help='Algorithm name (e.g., faiss_HNSW)')
    parser.add_argument('--runbook', required=True, help='Runbook name (e.g., general_experiment)')
    parser.add_argument('--params', help='Parameter folder name (e.g., ef-200_M-32_prefetch-custom_efsearch-40). Use --list-params to see available options.')
    parser.add_argument('--output-dir', default='results', help='Results directory (default: results)')
    parser.add_argument('--output-file', help='Output CSV file path (optional)')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets')
    parser.add_argument('--list-params', action='store_true', help='List available parameter folders for the specified algorithm')
    parser.add_argument('--all-params', action='store_true', help='Export results for all parameter combinations')
    
    args = parser.parse_args()
    
    if args.list_datasets:
        print("Available datasets:")
        for name in DATASETS.keys():
            print(f"  - {name}")
        return
    
    # 列出可用的参数组合
    if args.list_params:
        result_dir = Path(args.output_dir) / args.dataset / args.algorithm
        if not result_dir.exists():
            print(f"✗ 结果目录不存在: {result_dir}")
            sys.exit(1)
        
        print(f"算法 {args.algorithm} 在数据集 {args.dataset} 上的可用参数组合:")
        
        # 查找所有包含 .hdf5 文件的子目录
        param_dirs = []
        for item in result_dir.iterdir():
            if item.is_dir():
                hdf5_files = list(item.glob("*.hdf5"))
                if hdf5_files:
                    param_dirs.append(item.name)
        
        # 也检查根目录是否有直接的 hdf5 文件（旧格式）
        root_hdf5 = list(result_dir.glob("*.hdf5"))
        if root_hdf5:
            param_dirs.insert(0, "(root)")
        
        if not param_dirs:
            print("  (没有找到结果文件)")
        else:
            for pd_name in sorted(param_dirs):
                print(f"  - {pd_name}")
        return

    try:
        result_dir = Path(args.output_dir) / args.dataset / args.algorithm
        
        # 确定要处理的参数目录列表
        param_dirs_to_process = []
        
        if args.all_params:
            # 处理所有参数组合
            if result_dir.exists():
                for item in result_dir.iterdir():
                    if item.is_dir() and list(item.glob("*.hdf5")):
                        param_dirs_to_process.append(item.name)
                # 也检查根目录
                if list(result_dir.glob("*.hdf5")):
                    param_dirs_to_process.insert(0, None)
        elif args.params:
            # 指定的参数目录
            param_dirs_to_process = [args.params]
        else:
            # 尝试自动检测
            param_dirs_to_process = [None]  # 先尝试根目录（旧格式）
        
        if not param_dirs_to_process:
            print(f"✗ 没有找到结果文件，请使用 --list-params 查看可用的参数组合")
            sys.exit(1)
        
        # 收集所有参数组合的汇总统计
        all_summaries = []
        
        for param_dir in param_dirs_to_process:
            try:
                df, summary = export_results(
                    dataset_name=args.dataset,
                    algorithm=args.algorithm,
                    runbook_name=args.runbook,
                    output_dir=args.output_dir,
                    output_file=None,  # 详细结果由 export_results 内部保存
                    param_folder=param_dir
                )
                if summary is not None:
                    all_summaries.append(summary)
            except FileNotFoundError as e:
                if len(param_dirs_to_process) == 1 and param_dir is None:
                    # 如果只有一个（根目录）且找不到，提示使用 --list-params
                    print(f"\n✗ 错误: {e}")
                    print(f"\n提示: 可能需要指定参数目录，使用以下命令查看可用的参数组合:")
                    print(f"  python export_results.py --dataset {args.dataset} --algorithm {args.algorithm} --runbook {args.runbook} --list-params")
                    sys.exit(1)
                else:
                    print(f"  ⚠ 跳过 {param_dir}: {e}")
        
        # 生成汇总表（每个参数组合一行）
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            
            # 确定输出文件路径
            if args.output_file:
                summary_file = Path(args.output_file)
            else:
                summary_file = result_dir / f"{args.algorithm}_summary.csv"
            
            summary_df.to_csv(summary_file, index=False)
            
            print(f"\n{'='*80}")
            print(f"汇总结果")
            print(f"{'='*80}")
            print(f"✓ 共处理 {len(all_summaries)} 个参数组合")
            print(f"✓ 每个参数目录下已保存详细结果 (*_final_results.csv)")
            print(f"✓ 汇总表已保存: {summary_file}")
            
            # 打印汇总表
            print(f"\n汇总表 (每个参数组合一行):")
            print(summary_df.to_string(index=False))
            print(f"{'='*80}\n")
                    
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
