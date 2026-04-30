#!/usr/bin/env python3
"""
Ground Truth Computation Script for Benchmark ANNS

参考 big-ann-benchmarks/benchmark/congestion/compute_gt.py 的实现流程

This script computes ground truth for streaming/congestion benchmarks using DiskANN's
compute_groundtruth tool. It supports:
- Batch inserts with periodic ground truth snapshots
- Timestamped queries
- Deletion operations (delete, batch_insert_delete)
- Multiple search checkpoints
- Replace operations

Usage:
    python compute_gt.py --dataset sift --runbook_file runbooks/general_experiment.yaml \\
        --gt_cmdline_tool /path/to/algorithms_impl/DiskANN/build/apps/utils/compute_groundtruth

    Or let it auto-detect the tool:
    python compute_gt.py --dataset sift --runbook_file runbooks/general_experiment.yaml
"""

import argparse
import os
import sys
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple

from datasets.registry import DATASETS


def _dtype_str_to_numpy(dtype: str):
    if dtype == 'float32':
        return np.float32
    if dtype == 'int8':
        return np.int8
    if dtype == 'uint8':
        return np.uint8
    raise RuntimeError(f'Invalid datatype: {dtype}')


def ensure_query_file(ds, runbook_path: str, private_query: bool = False) -> str:
    """Return a query file path suitable for DiskANN compute_groundtruth.

    This repo's Dataset classes generally expose `get_queries()` (returns ndarray/mmap),
    but may not expose a `qs_fn` attribute like big-ann-benchmarks.

    Resolution strategy:
    1) If dataset exposes a query filename attribute and the file exists, use it.
    2) Otherwise, materialize `ds.get_queries()` into a binary file with header:
       [nq(uint32 little), dim(uint32 little)] + raw data.
    """
    candidate_attrs = []
    if private_query:
        candidate_attrs += [
            'qs_private_fn',
            'private_qs_fn',
            'private_queries_fn',
        ]
    candidate_attrs += [
        'qs_fn',
        'queries_fn',
        'query_fn',
    ]

    for attr in candidate_attrs:
        rel_or_abs = getattr(ds, attr, None)
        if not rel_or_abs:
            continue
        path = str(rel_or_abs)
        if not os.path.isabs(path):
            path = os.path.join(ds.basedir, path)
        if os.path.exists(path):
            return path

    # Fallback: write queries to a file under the ground-truth directory.
    out_dir = gt_dir(ds, runbook_path)
    os.makedirs(out_dir, exist_ok=True)

    dtype = _dtype_str_to_numpy(ds.dtype)
    suffix = {np.float32: 'fbin', np.int8: 'i8bin', np.uint8: 'u8bin'}[dtype]
    query_file = os.path.join(out_dir, f'queries_{ds.nq}_{ds.d}.{suffix}')

    if os.path.exists(query_file):
        return query_file

    queries = ds.get_queries()
    if queries is None:
        raise RuntimeError(
            f"Dataset {ds.__class__.__name__} returned None from get_queries(); "
            "cannot build query file for ground truth."
        )

    queries = np.ascontiguousarray(queries, dtype=dtype)
    if queries.ndim != 2:
        raise RuntimeError(f"Queries must be 2D (nq, d), got shape {queries.shape}")
    if queries.shape[1] != ds.d:
        raise RuntimeError(
            f"Query dim mismatch: ds.d={ds.d} but queries.shape[1]={queries.shape[1]}"
        )

    with open(query_file, 'wb') as f:
        f.write(int(queries.shape[0]).to_bytes(4, byteorder='little'))
        f.write(int(queries.shape[1]).to_bytes(4, byteorder='little'))
        queries.tofile(f)

    return query_file


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


def find_compute_groundtruth_tool():
    """
    Automatically find the compute_groundtruth binary.
    
    Searches in the following locations (relative to benchmark_anns):
    - ./algorithms_impl/DiskANN/build/apps/utils/compute_groundtruth
    - ../algorithms_impl/DiskANN/build/apps/utils/compute_groundtruth
    
    Returns:
        Path to compute_groundtruth binary, or None if not found
    """
    script_dir = Path(__file__).parent.resolve()
    
    # Possible relative paths to search
    search_paths = [
        script_dir / 'algorithms_impl' / 'DiskANN' / 'build' / 'apps' / 'utils' / 'compute_groundtruth',
        script_dir / 'algorithms_impl' / 'DiskANN' / 'build' / 'tests' / 'utils' / 'compute_groundtruth',
        script_dir.parent / 'algorithms_impl' / 'DiskANN' / 'build' / 'apps' / 'utils' / 'compute_groundtruth',
        script_dir.parent / 'algorithms_impl' / 'DiskANN' / 'build' / 'tests' / 'utils' / 'compute_groundtruth',
    ]
    
    for path in search_paths:
        if path.exists() and os.access(path, os.X_OK):
            return str(path.resolve())
    
    return None


def get_range_start_end(entry: dict, tag_to_id: dict) -> dict:
    """
    Initialize tag_to_id mapping for initial range.
    
    参考 big-ann-benchmarks 实现，用于初始化标签到ID的映射
    
    Args:
        entry: Runbook entry with 'start' and 'end' fields
        tag_to_id: Dictionary mapping tags to IDs
        
    Returns:
        Updated tag_to_id dictionary
    """
    for i in range(entry['end'] - entry['start']):
        tag_to_id[i + entry['start']] = i + entry['start']
    return tag_to_id


def get_next_set(tag_to_id: dict, entry: dict) -> dict:
    """
    Update tag_to_id mapping based on operation type.
    
    参考 big-ann-benchmarks 的实现，支持所有操作类型
    
    Args:
        tag_to_id: Current tag to ID mapping
        entry: Runbook entry describing the operation
        
    Returns:
        Updated tag_to_id dictionary
    """
    operation = entry['operation']
    
    if operation == 'initial':
        for i in range(entry['end'] - entry['start']):
            tag_to_id[i + entry['start']] = i + entry['start']
    
    elif operation == 'insert':
        for i in range(entry['end'] - entry['start']):
            tag_to_id[i + entry['start']] = i + entry['start']
    
    elif operation == 'delete':
        # Delete by key
        for i in range(entry['end'] - entry['start']):
            tag_to_id.pop(i + entry['start'], None)
    
    elif operation == 'batch_insert':
        for i in range(entry['end'] - entry['start']):
            tag_to_id[i + entry['start']] = i + entry['start']
    
    elif operation == 'batch_insert_delete':
        percentage = entry.get('deletion_percentage', 0)
        for i in range(entry['end'] - entry['start']):
            tag_to_id[i + entry['start']] = i + entry['start']
    
    elif operation == 'replace':
        # Replace key with value
        for i in range(entry['tags_end'] - entry['tags_start']):
            tag_to_id[i + entry['tags_start']] = entry['ids_start'] + i
    
    elif operation in ['search', 'startHPC', 'endHPC', 'waitPending', 'enableScenario']:
        # No-op operations for tag_to_id
        pass
    
    else:
        raise ValueError(f'Undefined entry in runbook: {operation}')
    
    return tag_to_id


def gt_dir(ds, runbook_path: str) -> str:
    """
    Get the directory path for storing ground truth files.
    
    参考 big-ann-benchmarks 的实现：使用 runbook 文件名作为子目录
    
    Args:
        ds: Dataset object
        runbook_path: Path to the runbook file
        
    Returns:
        Directory path for ground truth files
    """
    runbook_filename = os.path.split(runbook_path)[1]
    return os.path.join(ds.basedir, str(ds.nb), runbook_filename)


def output_gt(ds, tag_to_id: dict, step: int, gt_cmdline: str, query_file: str, runbook_path: str) -> None:
    """
    Output ground truth for a single search checkpoint.
    
    参考 big-ann-benchmarks 实现：
    1. 从 tag_to_id 提取 ids 和 tags 列表
    2. 根据 ids 从数据集获取对应的数据切片
    3. 写入 tags 文件和 data 文件
    4. 调用 compute_groundtruth 工具计算真值
    5. 删除临时 data 文件节省空间
    
    Args:
        ds: Dataset object
        tag_to_id: Current tag to ID mapping
        step: Step number in the runbook
        gt_cmdline: Base command line for compute_groundtruth tool
        runbook_path: Path to the runbook file
    """
    ids_list = []
    tags_list = []
    
    for tag, id_val in tag_to_id.items():
        ids_list.append(id_val)
        tags_list.append(tag)

    ids = np.array(ids_list, dtype=np.uint32)
    tags = np.array(tags_list, dtype=np.uint32)

    # Get data slice for active points
    data = ds.get_data_in_range(0, ds.nb)
    data_slice = data[np.array(ids)]

    # Create output directory and files
    dir_path = gt_dir(ds, runbook_path)
    prefix = os.path.join(dir_path, 'step') + str(step)
    os.makedirs(dir_path, exist_ok=True)

    tags_file = prefix + '.tags'
    data_file = prefix + '.data'
    gt_file = prefix + '.gt100'

    # Write tags file (format: [npts, dim] + data)
    with open(tags_file, 'wb') as tf:
        one = 1
        tf.write(tags.size.to_bytes(4, byteorder='little'))
        tf.write(one.to_bytes(4, byteorder='little'))
        tags.tofile(tf)
    
    # Write data file (format: [npts, dim] + data)
    with open(data_file, 'wb') as f:
        f.write(ids.size.to_bytes(4, byteorder='little'))  # npts
        f.write(ds.d.to_bytes(4, byteorder='little'))      # dimensions
        data_slice.tofile(f)
    
    # Construct and execute command
    cmdline = f"{gt_cmdline} {data_file} {query_file} 10 {gt_file} {tags_file}"
    
    print(f"Executing cmdline: {cmdline}")
    os.system(cmdline)
    
    # Clean up data file to save space
    print("Removing data file")
    rm_cmdline = "rm " + data_file
    os.system(rm_cmdline)


def output_gt_batch(ds, tag_to_id: dict, num_batch_insert: int, step: int, 
                    gt_cmdline: str, query_file: str, runbook_path: str, batch_size: int, 
                    with_deletion: bool = False) -> None:
    """
    Output ground truth for a batch insert checkpoint.
    
    参考 big-ann-benchmarks 实现：
    - 支持选择性输出（根据 batch_size 和 runbook 类型）
    - 文件命名格式：batch{num_batch_insert}_{step}
    - 支持 deletion 标记
    
    优化存储策略：对于某些实验类型，跳过大批量的 GT 输出以节省存储空间
    
    Args:
        ds: Dataset object
        tag_to_id: Current tag to ID mapping
        num_batch_insert: Batch insert counter
        step: Batch step number
        gt_cmdline: Base command line for compute_groundtruth tool
        runbook_path: Path to the runbook file
        batch_size: Size of each batch
        with_deletion: Whether this batch includes deletions
    """
    # Skip batch GT output for certain conditions to reduce storage
    # 对于特定批量大小和实验类型，跳过 GT 计算以节省存储
    # 通用的关键词检查，不依赖于具体的路径结构
    runbook_lower = runbook_path.lower()
    
    # 这些实验类型需要保留批量 GT
    important_experiments = [
        'simple',  # 简单测试
        'test_experiment', 'test_simple', 'test_congestion',  # 测试实验
        'general_experiment', 'baseline',  # 基础实验
        'deletion', 'bulk_deletion', 'batch_deletion',  # 删除相关实验
        'concept_drift', 'conceptdrift',  # 概念漂移实验
        'dimension',  # 维度相关实验
        'multi_modal', 'multimodal',  # 多模态实验
        'word_contamination', 'wordcontamination', 'contamination'  # 污染实验
    ]
    
    # 如果不包含 deletion 且 batch_size 为 2500，检查是否为重要实验
    if not with_deletion and batch_size == 2500:
        is_important = any(keyword in runbook_lower for keyword in important_experiments)
        if not is_important:
            # 跳过非重要实验的批量 GT 输出
            return

    ids_list = []
    tags_list = []
    
    for tag, id_val in tag_to_id.items():
        ids_list.append(id_val)
        tags_list.append(tag)

    ids = np.array(ids_list, dtype=np.uint32)
    tags = np.array(tags_list, dtype=np.uint32)

    # Get data slice for active points
    data = ds.get_data_in_range(0, ds.nb)
    data_slice = data[np.array(ids)]

    # Create output directory and files
    dir_path = gt_dir(ds, runbook_path)
    prefix = os.path.join(dir_path, 'batch') + str(num_batch_insert) + "_" + str(step)
    os.makedirs(dir_path, exist_ok=True)

    tags_file = prefix + '.tags'
    data_file = prefix + '.data'
    gt_file = prefix + '.gt100'

    # Write tags file (format: [npts, dim] + data)
    with open(tags_file, 'wb') as tf:
        one = 1
        tf.write(tags.size.to_bytes(4, byteorder='little'))
        tf.write(one.to_bytes(4, byteorder='little'))
        tags.tofile(tf)
    
    # Write data file (format: [npts, dim] + data)
    with open(data_file, 'wb') as f:
        f.write(ids.size.to_bytes(4, byteorder='little'))  # npts
        f.write(ds.d.to_bytes(4, byteorder='little'))      # dimensions
        data_slice.tofile(f)
    
    # Construct and execute command
    cmdline = f"{gt_cmdline} {data_file} {query_file} 10 {gt_file} {tags_file}"
    
    print(f"Executing cmdline: {cmdline}")
    os.system(cmdline)
    
    # Clean up data file to save space
    print("Removing data file")
    rm_cmdline = "rm " + data_file
    os.system(rm_cmdline)


def main():
    """
    Main function for ground truth computation.
    
    参考 big-ann-benchmarks 的 compute_gt.py 主流程：
    1. 解析命令行参数
    2. 加载数据集和 runbook
    3. 构建 compute_groundtruth 基础命令
    4. 遍历 runbook 条目，根据操作类型更新 tag_to_id
    5. 在 search 操作时调用 output_gt
    6. 处理 batch_insert 和 batch_insert_delete 操作
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        help='Dataset to benchmark on.',
        required=True
    )
    parser.add_argument(
        '--runbook_file',
        help='Runbook yaml file path',
        required=True
    )
    parser.add_argument(
        '--private_query',
        action='store_true',
        help='Use private query set (if available)'
    )
    parser.add_argument(
        '--gt_cmdline_tool',
        help='Path to DiskANN compute_groundtruth binary'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download dataset if not present'
    )
    args = parser.parse_args()

    # Load dataset
    ds = DATASETS[args.dataset]()

    # Optionally download/prepare dataset
    if args.download:
        ds.prepare()
    
    # Load runbook
    max_pts, runbook = load_runbook(args.dataset, ds.nb, args.runbook_file)
    
    # Get query file (path). Some Dataset implementations don't expose `qs_fn`.
    query_file = ensure_query_file(ds, args.runbook_file, private_query=args.private_query)
    
    gt_tool = args.gt_cmdline_tool or find_compute_groundtruth_tool()
    if not gt_tool:
        raise FileNotFoundError("Could not find DiskANN compute_groundtruth binary")

    # Build base command prefix for this DiskANN variant:
    #   compute_groundtruth <type> <base_file> <query_file> <K> <gt_file> [tag_file]
    common_cmd = gt_tool

    # Map data type
    common_cmd += ' '
    dtype = ds.dtype
    if dtype == 'float32':
        common_cmd += 'float'
    elif dtype == 'int8':
        common_cmd += 'int8'
    elif dtype == 'uint8':
        common_cmd += 'uint8'
    else:
        raise RuntimeError('Invalid datatype')
    
    # Process runbook - 参考 big-ann-benchmarks 的处理流程
    # runbook 是一个字典，key 是步骤号，需要按顺序处理
    step = 1
    ids = np.empty(0, dtype=np.uint32)
    num_batch_insert = 0
    tag_to_id = {}
    
    # 获取所有步骤号并排序
    step_keys = sorted([k for k in runbook.keys() if isinstance(k, int)])
    
    for step_key in step_keys:
        entry = runbook[step_key]
        if not isinstance(entry, dict) or 'operation' not in entry:
            continue
            
        # The first step must be an HPC and second must be initial
        if entry['operation'] == 'initial':
            tag_to_id = get_range_start_end(entry, {})
        elif entry['operation'] not in ['batch_insert', 'batch_insert_delete', 'startHPC', 'endHPC', 'waitPending']:
            tag_to_id = get_next_set(tag_to_id, entry)
        
        # Handle search operation
        if entry['operation'] == 'search':
            output_gt(ds, tag_to_id, step, common_cmd, query_file, args.runbook_file)
        
        # Handle batch_insert operation
        if entry['operation'] == 'batch_insert':
            batch_size = entry['batchSize']
            end = entry['end']
            start = entry['start']
            batch_step = (end - start) // batch_size
            continuous_counter = 0
            
            for i in range(batch_step):
                # Insert batch
                for j in range(start + i * batch_size, start + (i + 1) * batch_size):
                    tag_to_id[j] = j

                continuous_counter += batch_size
                # Output GT every 1% of progress
                if continuous_counter >= (end - start) / 100:
                    print(f"{i}: {start + i * batch_size}~{start + (i + 1) * batch_size} output gt")
                    output_gt_batch(ds, tag_to_id, num_batch_insert, i, common_cmd, query_file,
                                  args.runbook_file, batch_size)
                    continuous_counter = 0

            # Handle remaining points
            if start + batch_step * batch_size < end and start + (batch_step + 1) * batch_size > end:
                for j in range(start + batch_step * batch_size, end):
                    tag_to_id[j] = j

                continuous_counter += batch_size
                if continuous_counter >= (end - start) / 100:
                    print(f"{batch_step}: {start + batch_step * batch_size}~{end} output gt")
                    output_gt_batch(ds, tag_to_id, num_batch_insert, batch_step, common_cmd, query_file,
                                  args.runbook_file, batch_size)
                    continuous_counter = 0

            num_batch_insert += 1
        
        # Handle batch_insert_delete operation
        if entry['operation'] == 'batch_insert_delete':
            batch_size = entry['batchSize']
            end = entry['end']
            start = entry['start']
            percentage = entry['deletion_percentage']
            batch_step = (end - start) // batch_size
            continuous_counter = 0
            
            for i in range(batch_step):
                # Insert batch
                for j in range(start + i * batch_size, start + (i + 1) * batch_size):
                    tag_to_id[j] = j

                # Delete percentage of batch
                for j in range(int(start + (i + 1) * batch_size - batch_size * percentage), 
                             start + (i + 1) * batch_size):
                    tag_to_id.pop(j)

                continuous_counter += batch_size
                # Output GT every 1% of progress
                if continuous_counter >= (end - start) / 100:
                    print(f"{i}: {start + i * batch_size}~{start + (i + 1) * batch_size} output gt")
                    output_gt_batch(ds, tag_to_id, num_batch_insert, i, common_cmd, query_file,
                                  args.runbook_file, batch_size, True)
                    continuous_counter = 0

            # Handle remaining points
            if start + batch_step * batch_size < end and start + (batch_step + 1) * batch_size > end:
                for j in range(start + batch_step * batch_size, end):
                    tag_to_id[j] = j
                
                for j in range(int(start + end - batch_size * percentage), end):
                    tag_to_id.pop(j)

                continuous_counter += batch_size
                if continuous_counter >= (end - start) / 100:
                    print(f"{batch_step}: {start + batch_step * batch_size}~{end} output gt")
                    output_gt_batch(ds, tag_to_id, num_batch_insert, batch_step, common_cmd, query_file,
                                  args.runbook_file, batch_size, True)
                    continuous_counter = 0

            num_batch_insert += 1

        step += 1


if __name__ == '__main__':
    main()
