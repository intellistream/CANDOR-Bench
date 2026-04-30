"""
DiskANN Algorithm Implementation
使用 PyCANDYAlgo 封装的 diskannpy 接口
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
from ..base import BaseStreamingANN


def _prime_pycandy_environment() -> None:
    root = Path(__file__).resolve().parents[3]
    for path in (root / "algorithms_impl", root / "build-pycandy", root / "build-pycandy" / "src"):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    preload_candidates = [
        Path("/opt/intel/oneapi/mkl/2026.0/lib/libmkl_intel_ilp64.so.3"),
        Path("/opt/intel/oneapi/mkl/2026.0/lib/libmkl_intel_thread.so.3"),
        Path("/opt/intel/oneapi/mkl/2026.0/lib/libmkl_core.so.3"),
        Path("/opt/intel/oneapi/compiler/latest/lib/libiomp5.so"),
    ]
    for lib_path in preload_candidates:
        if lib_path.exists():
            try:
                ctypes.CDLL(str(lib_path), mode=os.RTLD_GLOBAL)
            except OSError:
                pass


_prime_pycandy_environment()

try:
    from PyCANDYAlgo import diskannpy
    DISKANN_AVAILABLE = True
except ImportError:
    DISKANN_AVAILABLE = False


class Diskann(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.name = "diskann"
        self._index_params = index_params
        self._metric = metric
        
        self.R = index_params.get("R", 32)
        self.L = index_params.get("L", 50)
        self.insert_threads = index_params.get("insert_threads", 16)
        self.consolidate_threads = index_params.get("consolidate_threads", 16)
        self.index = None
        self.active_indices = set()
        self.num_unprocessed_deletes = 0
        
    def translate_dist_fn(self, metric):
        if metric == 'euclidean':
            return diskannpy.Metric.L2
        elif metric == 'ip':
            return diskannpy.Metric.L2
        elif metric == 'angular':
            return diskannpy.Metric.COSINE
        else:
            raise Exception('Invalid metric')
        
    def setup(self, dtype, max_pts, ndim):
        if not DISKANN_AVAILABLE:
            raise RuntimeError("PyCANDYAlgo.diskannpy not available")
            
        if dtype == 'uint8':
            index_class = diskannpy.DynamicMemoryUInt8Index
        elif dtype == 'int8':
            index_class = diskannpy.DynamicMemoryInt8Index
        elif dtype == 'float32':
            index_class = diskannpy.DynamicMemoryFloatIndex
        else:
            raise Exception('Invalid dtype for index creation')
            
        self.index = index_class(
            algo_type=diskannpy.AlgoType.DISKANN,
            distance_metric=self.translate_dist_fn(self._metric),
            max_vectors=max_pts,
            dimensions=ndim,
            graph_degree=self.R,
            complexity=self.L,
            num_threads=self.insert_threads,
            initial_search_complexity=100
        )
        self.max_pts = max_pts
        print('DiskANN index constructed and ready')
        
    def insert(self, X, ids):
        self.active_indices.update(ids + 1)
        print(f'#active pts {len(self.active_indices)}, #unprocessed deletes {self.num_unprocessed_deletes}')
        
        if len(self.active_indices) + self.num_unprocessed_deletes >= self.max_pts:
            self.index.consolidate_delete()
            self.num_unprocessed_deletes = 0
        
        retvals = self.index.batch_insert(X, ids + 1, len(ids), self.insert_threads)
        if -1 in retvals:
            print('Insertion failed')
            print('Insertion return values:', retvals)
        
    def delete(self, ids):
        for id in ids:
            self.index.mark_deleted(id + 1)
        self.active_indices.difference_update(ids + 1)
        self.num_unprocessed_deletes += len(ids)
        
    def query(self, X, k):
        nq = X.shape[0]
        self.res, self.query_dists = self.index.batch_search(X, nq, k, self.Ls, self.search_threads)
        self.res = self.res - 1
        return self.res, self.query_dists
        
    def set_query_arguments(self, query_args):
        self.Ls = query_args.get("Ls", 50)
        self.search_threads = query_args.get("T", 8)
        
    def get_results(self):
        return self.res
