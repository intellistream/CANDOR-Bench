"""
Dataset Loaders

Utility functions for loading and processing datasets.
"""

import os
import numpy as np
from typing import Optional, Tuple
from .base import Dataset


def xbin_mmap(filename: str, dtype: str = "float32", maxn: Optional[int] = None) -> np.ndarray:
    """
    Memory-map a binary vector file.
    
    File format: [n, d] (uint32) + vector_data (dtype)
    
    Args:
        filename: Path to file
        dtype: Data type
        maxn: Maximum number of vectors to read
        
    Returns:
        (n, d) array (memory-mapped)
    """
    with open(filename, "rb") as f:
        n, d = np.fromfile(f, dtype=np.uint32, count=2)
        if maxn is not None:
            n = min(n, maxn)
        
        offset = 2 * np.dtype(np.uint32).itemsize
        data = np.memmap(f.name, dtype=dtype, mode='r', offset=offset, shape=(n, d))
    
    return data


def load_fvecs(filename: str, maxn: Optional[int] = None) -> np.ndarray:
    """Load .fvecs format file (float vectors)."""
    vectors = []
    with open(filename, "rb") as f:
        while True:
            d_bytes = f.read(4)
            if not d_bytes:
                break
            d = np.frombuffer(d_bytes, dtype=np.int32)[0]
            vec = np.frombuffer(f.read(d * 4), dtype=np.float32)
            vectors.append(vec)
            if maxn and len(vectors) >= maxn:
                break
    return np.array(vectors)


def load_ivecs(filename: str, maxn: Optional[int] = None) -> np.ndarray:
    """Load .ivecs format file (integer vectors)."""
    vectors = []
    with open(filename, "rb") as f:
        while True:
            d_bytes = f.read(4)
            if not d_bytes:
                break
            d = np.frombuffer(d_bytes, dtype=np.int32)[0]
            vec = np.frombuffer(f.read(d * 4), dtype=np.int32)
            vectors.append(vec)
            if maxn and len(vectors) >= maxn:
                break
    return np.array(vectors)


def knn_result_read(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read KNN groundtruth file.
    
    Supports formats: .ibin (binary), .ivecs
    
    Returns:
        (I, D): I is indices (nq, k), D is distances (nq, k)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Groundtruth file not found: {filename}")
    
    if filename.endswith('.ibin'):
        with open(filename, "rb") as f:
            nq, k = np.fromfile(f, dtype=np.uint32, count=2)
            I = np.fromfile(f, dtype=np.uint32).reshape(nq, k)
            remaining = f.read()
            if len(remaining) > 0:
                D = np.frombuffer(remaining, dtype=np.float32).reshape(nq, k)
            else:
                D = np.zeros_like(I, dtype=np.float32)
    elif filename.endswith('.ivecs'):
        I = load_ivecs(filename)
        D = np.zeros_like(I, dtype=np.float32)
    else:
        # Try ivecs format first, then binary
        try:
            I = load_ivecs(filename)
            D = np.zeros_like(I, dtype=np.float32)
        except:
            with open(filename, "rb") as f:
                nq, k = np.fromfile(f, dtype=np.uint32, count=2)
                I = np.fromfile(f, dtype=np.uint32).reshape(nq, k)
                D = np.zeros_like(I, dtype=np.float32)
    
    return I, D


def sanitize(x: np.ndarray) -> np.ndarray:
    """
    Keep array as-is (including memmap) to support large datasets.
    
    Note: memmap data triggers disk I/O on access, but this is necessary
    for datasets that don't fit in memory.
    """
    return x


def load_dataset(dataset_name: str) -> Dataset:
    """Load dataset by name."""
    from .registry import DATASETS
    
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASETS[dataset_name]()


def prepare_dataset(dataset_name: str, skip_data: bool = False) -> Dataset:
    """Prepare dataset (download and process)."""
    dataset = load_dataset(dataset_name)
    dataset.prepare(skip_data=skip_data)
    return dataset
