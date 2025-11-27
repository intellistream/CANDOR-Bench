"""
Datasets Module for Benchmark ANNS

This module provides dataset management functionality.
"""

from .base import Dataset
from .loaders import (
    load_dataset, 
    prepare_dataset,
    xbin_mmap,
    load_fvecs,
    load_ivecs,
    knn_result_read,
    sanitize,
)
from .registry import (
    DATASETS, 
    register_dataset,
    get_dataset,
    # SIFT 系列
    SIFTSmall,
    SIFT,
    SIFT100M,
    # 图像数据集
    OpenImagesStreaming,
    Sun,
    COCO,
    # 文本/词向量数据集
    Glove,
    Msong,
    MSTuring,
    WTE,
    # 测试数据集
    RandomDataset,
)

__all__ = [
    'Dataset',
    'load_dataset',
    'prepare_dataset',
    'DATASETS',
    'register_dataset',
    'get_dataset',
    # Dataset classes
    'SIFTSmall',
    'SIFT',
    'SIFT100M',
    'OpenImagesStreaming',
    'Sun',
    'COCO',
    'Glove',
    'Msong',
    'MSTuring',
    'WTE',
    'RandomDataset',
    # Loaders
    'xbin_mmap',
    'load_fvecs',
    'load_ivecs',
    'knn_result_read',
    'sanitize',
]
