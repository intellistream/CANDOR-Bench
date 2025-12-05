"""
Datasets Module

Provides unified interface for dataset loading and management.
"""

from .base import Dataset
from .loaders import (
    xbin_mmap,
    load_fvecs,
    load_ivecs,
    knn_result_read,
    sanitize,
    load_dataset,
    prepare_dataset,
)
from .registry import DATASETS, register_dataset, get_dataset
from .download_utils import download_dataset, DATASET_URLS

__all__ = [
    # Base class
    "Dataset",
    # Loaders
    "xbin_mmap",
    "load_fvecs",
    "load_ivecs",
    "knn_result_read",
    "sanitize",
    "load_dataset",
    "prepare_dataset",
    # Registry
    "DATASETS",
    "register_dataset",
    "get_dataset",
    # Download
    "download_dataset",
    "DATASET_URLS",
]
