"""
Base Dataset Class

Provides abstract interface for all dataset implementations.
"""

import numpy as np
from typing import Optional, Tuple, Iterator


class Dataset:
    """Base class defining the interface for all datasets."""
    
    def __init__(self):
        self.nb: int = 0        # Dataset size (number of base vectors)
        self.nq: int = 0        # Number of queries
        self.d: int = 0         # Vector dimension
        self.dtype: str = "float32"
        self.basedir: str = "raw_data/"
        
    def prepare(self, skip_data: bool = False) -> None:
        """Download and prepare the dataset."""
        raise NotImplementedError
    
    def get_dataset_fn(self) -> str:
        """Return the dataset file path."""
        raise NotImplementedError
    
    def get_dataset(self) -> np.ndarray:
        """Return full dataset (only for small datasets)."""
        raise NotImplementedError
    
    def get_dataset_iterator(self, bs: int = 512, split: Tuple[int, int] = (1, 0)) -> Iterator[np.ndarray]:
        """Return dataset iterator yielding batches of size bs."""
        raise NotImplementedError
    
    def get_queries(self) -> np.ndarray:
        """Return query vectors (nq, d)."""
        raise NotImplementedError
    
    def get_groundtruth(self, k: Optional[int] = None) -> np.ndarray:
        """Return groundtruth indices (nq, k)."""
        raise NotImplementedError
    
    def search_type(self) -> str:
        """Return search type: 'knn', 'range', or 'knn_filtered'."""
        return "knn"
    
    def distance(self) -> str:
        """Return distance metric: 'euclidean', 'ip', or 'angular'."""
        return "euclidean"
    
    def data_type(self) -> str:
        """Return data type: 'dense' or 'sparse'."""
        return "dense"
    
    def default_count(self) -> int:
        """Return default k for KNN search."""
        return 10
    
    def get_data_in_range(self, start: int, end: int) -> np.ndarray:
        """Return data slice [start:end]. Subclasses should override for efficiency."""
        dataset = self.get_dataset()
        return dataset[start:end]
    
    def short_name(self) -> str:
        """Return short dataset name."""
        return f"{self.__class__.__name__}-{self.nb}"
    
    def __str__(self) -> str:
        return (
            f"Dataset {self.__class__.__name__} in dimension {self.d}, "
            f"distance={self.distance()}, search_type={self.search_type()}, "
            f"size: Q={self.nq} B={self.nb}"
        )
