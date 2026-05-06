"""
Dataset Registry

Contains dataset implementations and registration system.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Iterator

from .base import Dataset
from .loaders import xbin_mmap, load_fvecs, load_ivecs, knn_result_read
from .download_utils import download_dataset


class SiftSmallDataset(Dataset):
    """SIFT Small (10K vectors) - for quick testing"""

    def __init__(self):
        super().__init__()
        self.nb = 10000
        self.nq = 100
        self.d = 128
        self.basedir = "raw_data/sift-small/"

    def prepare(self, skip_data: bool = False):
        os.makedirs(self.basedir, exist_ok=True)
        if not skip_data:
            ok = download_dataset("sift-small", self.basedir)
            if not ok and not os.path.exists(self.get_dataset_fn()):
                raise RuntimeError(
                    f"Failed to prepare SIFT-small dataset. "
                    f"Please install gdown (pip install gdown) and retry, "
                    f"or manually place data files under: {self.basedir}"
                )

    def get_dataset_fn(self):
        # Try both possible filenames
        candidates = [
            os.path.join(self.basedir, "data_10000_128"),
            os.path.join(self.basedir, "SIFTsmall", "data_10000_128"),
            os.path.join(self.basedir, "sift_base.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                return fn
        return candidates[0]  # Return first as default

    def get_dataset(self):
        fn = self.get_dataset_fn()
        if fn.endswith(".fvecs"):
            return load_fvecs(fn, maxn=self.nb)
        else:
            return xbin_mmap(fn, dtype=self.dtype, maxn=self.nb)

    def get_dataset_iterator(self, bs: int = 512, split: Tuple[int, int] = (1, 0)):
        data = self.get_dataset()
        for i in range(0, len(data), bs):
            yield data[i : i + bs]

    def get_queries(self):
        candidates = [
            os.path.join(self.basedir, "queries_100_128"),
            os.path.join(self.basedir, "SIFTsmall", "queries_100_128"),
            os.path.join(self.basedir, "sift_query.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                if fn.endswith(".fvecs"):
                    return load_fvecs(fn, maxn=self.nq)
                else:
                    return xbin_mmap(fn, dtype=self.dtype, maxn=self.nq)
        # Return None if not found
        return None

    def get_groundtruth(self, k: Optional[int] = None):
        # Groundtruth files are generated dynamically per runbook
        return None

    def distance(self):
        return "euclidean"


class SiftDataset(Dataset):
    """SIFT 1M - standard benchmark dataset"""

    def __init__(self):
        super().__init__()
        self.nb = 1000000
        self.nq = 10000
        self.d = 128
        self.basedir = "raw_data/sift/"

    def prepare(self, skip_data: bool = False):
        os.makedirs(self.basedir, exist_ok=True)
        if skip_data:
            return
        if os.path.exists(self.get_dataset_fn()):
            return  # Already downloaded
        ok = download_dataset("sift", self.basedir)
        if not ok and not os.path.exists(self.get_dataset_fn()):
            raise RuntimeError(
                f"Failed to prepare SIFT dataset. "
                f"Please install gdown (pip install gdown) and retry, "
                f"or manually place data files under: {self.basedir}"
            )

    def get_dataset_fn(self):
        candidates = [
            os.path.join(self.basedir, "data_1000000_128"),
            os.path.join(self.basedir, "SIFT", "data_1000000_128"),
            os.path.join(self.basedir, "sift_base.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                return fn
        return candidates[0]

    def get_dataset(self):
        fn = self.get_dataset_fn()
        if fn.endswith(".fvecs"):
            return load_fvecs(fn, maxn=self.nb)
        return xbin_mmap(fn, dtype=self.dtype)

    def get_dataset_iterator(self, bs: int = 512, split: Tuple[int, int] = (1, 0)):
        data = self.get_dataset()
        for i in range(0, len(data), bs):
            yield data[i : i + bs]

    def get_queries(self):
        candidates = [
            os.path.join(self.basedir, "queries_10000_128"),
            os.path.join(self.basedir, "SIFT", "queries_10000_128"),
            os.path.join(self.basedir, "sift_query.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                if fn.endswith(".fvecs"):
                    return load_fvecs(fn, maxn=self.nq)
                return xbin_mmap(fn, dtype=self.dtype, maxn=self.nq)
        raise FileNotFoundError(f"SIFT queries not found under {self.basedir}")

    def get_groundtruth(self, k: Optional[int] = None):
        # Groundtruth files are generated dynamically per runbook
        # Not loaded here - handled by benchmark runner
        return None

    def distance(self):
        return "euclidean"


class OpenImagesDataset(Dataset):
    """Open Images dataset"""

    def __init__(self):
        super().__init__()
        self.nb = 9000000
        self.nq = 10000
        self.d = 512
        self.basedir = "raw_data/openimages/"

    def prepare(self, skip_data: bool = False):
        os.makedirs(self.basedir, exist_ok=True)
        if not skip_data:
            ok = download_dataset("openimages", self.basedir)
            if not ok and not os.path.exists(self.get_dataset_fn()):
                raise RuntimeError(
                    f"Failed to prepare openimages dataset. "
                    f"Please install gdown (pip install gdown) and retry, "
                    f"or manually place data files under: {self.basedir}"
                )

    def get_dataset_fn(self):
        candidates = [
            os.path.join(self.basedir, f"data_{self.nb}_{self.d}"),
            os.path.join(self.basedir, "base.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                return fn
        return candidates[0]

    def get_dataset(self):
        fn = self.get_dataset_fn()
        if fn.endswith(".fvecs"):
            return load_fvecs(fn)
        return xbin_mmap(fn, dtype=self.dtype)

    def get_dataset_iterator(self, bs: int = 512, split: Tuple[int, int] = (1, 0)):
        data = self.get_dataset()
        for i in range(0, len(data), bs):
            yield data[i : i + bs]

    def get_queries(self):
        candidates = [
            os.path.join(self.basedir, f"queries_{self.nq}_{self.d}"),
            os.path.join(self.basedir, "query.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                if fn.endswith(".fvecs"):
                    return load_fvecs(fn)
                return xbin_mmap(fn, dtype=self.dtype, maxn=self.nq)
        return None

    def get_groundtruth(self, k: Optional[int] = None):
        return None

    def distance(self):
        return "euclidean"


class SunDataset(Dataset):
    """SUN397 scene recognition dataset"""

    def __init__(self):
        super().__init__()
        self.nb = 79106
        self.nq = 10000
        self.d = 512
        self.basedir = "raw_data/sun/"

    def prepare(self, skip_data: bool = False):
        os.makedirs(self.basedir, exist_ok=True)
        if not skip_data:
            ok = download_dataset("sun", self.basedir)
            if not ok and not os.path.exists(self.get_dataset_fn()):
                raise RuntimeError(
                    f"Failed to prepare sun dataset. "
                    f"Please install gdown (pip install gdown) and retry, "
                    f"or manually place data files under: {self.basedir}"
                )

    def get_dataset_fn(self):
        candidates = [
            os.path.join(self.basedir, f"data_{self.nb}_{self.d}"),
            os.path.join(self.basedir, "base.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                return fn
        return candidates[0]

    def get_dataset(self):
        fn = self.get_dataset_fn()
        if fn.endswith(".fvecs"):
            return load_fvecs(fn)
        return xbin_mmap(fn, dtype=self.dtype)

    def get_dataset_iterator(self, bs: int = 512, split: Tuple[int, int] = (1, 0)):
        data = self.get_dataset()
        for i in range(0, len(data), bs):
            yield data[i : i + bs]

    def get_queries(self):
        candidates = [
            os.path.join(self.basedir, f"queries_{self.nq}_{self.d}"),
            os.path.join(self.basedir, "query.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                if fn.endswith(".fvecs"):
                    return load_fvecs(fn)
                return xbin_mmap(fn, dtype=self.dtype, maxn=self.nq)
        return None

    def get_groundtruth(self, k: Optional[int] = None):
        return None

    def distance(self):
        return "euclidean"


class CocoDataset(Dataset):
    """COCO image caption dataset"""

    def __init__(self):
        super().__init__()
        self.nb = 117266
        self.nq = 5000
        self.d = 768
        self.basedir = "raw_data/coco/"

    def prepare(self, skip_data: bool = False):
        os.makedirs(self.basedir, exist_ok=True)
        if not skip_data:
            ok = download_dataset("coco", self.basedir)
            if not ok and not os.path.exists(self.get_dataset_fn()):
                raise RuntimeError(
                    f"Failed to prepare coco dataset. "
                    f"Please install gdown (pip install gdown) and retry, "
                    f"or manually place data files under: {self.basedir}"
                )

    def get_dataset_fn(self):
        candidates = [
            os.path.join(self.basedir, f"data_{self.nb}_{self.d}"),
            os.path.join(self.basedir, "base.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                return fn
        return candidates[0]

    def get_dataset(self):
        fn = self.get_dataset_fn()
        if fn.endswith(".fvecs"):
            return load_fvecs(fn)
        return xbin_mmap(fn, dtype=self.dtype)

    def get_dataset_iterator(self, bs: int = 512, split: Tuple[int, int] = (1, 0)):
        data = self.get_dataset()
        for i in range(0, len(data), bs):
            yield data[i : i + bs]

    def get_queries(self):
        candidates = [
            os.path.join(self.basedir, f"queries_{self.nq}_{self.d}"),
            os.path.join(self.basedir, "query.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                if fn.endswith(".fvecs"):
                    return load_fvecs(fn)
                return xbin_mmap(fn, dtype=self.dtype, maxn=self.nq)
        return None

    def get_groundtruth(self, k: Optional[int] = None):
        return None

    def distance(self):
        return "euclidean"


class GloveDataset(Dataset):
    """GloVe word embeddings (100d)"""

    def __init__(self):
        super().__init__()
        self.nb = 1183514
        self.nq = 10000
        self.d = 100
        self.basedir = "raw_data/glove/"

    def prepare(self, skip_data: bool = False):
        os.makedirs(self.basedir, exist_ok=True)
        if not skip_data:
            ok = download_dataset("glove", self.basedir)
            if not ok and not os.path.exists(self.get_dataset_fn()):
                raise RuntimeError(
                    f"Failed to prepare glove dataset. "
                    f"Please install gdown (pip install gdown) and retry, "
                    f"or manually place data files under: {self.basedir}"
                )

    def get_dataset_fn(self):
        candidates = [
            os.path.join(self.basedir, f"data_{self.nb}_{self.d}"),
            os.path.join(self.basedir, "base.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                return fn
        return candidates[0]

    def get_dataset(self):
        fn = self.get_dataset_fn()
        if fn.endswith(".fvecs"):
            return load_fvecs(fn)
        return xbin_mmap(fn, dtype=self.dtype)

    def get_dataset_iterator(self, bs: int = 512, split: Tuple[int, int] = (1, 0)):
        data = self.get_dataset()
        for i in range(0, len(data), bs):
            yield data[i : i + bs]

    def get_queries(self):
        candidates = [
            os.path.join(self.basedir, f"queries_{self.nq}_{self.d}"),
            os.path.join(self.basedir, "query.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                if fn.endswith(".fvecs"):
                    return load_fvecs(fn)
                return xbin_mmap(fn, dtype=self.dtype, maxn=self.nq)
        return None

    def get_groundtruth(self, k: Optional[int] = None):
        return None

    def distance(self):
        return "angular"


class MSongDataset(Dataset):
    """Million Song dataset"""

    def __init__(self):
        super().__init__()
        self.nb = 992272
        self.nq = 200
        self.d = 420
        self.basedir = "raw_data/msong/"

    def prepare(self, skip_data: bool = False):
        os.makedirs(self.basedir, exist_ok=True)
        if not skip_data:
            ok = download_dataset("msong", self.basedir)
            if not ok and not os.path.exists(self.get_dataset_fn()):
                raise RuntimeError(
                    f"Failed to prepare msong dataset. "
                    f"Please install gdown (pip install gdown) and retry, "
                    f"or manually place data files under: {self.basedir}"
                )

    def get_dataset_fn(self):
        candidates = [
            os.path.join(self.basedir, f"data_{self.nb}_{self.d}"),
            os.path.join(self.basedir, "base.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                return fn
        return candidates[0]

    def get_dataset(self):
        fn = self.get_dataset_fn()
        if fn.endswith(".fvecs"):
            return load_fvecs(fn)
        return xbin_mmap(fn, dtype=self.dtype)

    def get_dataset_iterator(self, bs: int = 512, split: Tuple[int, int] = (1, 0)):
        data = self.get_dataset()
        for i in range(0, len(data), bs):
            yield data[i : i + bs]

    def get_queries(self):
        candidates = [
            os.path.join(self.basedir, f"queries_{self.nq}_{self.d}"),
            os.path.join(self.basedir, "query.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                if fn.endswith(".fvecs"):
                    return load_fvecs(fn)
                return xbin_mmap(fn, dtype=self.dtype, maxn=self.nq)
        return None

    def get_groundtruth(self, k: Optional[int] = None):
        return None

    def distance(self):
        return "euclidean"


class RandomDataset(Dataset):
    """Random synthetic dataset for testing"""

    def __init__(self, nb: int = 10000, nq: int = 100, d: int = 128):
        super().__init__()
        self.nb = nb
        self.nq = nq
        self.d = d
        self.basedir = f"raw_data/random-{nb//1000}k/"
        self._data = None
        self._queries = None
        self._groundtruth = {}

    def prepare(self, skip_data: bool = False):
        os.makedirs(self.basedir, exist_ok=True)
        # Generate random data
        np.random.seed(42)
        self._data = np.random.randn(self.nb, self.d).astype(np.float32)
        self._queries = np.random.randn(self.nq, self.d).astype(np.float32)

    def get_dataset_fn(self):
        return os.path.join(self.basedir, "base.npy")

    def get_dataset(self):
        if self._data is None:
            self.prepare()
        return self._data

    def get_dataset_iterator(self, bs: int = 512, split: Tuple[int, int] = (1, 0)):
        data = self.get_dataset()
        for i in range(0, len(data), bs):
            yield data[i : i + bs]

    def get_queries(self):
        if self._queries is None:
            self.prepare()
        return self._queries

    def get_groundtruth(self, k: Optional[int] = None):
        k = k or self.default_count()
        k = min(int(k), self.nb)
        if k in self._groundtruth:
            return self._groundtruth[k]

        data = self.get_dataset()
        queries = self.get_queries()
        result = np.empty((self.nq, k), dtype=np.int32)
        for start in range(0, self.nq, 32):
            query_batch = queries[start : start + 32]
            distances = np.sum((query_batch[:, None, :] - data[None, :, :]) ** 2, axis=2)
            result[start : start + len(query_batch)] = np.argsort(distances, axis=1)[:, :k]
        self._groundtruth[k] = result
        return result

    def distance(self):
        return "euclidean"


class WTEDataset(Dataset):
    """Word-to-Embed concept drift dataset

    WTE datasets simulate concept drift scenarios with different drift rates.
    Drift rate indicates the proportion of data that undergoes distribution shift.
    """

    def __init__(self, drift_rate: float = 0.2):
        super().__init__()
        self.drift_rate = drift_rate
        self.nb = 1000000  # Adjust based on actual dataset size
        self.nq = 10000
        self.d = 100  # Adjust based on actual dimension
        self.basedir = f"raw_data/wte-{drift_rate}/"

    def prepare(self, skip_data: bool = False):
        os.makedirs(self.basedir, exist_ok=True)
        if not skip_data:
            dataset_name = f"wte-{self.drift_rate}"
            ok = download_dataset(dataset_name, self.basedir)
            if not ok and not os.path.exists(self.get_dataset_fn()):
                raise RuntimeError(
                    f"Failed to prepare {dataset_name} dataset. "
                    f"Please install gdown (pip install gdown) and retry, "
                    f"or manually place data files under: {self.basedir}"
                )

    def get_dataset_fn(self):
        candidates = [
            os.path.join(self.basedir, f"data_{self.nb}_{self.d}"),
            os.path.join(self.basedir, "base.fvecs"),
            os.path.join(self.basedir, "data.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                return fn
        return candidates[0]

    def get_dataset(self):
        fn = self.get_dataset_fn()
        if fn.endswith(".fvecs"):
            return load_fvecs(fn)
        return xbin_mmap(fn, dtype=self.dtype)

    def get_dataset_iterator(self, bs: int = 512, split: Tuple[int, int] = (1, 0)):
        data = self.get_dataset()
        for i in range(0, len(data), bs):
            yield data[i : i + bs]

    def get_queries(self):
        candidates = [
            os.path.join(self.basedir, f"queries_{self.nq}_{self.d}"),
            os.path.join(self.basedir, "query.fvecs"),
            os.path.join(self.basedir, "queries.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                if fn.endswith(".fvecs"):
                    return load_fvecs(fn)
                return xbin_mmap(fn, dtype=self.dtype, maxn=self.nq)
        return None

    def get_groundtruth(self, k: Optional[int] = None):
        return None

    def distance(self):
        return "euclidean"


class COCODriftDataset(Dataset):
    """COCO concept drift dataset

    COCO drift datasets simulate concept drift scenarios with different drift rates.
    Based on COCO image caption embeddings with controlled distribution shift.
    """

    def __init__(self, drift_rate: float = 0.2):
        super().__init__()
        self.drift_rate = drift_rate
        self.nb = 117266  # Same as base COCO dataset
        self.nq = 5000
        self.d = 768  # Same as base COCO dataset
        self.basedir = f"raw_data/coco-{drift_rate}/"

    def prepare(self, skip_data: bool = False):
        os.makedirs(self.basedir, exist_ok=True)
        if not skip_data:
            dataset_name = f"coco-{self.drift_rate}"
            ok = download_dataset(dataset_name, self.basedir)
            if not ok and not os.path.exists(self.get_dataset_fn()):
                raise RuntimeError(
                    f"Failed to prepare {dataset_name} dataset. "
                    f"Please install gdown (pip install gdown) and retry, "
                    f"or manually place data files under: {self.basedir}"
                )

    def get_dataset_fn(self):
        candidates = [
            os.path.join(self.basedir, f"data_{self.nb}_{self.d}"),
            os.path.join(self.basedir, "base.fvecs"),
            os.path.join(self.basedir, "data.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                return fn
        return candidates[0]

    def get_dataset(self):
        fn = self.get_dataset_fn()
        if fn.endswith(".fvecs"):
            return load_fvecs(fn)
        return xbin_mmap(fn, dtype=self.dtype)

    def get_dataset_iterator(self, bs: int = 512, split: Tuple[int, int] = (1, 0)):
        data = self.get_dataset()
        for i in range(0, len(data), bs):
            yield data[i : i + bs]

    def get_queries(self):
        candidates = [
            os.path.join(self.basedir, f"queries_{self.nq}_{self.d}"),
            os.path.join(self.basedir, "query.fvecs"),
            os.path.join(self.basedir, "queries.fvecs"),
        ]
        for fn in candidates:
            if os.path.exists(fn):
                if fn.endswith(".fvecs"):
                    return load_fvecs(fn)
                return xbin_mmap(fn, dtype=self.dtype, maxn=self.nq)
        return None

    def get_groundtruth(self, k: Optional[int] = None):
        return None

    def distance(self):
        return "euclidean"


# Dataset registry
DATASETS = {
    # SIFT series
    "sift-small": SiftSmallDataset,
    "sift": SiftDataset,
    # Image datasets
    "openimages": OpenImagesDataset,
    "sun": SunDataset,
    "coco": CocoDataset,
    # Text/embedding datasets
    "glove": GloveDataset,
    "msong": MSongDataset,
    # Synthetic datasets
    "random-xs": lambda: RandomDataset(nb=1000, nq=10, d=128),
    "random-s": lambda: RandomDataset(nb=10000, nq=100, d=128),
    "random-m": lambda: RandomDataset(nb=100000, nq=1000, d=128),
    # Concept drift datasets (WTE - Word-to-Embed)
    "wte-0.05": lambda: WTEDataset(drift_rate=0.05),
    "wte-0.2": lambda: WTEDataset(drift_rate=0.2),
    "wte-0.4": lambda: WTEDataset(drift_rate=0.4),
    "wte-0.6": lambda: WTEDataset(drift_rate=0.6),
    "wte-0.8": lambda: WTEDataset(drift_rate=0.8),
    # Concept drift datasets (COCO)
    "coco-0.05": lambda: COCODriftDataset(drift_rate=0.05),
    "coco-0.2": lambda: COCODriftDataset(drift_rate=0.2),
    "coco-0.4": lambda: COCODriftDataset(drift_rate=0.4),
    "coco-0.6": lambda: COCODriftDataset(drift_rate=0.6),
    "coco-0.8": lambda: COCODriftDataset(drift_rate=0.8),
}


def register_dataset(name: str, dataset_class):
    """Register a new dataset."""
    DATASETS[name] = dataset_class


def get_dataset(name: str) -> Dataset:
    """Get dataset instance by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    dataset_factory = DATASETS[name]
    if callable(dataset_factory):
        return dataset_factory()
    return dataset_factory
