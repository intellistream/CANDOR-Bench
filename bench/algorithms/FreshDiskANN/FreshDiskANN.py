"""FreshDiskANN benchmark adapter backed by DiskANN DynamicMemoryIndex."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..base import BaseStreamingANN

_DISKANN_IMPORT_ERROR: Exception | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _import_diskannpy():
    global _DISKANN_IMPORT_ERROR

    try:
        from PyCANDYAlgo import diskannpy as dap  # type: ignore
        return dap
    except Exception as exc:
        _DISKANN_IMPORT_ERROR = exc

    loaded = sys.modules.get("diskannpy")
    if loaded is not None and (
        hasattr(loaded, "DynamicMemoryIndex") or hasattr(loaded, "DynamicMemoryFloatIndex")
    ):
        return loaded

    try:
        import diskannpy as dap  # type: ignore
        return dap
    except ModuleNotFoundError as exc:
        _DISKANN_IMPORT_ERROR = exc
    except ImportError as exc:
        _DISKANN_IMPORT_ERROR = exc
        loaded = sys.modules.get("diskannpy")
        if loaded is not None and (
            hasattr(loaded, "DynamicMemoryIndex") or hasattr(loaded, "DynamicMemoryFloatIndex")
        ):
            return loaded
        return None
    except Exception as exc:
        _DISKANN_IMPORT_ERROR = exc
        return None

    local_src = _repo_root() / "DiskANN" / "python" / "src"
    if local_src.exists() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))
    try:
        import diskannpy as dap  # type: ignore
        return dap
    except Exception as exc:
        _DISKANN_IMPORT_ERROR = exc
        loaded = sys.modules.get("diskannpy")
        if loaded is not None and (
            hasattr(loaded, "DynamicMemoryIndex") or hasattr(loaded, "DynamicMemoryFloatIndex")
        ):
            return loaded
        return None

_diskannpy = None


class FreshDiskANN(BaseStreamingANN):
    """Dynamic in-memory DiskANN adapter for streaming benchmark workloads."""

    def __init__(self, metric: str, index_params: dict[str, Any]):
        self.name = "FreshDiskANN"
        self.metric = metric
        self.index_params = index_params

        self.R = int(index_params.get("R", index_params.get("graph_degree", 32)))
        self.L = int(index_params.get("L", index_params.get("complexity", 100)))
        self.alpha = float(index_params.get("alpha", 1.2))
        self.num_threads = int(index_params.get("num_threads", index_params.get("insert_threads", 16)))
        self.insert_threads = int(index_params.get("insert_threads", self.num_threads))
        self.search_threads = int(index_params.get("search_threads", index_params.get("T", self.num_threads)))
        self.initial_search_complexity = int(
            index_params.get("initial_search_complexity", max(self.L, 100))
        )
        self.filter_complexity = int(index_params.get("filter_complexity", 0))
        self.num_frozen_points = int(index_params.get("num_frozen_points", 1))
        self.max_occlusion_size = int(index_params.get("max_occlusion_size", 750))
        self.saturate_graph = bool(index_params.get("saturate_graph", False))
        self.concurrent_consolidation = bool(index_params.get("concurrent_consolidation", True))
        self.consolidate_every = int(index_params.get("consolidate_every", 0))
        self.verbose = bool(index_params.get("verbose", False))

        self.index = None
        self.max_pts = 0
        self.ndim = 0
        self.active_mask: np.ndarray | None = None
        self.pending_deletes = 0
        self.res = None
        self.query_dists = None
        self.Ls = self.L

    @staticmethod
    def _metric_name(metric: str) -> str:
        metric = str(metric).lower()
        if metric in ("euclidean", "l2"):
            return "l2"
        if metric in ("angular", "cosine"):
            return "cosine"
        if metric in ("ip", "inner_product", "mips"):
            return "mips"
        raise ValueError(f"FreshDiskANN unsupported metric: {metric}")

    @staticmethod
    def _native_metric(dap, metric: str):
        metric = str(metric).lower()
        if metric in ("euclidean", "l2"):
            return dap.Metric.L2
        if metric in ("angular", "cosine"):
            return dap.Metric.COSINE
        if metric in ("ip", "inner_product", "mips"):
            return getattr(dap.Metric, "INNER_PRODUCT", dap.Metric.L2)
        raise ValueError(f"FreshDiskANN unsupported metric: {metric}")

    @staticmethod
    def _vector_dtype(dtype: str):
        dtype = str(dtype).lower()
        if dtype in ("float", "float32", "np.float32"):
            return np.float32
        if dtype in ("uint8", "np.uint8"):
            return np.uint8
        if dtype in ("int8", "np.int8"):
            return np.int8
        raise ValueError(f"FreshDiskANN unsupported dtype: {dtype}")

    def _check_diskannpy(self):
        global _diskannpy
        if _diskannpy is None:
            _diskannpy = _import_diskannpy()
        if _diskannpy is None:
            raise RuntimeError(
                "FreshDiskANN requires either PyCANDYAlgo.diskannpy or diskannpy. "
                "In benchmark runs PyCANDYAlgo is preferred to avoid pybind duplicate registration; "
                "otherwise install with `pip install -e DiskANN` using the same Python executable. "
                f"Original import error: {_DISKANN_IMPORT_ERROR!r}"
            )
        return _diskannpy

    def setup(self, dtype, max_pts, ndim):
        dap = self._check_diskannpy()
        self.max_pts = int(max_pts)
        self.ndim = int(ndim)
        self.active_mask = np.zeros(self.max_pts + 1, dtype=bool)
        vector_dtype = self._vector_dtype(dtype)
        if hasattr(dap, "DynamicMemoryIndex"):
            self.index = dap.DynamicMemoryIndex(
                distance_metric=self._metric_name(self.metric),
                vector_dtype=vector_dtype,
                dimensions=self.ndim,
                max_vectors=self.max_pts + self.num_frozen_points,
                complexity=self.L,
                graph_degree=self.R,
                saturate_graph=self.saturate_graph,
                max_occlusion_size=self.max_occlusion_size,
                alpha=self.alpha,
                num_threads=self.num_threads,
                filter_complexity=self.filter_complexity,
                num_frozen_points=self.num_frozen_points,
                initial_search_complexity=self.initial_search_complexity,
                search_threads=self.search_threads,
                concurrent_consolidation=self.concurrent_consolidation,
            )
        else:
            if vector_dtype is np.float32:
                index_class = dap.DynamicMemoryFloatIndex
            elif vector_dtype is np.uint8:
                index_class = dap.DynamicMemoryUInt8Index
            elif vector_dtype is np.int8:
                index_class = dap.DynamicMemoryInt8Index
            else:
                raise ValueError(f"FreshDiskANN unsupported dtype: {dtype}")
            self.index = index_class(
                algo_type=dap.AlgoType.DISKANN,
                distance_metric=self._native_metric(dap, self.metric),
                max_vectors=self.max_pts + self.num_frozen_points,
                dimensions=self.ndim,
                graph_degree=self.R,
                complexity=self.L,
                num_threads=self.num_threads,
                initial_search_complexity=self.initial_search_complexity,
            )
        self.pending_deletes = 0
        if self.verbose:
            print(
                "FreshDiskANN index constructed "
                f"max_pts={self.max_pts} dim={self.ndim} R={self.R} L={self.L} "
                f"insert_threads={self.insert_threads} search_threads={self.search_threads}"
            )

    def _check_ready(self):
        if self.index is None:
            raise RuntimeError("FreshDiskANN index is not initialized. Call setup first.")

    def insert(self, X, ids):
        self._check_ready()
        if X is None or len(X) == 0:
            return
        vectors = np.ascontiguousarray(X)
        ids_arr = np.asarray(ids, dtype=np.uint32).reshape(-1)
        if vectors.shape[0] != ids_arr.shape[0]:
            n = min(vectors.shape[0], ids_arr.shape[0])
            if n == 0:
                return
            vectors = vectors[:n]
            ids_arr = ids_arr[:n]

        tags = ids_arr + np.uint32(1)
        if self.active_mask is not None and tags.size > 0 and int(tags.max()) < self.active_mask.shape[0]:
            keep = ~self.active_mask[tags]
            if not np.any(keep):
                return
            vectors = np.ascontiguousarray(vectors[keep])
            tags = np.ascontiguousarray(tags[keep], dtype=np.uint32)
        else:
            tags = np.ascontiguousarray(tags, dtype=np.uint32)

        try:
            retvals = self.index.batch_insert(
                vectors=vectors,
                vector_ids=tags,
                num_threads=self.insert_threads,
            )
        except TypeError:
            retvals = self.index.batch_insert(vectors, tags, int(tags.shape[0]), self.insert_threads)
        if retvals is not None:
            retvals = np.asarray(retvals)
            if np.any(retvals == -1):
                failed = int(np.sum(retvals == -1))
                raise RuntimeError(f"FreshDiskANN insertion failed for {failed}/{retvals.size} vectors")
        if self.active_mask is not None and tags.size > 0 and int(tags.max()) < self.active_mask.shape[0]:
            self.active_mask[tags] = True

    def delete(self, ids):
        self._check_ready()
        ids_arr = np.asarray(ids, dtype=np.uint32).reshape(-1)
        if ids_arr.size == 0:
            return
        tags = ids_arr + np.uint32(1)
        for tag in tags:
            tag_int = int(tag)
            if self.active_mask is not None and tag_int < self.active_mask.shape[0] and not self.active_mask[tag_int]:
                continue
            rc = self.index.mark_deleted(np.uint32(tag_int))
            if rc is None or rc == 0:
                self.pending_deletes += 1
                if self.active_mask is not None and tag_int < self.active_mask.shape[0]:
                    self.active_mask[tag_int] = False
        if self.consolidate_every > 0 and self.pending_deletes >= self.consolidate_every:
            self.index.consolidate_delete()
            self.pending_deletes = 0

    def query(self, X, k):
        self._check_ready()
        queries = np.ascontiguousarray(X)
        nq = int(queries.shape[0])
        try:
            ids, dists = self.index.batch_search(
                queries=queries,
                k_neighbors=int(k),
                complexity=int(self.Ls),
                num_threads=int(self.search_threads),
            )
        except TypeError:
            ids, dists = self.index.batch_search(queries, nq, int(k), int(self.Ls), int(self.search_threads))
        ids = np.asarray(ids, dtype=np.int64) - 1
        dists = np.asarray(dists, dtype=np.float32)
        self.res = ids.reshape(nq, int(k))
        self.query_dists = dists.reshape(nq, int(k))
        return self.res, self.query_dists

    def set_query_arguments(self, query_args):
        self.Ls = int(query_args.get("Ls", query_args.get("L", query_args.get("ef", self.L))))
        self.search_threads = int(query_args.get("T", query_args.get("search_threads", self.search_threads)))

    def get_results(self):
        return self.res
