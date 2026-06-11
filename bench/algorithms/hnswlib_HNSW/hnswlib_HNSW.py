"""Benchmark adapter for upstream hnswlib via a small C API."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

from ..base import BaseStreamingANN


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class _HnswlibLib:
    _lib = None
    _initialized = False

    @classmethod
    def get(cls):
        if cls._lib is not None:
            return cls._lib

        root = _repo_root()
        hnsw_dir = root / "algorithms_impl" / "hnswlib"
        so_path = hnsw_dir / "libhnswlib_capi.so"
        cpp_path = hnsw_dir / "hnswlib_c_api.cpp"
        needs_build = (
            not so_path.exists()
            or (cpp_path.exists() and cpp_path.stat().st_mtime > so_path.stat().st_mtime)
        )
        if needs_build:
            if not cpp_path.exists():
                raise RuntimeError(f"hnswlib C API source not found: {cpp_path}")
            compile_cmd = [
                "g++",
                "-std=c++17",
                "-O3",
                "-fPIC",
                "-shared",
                "-pthread",
                "-D__AVX__",
                "-mavx2",
                "-I",
                str(hnsw_dir),
                str(cpp_path),
                "-o",
                str(so_path),
            ]
            try:
                subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    "Failed to build hnswlib C API shared library:\n"
                    f"cmd: {' '.join(compile_cmd)}\n"
                    f"stdout:\n{exc.stdout}\n"
                    f"stderr:\n{exc.stderr}"
                ) from exc

        lib = ctypes.CDLL(str(so_path))
        if not cls._initialized:
            lib.hnswlib_last_error.restype = ctypes.c_char_p
            lib.hnswlib_create.argtypes = [
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.hnswlib_create.restype = ctypes.c_void_p
            lib.hnswlib_destroy.argtypes = [ctypes.c_void_p]
            lib.hnswlib_destroy.restype = None
            lib.hnswlib_set_ef.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            lib.hnswlib_set_ef.restype = ctypes.c_int
            lib.hnswlib_add_points.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            lib.hnswlib_add_points.restype = ctypes.c_int
            lib.hnswlib_mark_delete.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
            ]
            lib.hnswlib_mark_delete.restype = ctypes.c_int
            lib.hnswlib_search.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.hnswlib_search.restype = ctypes.c_int
            lib.hnswlib_search_warm.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.hnswlib_search_warm.restype = ctypes.c_int
            cls._initialized = True

        cls._lib = lib
        return lib

    @classmethod
    def last_error(cls):
        err = cls.get().hnswlib_last_error()
        if err is None:
            return "Unknown hnswlib C API error"
        return err.decode("utf-8", errors="replace")


class HnswlibHnsw(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.metric = metric
        self.index_params = index_params or {}
        self.name = "hnswlib_HNSW"

        self.M = int(self.index_params.get("M", 16))
        self.efConstruction = int(self.index_params.get("efConstruction", 120))
        self.ef = int(self.index_params.get("ef", 40))
        self.random_seed = int(self.index_params.get("random_seed", 100))
        self.allow_replace_deleted = bool(self.index_params.get("allow_replace_deleted", True))
        self.replace_deleted = bool(self.index_params.get("replace_deleted", False))
        self.num_threads = int(self.index_params.get("num_threads", 8))

        self.streamseed_mode = "off"
        self.hint_level1_only = 0
        self.hint_adaptive_gate_mode = 0
        self.hint_hops = 1
        self.hint_max_candidates = 256
        self.hint_gate = -1.0
        self.hint_qual_gate = -1.0
        self.hint_cons_gate = -1.0
        self.hint_gate_m_quantile = 0.25
        self.hint_gate_o_quantile = 0.30
        self.hint_gate_min_samples = 128
        self.hint_table_slots = 1024
        self.hint_slot_capacity = 2

        self.metric_kind = self._metric_kind(metric)
        self.normalize = str(metric).lower() in ("angular", "cosine")
        self.lib = _HnswlibLib.get()
        self.handle = None
        self.ndim = None
        self.active_ids = set()
        self.res = None
        self.dists = None

    @staticmethod
    def _metric_kind(metric):
        metric = str(metric).lower()
        if metric in ("euclidean", "l2"):
            return 0
        if metric in ("angular", "cosine", "ip", "inner_product"):
            return 1
        raise ValueError(f"hnswlib_HNSW unsupported metric: {metric}")

    @staticmethod
    def _normalize_rows(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return x / norms

    def _prepare_vectors(self, x):
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if self.normalize:
            arr = self._normalize_rows(arr)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def setup(self, dtype, max_pts, ndim):
        if dtype not in ("float32", "float", np.float32):
            raise ValueError(f"hnswlib_HNSW requires float32 input, got {dtype}")
        self.ndim = int(ndim)
        self.handle = self.lib.hnswlib_create(
            ctypes.c_size_t(self.ndim),
            ctypes.c_size_t(int(max_pts)),
            ctypes.c_size_t(self.M),
            ctypes.c_size_t(self.efConstruction),
            ctypes.c_size_t(self.random_seed),
            ctypes.c_int(1 if self.allow_replace_deleted else 0),
            ctypes.c_int(self.metric_kind),
        )
        if not self.handle:
            raise RuntimeError(_HnswlibLib.last_error())
        self.active_ids = set()
        self.set_query_arguments({"ef": self.ef})

    def insert(self, X, ids):
        self._ensure_ready()
        X = self._prepare_vectors(X)
        ids = np.ascontiguousarray(ids, dtype=np.uint64).reshape(-1)
        if ids.shape[0] < X.shape[0]:
            raise ValueError("ids count is smaller than vectors in hnswlib_HNSW.insert")
        if ids.shape[0] > X.shape[0]:
            ids = ids[: X.shape[0]]

        keep = [i for i, idx in enumerate(ids.tolist()) if idx not in self.active_ids]
        if not keep:
            return
        X_new = np.ascontiguousarray(X[keep], dtype=np.float32)
        ids_new = np.ascontiguousarray(ids[keep], dtype=np.uint64)
        rc = self.lib.hnswlib_add_points(
            self.handle,
            X_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ids_new.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_size_t(X_new.shape[0]),
            ctypes.c_size_t(self.ndim),
            ctypes.c_int(1 if self.replace_deleted else 0),
        )
        if rc != 0:
            raise RuntimeError(_HnswlibLib.last_error())
        self.active_ids.update(ids_new.tolist())

    def delete(self, ids):
        self._ensure_ready()
        ids = np.ascontiguousarray(ids, dtype=np.uint64).reshape(-1)
        existing = [idx for idx in ids.tolist() if idx in self.active_ids]
        if not existing:
            return
        ids_arr = np.ascontiguousarray(np.array(existing, dtype=np.uint64))
        rc = self.lib.hnswlib_mark_delete(
            self.handle,
            ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_size_t(ids_arr.shape[0]),
        )
        if rc != 0:
            raise RuntimeError(_HnswlibLib.last_error())
        for idx in existing:
            self.active_ids.discard(idx)

    def query(self, X, k):
        self._ensure_ready()
        X = self._prepare_vectors(X)
        nq = int(X.shape[0])
        topk = int(k)
        out_ids = np.empty((nq, topk), dtype=np.uint64)
        out_dists = np.empty((nq, topk), dtype=np.float32)
        rc = self.lib.hnswlib_search_warm(
            self.handle,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(nq),
            ctypes.c_size_t(self.ndim),
            ctypes.c_size_t(topk),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(self._streamseed_mode_value()),
            ctypes.c_int(int(self.hint_level1_only)),
            ctypes.c_int(int(self.hint_adaptive_gate_mode)),
            ctypes.c_int(int(self.hint_hops)),
            ctypes.c_int(int(self.hint_max_candidates)),
            ctypes.c_float(float(self.hint_gate)),
            ctypes.c_float(float(self.hint_qual_gate)),
            ctypes.c_float(float(self.hint_cons_gate)),
            ctypes.c_float(float(self.hint_gate_m_quantile)),
            ctypes.c_float(float(self.hint_gate_o_quantile)),
            ctypes.c_int(int(self.hint_gate_min_samples)),
            ctypes.c_int(int(self.hint_table_slots)),
            ctypes.c_int(int(self.hint_slot_capacity)),
            out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            out_dists.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(_HnswlibLib.last_error())

        invalid = np.iinfo(np.uint64).max
        invalid_mask = out_ids == invalid
        ids = out_ids.astype(np.int64)
        ids[invalid_mask] = -1
        self.res = ids
        self.dists = out_dists
        return self.res, self.dists

    def _streamseed_mode_value(self):
        if isinstance(self.streamseed_mode, str):
            mode_map = {"off": 0, "streamseed_core": 1}
            return mode_map.get(self.streamseed_mode.lower(), 1)
        return int(self.streamseed_mode)

    def set_query_arguments(self, query_args):
        self.ef = int(query_args.get("ef", self.ef))
        self.streamseed_mode = query_args.get("streamseed_mode", self.streamseed_mode)
        self.hint_level1_only = int(query_args.get("hint_level1_only", self.hint_level1_only))
        self.hint_adaptive_gate_mode = int(query_args.get("hint_adaptive_gate_mode", self.hint_adaptive_gate_mode))
        self.hint_hops = int(query_args.get("hint_hops", self.hint_hops))
        self.hint_max_candidates = int(query_args.get("hint_max_candidates", self.hint_max_candidates))
        self.hint_gate = float(query_args.get("hint_gate", self.hint_gate))
        self.hint_qual_gate = float(query_args.get("hint_qual_gate", self.hint_qual_gate))
        self.hint_cons_gate = float(query_args.get("hint_cons_gate", self.hint_cons_gate))
        self.hint_gate_m_quantile = float(query_args.get("hint_gate_m_quantile", self.hint_gate_m_quantile))
        self.hint_gate_o_quantile = float(query_args.get("hint_gate_o_quantile", self.hint_gate_o_quantile))
        self.hint_gate_min_samples = int(query_args.get("hint_gate_min_samples", self.hint_gate_min_samples))
        self.hint_table_slots = int(query_args.get("hint_table_slots", self.hint_table_slots))
        self.hint_slot_capacity = int(query_args.get("hint_slot_capacity", self.hint_slot_capacity))
        if self.handle:
            rc = self.lib.hnswlib_set_ef(self.handle, ctypes.c_size_t(self.ef))
            if rc != 0:
                raise RuntimeError(_HnswlibLib.last_error())

    def get_results(self):
        return self.res

    def _ensure_ready(self):
        if not self.handle:
            raise RuntimeError("hnswlib_HNSW index is not initialized. Call setup first.")

    def __del__(self):
        try:
            if self.handle:
                self.lib.hnswlib_destroy(self.handle)
                self.handle = None
        except Exception:
            pass


__all__ = ["HnswlibHnsw"]
