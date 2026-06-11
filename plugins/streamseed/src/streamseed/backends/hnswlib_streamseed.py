from __future__ import annotations

import ctypes
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .base import StreamSeedBackend, StreamSeedConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


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
            lib.hnswlib_add_points_parallel.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.hnswlib_add_points_parallel.restype = ctypes.c_int
            lib.hnswlib_mark_delete.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
            ]
            lib.hnswlib_mark_delete.restype = ctypes.c_int
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


class HnswlibStreamSeedBackend(StreamSeedBackend):
    """hnswlib backend with StreamSeed hint-table warm search."""

    def __init__(
        self,
        metric: str = "euclidean",
        M: int = 16,
        ef_construction: int = 120,
        random_seed: int = 100,
        allow_replace_deleted: bool = True,
        replace_deleted: bool = False,
        num_threads: int = 8,
    ):
        self.metric = metric
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.random_seed = int(random_seed)
        self.allow_replace_deleted = bool(allow_replace_deleted)
        self.replace_deleted = bool(replace_deleted)
        self.num_threads = int(num_threads)
        self.verbose = False
        self.metric_kind = self._metric_kind(metric)
        self.normalize = str(metric).lower() in ("angular", "cosine")
        self.config = StreamSeedConfig()
        self.lib = _HnswlibLib.get()
        self.handle = None
        self.ndim = 0
        self.max_pts = 0
        self.active_ids = set()
        self._active_mask = None
        self._last_result = None
        self._last_dists = None

    @staticmethod
    def _metric_kind(metric):
        metric = str(metric).lower()
        if metric in ("euclidean", "l2"):
            return 0
        if metric in ("angular", "cosine", "ip", "inner_product"):
            return 1
        raise ValueError(f"HnswlibStreamSeedBackend unsupported metric: {metric}")

    @staticmethod
    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return x / norms

    def _prepare_vectors(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.ndim:
            raise ValueError(f"Vector dim mismatch: expect {self.ndim}, got {arr.shape[1]}")
        if self.normalize:
            arr = self._normalize_rows(arr)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _check_ready(self) -> None:
        if not self.handle:
            raise RuntimeError("hnswlib backend not initialized. Call setup() first.")

    def setup(self, max_pts: int, ndim: int, verbose: bool = False) -> None:
        self.verbose = bool(verbose)
        self.max_pts = int(max_pts)
        self.ndim = int(ndim)
        self.handle = self.lib.hnswlib_create(
            ctypes.c_size_t(self.ndim),
            ctypes.c_size_t(self.max_pts),
            ctypes.c_size_t(self.M),
            ctypes.c_size_t(self.ef_construction),
            ctypes.c_size_t(self.random_seed),
            ctypes.c_int(1 if self.allow_replace_deleted else 0),
            ctypes.c_int(self.metric_kind),
        )
        if not self.handle:
            raise RuntimeError(_HnswlibLib.last_error())
        self.active_ids = set()
        self._active_mask = np.zeros(self.max_pts, dtype=bool)
        self.configure(self.config)
        if self.verbose:
            print(
                f"[hnswlib_streamseed] setup max_pts={self.max_pts} ndim={self.ndim} "
                f"M={self.M} efConstruction={self.ef_construction} num_threads={self.num_threads}"
            )

    def _ids_within_active_mask(self, ids: np.ndarray) -> bool:
        if self._active_mask is None:
            return False
        if ids.size == 0:
            return True
        return int(ids.min()) >= 0 and int(ids.max()) < self.max_pts

    def configure(self, config: StreamSeedConfig) -> None:
        if isinstance(config.streamseed_mode, str):
            mode_map = {"off": 0, "streamseed_core": 1}
            config.streamseed_mode = mode_map.get(config.streamseed_mode.lower(), 1)
        self.config = config
        if self.handle:
            rc = self.lib.hnswlib_set_ef(self.handle, ctypes.c_size_t(int(config.ef_search)))
            if rc != 0:
                raise RuntimeError(_HnswlibLib.last_error())

    def insert(self, x: np.ndarray, ids: np.ndarray) -> None:
        self._check_ready()
        vecs = self._prepare_vectors(x)
        ids = np.ascontiguousarray(ids, dtype=np.uint64).reshape(-1)
        if vecs.shape[0] != ids.shape[0]:
            n = min(vecs.shape[0], ids.shape[0])
            if n == 0:
                return
            vecs = vecs[:n]
            ids = ids[:n]

        if self._ids_within_active_mask(ids):
            keep_mask = ~self._active_mask[ids]
            if not np.any(keep_mask):
                return
            vecs_new = np.ascontiguousarray(vecs[keep_mask], dtype=np.float32)
            ids_new = np.ascontiguousarray(ids[keep_mask], dtype=np.uint64)
        else:
            id_list = ids.tolist()
            keep = []
            for i, idx in enumerate(id_list):
                idx_int = int(idx)
                if self._active_mask is not None and 0 <= idx_int < self.max_pts:
                    if not self._active_mask[idx_int]:
                        keep.append(i)
                elif idx_int not in self.active_ids:
                    keep.append(i)
            if not keep:
                return
            vecs_new = np.ascontiguousarray(vecs[keep], dtype=np.float32)
            ids_new = np.ascontiguousarray(ids[keep], dtype=np.uint64)
        log_insert = self.verbose and vecs_new.shape[0] >= 100000
        if log_insert:
            print(
                f"[hnswlib_streamseed] parallel insert start n={vecs_new.shape[0]} "
                f"dim={self.ndim} num_threads={self.num_threads}"
            )
        t0 = time.time() if log_insert else 0.0
        rc = self.lib.hnswlib_add_points_parallel(
            self.handle,
            vecs_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ids_new.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_size_t(vecs_new.shape[0]),
            ctypes.c_size_t(self.ndim),
            ctypes.c_int(1 if self.replace_deleted else 0),
            ctypes.c_int(self.num_threads),
        )
        if rc != 0:
            raise RuntimeError(_HnswlibLib.last_error())
        if log_insert:
            elapsed = time.time() - t0
            qps = vecs_new.shape[0] / elapsed if elapsed > 0 else 0.0
            print(
                f"[hnswlib_streamseed] parallel insert done n={vecs_new.shape[0]} "
                f"elapsed={elapsed:.3f}s qps={qps:.1f}"
            )
        if self._ids_within_active_mask(ids_new):
            self._active_mask[ids_new] = True
        else:
            self.active_ids.update(ids_new.tolist())

    def delete(self, ids: np.ndarray) -> None:
        self._check_ready()
        ids = np.ascontiguousarray(ids, dtype=np.uint64).reshape(-1)
        if self._ids_within_active_mask(ids):
            existing_mask = self._active_mask[ids]
            if not np.any(existing_mask):
                return
            ids_arr = np.ascontiguousarray(ids[existing_mask], dtype=np.uint64)
        else:
            existing = []
            for idx in ids.tolist():
                idx_int = int(idx)
                if self._active_mask is not None and 0 <= idx_int < self.max_pts:
                    if self._active_mask[idx_int]:
                        existing.append(idx_int)
                elif idx_int in self.active_ids:
                    existing.append(idx_int)
            if not existing:
                return
            ids_arr = np.ascontiguousarray(np.asarray(existing, dtype=np.uint64))
        rc = self.lib.hnswlib_mark_delete(
            self.handle,
            ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_size_t(ids_arr.shape[0]),
        )
        if rc != 0:
            raise RuntimeError(_HnswlibLib.last_error())
        if self._ids_within_active_mask(ids_arr):
            self._active_mask[ids_arr] = False
        else:
            for idx in ids_arr.tolist():
                self.active_ids.discard(idx)

    def query(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self._check_ready()
        q = self._prepare_vectors(x)
        nq = int(q.shape[0])
        topk = int(k)
        out_ids = np.empty((nq, topk), dtype=np.uint64)
        out_dists = np.empty((nq, topk), dtype=np.float32)
        rc = self.lib.hnswlib_search_warm(
            self.handle,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(nq),
            ctypes.c_size_t(self.ndim),
            ctypes.c_size_t(topk),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(int(self.config.streamseed_mode)),
            ctypes.c_int(int(self.config.hint_level1_only)),
            ctypes.c_int(int(self.config.hint_adaptive_gate_mode)),
            ctypes.c_int(int(self.config.hint_hops)),
            ctypes.c_int(int(self.config.hint_max_candidates)),
            ctypes.c_float(float(self.config.hint_gate)),
            ctypes.c_float(float(self.config.hint_qual_gate)),
            ctypes.c_float(float(self.config.hint_cons_gate)),
            ctypes.c_float(float(self.config.hint_gate_m_quantile)),
            ctypes.c_float(float(self.config.hint_gate_o_quantile)),
            ctypes.c_int(int(self.config.hint_gate_min_samples)),
            ctypes.c_int(int(self.config.hint_table_slots)),
            ctypes.c_int(int(self.config.hint_slot_capacity)),
            out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            out_dists.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(_HnswlibLib.last_error())

        invalid = np.iinfo(np.uint64).max
        invalid_mask = out_ids == invalid
        ids = out_ids.astype(np.int64)
        ids[invalid_mask] = -1
        self._last_result = ids
        self._last_dists = out_dists
        return ids, out_dists

    def get_results(self) -> Optional[np.ndarray]:
        return self._last_result

    def __del__(self):
        try:
            if self.handle:
                self.lib.hnswlib_destroy(self.handle)
                self.handle = None
        except Exception:
            pass
