from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .base import StreamSeedBackend, StreamSeedConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


class _WolverineLib:
    _lib = None
    _initialized = False

    @classmethod
    def get(cls):
        if cls._lib is not None:
            return cls._lib

        root = _repo_root()
        wolverine_dir = root / "algorithms_impl" / "Wolverine"
        so_path = wolverine_dir / "libwolverine_capi.so"
        cpp_path = wolverine_dir / "wolverine_c_api.cpp"
        needs_build = (
            not so_path.exists()
            or (cpp_path.exists() and cpp_path.stat().st_mtime > so_path.stat().st_mtime)
        )
        if needs_build:
            if not cpp_path.exists():
                raise RuntimeError(f"Wolverine C API source not found: {cpp_path}")
            compile_cmd = [
                "g++",
                "-std=c++17",
                "-O3",
                "-fPIC",
                "-shared",
                "-fopenmp",
                "-D__AVX__",
                "-mavx2",
                "-I",
                str(wolverine_dir),
                str(cpp_path),
                "-o",
                str(so_path),
            ]
            try:
                subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    "Failed to build Wolverine C API shared library:\n"
                    f"cmd: {' '.join(compile_cmd)}\n"
                    f"stdout:\n{exc.stdout}\n"
                    f"stderr:\n{exc.stderr}"
                ) from exc

        lib = ctypes.CDLL(str(so_path))
        if not cls._initialized:
            lib.wolverine_last_error.restype = ctypes.c_char_p
            lib.wolverine_create.argtypes = [
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            lib.wolverine_create.restype = ctypes.c_void_p
            lib.wolverine_destroy.argtypes = [ctypes.c_void_p]
            lib.wolverine_destroy.restype = None
            lib.wolverine_set_ef.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            lib.wolverine_set_ef.restype = ctypes.c_int
            lib.wolverine_add_points.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.c_size_t,
            ]
            lib.wolverine_add_points.restype = ctypes.c_int
            lib.wolverine_patch_delete.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.wolverine_patch_delete.restype = ctypes.c_int
            lib.wolverine_search_warm.argtypes = [
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
            lib.wolverine_search_warm.restype = ctypes.c_int
            cls._initialized = True

        cls._lib = lib
        return lib

    @classmethod
    def last_error(cls):
        err = cls.get().wolverine_last_error()
        if err is None:
            return "Unknown Wolverine C API error"
        return err.decode("utf-8", errors="replace")


class WolverineStreamSeedBackend(StreamSeedBackend):
    """Wolverine backend with StreamSeed hint-table warm search."""

    def __init__(
        self,
        metric: str = "euclidean",
        M: int = 16,
        ef_construction: int = 120,
        random_seed: int = 100,
        allow_replace_deleted: bool = True,
        delete_model: int = 4,
        new_link_size: int = 16,
        num_threads: int = 8,
    ):
        if metric != "euclidean":
            raise ValueError("WolverineStreamSeedBackend supports euclidean only")
        self.metric = metric
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.random_seed = int(random_seed)
        self.allow_replace_deleted = bool(allow_replace_deleted)
        self.delete_model = int(delete_model)
        self.new_link_size = int(new_link_size)
        self.num_threads = int(num_threads)
        self.config = StreamSeedConfig()
        self.lib = _WolverineLib.get()
        self.handle = None
        self.ndim = 0
        self.max_pts = 0
        self.active_ids = set()
        self._last_result = None
        self._last_dists = None

    def _check_ready(self) -> None:
        if not self.handle:
            raise RuntimeError("Wolverine backend not initialized. Call setup() first.")

    def setup(self, max_pts: int, ndim: int, verbose: bool = False) -> None:
        del verbose
        self.max_pts = int(max_pts)
        self.ndim = int(ndim)
        self.handle = self.lib.wolverine_create(
            ctypes.c_size_t(self.ndim),
            ctypes.c_size_t(self.max_pts),
            ctypes.c_size_t(self.M),
            ctypes.c_size_t(self.ef_construction),
            ctypes.c_size_t(self.random_seed),
            ctypes.c_int(1 if self.allow_replace_deleted else 0),
        )
        if not self.handle:
            raise RuntimeError(_WolverineLib.last_error())
        self.active_ids = set()
        self.configure(self.config)

    def configure(self, config: StreamSeedConfig) -> None:
        if isinstance(config.streamseed_mode, str):
            mode_map = {"off": 0, "streamseed_core": 1}
            config.streamseed_mode = mode_map.get(config.streamseed_mode.lower(), 1)
        self.config = config
        if self.handle:
            rc = self.lib.wolverine_set_ef(self.handle, ctypes.c_size_t(int(config.ef_search)))
            if rc != 0:
                raise RuntimeError(_WolverineLib.last_error())

    def insert(self, x: np.ndarray, ids: np.ndarray) -> None:
        self._check_ready()
        x = np.ascontiguousarray(x, dtype=np.float32)
        ids = np.ascontiguousarray(ids, dtype=np.uint64).reshape(-1)
        if x.shape[0] != ids.shape[0]:
            n = min(x.shape[0], ids.shape[0])
            if n == 0:
                return
            x = x[:n]
            ids = ids[:n]

        keep = [i for i, idx in enumerate(ids.tolist()) if idx not in self.active_ids]
        if not keep:
            return
        x_new = np.ascontiguousarray(x[keep], dtype=np.float32)
        ids_new = np.ascontiguousarray(ids[keep], dtype=np.uint64)
        rc = self.lib.wolverine_add_points(
            self.handle,
            x_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ids_new.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_size_t(x_new.shape[0]),
            ctypes.c_size_t(self.ndim),
        )
        if rc != 0:
            raise RuntimeError(_WolverineLib.last_error())
        self.active_ids.update(ids_new.tolist())

    def delete(self, ids: np.ndarray) -> None:
        self._check_ready()
        ids = np.asarray(ids, dtype=np.uint64).reshape(-1)
        existing = [idx for idx in ids.tolist() if idx in self.active_ids]
        if not existing:
            return
        ids_arr = np.ascontiguousarray(np.asarray(existing, dtype=np.uint64))
        rc = self.lib.wolverine_patch_delete(
            self.handle,
            ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_size_t(ids_arr.shape[0]),
            ctypes.c_int(self.delete_model),
            ctypes.c_int(self.new_link_size),
            ctypes.c_int(self.num_threads),
        )
        if rc != 0:
            raise RuntimeError(_WolverineLib.last_error())
        for idx in existing:
            self.active_ids.discard(idx)

    def query(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self._check_ready()
        q = np.ascontiguousarray(x, dtype=np.float32)
        nq = int(q.shape[0])
        topk = int(k)
        out_ids = np.empty((nq, topk), dtype=np.uint64)
        out_dists = np.empty((nq, topk), dtype=np.float32)
        rc = self.lib.wolverine_search_warm(
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
            raise RuntimeError(_WolverineLib.last_error())

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
                self.lib.wolverine_destroy(self.handle)
                self.handle = None
        except Exception:
            pass
