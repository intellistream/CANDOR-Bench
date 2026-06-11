"""
Wolverine Algorithm Implementation
通过 ctypes 调用 algorithms_impl/Wolverine 的 C API 动态库
"""

import ctypes
import subprocess
from pathlib import Path

import numpy as np

from ..base import BaseStreamingANN


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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

            lib.wolverine_search.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.wolverine_search.restype = ctypes.c_int

            cls._initialized = True

        cls._lib = lib
        return lib

    @classmethod
    def _last_error(cls):
        lib = cls.get()
        err = lib.wolverine_last_error()
        if err is None:
            return "Unknown Wolverine C API error"
        return err.decode("utf-8", errors="replace")


class Wolverine(BaseStreamingANN):
    def __init__(self, metric, index_params):
        if metric != "euclidean":
            raise ValueError("Wolverine currently supports only euclidean metric")

        self.metric = metric
        self.index_params = index_params or {}
        self.name = "Wolverine"

        self.M = int(self.index_params.get("M", 16))
        self.efConstruction = int(self.index_params.get("efConstruction", 120))
        self.ef = int(self.index_params.get("ef", 40))
        self.random_seed = int(self.index_params.get("random_seed", 100))
        self.allow_replace_deleted = bool(self.index_params.get("allow_replace_deleted", True))

        self.delete_model = int(self.index_params.get("delete_model", 4))
        self.new_link_size = int(self.index_params.get("new_link_size", self.M))
        self.num_threads = int(self.index_params.get("num_threads", 8))

        self.lib = _WolverineLib.get()
        self.handle = None
        self.ndim = None
        self.res = None
        self.dists = None
        self.active_ids = set()

    def setup(self, dtype, max_pts, ndim):
        if dtype not in ("float32", "float", np.float32):
            raise ValueError(f"Wolverine requires float32 input, got {dtype}")

        self.ndim = int(ndim)
        handle = self.lib.wolverine_create(
            ctypes.c_size_t(self.ndim),
            ctypes.c_size_t(int(max_pts)),
            ctypes.c_size_t(self.M),
            ctypes.c_size_t(self.efConstruction),
            ctypes.c_size_t(self.random_seed),
            ctypes.c_int(1 if self.allow_replace_deleted else 0),
        )
        if not handle:
            raise RuntimeError(_WolverineLib._last_error())

        self.handle = handle
        self.active_ids = set()
        self.set_query_arguments({"ef": self.ef})

    def insert(self, X, ids):
        self._ensure_ready()
        X = np.ascontiguousarray(X, dtype=np.float32)
        ids = np.ascontiguousarray(ids, dtype=np.uint64)

        if ids.shape[0] < X.shape[0]:
            raise ValueError("ids count is smaller than vectors in Wolverine.insert")
        if ids.shape[0] > X.shape[0]:
            # Some runbooks may request ranges beyond dataset size on tiny datasets.
            # Keep insertion aligned with loaded vectors by trimming extra ids.
            ids = ids[: X.shape[0]]

        keep = [i for i, idx in enumerate(ids.tolist()) if idx not in self.active_ids]
        if not keep:
            return
        X_new = np.ascontiguousarray(X[keep], dtype=np.float32)
        ids_new = np.ascontiguousarray(ids[keep], dtype=np.uint64)

        rc = self.lib.wolverine_add_points(
            self.handle,
            X_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ids_new.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_size_t(X_new.shape[0]),
            ctypes.c_size_t(self.ndim),
        )
        if rc != 0:
            raise RuntimeError(_WolverineLib._last_error())

        self.active_ids.update(ids_new.tolist())

    def delete(self, ids):
        self._ensure_ready()
        ids = np.ascontiguousarray(ids, dtype=np.uint64)
        if ids.size == 0:
            return

        existing = [idx for idx in ids.tolist() if idx in self.active_ids]
        if not existing:
            return

        ids_arr = np.ascontiguousarray(np.array(existing, dtype=np.uint64))
        rc = self.lib.wolverine_patch_delete(
            self.handle,
            ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_size_t(ids_arr.shape[0]),
            ctypes.c_int(self.delete_model),
            ctypes.c_int(self.new_link_size),
            ctypes.c_int(self.num_threads),
        )
        if rc != 0:
            raise RuntimeError(_WolverineLib._last_error())

        for idx in existing:
            self.active_ids.discard(idx)

    def query(self, X, k):
        self._ensure_ready()
        X = np.ascontiguousarray(X, dtype=np.float32)
        nq = int(X.shape[0])
        k = int(k)

        out_ids = np.empty((nq, k), dtype=np.uint64)
        out_dists = np.empty((nq, k), dtype=np.float32)

        rc = self.lib.wolverine_search(
            self.handle,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(nq),
            ctypes.c_size_t(self.ndim),
            ctypes.c_size_t(k),
            ctypes.c_int(self.num_threads),
            out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            out_dists.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(_WolverineLib._last_error())

        invalid = np.iinfo(np.uint64).max
        invalid_mask = out_ids == invalid
        out_ids = out_ids.astype(np.int64)
        out_ids[invalid_mask] = -1

        self.res = out_ids
        self.dists = out_dists
        return self.res, self.dists

    def set_query_arguments(self, query_args):
        self.ef = int(query_args.get("ef", self.ef))
        if self.handle:
            rc = self.lib.wolverine_set_ef(self.handle, ctypes.c_size_t(self.ef))
            if rc != 0:
                raise RuntimeError(_WolverineLib._last_error())

    def get_results(self):
        return self.res

    def _ensure_ready(self):
        if not self.handle:
            raise RuntimeError("Wolverine index is not initialized. Call setup first.")

    def __del__(self):
        try:
            if self.handle:
                self.lib.wolverine_destroy(self.handle)
                self.handle = None
        except Exception:
            pass
