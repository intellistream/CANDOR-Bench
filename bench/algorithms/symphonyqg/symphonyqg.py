"""SymphonyQG streaming adapter for SAGE benchmark."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..base import BaseStreamingANN


try:
    import symphonyqg

    SYMPHONYQG_AVAILABLE = True
except ImportError:
    SYMPHONYQG_AVAILABLE = False


class Symphonyqg(BaseStreamingANN):
    """Expose SymphonyQG with the common streaming interface.

    SymphonyQG is a build-once index. To fit streaming benchmarks, we keep an
    active vector store and rebuild the index after insert/delete batches.
    """

    def __init__(self, metric: str, index_params: Optional[Dict] = None):
        self.metric = metric
        self.name = "symphonyqg"

        params = index_params or {}
        self.index_type = params.get("index_type", "QG")
        self.degree_bound = int(params.get("degree_bound", 32))
        self.ef_construction = int(params.get("efConstruction", 200))
        self.ef_search = int(params.get("efSearch", 100))
        self.num_iter = int(params.get("num_iter", 3))
        self.num_threads = int(params.get("num_threads", 0))
        self.enable_batch_search = bool(params.get("enable_batch_search", True))
        self.rebuild_every_n_updates = max(1, int(params.get("rebuild_every_n_updates", 1)))
        self.rebuild_on_query_if_dirty = bool(params.get("rebuild_on_query_if_dirty", True))
        self.reserve_update_batches = max(
            1,
            int(params.get("reserve_update_batches", self.rebuild_every_n_updates)),
        )

        self.max_pts = 0
        self.ndim = 0

        self._data = None
        self._alive = None
        self._active_ids = np.empty(0, dtype=np.int64)
        self._ext_to_int = np.empty(0, dtype=np.int64)
        self._int_to_ext = np.empty(0, dtype=np.int64)
        self._index_capacity = 0
        self._last_insert_batch_size = 1

        self._index = None
        self._dirty = False
        self._pending_update_batches = 0
        self.res = None

    def _check_backend(self):
        if not SYMPHONYQG_AVAILABLE:
            raise RuntimeError(
                "symphonyqg python module not available. Build/install via: "
                "cd algorithms_impl/SymphonyQG/python && pip install ."
            )

    def setup(self, dtype, max_pts, ndim):
        self._check_backend()

        if self.metric not in ("euclidean", "angular"):
            raise ValueError("SymphonyQG wrapper currently supports euclidean/angular only")

        if dtype != "float32":
            raise ValueError("SymphonyQG wrapper currently supports float32 only")

        self.max_pts = int(max_pts)
        self.ndim = int(ndim)
        self._data = np.zeros((self.max_pts, self.ndim), dtype=np.float32)
        self._alive = np.zeros(self.max_pts, dtype=bool)
        self._active_ids = np.empty(0, dtype=np.int64)
        self._ext_to_int = -1 * np.ones(self.max_pts, dtype=np.int64)
        self._int_to_ext = np.empty(0, dtype=np.int64)
        self._index_capacity = 0
        self._last_insert_batch_size = 1
        self._index = None
        self._dirty = False
        self._pending_update_batches = 0
        self.res = None

    def _mark_dirty_and_maybe_rebuild(self):
        self._dirty = True
        self._pending_update_batches += 1

        # Ensure the first usable index is built promptly; after that, rebuild by cadence.
        if self._index is None or self._pending_update_batches >= self.rebuild_every_n_updates:
            self._rebuild_index_if_needed()

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
        if self.metric == "angular":
            arr = self._normalize_rows(arr)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _rebuild_index_if_needed(self):
        if not self._dirty:
            return

        self._active_ids = np.flatnonzero(self._alive).astype(np.int64)
        active_count = int(self._active_ids.shape[0])

        if active_count == 0:
            self._index = None
            self._int_to_ext = np.empty(0, dtype=np.int64)
            self._ext_to_int.fill(-1)
            self._index_capacity = 0
            self._dirty = False
            self._pending_update_batches = 0
            return

        reserve_slots = self._last_insert_batch_size * self.reserve_update_batches
        target_capacity = min(self.max_pts, active_count + reserve_slots)
        target_capacity = max(active_count, target_capacity)

        build_data = np.zeros((target_capacity, self.ndim), dtype=np.float32)
        build_data[:active_count] = self._data[self._active_ids]

        self._ext_to_int.fill(-1)
        self._int_to_ext = -1 * np.ones(target_capacity, dtype=np.int64)
        self._ext_to_int[self._active_ids] = np.arange(active_count, dtype=np.int64)
        self._int_to_ext[:active_count] = self._active_ids

        self._index = symphonyqg.Index(
            index_type=self.index_type,
            metric="L2",
            num_elements=target_capacity,
            dimension=self.ndim,
            degree_bound=self.degree_bound,
        )

        if self.num_threads > 0:
            self._index.build_index(
                build_data,
                self.ef_construction,
                num_iter=self.num_iter,
                num_thread=self.num_threads,
            )
        else:
            self._index.build_index(
                build_data,
                self.ef_construction,
                num_iter=self.num_iter,
            )

        self._index.set_ef(self.ef_search)
        self._index_capacity = target_capacity
        self._dirty = False
        self._pending_update_batches = 0

    def _filter_alive_results(self, candidates: np.ndarray, topk: int) -> np.ndarray:
        ids = -1 * np.ones((candidates.shape[0], topk), dtype=np.int64)
        for i in range(candidates.shape[0]):
            write = 0
            for cid in candidates[i]:
                c = int(cid)
                if c < 0 or c >= self._index_capacity:
                    continue
                ext_id = int(self._int_to_ext[c])
                if ext_id < 0 or ext_id >= self.max_pts:
                    continue
                if not self._alive[ext_id]:
                    continue
                ids[i, write] = ext_id
                write += 1
                if write >= topk:
                    break
        return ids

    def insert(self, X, ids):
        vecs = self._prepare_vectors(X)
        ext_ids = np.asarray(ids, dtype=np.int64).reshape(-1)

        if vecs.shape[0] != ext_ids.shape[0]:
            n = min(vecs.shape[0], ext_ids.shape[0])
            if n == 0:
                return
            print(
                f"[symphonyqg] warning: X/ids size mismatch "
                f"({vecs.shape[0]} vs {ext_ids.shape[0]}), clipping to {n}"
            )
            vecs = vecs[:n]
            ext_ids = ext_ids[:n]
        if np.any(ext_ids < 0) or np.any(ext_ids >= self.max_pts):
            raise ValueError("Insert ids out of range")

        self._data[ext_ids] = vecs
        self._alive[ext_ids] = True
        self._last_insert_batch_size = max(1, int(ext_ids.shape[0]))

        # Prefer online graph updates when C++ insert is available and there is free capacity.
        if self._index is not None and hasattr(self._index, "insert"):
            try:
                int_ids = np.asarray(self._ext_to_int[ext_ids], dtype=np.int64)
                unseen_mask = int_ids < 0
                if np.any(unseen_mask):
                    need_new = int(np.sum(unseen_mask))
                    used = int(np.sum(self._int_to_ext >= 0))
                    free_slots = int(self._index_capacity - used)
                    if free_slots >= need_new:
                        new_int_ids = np.arange(used, used + need_new, dtype=np.int64)
                        unseen_ext = ext_ids[unseen_mask]
                        self._ext_to_int[unseen_ext] = new_int_ids
                        self._int_to_ext[new_int_ids] = unseen_ext
                        int_ids[unseen_mask] = new_int_ids

                if np.all(int_ids >= 0):
                    self._index.insert(vecs, int_ids, self.ef_search)
            except Exception as exc:
                print(f"[symphonyqg] online insert failed, fallback to periodic rebuild path: {exc}")

        self._mark_dirty_and_maybe_rebuild()

    def delete(self, ids):
        ext_ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ext_ids.size == 0:
            return

        # Some datasets may provide duplicate or oversized delete ID lists.
        # Keep the operation best-effort and bounded.
        if ext_ids.size > self.max_pts:
            ext_ids = ext_ids[: self.max_pts]

        valid = (ext_ids >= 0) & (ext_ids < self.max_pts)
        ext_ids = ext_ids[valid]
        if ext_ids.size == 0:
            return

        self._alive[ext_ids] = False
        self._mark_dirty_and_maybe_rebuild()

    def query(self, X, k):
        q = self._prepare_vectors(X)
        topk = int(k)
        nq = q.shape[0]

        if self.rebuild_on_query_if_dirty:
            self._rebuild_index_if_needed()

        if self._index is None or self._active_ids.size == 0:
            ids = -1 * np.ones((nq, topk), dtype=np.int64)
            self.res = ids
            return ids, None

        if self.enable_batch_search and hasattr(self._index, "search_batch"):
            cand_all = np.asarray(self._index.search_batch(q, topk), dtype=np.int64)
            ids = self._filter_alive_results(cand_all, topk)
            valid_mask = (cand_all >= 0) & (cand_all < self._index_capacity)
            alive_mask = np.zeros_like(valid_mask, dtype=bool)
            mapped_ext = np.full(cand_all.shape, -1, dtype=np.int64)
            mapped_ext[valid_mask] = self._int_to_ext[cand_all[valid_mask]]
            ext_valid_mask = (mapped_ext >= 0) & (mapped_ext < self.max_pts)
            alive_mask[ext_valid_mask] = self._alive[mapped_ext[ext_valid_mask]]
            filtered_count = int(np.sum(~(valid_mask & alive_mask)))
            total_count = int(cand_all.size)
            print(f"[symphonyqg] filtered candidates: {filtered_count}/{total_count}")
            self.res = ids
            return ids, None

        ids = -1 * np.ones((nq, topk), dtype=np.int64)

        for i in range(nq):
            cand = np.asarray(self._index.search(q[i], topk), dtype=np.int64).reshape(1, -1)
            ids[i] = self._filter_alive_results(cand, topk)[0]

        self.res = ids
        return ids, None

    def set_query_arguments(self, query_args):
        self.ef_search = int(query_args.get("ef", query_args.get("efSearch", self.ef_search)))
        if self._index is not None:
            self._index.set_ef(self.ef_search)

    def get_results(self):
        return self.res
