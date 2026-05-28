from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import StreamSeedBackend, StreamSeedConfig


class SymphonyQGBackend(StreamSeedBackend):
    """SymphonyQG backend for StreamSeed plugin API.
    """

    _IGNORED_HINT_FIELDS = (
        "hint_level1_only",
        "hint_adaptive_gate_mode",
        "hint_hops",
        "hint_max_candidates",
        "hint_gate",
        "hint_qual_gate",
        "hint_cons_gate",
        "hint_gate_m_quantile",
        "hint_gate_o_quantile",
        "hint_gate_min_samples",
    )

    def __init__(
        self,
        metric: str = "euclidean",
        index_type: str = "QG",
        degree_bound: int = 32,
        ef_construction: int = 200,
        num_iter: int = 3,
        num_threads: int = 0,
        enable_batch_search: bool = True,
        rebuild_every_n_updates: int = 1,
        rebuild_on_query_if_dirty: bool = True,
    ):
        self.metric = metric
        self.index_type = index_type
        self.degree_bound = int(degree_bound)
        self.ef_construction = int(ef_construction)
        self.num_iter = int(num_iter)
        self.num_threads = int(num_threads)
        self.enable_batch_search = bool(enable_batch_search)
        self.rebuild_every_n_updates = max(1, int(rebuild_every_n_updates))
        self.rebuild_on_query_if_dirty = bool(rebuild_on_query_if_dirty)

        self.config = StreamSeedConfig()
        self._ignored_hint_warned = False

        self.max_pts = 0
        self.ndim = 0
        self._data = None
        self._alive = None
        self._active_ids = np.empty(0, dtype=np.int64)
        self._index = None
        self._dirty = False
        self._pending_update_batches = 0
        self._last_result = None

    def _mark_dirty_and_maybe_rebuild(self) -> None:
        self._dirty = True
        self._pending_update_batches += 1

        # Build the first usable index promptly; after that, rebuild by cadence.
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

    def setup(self, max_pts: int, ndim: int, verbose: bool = False) -> None:
        del verbose
        try:
            import symphonyqg
        except ImportError as exc:
            raise RuntimeError("symphonyqg module is required for SymphonyQGBackend") from exc

        if self.metric not in ("euclidean", "angular"):
            raise ValueError("SymphonyQGBackend supports euclidean/angular only")

        if self.degree_bound <= 0 or (self.degree_bound % 32) != 0:
            raise ValueError(
                "SymphonyQGBackend requires degree_bound to be a positive multiple of 32"
            )

        if self.ef_construction <= 0:
            raise ValueError("SymphonyQGBackend requires ef_construction > 0")

        if self.num_iter < 3:
            raise ValueError("SymphonyQGBackend requires num_iter >= 3")

        # Upstream SymphonyQG FHT rotator only supports dimensions whose
        # ceil(log2(dim)) is in [6, 11], i.e. dim in [64, 2048].
        if int(ndim) < 64 or int(ndim) > 2048:
            raise ValueError(
                "SymphonyQGBackend requires ndim in [64, 2048] due to upstream rotator constraints"
            )

        self._symphonyqg = symphonyqg
        self.max_pts = int(max_pts)
        self.ndim = int(ndim)
        self._data = np.zeros((self.max_pts, self.ndim), dtype=np.float32)
        self._alive = np.zeros(self.max_pts, dtype=bool)
        self._active_ids = np.empty(0, dtype=np.int64)
        self._index = None
        self._dirty = False
        self._pending_update_batches = 0
        self._last_result = None

    def configure(self, config: StreamSeedConfig) -> None:
        if isinstance(config.streamseed_mode, str):
            mode_map = {
                "off": 0,
                "streamseed_core": 1,
            }
            config.streamseed_mode = mode_map.get(config.streamseed_mode.lower(), 1)

        # For fixed 10k-query workloads, default to 1:1 slot mapping by query id.
        if int(config.streamseed_mode) != 0:
            base_default_slots = StreamSeedConfig().hint_table_slots
            if int(config.hint_table_slots) == int(base_default_slots):
                config.hint_table_slots = 10000

        self.config = config
        if not self._ignored_hint_warned:
            ignored = []
            for key in self._IGNORED_HINT_FIELDS:
                value = getattr(config, key)
                default_value = getattr(StreamSeedConfig(), key)
                if value != default_value:
                    ignored.append(key)
            if ignored:
                print(
                    "[streamseed:symphonyqg] hint parameters are currently ignored: "
                    + ", ".join(ignored)
                )
                self._ignored_hint_warned = True
        if self._index is not None:
            self._index.set_ef(int(self.config.ef_search))

    def _rebuild_index_if_needed(self) -> None:
        if not self._dirty:
            return

        self._active_ids = np.flatnonzero(self._alive).astype(np.int64)
        active_count = int(self._active_ids.shape[0])

        if active_count == 0:
            self._index = None
            self._dirty = False
            self._pending_update_batches = 0
            return

        build_data = np.ascontiguousarray(self._data[self._active_ids], dtype=np.float32)
        self._index = self._symphonyqg.Index(
            index_type=self.index_type,
            metric="L2",
            num_elements=active_count,
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

        self._index.set_ef(int(self.config.ef_search))
        self._dirty = False
        self._pending_update_batches = 0

    def insert(self, x: np.ndarray, ids: np.ndarray) -> None:
        vecs = self._prepare_vectors(x)
        ext_ids = np.asarray(ids, dtype=np.int64).reshape(-1)

        if vecs.shape[0] != ext_ids.shape[0]:
            n = min(vecs.shape[0], ext_ids.shape[0])
            if n == 0:
                return
            print(
                f"[streamseed:symphonyqg] warning: x/ids size mismatch "
                f"({vecs.shape[0]} vs {ext_ids.shape[0]}), clipping to {n}"
            )
            vecs = vecs[:n]
            ext_ids = ext_ids[:n]

        if np.any(ext_ids < 0) or np.any(ext_ids >= self.max_pts):
            raise ValueError("Insert ids out of range")

        self._data[ext_ids] = vecs
        self._alive[ext_ids] = True
        self._mark_dirty_and_maybe_rebuild()

    def delete(self, ids: np.ndarray) -> None:
        ext_ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ext_ids.size == 0:
            return

        if ext_ids.size > self.max_pts:
            ext_ids = ext_ids[: self.max_pts]

        valid = (ext_ids >= 0) & (ext_ids < self.max_pts)
        ext_ids = ext_ids[valid]
        if ext_ids.size == 0:
            return

        self._alive[ext_ids] = False
        self._mark_dirty_and_maybe_rebuild()

    def query(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        q = self._prepare_vectors(x)
        topk = int(k)
        nq = q.shape[0]

        if self.rebuild_on_query_if_dirty:
            self._rebuild_index_if_needed()

        if self._index is None or self._active_ids.size == 0:
            ids = -1 * np.ones((nq, topk), dtype=np.int64)
            self._last_result = ids
            return ids, None

        active_size = self._active_ids.shape[0]
        ids = -1 * np.ones((nq, topk), dtype=np.int64)

        if hasattr(self._index, "search_warm"):
            labels = np.asarray(
                self._index.search_warm(
                    nq,
                    q.ravel(),
                    topk,
                    int(self.config.ef_search),
                    int(self.config.streamseed_mode),
                    int(self.config.hint_level1_only),
                    int(self.config.hint_adaptive_gate_mode),
                    int(self.config.hint_hops),
                    int(self.config.hint_max_candidates),
                    float(self.config.hint_gate),
                    float(self.config.hint_qual_gate),
                    float(self.config.hint_cons_gate),
                    float(self.config.hint_gate_m_quantile),
                    float(self.config.hint_gate_o_quantile),
                    int(self.config.hint_gate_min_samples),
                    int(self.config.hint_table_slots),
                    int(self.config.hint_slot_capacity),
                ),
                dtype=np.int64,
            )
            labels = labels.reshape(nq, topk)
            valid = (labels >= 0) & (labels < active_size)
            ids[valid] = self._active_ids[labels[valid]]
            self._last_result = ids
            return ids, None

        if self.enable_batch_search and hasattr(self._index, "search_batch"):
            local_all = np.asarray(self._index.search_batch(q, topk), dtype=np.int64)
            valid = (local_all >= 0) & (local_all < active_size)
            ids[valid] = self._active_ids[local_all[valid]]
            self._last_result = ids
            return ids, None

        for i in range(nq):
            local = np.asarray(self._index.search(q[i], topk), dtype=np.int64)
            valid = (local >= 0) & (local < active_size)
            ids[i, valid] = self._active_ids[local[valid]]

        self._last_result = ids
        return ids, None

    def get_results(self) -> Optional[np.ndarray]:
        return self._last_result
