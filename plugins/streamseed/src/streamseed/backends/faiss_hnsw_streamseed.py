from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import StreamSeedBackend, StreamSeedConfig


class FaissHnswStreamSeedBackend(StreamSeedBackend):
    """Faiss incremental HNSW backend with StreamSeed query-hint controls."""

    def __init__(self, metric: str = "euclidean", indexkey: str = "HNSWIncremental16"):
        self.metric = metric
        self.indexkey = indexkey
        self.index = None
        self.config = StreamSeedConfig()
        self.ntotal = 0

        self.id_map = None
        self.inverse_id_map = None
        self._last_result = None

    def setup(self, max_pts: int, ndim: int, verbose: bool = False) -> None:
        try:
            import PyCANDYAlgo
        except ImportError as exc:
            raise RuntimeError("PyCANDYAlgo is required for FaissHnswStreamSeedBackend") from exc

        if self.metric == "euclidean":
            self.index = PyCANDYAlgo.index_factory_l2(ndim, self.indexkey)
        else:
            self.index = PyCANDYAlgo.index_factory_ip(ndim, self.indexkey)

        self.index.verbose = bool(verbose)
        self.id_map = -1 * np.ones(max_pts, dtype=np.int64)
        self.inverse_id_map = -1 * np.ones(max_pts, dtype=np.int64)
        self.ntotal = 0

    def configure(self, config: StreamSeedConfig) -> None:
        if isinstance(config.streamseed_mode, str):
            mode_map = {
                "off": 0,
                "streamseed_core": 1,
            }
            config.streamseed_mode = mode_map.get(config.streamseed_mode.lower(), 1)
        self.config = config

    def insert(self, x: np.ndarray, ids: np.ndarray) -> None:
        if self.index is None:
            raise RuntimeError("Backend index not initialized. Call setup() first.")

        x = np.ascontiguousarray(x, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int64)

        if x.shape[0] != ids.shape[0]:
            raise ValueError("x and ids must have the same number of rows")

        mask = self.inverse_id_map[ids] == -1
        new_ids = ids[mask]
        new_data = x[mask]
        if new_data.shape[0] == 0:
            return

        self.index.train(new_data.shape[0], new_data.ravel())
        self.index.add(new_data.shape[0], new_data.ravel())

        internal = np.arange(self.ntotal, self.ntotal + new_data.shape[0], dtype=np.int64)
        self.id_map[internal] = new_ids
        self.inverse_id_map[new_ids] = internal
        self.ntotal += new_data.shape[0]

    def delete(self, ids: np.ndarray) -> None:
        if self.inverse_id_map is None or self.id_map is None:
            return
        ids = np.asarray(ids, dtype=np.int64)
        for ext_id in ids:
            if ext_id < 0 or ext_id >= self.inverse_id_map.shape[0]:
                continue
            internal = self.inverse_id_map[ext_id]
            if internal >= 0:
                self.inverse_id_map[ext_id] = -1
                if internal < self.id_map.shape[0]:
                    self.id_map[internal] = -1

    def query(self, x: np.ndarray, k: int, query_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.index is None:
            raise RuntimeError("Backend index not initialized. Call setup() first.")

        x = np.ascontiguousarray(x, dtype=np.float32)
        query_ids_arg = None
        if query_ids is not None:
            query_ids_arg = np.ascontiguousarray(query_ids, dtype=np.int64)
            if query_ids_arg.shape[0] != x.shape[0]:
                raise ValueError("query_ids must have the same length as query rows")

        labels = np.array(
            self.index.search_warm(
                x.shape[0],
                x.ravel(),
                k,
                self.config.ef_search,
                int(self.config.streamseed_mode),
                int(self.config.hint_level1_only),
                int(self.config.hint_adaptive_gate_mode),
                self.config.hint_hops,
                self.config.hint_max_candidates,
                self.config.hint_gate,
                self.config.hint_qual_gate,
                self.config.hint_cons_gate,
                self.config.hint_gate_m_quantile,
                self.config.hint_gate_o_quantile,
                int(self.config.hint_gate_min_samples),
                self.config.hint_table_slots,
                self.config.hint_slot_capacity,
                query_ids_arg,
            )
        )

        ids = self.id_map[labels]
        self._last_result = ids.reshape(x.shape[0], k)
        return self._last_result, None

    def get_results(self) -> Optional[np.ndarray]:
        return self._last_result
