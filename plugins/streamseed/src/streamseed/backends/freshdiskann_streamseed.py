from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .base import StreamSeedBackend, StreamSeedConfig

_DISKANN_IMPORT_ERROR: Exception | None = None
_diskannpy = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


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


class FreshDiskANNStreamSeedBackend(StreamSeedBackend):
    """FreshDiskANN backend with StreamSeed history-topk plus graph-neighbor hints."""

    def __init__(
        self,
        metric: str = "euclidean",
        R: int = 32,
        L: int = 100,
        alpha: float = 1.2,
        num_threads: int = 16,
        insert_threads: Optional[int] = None,
        search_threads: Optional[int] = None,
        initial_search_complexity: Optional[int] = None,
        filter_complexity: int = 0,
        num_frozen_points: int = 1,
        max_occlusion_size: int = 750,
        saturate_graph: bool = False,
        concurrent_consolidation: bool = True,
        consolidate_every: int = 0,
    ):
        self.metric = metric
        self.R = int(R)
        self.L = int(L)
        self.alpha = float(alpha)
        self.num_threads = int(num_threads)
        self.insert_threads = int(insert_threads if insert_threads is not None else self.num_threads)
        self.search_threads = int(search_threads if search_threads is not None else self.num_threads)
        self.initial_search_complexity = int(
            initial_search_complexity if initial_search_complexity is not None else max(self.L, 100)
        )
        self.filter_complexity = int(filter_complexity)
        self.num_frozen_points = int(num_frozen_points)
        self.max_occlusion_size = int(max_occlusion_size)
        self.saturate_graph = bool(saturate_graph)
        self.concurrent_consolidation = bool(concurrent_consolidation)
        self.consolidate_every = int(consolidate_every)

        self.index = None
        self.config = StreamSeedConfig()
        self.verbose = False
        self.max_pts = 0
        self.ndim = 0
        self.active_mask: np.ndarray | None = None
        self.vectors: np.ndarray | None = None
        self.pending_deletes = 0
        self._last_result = None
        self._last_dists = None
        self._hint_labels: np.ndarray | None = None
        self._hint_counts: np.ndarray | None = None
        self._hint_owner_queries: np.ndarray | None = None
        self._hint_slots = 0
        self._hint_capacity = 0
        self._warned_missing_neighbor_api = False
        self._warned_missing_warm_api = False
        self._warned_neighbor_error = False

    @staticmethod
    def _metric_name(metric: str) -> str:
        metric = str(metric).lower()
        if metric in ("euclidean", "l2"):
            return "l2"
        if metric in ("angular", "cosine"):
            return "cosine"
        if metric in ("ip", "inner_product", "mips"):
            return "mips"
        raise ValueError(f"FreshDiskANNStreamSeedBackend unsupported metric: {metric}")

    @staticmethod
    def _native_metric(dap, metric: str):
        metric = str(metric).lower()
        if metric in ("euclidean", "l2"):
            return dap.Metric.L2
        if metric in ("angular", "cosine"):
            return dap.Metric.COSINE
        if metric in ("ip", "inner_product", "mips"):
            return getattr(dap.Metric, "INNER_PRODUCT", dap.Metric.L2)
        raise ValueError(f"FreshDiskANNStreamSeedBackend unsupported metric: {metric}")

    def _check_diskannpy(self):
        global _diskannpy
        if _diskannpy is None:
            _diskannpy = _import_diskannpy()
        if _diskannpy is None:
            raise RuntimeError(
                "FreshDiskANNStreamSeedBackend requires PyCANDYAlgo.diskannpy or diskannpy. "
                f"Original import error: {_DISKANN_IMPORT_ERROR!r}"
            )
        return _diskannpy

    def _check_ready(self) -> None:
        if self.index is None:
            raise RuntimeError("FreshDiskANN backend not initialized. Call setup() first.")

    def setup(self, max_pts: int, ndim: int, verbose: bool = False) -> None:
        dap = self._check_diskannpy()
        self.max_pts = int(max_pts)
        self.ndim = int(ndim)
        self.verbose = bool(verbose)
        self.active_mask = np.zeros(self.max_pts, dtype=bool)
        self.vectors = np.zeros((self.max_pts, self.ndim), dtype=np.float32)

        if hasattr(dap, "DynamicMemoryIndex"):
            self.index = dap.DynamicMemoryIndex(
                distance_metric=self._metric_name(self.metric),
                vector_dtype=np.float32,
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
            self.index = dap.DynamicMemoryFloatIndex(
                algo_type=dap.AlgoType.DISKANN,
                distance_metric=self._native_metric(dap, self.metric),
                max_vectors=self.max_pts + self.num_frozen_points,
                dimensions=self.ndim,
                graph_degree=self.R,
                complexity=self.L,
                num_threads=self.num_threads,
                initial_search_complexity=self.initial_search_complexity,
            )
        if self.verbose:
            print(
                "[freshdiskann_streamseed] setup "
                f"max_pts={self.max_pts} dim={self.ndim} R={self.R} L={self.L} "
                f"insert_threads={self.insert_threads} search_threads={self.search_threads}"
            )

    def configure(self, config: StreamSeedConfig) -> None:
        if isinstance(config.streamseed_mode, str):
            mode_map = {"off": 0, "streamseed_core": 1}
            config.streamseed_mode = mode_map.get(config.streamseed_mode.lower(), 1)
        self.config = config
        if int(config.streamseed_mode) != 0:
            self._ensure_hint_table(int(config.hint_table_slots), int(config.hint_slot_capacity))
        else:
            self._ensure_hint_table(0, 0)

    def _ensure_hint_table(self, slots: int, capacity: int) -> None:
        slots = max(int(slots), 0)
        capacity = max(int(capacity), 0)
        if slots == 0 or capacity == 0:
            self._hint_labels = None
            self._hint_counts = None
            self._hint_owner_queries = None
            self._hint_slots = 0
            self._hint_capacity = 0
            return
        if self._hint_slots == slots and self._hint_capacity == capacity:
            return
        self._hint_labels = -1 * np.ones((slots, capacity), dtype=np.int64)
        self._hint_counts = np.zeros(slots, dtype=np.int32)
        self._hint_owner_queries = -1 * np.ones(slots, dtype=np.int64)
        self._hint_slots = slots
        self._hint_capacity = capacity

    def insert(self, x: np.ndarray, ids: np.ndarray) -> None:
        self._check_ready()
        vecs = np.ascontiguousarray(x, dtype=np.float32)
        ids_arr = np.asarray(ids, dtype=np.int64).reshape(-1)
        if vecs.shape[0] != ids_arr.shape[0]:
            n = min(vecs.shape[0], ids_arr.shape[0])
            if n == 0:
                return
            vecs = vecs[:n]
            ids_arr = ids_arr[:n]

        valid = (ids_arr >= 0) & (ids_arr < self.max_pts)
        if not np.any(valid):
            return
        vecs = np.ascontiguousarray(vecs[valid], dtype=np.float32)
        ids_arr = ids_arr[valid]
        if self.active_mask is not None:
            keep = ~self.active_mask[ids_arr]
            if not np.any(keep):
                return
            vecs = np.ascontiguousarray(vecs[keep], dtype=np.float32)
            ids_arr = ids_arr[keep]

        tags = np.ascontiguousarray(ids_arr + 1, dtype=np.uint32)
        try:
            retvals = self.index.batch_insert(vectors=vecs, vector_ids=tags, num_threads=self.insert_threads)
        except TypeError:
            retvals = self.index.batch_insert(vecs, tags, int(tags.shape[0]), self.insert_threads)
        if retvals is not None:
            retvals = np.asarray(retvals)
            if retvals.size and np.any(retvals == -1):
                failed = int(np.sum(retvals == -1))
                raise RuntimeError(f"FreshDiskANN insertion failed for {failed}/{retvals.size} vectors")
        if self.active_mask is not None:
            self.active_mask[ids_arr] = True
        if self.vectors is not None:
            self.vectors[ids_arr] = vecs

    def delete(self, ids: np.ndarray) -> None:
        self._check_ready()
        ids_arr = np.asarray(ids, dtype=np.int64).reshape(-1)
        valid = ids_arr[(ids_arr >= 0) & (ids_arr < self.max_pts)]
        if valid.size == 0:
            return
        if self.active_mask is not None:
            valid = valid[self.active_mask[valid]]
            if valid.size == 0:
                return
        for ext_id in valid.tolist():
            rc = self.index.mark_deleted(np.uint32(ext_id + 1))
            if rc is None or rc == 0:
                self.pending_deletes += 1
                if self.active_mask is not None:
                    self.active_mask[ext_id] = False
        if self.consolidate_every > 0 and self.pending_deletes >= self.consolidate_every:
            self.index.consolidate_delete()
            self.pending_deletes = 0

    def _search_diskann(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        nq = int(queries.shape[0])
        complexity = max(int(self.config.ef_search), int(k))
        try:
            result = self.index.batch_search(
                queries=queries,
                k_neighbors=int(k),
                complexity=complexity,
                num_threads=int(self.search_threads),
            )
        except TypeError:
            result = self.index.batch_search(queries, nq, int(k), complexity, int(self.search_threads))
        ids = getattr(result, "identifiers", result[0])
        dists = getattr(result, "distances", result[1])
        ids = np.asarray(ids, dtype=np.int64).reshape(nq, int(k)) - 1
        dists = np.asarray(dists, dtype=np.float32).reshape(nq, int(k))
        if self.active_mask is not None:
            invalid = (ids < 0) | (ids >= self.max_pts)
            in_range = ~invalid
            if np.any(in_range):
                invalid[in_range] = ~self.active_mask[ids[in_range]]
            ids[invalid] = -1
            dists[invalid] = np.inf
        return ids, dists

    def _has_neighbor_api(self) -> bool:
        inner = getattr(self.index, "_index", None)
        return bool(hasattr(self.index, "get_neighbors") or (inner is not None and hasattr(inner, "get_neighbors")))

    def _warm_search_target(self):
        if hasattr(self.index, "batch_search_warm"):
            return self.index
        inner = getattr(self.index, "_index", None)
        if inner is not None and hasattr(inner, "batch_search_warm"):
            return inner
        return None

    def _search_diskann_warm(self, queries: np.ndarray, k: int, query_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        target = self._warm_search_target()
        if target is None:
            raise RuntimeError("DiskANN batch_search_warm API is unavailable")
        nq = int(queries.shape[0])
        complexity = max(int(self.config.ef_search), int(k))
        query_ids_arg = None
        if query_ids is not None:
            query_ids_arg = np.ascontiguousarray(query_ids, dtype=np.int64).reshape(-1)
            if query_ids_arg.shape[0] != nq:
                raise ValueError("query_ids must have the same length as query rows")
        result = target.batch_search_warm(
            queries,
            nq,
            int(k),
            complexity,
            int(self.search_threads),
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
            query_ids_arg,
        )
        ids = getattr(result, "identifiers", result[0])
        dists = getattr(result, "distances", result[1])
        ids = np.asarray(ids, dtype=np.int64).reshape(nq, int(k)) - 1
        dists = np.asarray(dists, dtype=np.float32).reshape(nq, int(k))
        if self.active_mask is not None:
            invalid = (ids < 0) | (ids >= self.max_pts)
            in_range = ~invalid
            if np.any(in_range):
                invalid[in_range] = ~self.active_mask[ids[in_range]]
            ids[invalid] = -1
            dists[invalid] = np.inf
        return ids, dists

    def _raw_get_neighbors(self, tag: np.uint32) -> np.ndarray:
        if hasattr(self.index, "get_neighbors"):
            return np.asarray(self.index.get_neighbors(tag), dtype=np.int64)
        inner = getattr(self.index, "_index", None)
        if inner is not None and hasattr(inner, "get_neighbors"):
            return np.asarray(inner.get_neighbors(tag), dtype=np.int64)
        return np.empty(0, dtype=np.int64)

    def _neighbors_for_seed(self, seed_id: int) -> list[int]:
        if self.active_mask is None or seed_id < 0 or seed_id >= self.max_pts or not self.active_mask[seed_id]:
            return []
        try:
            tags = self._raw_get_neighbors(np.uint32(seed_id + 1))
        except Exception as exc:
            if self.verbose and not self._warned_neighbor_error:
                print(f"[freshdiskann_streamseed] get_neighbors failed; falling back for affected queries: {exc!r}")
                self._warned_neighbor_error = True
            return []
        out = []
        for tag in tags.tolist():
            idx = int(tag) - 1
            if 0 <= idx < self.max_pts and idx != seed_id and self.active_mask[idx]:
                out.append(idx)
        return out

    def _candidate_ids(self, qi: int) -> list[int]:
        if (
            self._hint_labels is None
            or self._hint_counts is None
            or self._hint_owner_queries is None
            or self.active_mask is None
            or self._hint_slots <= 0
        ):
            return []
        slot = qi % self._hint_slots
        if self._hint_owner_queries[slot] != qi or self._hint_counts[slot] <= 0:
            return []

        max_candidates = int(self.config.hint_max_candidates)
        if max_candidates <= 0:
            max_candidates = 256
        hops = max(int(self.config.hint_hops), 1)
        if int(self.config.hint_level1_only):
            hops = 1

        seen: set[int] = set()
        frontier: list[int] = []
        seed_count = min(int(self._hint_counts[slot]), self._hint_capacity)
        for seed in self._hint_labels[slot, :seed_count].tolist():
            seed = int(seed)
            if 0 <= seed < self.max_pts and self.active_mask[seed] and seed not in seen:
                seen.add(seed)
                frontier.append(seed)
                if len(seen) >= max_candidates:
                    return list(seen)

        for _ in range(hops):
            next_frontier: list[int] = []
            for seed in frontier:
                for nbr in self._neighbors_for_seed(seed):
                    if nbr not in seen:
                        seen.add(nbr)
                        next_frontier.append(nbr)
                        if len(seen) >= max_candidates:
                            return list(seen)
            if not next_frontier:
                break
            frontier = next_frontier
        return list(seen)

    def _stored_seed_ids(self, qi: int) -> np.ndarray:
        if (
            self._hint_labels is None
            or self._hint_counts is None
            or self._hint_owner_queries is None
            or self.active_mask is None
            or self._hint_slots <= 0
            or self._hint_capacity <= 0
        ):
            return np.empty(0, dtype=np.int64)
        slot = qi % self._hint_slots
        if self._hint_owner_queries[slot] != qi or self._hint_counts[slot] <= 0:
            return np.empty(0, dtype=np.int64)
        count = min(int(self._hint_counts[slot]), self._hint_capacity)
        seeds = np.asarray(self._hint_labels[slot, :count], dtype=np.int64)
        valid = (seeds >= 0) & (seeds < self.max_pts)
        if np.any(valid):
            valid[valid] = self.active_mask[seeds[valid]]
        return seeds[valid]

    def _reject_by_consistency(self, qi: int, labels: np.ndarray, k: int) -> bool:
        gate = float(self.config.hint_cons_gate)
        if gate < 0.0:
            return False
        gate = min(gate, 1.0)
        if k <= 0:
            return False
        seeds = self._stored_seed_ids(qi)
        if seeds.size == 0:
            return False
        top = np.asarray(labels, dtype=np.int64).reshape(-1)[:k]
        top = top[top >= 0]
        if top.size == 0:
            return True
        overlap = np.intersect1d(top, seeds, assume_unique=False).size
        return (float(overlap) / float(k)) < gate

    def _rerank_candidates(self, query: np.ndarray, candidates: list[int], k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.vectors is None or len(candidates) < k:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        cand = np.asarray(candidates, dtype=np.int64)
        vecs = self.vectors[cand]
        metric = str(self.metric).lower()
        if metric in ("ip", "inner_product", "mips"):
            dists = -(vecs @ query)
        elif metric in ("angular", "cosine"):
            q_norm = max(float(np.linalg.norm(query)), 1e-12)
            v_norm = np.maximum(np.linalg.norm(vecs, axis=1), 1e-12)
            dists = 1.0 - ((vecs @ query) / (v_norm * q_norm))
        else:
            diff = vecs - query.reshape(1, -1)
            dists = np.einsum("ij,ij->i", diff, diff)
        order = np.argsort(dists, kind="stable")[:k]
        labels = cand[order].astype(np.int64)
        dists = np.asarray(dists[order], dtype=np.float32)
        if dists.size and float(self.config.hint_gate) >= 0.0 and float(dists[0]) > float(self.config.hint_gate):
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        return labels, dists

    def _update_hint_slot(self, qi: int, labels: np.ndarray) -> None:
        if self._hint_labels is None or self._hint_counts is None or self._hint_owner_queries is None:
            return
        if self.active_mask is None or self._hint_slots <= 0 or self._hint_capacity <= 0:
            return
        slot = qi % self._hint_slots
        store = []
        for label in labels.reshape(-1).tolist():
            idx = int(label)
            if 0 <= idx < self.max_pts and self.active_mask[idx]:
                store.append(idx)
            if len(store) >= self._hint_capacity:
                break
        self._hint_labels[slot, :] = -1
        if store:
            self._hint_labels[slot, : len(store)] = np.asarray(store, dtype=np.int64)
        self._hint_counts[slot] = len(store)
        self._hint_owner_queries[slot] = qi

    def _update_hint_slots(self, ids: np.ndarray) -> None:
        for qi in range(int(ids.shape[0])):
            self._update_hint_slot(qi, ids[qi])

    def query(self, x: np.ndarray, k: int, query_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self._check_ready()
        q = np.ascontiguousarray(x, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        nq = int(q.shape[0])
        topk = int(k)

        use_hints = (
            int(self.config.streamseed_mode) != 0
            and int(self.config.hint_table_slots) > 0
            and int(self.config.hint_slot_capacity) > 0
        )
        if use_hints and self._warm_search_target() is not None:
            ids, dists = self._search_diskann_warm(q, topk, query_ids=query_ids)
            self._last_result = ids
            self._last_dists = dists
            return ids, dists

        if use_hints and self.verbose:
            if self._warm_search_target() is None and not self._warned_missing_warm_api:
                print("[freshdiskann_streamseed] streamseed_core requested but DiskANN batch_search_warm API is unavailable; using normal FreshDiskANN search")
                self._warned_missing_warm_api = True

        ids, dists = self._search_diskann(q, topk)
        if use_hints:
            self._update_hint_slots(ids)
        self._last_result = ids
        self._last_dists = dists
        return ids, dists

    def get_results(self) -> Optional[np.ndarray]:
        return self._last_result
