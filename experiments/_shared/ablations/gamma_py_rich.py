"""GammaPyHybridRich — full rich version of the router with all the mechanisms
that the C++ GammaFresh's control/ directory has, in Python form.

Mirrors the design of:
  - control/vector_life_table.h   (VectorLifeState per vector)
  - control/placement_controller.h (cost-model placement)
  - heat_monitor.cpp              (per-vector + per-partition heat)
  - gamma estimation              (mortality rate per partition)

Mechanisms (each independently toggleable for ablation):
  - n_partitions          : K spatial partitions, centroid routing (e24-style)
  - use_per_vector_cost   : per-vector lifetime EMA → per-vector admit decision
                             (instead of e27's per-batch decision)
  - use_hot_tier          : top-N most-queried vectors get pinned in a fast
                             tier (small dict, brute-force scan first)
  - use_gamma_migration   : migration only flushes vectors whose estimated
                             gamma (mortality rate) is BELOW threshold;
                             likely-to-die vectors stay in buffer until they
                             die (or are forced out)
  - use_tombstone_rebuild : per-partition rebuild when tombstone-fraction
                             crosses threshold (e25-style, but per-partition)

Workload-aware lifetime-tag support: if `lifetime_tags` dict is provided to
add(), each vector's tag enters its VectorLifeState and the cost-admit
decision uses tag-conditional EMAs. This is for testing on
mixed-lifetime workloads (e29) where the C++ design should clearly win
over a single global EMA.
"""
from __future__ import annotations
from collections import deque, defaultdict
from typing import Callable, Optional
import numpy as np
from ..gamma_py import GraphBackend


class VectorLifeState:
    __slots__ = ("id", "insert_op_id", "delete_op_id", "owner_partition_id",
                  "placement", "query_hit_count", "lifetime_tag")

    def __init__(self, vid: int, insert_op_id: int, owner_partition_id: int,
                 placement: str, lifetime_tag: str = ""):
        self.id = vid
        self.insert_op_id = insert_op_id
        self.delete_op_id: Optional[int] = None
        self.owner_partition_id = owner_partition_id
        self.placement = placement   # "buffer" | "graph" | "hot"
        self.query_hit_count = 0
        self.lifetime_tag = lifetime_tag


class GammaPyHybridRich:
    def __init__(self, backend_factory: Callable[[int], GraphBackend],
                 dim: int, buf_capacity: int,
                 *,
                 # --- mechanism toggles ---
                 n_partitions: int = 1,
                 search_partitions: int = 1,
                 use_per_vector_cost: bool = False,
                 use_hot_tier: bool = False,
                 use_gamma_migration: bool = False,
                 use_tombstone_rebuild: bool = False,
                 # --- knobs ---
                 hot_tier_size: int = 500,
                 hot_tier_promote_threshold: int = 5,
                 cost_admit_threshold_ops: int = 50_000,
                 gamma_migration_threshold: float = 1.5,
                 rebuild_threshold: float = 0.5,
                 min_alive_for_rebuild: int = 1000,
                 default_pred_lifetime: int = 1_000_000,
                 history_window: int = 1000,
                 max_elements_per_part: Optional[int] = None):
        if not (1 <= search_partitions <= n_partitions):
            raise ValueError(f"search_partitions {search_partitions} out of [1, {n_partitions}]")
        self.dim = dim
        self.k = n_partitions
        self.m_search = search_partitions
        self.use_per_vector_cost = use_per_vector_cost
        self.use_hot_tier = use_hot_tier
        self.use_gamma_migration = use_gamma_migration
        self.use_tombstone_rebuild = use_tombstone_rebuild
        self.hot_tier_size = hot_tier_size
        self.hot_tier_promote_threshold = hot_tier_promote_threshold
        self.cost_admit_threshold = int(cost_admit_threshold_ops)
        self.gamma_migration_threshold = float(gamma_migration_threshold)
        self.rebuild_threshold = rebuild_threshold
        self.min_alive_for_rebuild = min_alive_for_rebuild
        self.default_pred_lifetime = default_pred_lifetime
        self.backend_factory = backend_factory

        # --- state ---
        self.graphs: list[GraphBackend] = [backend_factory(p) for p in range(self.k)]
        per_part_cap = max(1, buf_capacity // self.k * 2)
        self.buf_capacity_per_part = per_part_cap
        self.buf_vecs = [np.empty((per_part_cap, dim), dtype=np.float32) for _ in range(self.k)]
        self.buf_ids = [np.empty(per_part_cap, dtype=np.int64) for _ in range(self.k)]
        self.buf_alive = [np.zeros(per_part_cap, dtype=np.bool_) for _ in range(self.k)]
        self.buf_size = [0 for _ in range(self.k)]
        self.centroids = np.zeros((self.k, dim), dtype=np.float32)
        self._centroid_initialized = False
        self.part_vec_count = np.zeros(self.k, dtype=np.int64)
        self.part_vec_sum = np.zeros((self.k, dim), dtype=np.float64)

        # vector life table — per-vector state
        self.life_table: dict[int, VectorLifeState] = {}
        # secondary indexes
        self._buffer_pos: dict[int, tuple[int, int]] = {}  # id -> (partition, pos)
        self._graph_vecs: dict[int, np.ndarray] = {}  # for rebuild

        # heat tracking
        # key = (partition, id) — actually just id is fine since partition is
        # in life_table; but tracking access count
        self.hot_tier_ids: set[int] = set()
        self.hot_tier_vecs: dict[int, np.ndarray] = {}

        # per-partition gamma (mortality rate) estimate
        # gamma = (deletes_in_partition_recently) / (vector_count_in_partition)
        # higher gamma = vectors die fast there
        self.part_inserted: list[int] = [0 for _ in range(self.k)]
        self.part_deleted_in_graph: list[int] = [0 for _ in range(self.k)]
        self.part_recent_lifetimes: list[deque] = [deque(maxlen=history_window) for _ in range(self.k)]
        self.global_recent_lifetimes: deque = deque(maxlen=history_window)

        # operation counter (matches C++ OperationSequenceTracker semantics)
        self.global_op = 0

        # Stats
        self.admit_buffer_count = 0
        self.admit_direct_count = 0
        self.hot_promote_count = 0
        self.rebuild_count = 0
        self.gamma_migration_skip_count = 0  # vectors held back from migration

    # ------------------------------------------------------------------
    # Routing / centroid maintenance
    # ------------------------------------------------------------------
    def _seed_centroids_from_data(self, data: np.ndarray):
        if len(data) == 0:
            return
        if len(data) < self.k:
            for j in range(self.k):
                self.centroids[j] = data[j % len(data)]
        else:
            stride = len(data) // self.k
            for j in range(self.k):
                self.centroids[j] = data[j * stride]
        self._centroid_initialized = True

    def _route(self, vectors: np.ndarray) -> np.ndarray:
        if self.k == 1:
            return np.zeros(len(vectors), dtype=np.int64)
        if not self._centroid_initialized:
            return np.arange(len(vectors), dtype=np.int64) % self.k
        q_sq = (vectors * vectors).sum(axis=1, keepdims=True)
        c_sq = (self.centroids * self.centroids).sum(axis=1)[None, :]
        d2 = q_sq + c_sq - 2.0 * (vectors @ self.centroids.T)
        return d2.argmin(axis=1).astype(np.int64)

    def _update_centroid_running(self, p: int, vectors: np.ndarray):
        n_new = len(vectors)
        if n_new == 0:
            return
        self.part_vec_sum[p] += vectors.sum(axis=0)
        self.part_vec_count[p] += n_new
        self.centroids[p] = (self.part_vec_sum[p] / max(1, self.part_vec_count[p])).astype(np.float32)
        self._centroid_initialized = True

    # ------------------------------------------------------------------
    # Cost-admit (per-vector)
    # ------------------------------------------------------------------
    def _predict_lifetime_for(self, partition: int, lifetime_tag: str = "") -> float:
        # Tag-conditional EMA if tag provided and tag-history non-empty
        if lifetime_tag and lifetime_tag in self._tag_lifetimes:
            tlist = self._tag_lifetimes[lifetime_tag]
            if tlist:
                return sum(tlist) / len(tlist)
        # Per-partition EMA if partition has observed lifetimes
        if self.part_recent_lifetimes[partition]:
            return sum(self.part_recent_lifetimes[partition]) / len(self.part_recent_lifetimes[partition])
        # Global EMA
        if self.global_recent_lifetimes:
            return sum(self.global_recent_lifetimes) / len(self.global_recent_lifetimes)
        return float(self.default_pred_lifetime)

    @property
    def _tag_lifetimes(self) -> dict[str, deque]:
        if not hasattr(self, "_tag_lifetimes_dict"):
            self._tag_lifetimes_dict = defaultdict(lambda: deque(maxlen=200))
        return self._tag_lifetimes_dict

    # ------------------------------------------------------------------
    # γ (mortality) estimation per partition
    # ------------------------------------------------------------------
    def _partition_gamma(self, p: int) -> float:
        # gamma = expected deletes / expected lifetime
        # Use observed median of recent lifetimes for that partition
        # If short → high mortality (gamma high)
        if not self.part_recent_lifetimes[p]:
            return 0.0
        median_life = sorted(self.part_recent_lifetimes[p])[len(self.part_recent_lifetimes[p]) // 2]
        # Normalize by op count to get rate per op
        return 1.0 / max(1.0, median_life)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initial_load(self, ids, vectors, lifetime_tags: Optional[list[str]] = None):
        ids_arr = np.asarray(ids, dtype=np.int64)
        vec_arr = np.ascontiguousarray(vectors, dtype=np.float32)
        if lifetime_tags is None:
            lifetime_tags = [""] * len(ids_arr)
        self._seed_centroids_from_data(vec_arr)
        assignments = self._route(vec_arr)
        for p in range(self.k):
            mask = assignments == p
            if not mask.any():
                continue
            sub_ids = ids_arr[mask]
            sub_vecs = np.ascontiguousarray(vec_arr[mask])
            self.graphs[p].add(sub_vecs, sub_ids)
            for i, v in zip(sub_ids, sub_vecs):
                vid = int(i)
                tag = lifetime_tags[int(np.where(ids_arr == i)[0][0])] if lifetime_tags else ""
                self.life_table[vid] = VectorLifeState(vid, self.global_op, p, "graph", tag)
                self._graph_vecs[vid] = v
                self.global_op += 1
            self._update_centroid_running(p, sub_vecs)
            self.part_inserted[p] += len(sub_ids)

    def add(self, ids, vectors, lifetime_tags: Optional[list[str]] = None):
        ids_arr = np.asarray(ids, dtype=np.int64)
        vec_arr = np.ascontiguousarray(vectors, dtype=np.float32)
        if lifetime_tags is None:
            lifetime_tags = [""] * len(ids_arr)
        assignments = self._route(vec_arr)

        # Group by partition
        for p in range(self.k):
            mask = assignments == p
            if not mask.any():
                continue
            sub_indices = np.where(mask)[0]
            sub_ids = ids_arr[mask]
            sub_vecs = vec_arr[mask]
            sub_tags = [lifetime_tags[i] for i in sub_indices]

            # Per-vector cost-admit decision
            if self.use_per_vector_cost:
                # Each vector decides individually
                buf_indices = []
                direct_indices = []
                for i, tag in enumerate(sub_tags):
                    pred = self._predict_lifetime_for(p, tag)
                    if pred < self.cost_admit_threshold:
                        buf_indices.append(i)
                    else:
                        direct_indices.append(i)
                # Buffer-admit batch
                if buf_indices:
                    self._buffer_add(p, sub_ids[buf_indices], sub_vecs[buf_indices],
                                      [sub_tags[i] for i in buf_indices])
                    self.admit_buffer_count += len(buf_indices)
                # Direct admit batch
                if direct_indices:
                    self._direct_add(p, sub_ids[direct_indices], sub_vecs[direct_indices],
                                      [sub_tags[i] for i in direct_indices])
                    self.admit_direct_count += len(direct_indices)
            else:
                # Default: buffer everything (gamma_v2 behavior)
                self._buffer_add(p, sub_ids, sub_vecs, sub_tags)
                self.admit_buffer_count += len(sub_ids)

    def _buffer_add(self, p: int, ids_arr: np.ndarray, vecs: np.ndarray,
                     tags: list[str]):
        n_new = len(ids_arr)
        cur = self.buf_size[p]
        end = cur + n_new
        if end > self.buf_capacity_per_part:
            new_cap = max(end, self.buf_capacity_per_part * 2)
            nv = np.empty((new_cap, self.dim), dtype=np.float32)
            ni = np.empty(new_cap, dtype=np.int64)
            na = np.zeros(new_cap, dtype=np.bool_)
            nv[:cur] = self.buf_vecs[p][:cur]
            ni[:cur] = self.buf_ids[p][:cur]
            na[:cur] = self.buf_alive[p][:cur]
            self.buf_vecs[p] = nv
            self.buf_ids[p] = ni
            self.buf_alive[p] = na
            self.buf_capacity_per_part = max(self.buf_capacity_per_part, new_cap)
        self.buf_vecs[p][cur:end] = vecs
        self.buf_ids[p][cur:end] = ids_arr
        self.buf_alive[p][cur:end] = True
        for off, (vid, tag) in enumerate(zip(ids_arr.tolist(), tags)):
            self.life_table[vid] = VectorLifeState(vid, self.global_op + off, p, "buffer", tag)
            self._buffer_pos[vid] = (p, cur + off)
        self.buf_size[p] = end
        self.global_op += n_new

    def _direct_add(self, p: int, ids_arr: np.ndarray, vecs: np.ndarray,
                     tags: list[str]):
        vecs_c = np.ascontiguousarray(vecs)
        self.graphs[p].add(vecs_c, ids_arr)
        for off, (vid, v, tag) in enumerate(zip(ids_arr.tolist(), vecs_c, tags)):
            self.life_table[vid] = VectorLifeState(vid, self.global_op + off, p, "graph", tag)
            self._graph_vecs[vid] = v
            self.part_inserted[p] += 1
        self._update_centroid_running(p, vecs_c)
        self.global_op += n_new if False else len(ids_arr)

    def delete(self, ids):
        for i in ids:
            id_int = int(i)
            state = self.life_table.get(id_int)
            if state is None:
                continue
            state.delete_op_id = self.global_op
            lifetime = self.global_op - state.insert_op_id
            self.global_recent_lifetimes.append(lifetime)
            self.part_recent_lifetimes[state.owner_partition_id].append(lifetime)
            if state.lifetime_tag:
                self._tag_lifetimes[state.lifetime_tag].append(lifetime)

            if state.placement == "buffer":
                bp_pos = self._buffer_pos.get(id_int)
                if bp_pos is not None:
                    p, pos = bp_pos
                    if self.buf_alive[p][pos]:
                        self.buf_alive[p][pos] = False
                    del self._buffer_pos[id_int]
            elif state.placement == "graph":
                self.graphs[state.owner_partition_id].mark_deleted(id_int)
                self.part_deleted_in_graph[state.owner_partition_id] += 1
                # Hot tier is a parallel cache — drop entry too if present
                self.hot_tier_ids.discard(id_int)
                self.hot_tier_vecs.pop(id_int, None)
            del self.life_table[id_int]
            self.global_op += 1

    def maintain(self):
        for p in range(self.k):
            if self.buf_size[p] == 0:
                continue
            alive_mask = self.buf_alive[p][:self.buf_size[p]]
            if not alive_mask.any():
                self.buf_size[p] = 0
                self.buf_alive[p][:] = False
                continue
            ids_alive = self.buf_ids[p][:self.buf_size[p]][alive_mask]
            vecs_alive = self.buf_vecs[p][:self.buf_size[p]][alive_mask]

            if self.use_gamma_migration and self.part_recent_lifetimes[p]:
                # Only migrate vectors whose predicted-remaining-life > threshold
                migrate_idx = []
                for i, vid in enumerate(ids_alive.tolist()):
                    state = self.life_table.get(vid)
                    if state is None:
                        continue
                    age = self.global_op - state.insert_op_id
                    pred = self._predict_lifetime_for(p, state.lifetime_tag)
                    remaining = pred - age
                    # If the vector is expected to live well past now, migrate; else hold
                    if remaining >= self.gamma_migration_threshold * pred * 0.5:
                        migrate_idx.append(i)
                    else:
                        self.gamma_migration_skip_count += 1
                if migrate_idx:
                    sub_ids = ids_alive[migrate_idx]
                    sub_vecs = np.ascontiguousarray(vecs_alive[migrate_idx])
                    self.graphs[p].add(sub_vecs, sub_ids)
                    for vid, v in zip(sub_ids.tolist(), sub_vecs):
                        st = self.life_table.get(vid)
                        if st is not None:
                            st.placement = "graph"
                            self._graph_vecs[vid] = v
                        self._buffer_pos.pop(vid, None)
                        self.part_inserted[p] += 1
                # The rest stay in buffer (we DON'T clear buffer entries that didn't migrate)
                # Implementation: rebuild compact buffer with un-migrated
                kept_idx = [i for i in range(len(ids_alive)) if i not in set(migrate_idx)]
                if kept_idx:
                    self.buf_size[p] = len(kept_idx)
                    self.buf_ids[p][:self.buf_size[p]] = ids_alive[kept_idx]
                    self.buf_vecs[p][:self.buf_size[p]] = vecs_alive[kept_idx]
                    self.buf_alive[p][:self.buf_size[p]] = True
                    self.buf_alive[p][self.buf_size[p]:] = False
                    # Update positions
                    for new_pos, vid in enumerate(ids_alive[kept_idx].tolist()):
                        self._buffer_pos[vid] = (p, new_pos)
                else:
                    self.buf_size[p] = 0
                    self.buf_alive[p][:] = False
            else:
                # Default: migrate everything alive
                self.graphs[p].add(np.ascontiguousarray(vecs_alive), ids_alive)
                for vid, v in zip(ids_alive.tolist(), vecs_alive):
                    st = self.life_table.get(vid)
                    if st is not None:
                        st.placement = "graph"
                        self._graph_vecs[vid] = v
                    self._buffer_pos.pop(vid, None)
                    self.part_inserted[p] += 1
                self.buf_size[p] = 0
                self.buf_alive[p][:] = False

        if self.use_tombstone_rebuild:
            self._maybe_rebuild()
        if self.use_hot_tier:
            self._refresh_hot_tier()

    def _maybe_rebuild(self):
        for p in range(self.k):
            denom = max(1, self.part_inserted[p])
            tomb_frac = self.part_deleted_in_graph[p] / denom
            n_alive = self.part_inserted[p] - self.part_deleted_in_graph[p]
            if n_alive < self.min_alive_for_rebuild:
                continue
            if tomb_frac < self.rebuild_threshold:
                continue
            # Rebuild this partition from its alive graph_vecs
            alive_ids = [vid for vid, st in self.life_table.items()
                          if st.owner_partition_id == p and st.placement == "graph"]
            if len(alive_ids) < self.min_alive_for_rebuild:
                continue
            alive_arr = np.array(alive_ids, dtype=np.int64)
            alive_vecs = np.stack([self._graph_vecs[vid] for vid in alive_ids])
            self.graphs[p] = self.backend_factory(p)
            self.graphs[p].add(np.ascontiguousarray(alive_vecs), alive_arr)
            # Drop only entries that are dead (no longer in life_table) — keep
            # everything alive whether placement is graph (in this or any other
            # partition) or hot (parallel cache, vector still in graph too).
            self._graph_vecs = {vid: v for vid, v in self._graph_vecs.items()
                                if vid in self.life_table}
            self.part_inserted[p] = len(alive_ids)
            self.part_deleted_in_graph[p] = 0
            self.rebuild_count += 1

    def _refresh_hot_tier(self):
        # Hot tier is a PARALLEL CACHE: vectors stay in graph too. Don't change
        # placement; just maintain a separate dict of hot copies. (Old design
        # changed placement to "hot" which then confused the rebuild filter
        # — see bug fix in 2026-05-10.)
        candidates = sorted(
            (s for s in self.life_table.values()
             if s.placement == "graph" and s.query_hit_count >= self.hot_tier_promote_threshold),
            key=lambda s: s.query_hit_count, reverse=True
        )[:self.hot_tier_size]
        new_hot_ids = {s.id for s in candidates}
        # demote old hot (just drop from cache; vector still in graph)
        for vid in list(self.hot_tier_ids):
            if vid not in new_hot_ids:
                self.hot_tier_ids.discard(vid)
                self.hot_tier_vecs.pop(vid, None)
        # promote new hot (just add copy to cache; vector still in graph)
        for s in candidates:
            if s.id not in self.hot_tier_ids:
                self.hot_tier_ids.add(s.id)
                if s.id in self._graph_vecs:
                    self.hot_tier_vecs[s.id] = self._graph_vecs[s.id]
                    self.hot_promote_count += 1

    def search(self, queries, k):
        Q = len(queries)
        # Decide per-query partition list
        if self.k == 1 or self.m_search >= self.k or not self._centroid_initialized:
            partitions_per_query = [list(range(self.k)) for _ in range(Q)]
        else:
            q_sq = (queries * queries).sum(axis=1, keepdims=True)
            c_sq = (self.centroids * self.centroids).sum(axis=1)[None, :]
            d2 = q_sq + c_sq - 2.0 * (queries @ self.centroids.T)
            top_idx = np.argpartition(d2, self.m_search, axis=1)[:, :self.m_search]
            partitions_per_query = top_idx.tolist()

        result_labels = np.full((Q, k), -1, dtype=np.int64)
        result_dists = np.full((Q, k), np.inf, dtype=np.float32)

        # Hot tier brute-force scan (vectorized over Q)
        hot_d = None
        hot_ids_arr = None
        if self.use_hot_tier and self.hot_tier_vecs:
            hot_ids_list = list(self.hot_tier_vecs)
            hot_ids_arr = np.array(hot_ids_list, dtype=np.int64)
            hot_vec_arr = np.stack([self.hot_tier_vecs[vid] for vid in hot_ids_list])
            q_sq2 = (queries * queries).sum(axis=1, keepdims=True)
            v_sq2 = (hot_vec_arr * hot_vec_arr).sum(axis=1)
            cross = queries @ hot_vec_arr.T
            hot_d = q_sq2 + v_sq2[None, :] - 2.0 * cross  # (Q, n_hot)

        for q_i in range(Q):
            cands_labels = []
            cands_dists = []
            q_single = queries[q_i:q_i + 1]
            # Hot tier (if any) — already computed
            if hot_d is not None:
                cands_labels.append(hot_ids_arr)
                cands_dists.append(hot_d[q_i])
            # Graph partitions
            for p in partitions_per_query[q_i]:
                try:
                    lbls, dsts = self.graphs[p].search(q_single, k)
                    cands_labels.append(lbls[0])
                    cands_dists.append(dsts[0])
                except Exception:
                    pass
                # Buffer scan in this partition
                if self.buf_size[p] > 0:
                    am = self.buf_alive[p][:self.buf_size[p]]
                    if am.any():
                        bv = self.buf_vecs[p][:self.buf_size[p]][am]
                        bi = self.buf_ids[p][:self.buf_size[p]][am]
                        v_sq = (bv * bv).sum(axis=1)
                        cross = q_single @ bv.T
                        bd = (q_single * q_single).sum(axis=1, keepdims=True) + v_sq[None, :] - 2.0 * cross
                        cands_labels.append(bi)
                        cands_dists.append(bd[0])
            if not cands_labels:
                continue
            all_l = np.concatenate(cands_labels)
            all_d = np.concatenate(cands_dists)
            valid = all_l >= 0
            if not valid.any():
                continue
            all_l = all_l[valid]
            all_d = all_d[valid]
            # Dedupe by id (a vector promoted to hot tier ALSO lives in the
            # graph; without dedup it'd appear twice in top-k and crowd out
            # other neighbors). Keep the smallest distance per id.
            if len(all_l) > k:
                # Sort by distance asc, then take unique-by-id keeping first
                sort_idx = np.argsort(all_d)
                sorted_l = all_l[sort_idx]
                sorted_d = all_d[sort_idx]
                _, unique_idx = np.unique(sorted_l, return_index=True)
                # Preserve sort order (np.unique returns first occurrence in sorted_l)
                unique_idx_sorted = np.sort(unique_idx)
                all_l = sorted_l[unique_idx_sorted]
                all_d = sorted_d[unique_idx_sorted]
            kk = min(k, len(all_l))
            order = np.argpartition(all_d, kk - 1)[:kk] if kk < len(all_d) else np.arange(len(all_d))
            order = order[np.argsort(all_d[order])]
            result_labels[q_i, :kk] = all_l[order]
            result_dists[q_i, :kk] = all_d[order]
            # Update query_hit_count for top-k results
            if self.use_hot_tier:
                for vid in result_labels[q_i, :kk]:
                    if vid >= 0:
                        st = self.life_table.get(int(vid))
                        if st is not None:
                            st.query_hit_count += 1
        return result_labels, result_dists
