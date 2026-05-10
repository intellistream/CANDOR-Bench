"""Cluster-correlated lifetime workload generator.

The hypothesis under test (see e35): when a vector's lifetime depends
on its cluster identity (e.g. spam vs. verified accounts in embedding
space), per-cluster lifetime EMA is a usable signal for an admission
controller. Existing patterns in `workloads.py` have no lifetime
structure, which is why we couldn't show value for the C++ team's
lifetime-aware modules in e27/e30.

Design
------
1. Run kmeans(K) on the first `init_n` rows of `data` to fix
   `K` ground-truth centroids.
2. Assign every vector its ground-truth cluster_id by nearest centroid.
3. Mark `K // 2` clusters as "long-lived" (mean lifetime
   = `long_mean_ops` ops) and the rest as "short-lived" (mean lifetime
   = `short_mean_ops` ops).
4. At insert time of vector v in cluster c, sample
       lifetime ~ Exp(1 / mean_lifetime_ops[c])
   and schedule the death at op = insert_op + lifetime.
5. At each `run_workload` step, the delete_fn returned here:
     - schedules deaths for any newly-seen alive IDs
     - returns IDs whose scheduled death has elapsed (capped at n_to_del
       to fit the run_workload contract; surplus deaths get returned
       in subsequent steps).

Note we use *vector id* as a proxy for op_id, since `run_workload`
inserts in id order. This means a vector with id=12000 was inserted
at op 12000.
"""
from __future__ import annotations
import heapq
import numpy as np
import faiss


def assign_to_centroids(vecs: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return cluster_id for each row of vecs (nearest centroid)."""
    c_sq = (centroids * centroids).sum(axis=1)
    v_sq = (vecs * vecs).sum(axis=1, keepdims=True)
    cross = vecs @ centroids.T
    d = v_sq + c_sq[None, :] - 2.0 * cross
    return np.argmin(d, axis=1)


def fit_workload_centroids(data: np.ndarray, K: int, sample_n: int = 20_000,
                           seed: int = 1) -> np.ndarray:
    """faiss kmeans on first `sample_n` rows of data → (K, d) centroids."""
    sample = np.ascontiguousarray(data[:min(sample_n, len(data))], dtype=np.float32)
    km = faiss.Kmeans(data.shape[1], K, niter=20, verbose=False, seed=seed)
    km.train(sample)
    return km.centroids.astype(np.float32, copy=False)


def make_cluster_lifetime_pattern(*, data: np.ndarray, K: int = 16,
                                  long_mean_ops: float = 50_000.0,
                                  short_mean_ops: float = 5_000.0,
                                  fraction_long: float = 0.5,
                                  centroid_seed: int = 1,
                                  workload_seed: int = 42):
    """Returns (delete_fn, info) where delete_fn is run_workload-compatible.

    info dict contains:
      'centroids', 'id_to_cluster', 'cluster_is_long', 'cluster_mean_ops'
    """
    rng = np.random.default_rng(workload_seed)
    n, d = data.shape
    centroids = fit_workload_centroids(data, K, seed=centroid_seed)
    id_to_cluster = assign_to_centroids(data, centroids)  # (n,)

    n_long = int(round(K * fraction_long))
    rng_perm = np.random.default_rng(centroid_seed)
    cluster_perm = rng_perm.permutation(K)
    long_clusters = set(int(c) for c in cluster_perm[:n_long])
    cluster_is_long = np.array([c in long_clusters for c in range(K)], dtype=bool)
    cluster_mean_ops = np.where(cluster_is_long, long_mean_ops, short_mean_ops)

    # State that persists across delete_fn calls
    scheduled: set[int] = set()
    deaths: list[tuple[int, int]] = []  # min-heap of (death_op, vec_id)
    backlog: list[int] = []              # overflow queue if n_to_del cap

    def delete_fn(alive_set, n_to_del, rng_unused, data_arg, deleted_so_far,
                  batch_idx=0, **kwargs):
        # Schedule deaths for newly-seen alive vectors
        # (alive_set is a Python set of int, possibly large — only iterate the diff)
        new_ids = alive_set - scheduled
        for vid in new_ids:
            c = int(id_to_cluster[vid])
            mean = float(cluster_mean_ops[c])
            life = max(1, int(np.ceil(rng.exponential(mean))))
            heapq.heappush(deaths, (vid + life, vid))
        scheduled.update(new_ids)

        # Current op_id ≈ max alive id (we insert in id order)
        cur_op = max(alive_set) if alive_set else 0

        # Drain due deaths into a return list, capped at n_to_del
        out: list[int] = []
        # First, drain backlog
        while backlog and len(out) < n_to_del:
            vid = backlog.pop(0)
            if vid in alive_set:
                out.append(vid)
        # Then, due deaths from heap
        while deaths and deaths[0][0] <= cur_op:
            _, vid = heapq.heappop(deaths)
            if vid not in alive_set:
                continue
            if len(out) < n_to_del:
                out.append(vid)
            else:
                backlog.append(vid)

        return np.asarray(out, dtype=np.int64)

    info = {
        "centroids": centroids,
        "id_to_cluster": id_to_cluster,
        "cluster_is_long": cluster_is_long,
        "cluster_mean_ops": cluster_mean_ops,
        "long_mean_ops": long_mean_ops,
        "short_mean_ops": short_mean_ops,
        "K": K,
    }
    return delete_fn, info
