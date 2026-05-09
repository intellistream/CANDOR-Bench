"""SIFT dataset loading + ground-truth computation with /tmp caching."""
import os
import numpy as np
from datasets.registry import get_dataset


def load_sift(n_queries: int = 1000, slice_n: int | None = None):
    """Load SIFT 1M dataset. Optionally slice to first `slice_n` vectors."""
    ds = get_dataset("sift")
    data = np.asarray(ds.get_dataset(), dtype=np.float32)
    if slice_n is not None:
        data = data[:slice_n]
    queries = np.asarray(ds.get_queries(), dtype=np.float32)[:n_queries]
    return data, queries


def compute_gt(data: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    """Brute-force k-NN ground truth via blocked NumPy."""
    nq = queries.shape[0]
    qn = (queries**2).sum(1, keepdims=True)
    gt = np.empty((nq, k), dtype=np.uint32)
    block = 50000
    dn_full = (data**2).sum(1)
    for i0 in range(0, nq, 100):
        i1 = min(i0 + 100, nq)
        q = queries[i0:i1]
        qni = qn[i0:i1]
        best_dist = np.full((i1 - i0, k), np.inf, dtype=np.float32)
        best_idx = np.full((i1 - i0, k), -1, dtype=np.int64)
        for j0 in range(0, data.shape[0], block):
            j1 = min(j0 + block, data.shape[0])
            d2 = qni + dn_full[j0:j1] - 2 * (q @ data[j0:j1].T)
            combined_dist = np.concatenate([best_dist, d2.astype(np.float32)], axis=1)
            combined_idx = np.concatenate(
                [best_idx, np.broadcast_to(np.arange(j0, j1, dtype=np.int64),
                                            (i1 - i0, j1 - j0))],
                axis=1,
            )
            order = np.argpartition(combined_dist, k, axis=1)[:, :k]
            row_idx = np.arange(i1 - i0)[:, None]
            best_dist = combined_dist[row_idx, order]
            best_idx = combined_idx[row_idx, order]
        for r in range(i1 - i0):
            o = np.argsort(best_dist[r])
            gt[i0 + r] = best_idx[r, o].astype(np.uint32)
    return gt


def cached_gt(data: np.ndarray, queries: np.ndarray, k: int = 10, tag: str = "default"):
    """Compute GT once, cache in /tmp keyed by `tag` + sizes."""
    cache = f"/tmp/sift_gt_{tag}_n{len(data)}_q{len(queries)}_k{k}.npy"
    if os.path.exists(cache):
        return np.load(cache)
    gt = compute_gt(data, queries, k)
    np.save(cache, gt)
    return gt


def gt_for_surviving(data: np.ndarray, queries: np.ndarray, deleted_so_far: int, k: int = 10):
    """GT against the surviving slice data[deleted_so_far:], offset back to original ids."""
    surviving = data[deleted_so_far:]
    gt = compute_gt(surviving, queries, k)
    return (gt + deleted_so_far).astype(np.uint32)
