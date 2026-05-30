#!/usr/bin/env python3
# gen_multicluster_total.py
# Generate base.bin & query.bin with K>=3 clusters and sparse bridges,
# given a *total* vector count N (e.g., 1_000_000).

import argparse
import math
from pathlib import Path

import numpy as np


# ---------- I/O formats ----------
def save_annbin(path, X: np.ndarray):
    """annbin: [uint32 n, uint32 d] header, then row-major float32 data"""
    X = np.asarray(X, dtype=np.float32, order="C")
    n, d = X.shape
    with open(path, "wb") as f:
        f.write(np.uint32(n).tobytes())
        f.write(np.uint32(d).tobytes())
        f.write(X.tobytes())


def save_fvecs(path, X: np.ndarray):
    """fvecs: per-row [int32 d][d x float32]"""
    X = np.asarray(X, dtype=np.float32, order="C")
    d = np.int32(X.shape[1])
    with open(path, "wb") as f:
        for row in X:
            f.write(d.tobytes())
            f.write(row.tobytes())


# ---------- math helpers ----------
def two_norm_sq(A, B):
    """Squared L2 distances between rows of A (m,d) and B (n,d) -> (m,n)"""
    A = np.asarray(A, dtype=np.float32, order="C")
    B = np.asarray(B, dtype=np.float32, order="C")
    a2 = np.sum(A * A, axis=1, keepdims=True)
    b2 = np.sum(B * B, axis=1, keepdims=True).T
    return a2 + b2 - 2.0 * (A @ B.T)


def gt_topk_indices(X, Q, k=10, batch=1000):
    """Brute-force top-k (indices) for each q in Q against base X (heavy for huge N)."""
    tops = []
    for i in range(0, Q.shape[0], batch):
        q = Q[i : i + batch]
        D = two_norm_sq(q, X)  # (b, n)
        part = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(part.shape[0])[:, None]
        dpart = D[rows, part]
        order = np.argsort(dpart, axis=1)
        topk = part[rows, order]
        tops.append(topk)
    return np.vstack(tops)


# ---------- cluster sizing ----------
def cluster_sizes_from_total(
    N, K, mode="equal", alpha=1.0, zipf_s=1.2, seed=0, min_per=1
):
    rng = np.random.default_rng(seed)
    if mode == "equal":
        base = [N // K] * K
        for i in range(N % K):
            base[i] += 1
        return np.array(base, dtype=np.int64)
    elif mode == "dirichlet":
        p = rng.dirichlet(alpha * np.ones(K, dtype=np.float64))
        raw = np.floor(N * p).astype(np.int64)
        # Distribute leftovers to the largest residuals
        leftover = N - raw.sum()
        if leftover > 0:
            order = np.argsort(-p)  # descending
            raw[order[:leftover]] += 1
        raw = np.maximum(raw, min_per)
        # fix any overflow from min_per
        extra = raw.sum() - N
        if extra > 0:
            order = np.argsort(raw)  # take from biggest first (reverse at end)
            idx = order[::-1]
            for j in idx:
                take = min(extra, raw[j] - min_per)
                raw[j] -= take
                extra -= take
                if extra == 0:
                    break
        return raw
    elif mode == "zipf":
        ranks = np.arange(1, K + 1, dtype=np.float64)
        w = 1.0 / (ranks**zipf_s)
        w /= w.sum()
        raw = np.floor(N * w).astype(np.int64)
        leftover = N - raw.sum()
        if leftover > 0:
            order = np.argsort(-w)
            raw[order[:leftover]] += 1
        raw = np.maximum(raw, min_per)
        extra = raw.sum() - N
        if extra > 0:
            order = np.argsort(raw)
            idx = order[::-1]
            for j in idx:
                take = min(extra, raw[j] - min_per)
                raw[j] -= take
                extra -= take
                if extra == 0:
                    break
        return raw
    else:
        raise ValueError("mode must be one of: equal, dirichlet, zipf")


# ---------- data generation ----------
def gen_k_clusters_with_sparse_bridges_from_sizes(
    sizes,
    d=2,
    sep=10.0,
    sigma=0.5,
    bridges_per_pair=2,
    bridge_sigma=0.15,
    seed=42,
    layout="ring",
):
    """
    Returns:
      X: (N, d) float32 base vectors
      labels: (N,) int32 cluster id in [0..K-1] (bridges labeled as 'left' cluster id)
      meta: dict with centers, pairs, cluster index ranges, bridge indices
    """
    rng = np.random.default_rng(seed)
    K = len(sizes)

    # Cluster centers on a circle in first 2 dims
    centers = np.zeros((K, d), dtype=np.float32)
    for k in range(K):
        theta = 2.0 * math.pi * k / K
        centers[k, 0] = sep * math.cos(theta)
        centers[k, 1] = sep * math.sin(theta)

    data = []
    labels = []
    idx_clusters = []
    start = 0
    for k in range(K):
        pts = rng.normal(loc=centers[k], scale=sigma, size=(int(sizes[k]), d)).astype(
            np.float32
        )
        data.append(pts)
        labels.append(np.full(pts.shape[0], k, np.int32))
        idx_clusters.append(np.arange(start, start + pts.shape[0]))
        start += pts.shape[0]

    # Bridges between pairs (sparse)
    if layout == "ring":
        pairs = [(k, (k + 1) % K) for k in range(K)]
    elif layout == "chain":
        pairs = [(k, k + 1) for k in range(K - 1)]
    elif layout == "star":
        hub = 0
        pairs = [(hub, k) for k in range(1, K)]
    else:
        pairs = [(k, (k + 1) % K) for k in range(K)]  # default ring

    bridge_points = []
    bridge_labels = []
    for a, b in pairs:
        mid = (centers[a] + centers[b]) / 2.0
        B = rng.normal(loc=mid, scale=bridge_sigma, size=(bridges_per_pair, d)).astype(
            np.float32
        )
        bridge_points.append(B)
        bridge_labels.append(np.full(B.shape[0], a, np.int32))  # tag to 'a'

    if bridge_points:
        B = np.vstack(bridge_points)
        Lb = np.concatenate(bridge_labels)
        X = np.vstack([*data, B]).astype(np.float32, order="C")
        labels = np.concatenate([*labels, Lb])
        idx_bridge = np.arange(X.shape[0] - B.shape[0], X.shape[0])
    else:
        X = np.vstack(data).astype(np.float32, order="C")
        labels = np.concatenate(labels)
        idx_bridge = np.array([], dtype=np.int64)

    meta = {
        "centers": centers,
        "pairs": pairs,
        "idx_clusters": idx_clusters,
        "idx_bridge": idx_bridge,
    }
    return X, labels.astype(np.int32), meta


def boundary_queries_from_cluster(
    centers,
    source_cluster=0,
    neighbor_cluster=1,
    nq=1000,
    sigma=0.5,
    t_low=0.6,
    t_high=0.9,
    sigma_q_ratio=0.4,
    d=None,
    seed=123,
):
    """Sample queries near boundary toward a neighbor."""
    rng = np.random.default_rng(seed)
    if d is None:
        d = centers.shape[1]
    mu_s = centers[source_cluster]
    mu_n = centers[neighbor_cluster]
    t = rng.uniform(t_low, t_high, size=(nq, 1)).astype(np.float32)
    dirv = (mu_n - mu_s)[None, :]
    eps = rng.normal(loc=0.0, scale=sigma_q_ratio * sigma, size=(nq, d)).astype(
        np.float32
    )
    Q = mu_s[None, :] + t * dirv + eps
    return Q


# ---------- query filtering (optional) ----------
def filter_queries_cross_cluster(
    Q,
    X,
    labels,
    source_cluster,
    k=10,
    min_out_frac=0.3,
    mode="none",
    sample_size=20000,
    seed=0,
):
    """
    Filter queries whose GT/sampled top-k has >= min_out_frac outside source_cluster.
    mode:
      - 'none'     : no filtering (fastest; fine for huge N)
      - 'sampled'  : top-k against a random subset of size sample_size
      - 'bf'       : full brute-force (heavy; avoid for huge N)
    """
    if mode == "none":
        mask = np.ones(Q.shape[0], dtype=bool)
        out_frac = np.zeros(Q.shape[0], dtype=np.float32)
        return Q, mask, out_frac

    rng = np.random.default_rng(seed)
    if mode == "sampled":
        n = X.shape[0]
        sample_size = min(sample_size, n)
        idx = rng.choice(n, size=sample_size, replace=False)
        Xs = X[idx]
        ls = labels[idx]
        topk = gt_topk_indices(Xs, Q, k=k)
        lbl_topk = ls[topk]
    elif mode == "bf":
        topk = gt_topk_indices(X, Q, k=k)
        lbl_topk = labels[topk]
    else:
        raise ValueError("filter mode must be one of: none, sampled, bf")

    out_frac = np.mean(lbl_topk != source_cluster, axis=1)
    mask = out_frac >= min_out_frac
    return Q[mask], mask, out_frac


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Generate base.bin and query.bin from total N with K clusters + sparse bridges."
    )
    # Output
    ap.add_argument("--outdir", type=str, default=".", help="output directory")
    ap.add_argument(
        "--name-prefix",
        type=str,
        default=None,
        help="filename prefix (defaults to '<K>cluster')",
    )
    ap.add_argument("--format", type=str, choices=["annbin", "fvecs"], default="annbin")

    # Data size & shape
    ap.add_argument(
        "--N",
        type=int,
        required=True,
        help="total number of base vectors (e.g., 1000000)",
    )
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument(
        "--K", type=int, default=3, help="number of clusters (>=3 recommended)"
    )

    # Cluster size distribution
    ap.add_argument(
        "--size-mode", type=str, choices=["equal", "dirichlet", "zipf"], default="equal"
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Dirichlet alpha (lower => more imbalance)",
    )
    ap.add_argument("--zipf-s", type=float, default=1.2, help="Zipf exponent")

    # Geometry & noise
    ap.add_argument("--sep", type=float, default=15.0, help="cluster center separation")
    ap.add_argument("--sigma", type=float, default=0.6, help="cluster stddev")
    ap.add_argument("--bridges-per-pair", type=int, default=2)
    ap.add_argument("--bridge-sigma", type=float, default=0.2)
    ap.add_argument(
        "--layout", type=str, choices=["ring", "chain", "star"], default="ring"
    )

    # Queries
    ap.add_argument(
        "--q-source", type=int, default=0, help="cluster id to draw queries from"
    )
    ap.add_argument(
        "--q-neighbor", type=int, default=1, help="neighbor cluster id to bias toward"
    )
    ap.add_argument("--nq", type=int, default=10000)
    ap.add_argument("--t-low", type=float, default=0.6)
    ap.add_argument("--t-high", type=float, default=0.9)
    ap.add_argument("--sigma-q-ratio", type=float, default=0.4)

    # Filtering (optional, can be heavy)
    ap.add_argument(
        "--filter", type=str, choices=["none", "sampled", "bf"], default="none"
    )
    ap.add_argument("--k", type=int, default=10, help="top-k for filtering")
    ap.add_argument("--min-out-frac", type=float, default=0.3)
    ap.add_argument("--filter-sample", type=int, default=20000)

    # Seeds
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qseed", type=int, default=123)

    args = ap.parse_args()

    def count_pairs(k: int, layout: str) -> int:
        if layout == "ring":
            return k
        if layout == "chain":
            return max(k - 1, 0)
        if layout == "star":
            return max(k - 1, 0)
        return k

    pair_count = count_pairs(args.K, args.layout)
    total_bridges = pair_count * args.bridges_per_pair
    if args.N < total_bridges:
        raise ValueError(
            f"N={args.N} is too small to accommodate {total_bridges} bridge points; "
            "increase N or reduce --bridges-per-pair."
        )

    # Compute per-cluster sizes from total N
    sizes = cluster_sizes_from_total(
        N=args.N - total_bridges,
        K=args.K,
        mode=args.size_mode,
        alpha=args.alpha,
        zipf_s=args.zipf_s,
        seed=args.seed,
        min_per=1,
    )
    if sizes.sum() != args.N - total_bridges:
        raise RuntimeError("Internal error: cluster size computation mismatch.")

    # Build base with sparse bridges
    X, labels, meta = gen_k_clusters_with_sparse_bridges_from_sizes(
        sizes=sizes,
        d=args.d,
        sep=args.sep,
        sigma=args.sigma,
        bridges_per_pair=args.bridges_per_pair,
        bridge_sigma=args.bridge_sigma,
        seed=args.seed,
        layout=args.layout,
    )

    # Queries (boundary-biased from one cluster)
    neighbor = args.q_neighbor % args.K
    if neighbor == args.q_source:
        neighbor = (args.q_source + 1) % args.K

    Q0 = boundary_queries_from_cluster(
        centers=meta["centers"],
        source_cluster=args.q_source,
        neighbor_cluster=neighbor,
        nq=args.nq,
        sigma=args.sigma,
        t_low=args.t_low,
        t_high=args.t_high,
        sigma_q_ratio=args.sigma_q_ratio,
        d=args.d,
        seed=args.qseed,
    )

    # Optional filtering to ensure cross-cluster reliance
    Q, mask, out_frac = filter_queries_cross_cluster(
        Q0,
        X,
        labels,
        source_cluster=args.q_source,
        k=args.k,
        min_out_frac=args.min_out_frac,
        mode=args.filter,
        sample_size=args.filter_sample,
        seed=args.qseed,
    )

    prefix = args.name_prefix or f"{args.K}cluster_d{args.d}"
    base_filename = (
        f"{prefix}_base.bin" if args.format == "annbin" else f"{prefix}_base.fvecs"
    )
    query_filename = (
        f"{prefix}_query.bin" if args.format == "annbin" else f"{prefix}_query.fvecs"
    )
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    base_path = f"{args.outdir}/{base_filename}"
    query_path = f"{args.outdir}/{query_filename}"
    if args.format == "annbin":
        save_annbin(base_path, X)
        save_annbin(query_path, Q)
    else:
        save_fvecs(base_path, X)
        save_fvecs(query_path, Q)

    # Meta for reproducibility
    meta_path = f"{args.outdir}/{prefix}_meta.npz"
    np.savez(
        meta_path,
        sizes=sizes,
        labels=labels,
        centers=meta["centers"],
        idx_bridge=meta["idx_bridge"],
        pairs=np.array(meta["pairs"], dtype=np.int32),
        q_mask=mask if mask is not None else np.ones(Q.shape[0], dtype=bool),
        q_out_frac=out_frac,
    )

    N_total = X.shape[0]
    bytes_data = N_total * args.d * 4
    print(f"[ok] base: {X.shape}  (~{bytes_data/1e9:.2f} GB floats)")
    print(f"[ok] query: {Q.shape}")
    print(f"[ok] wrote {base_path} and {query_path}")
    print(f"[hint] {meta_path} has sizes/labels/centers/bridges and query filter info.")


if __name__ == "__main__":
    main()
