"""e37 — Lifetime-aware admission on hot-vector workload.

THE question (PhD-level): does per-cluster lifetime-aware admission
EVER win, given a workload that should favor it?

Workload designed to favor lifetime admission:
  * Cluster-correlated lifetime (8 long, 8 short) — same as e35
  * Biased queries: 80% of queries are sampled near long-lived
    cluster centroids. So:
    - long-lived clusters get queried HEAVILY (~10× more per cluster)
    - hot vectors live in those clusters (predictable, stable)
    - short-lived clusters barely touched
  * High churn: 1:1 insert:delete
  * SIFT 200K, K=16

If V9/V10 don't win HERE, lifetime admission is dominated everywhere.

Variants
--------
V0  hnswlib direct           : reference
V2  flat + rebuild            : strong baseline
V4  cluster + rebuild         : e33 champion (no lifetime)
V9  cluster + lifetime ADMISSION + lazy flush  ★ new
V10 cluster + lifetime ADMISSION + rebuild     ★ new combo
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import _shared
from _shared import load_dataset, run_workload
from _shared.backends import HnswlibBackend
from _shared.buffers import FlatBuffer, ClusterBuffer
from _shared.router_modular import ModularRouter
from _shared.workloads_lifetime import (
    make_cluster_lifetime_pattern, fit_workload_centroids, assign_to_centroids
)


def make_biased_queries(data: np.ndarray, n_queries: int, K: int,
                        long_clusters: set, centroids: np.ndarray,
                        long_query_fraction: float = 0.8,
                        seed: int = 42):
    """Synthesize a query set where `long_query_fraction` of queries
    fall near long-lived cluster centroids; the rest are uniform-random.

    Mimics a hot-vector workload: heavy traffic against the long-lived
    region of embedding space, light traffic against the short-lived
    region.
    """
    rng = np.random.default_rng(seed)
    queries = np.empty((n_queries, data.shape[1]), dtype=np.float32)
    n_long = int(round(n_queries * long_query_fraction))
    long_arr = sorted(long_clusters)
    # Long queries: pick a long centroid + small random perturbation
    for i in range(n_long):
        c = long_arr[rng.integers(len(long_arr))]
        q = centroids[c] + rng.standard_normal(data.shape[1]).astype(np.float32) * 5.0
        queries[i] = q
    # Cold queries: pick from data uniformly (samples random regions)
    cold_idx = rng.choice(len(data), size=n_queries - n_long, replace=False)
    queries[n_long:] = data[cold_idx]
    rng.shuffle(queries)
    return queries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="200K", choices=["200K"])
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--rebuild_threshold", type=float, default=0.5)
    p.add_argument("--long_mean_ops", type=float, default=50_000.0)
    p.add_argument("--short_mean_ops", type=float, default=5_000.0)
    p.add_argument("--admission_horizon", type=float, default=20_000.0,
                   help="cluster_avg_lifetime above this → direct to graph at admit")
    p.add_argument("--long_query_fraction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    slice_n, init_n, batch, qstride, n_queries = 200_000, 20_000, 2_500, 10_000, 500
    data, _ = load_dataset("sift", n_queries=10, slice_n=slice_n)  # we'll override queries
    n, d = data.shape
    K = args.K

    # Pre-fit centroids and identify long vs short clusters
    centroids = fit_workload_centroids(data, K, seed=1)
    rng_perm = np.random.default_rng(1)
    cluster_perm = rng_perm.permutation(K)
    n_long = K // 2
    long_clusters = set(int(c) for c in cluster_perm[:n_long])

    queries = make_biased_queries(data, n_queries, K, long_clusters, centroids,
                                  long_query_fraction=args.long_query_fraction,
                                  seed=args.seed)
    print(f"[e37] biased queries built: {int(n_queries * args.long_query_fraction)} hot / "
          f"{n_queries - int(n_queries * args.long_query_fraction)} cold", flush=True)
    print(f"[e37] long clusters: {sorted(long_clusters)}", flush=True)

    factory = lambda: HnswlibBackend(d, max_elements=n)
    buf_cap = batch * 50

    out_path = os.path.join(os.path.dirname(__file__),
                            f"output_sift_lifetime_zipf_K{K}_{args.scale}.json")
    rows = []

    def write():
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    def make_delete_fn():
        df, info = make_cluster_lifetime_pattern(
            data=data, K=K,
            long_mean_ops=args.long_mean_ops,
            short_mean_ops=args.short_mean_ops,
            centroid_seed=1, workload_seed=args.seed)
        return df

    # V0 hnswlib direct
    print(f"\n========== V0 hnswlib direct", flush=True)
    delete_fn = make_delete_fn()
    be = factory()
    r = run_workload(be, "V0_hnswlib_direct", data, queries, init_n, batch, qstride,
                     delete_fn, use_gamma=False, has_mark_deleted=True, seed=args.seed)
    rows.append({"variant": "V0_hnswlib_direct", **r}); write()
    print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f}", flush=True)

    # V2 flat + rebuild
    print(f"\n========== V2 flat + rebuild", flush=True)
    delete_fn = make_delete_fn()
    buf = FlatBuffer(d, init_capacity=buf_cap)
    g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                      rebuild_threshold=args.rebuild_threshold,
                      backend_factory=factory)
    r = run_workload(g, "V2_flat_rebuild", data, queries, init_n, batch, qstride,
                     delete_fn, use_gamma=True, has_mark_deleted=False, seed=args.seed)
    rows.append({"variant": "V2_flat_rebuild", "rebuild_count": int(g.rebuild_count), **r}); write()
    print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} rebuilds={g.rebuild_count}", flush=True)

    # V4 cluster + rebuild
    print(f"\n========== V4 cluster + rebuild", flush=True)
    delete_fn = make_delete_fn()
    buf = ClusterBuffer(d, n_clusters=K, init_capacity_per_cluster=max(buf_cap // K, 64), n_probe=4)
    g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                      rebuild_threshold=args.rebuild_threshold,
                      backend_factory=factory)
    r = run_workload(g, "V4_cluster_rebuild", data, queries, init_n, batch, qstride,
                     delete_fn, use_gamma=True, has_mark_deleted=False, seed=args.seed)
    rows.append({"variant": "V4_cluster_rebuild", "K": K, "rebuild_count": int(g.rebuild_count), **r}); write()
    print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} rebuilds={g.rebuild_count}", flush=True)

    # V9 cluster + lifetime ADMISSION + lazy_flush (no rebuild)
    print(f"\n========== V9 cluster + lifetime_admission + lazy_flush", flush=True)
    delete_fn = make_delete_fn()
    buf = ClusterBuffer(d, n_clusters=K, init_capacity_per_cluster=max(buf_cap // K, 64), n_probe=4)
    g = ModularRouter(buf, factory(), d, maintain_strategy="lazy_flush",
                      use_lifetime_admission=True,
                      admission_horizon=args.admission_horizon,
                      admission_min_observations=5)
    r = run_workload(g, "V9_lifetime_admission", data, queries, init_n, batch, qstride,
                     delete_fn, use_gamma=True, has_mark_deleted=False, seed=args.seed)
    rows.append({"variant": "V9_lifetime_admission", "K": K,
                 "admit_to_graph": int(g.admit_to_graph_count),
                 "admit_to_buffer": int(g.admit_to_buffer_count), **r}); write()
    print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} "
          f"admit_graph={g.admit_to_graph_count} admit_buf={g.admit_to_buffer_count}", flush=True)

    # V10 V9 + rebuild
    print(f"\n========== V10 cluster + lifetime_admission + rebuild", flush=True)
    delete_fn = make_delete_fn()
    buf = ClusterBuffer(d, n_clusters=K, init_capacity_per_cluster=max(buf_cap // K, 64), n_probe=4)
    g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                      rebuild_threshold=args.rebuild_threshold,
                      backend_factory=factory,
                      use_lifetime_admission=True,
                      admission_horizon=args.admission_horizon,
                      admission_min_observations=5)
    r = run_workload(g, "V10_lifetime_admission_rebuild", data, queries, init_n, batch, qstride,
                     delete_fn, use_gamma=True, has_mark_deleted=False, seed=args.seed)
    rows.append({"variant": "V10_lifetime_admission_rebuild", "K": K,
                 "rebuild_count": int(g.rebuild_count),
                 "admit_to_graph": int(g.admit_to_graph_count),
                 "admit_to_buffer": int(g.admit_to_buffer_count), **r}); write()
    print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} rebuilds={g.rebuild_count} "
          f"admit_graph={g.admit_to_graph_count} admit_buf={g.admit_to_buffer_count}", flush=True)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
