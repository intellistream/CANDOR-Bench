"""e39 — Construct a workload where lifetime admission CAN win.

Cost equation from COST_ANALYSIS.md says lifetime admission wins
iff:
    Q × time_in_buf × c_buf_search × (n_probe/K) > c_g_admit

In e37/e38 (default workload, queries every qstride=10K) per-vector
expected query count is too low (~625 queries × 0.175 μs ≈ 109 μs)
to offset c_g_admit ≈ 600 μs. So admission breaks even or loses.

This experiment constructs a READ-HEAVY workload by issuing more
queries per search call:
  - default e38: 500 queries/call, qstride=10K → 1 search per 10K
                 inserts × 500 = 500 queries per 10K inserts (1:20)
  - e39: 5000 queries/call, qstride=10K → 5000 queries per 10K
         inserts (1:2 — much more read-heavy)

Plus we use biased queries (80% to long clusters), so long clusters
get truly hammered.

If admission still doesn't win at 10× query rate, then the boundary
is even further out.

Variants:
  V4    cluster + rebuild (baseline)
  V10   lifetime_admission no fix (e37 baseline; admission = 0)
  V13   lifetime_admission + Fix A + Fix B (best version)
  V13_full_probe   V13 with n_probe = K (full buffer scan)
                   — to isolate "is the limitation partial-probe or admission?"
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import _shared
from _shared import load_dataset, run_workload
from _shared.backends import HnswlibBackend
from _shared.buffers import ClusterBuffer
from _shared.router_modular import ModularRouter
from _shared.workloads_lifetime import (
    make_cluster_lifetime_pattern, fit_workload_centroids
)


def make_biased_queries(data, n_queries, K, long_clusters, centroids,
                        long_query_fraction=0.95, seed=42):
    """Construct n_queries vectors heavily weighted toward long-lived
    cluster centroids."""
    rng = np.random.default_rng(seed)
    queries = np.empty((n_queries, data.shape[1]), dtype=np.float32)
    n_long = int(round(n_queries * long_query_fraction))
    long_arr = sorted(long_clusters)
    for i in range(n_long):
        c = long_arr[rng.integers(len(long_arr))]
        queries[i] = centroids[c] + rng.standard_normal(data.shape[1]).astype(np.float32) * 5.0
    cold_idx = rng.choice(len(data), size=n_queries - n_long, replace=False)
    queries[n_long:] = data[cold_idx]
    rng.shuffle(queries)
    return queries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--rebuild_threshold", type=float, default=0.5)
    p.add_argument("--long_mean_ops", type=float, default=50_000.0)
    p.add_argument("--short_mean_ops", type=float, default=5_000.0)
    p.add_argument("--admission_horizon", type=float, default=20_000.0)
    p.add_argument("--long_query_fraction", type=float, default=0.95)
    p.add_argument("--n_queries", type=int, default=5000,
                   help="queries per search call (default 5000 = 10× e38)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    slice_n, init_n, batch, qstride = 200_000, 20_000, 2_500, 10_000
    data, _ = load_dataset("sift", n_queries=10, slice_n=slice_n)
    n, d = data.shape
    K = args.K

    centroids = fit_workload_centroids(data, K, seed=1)
    rng_perm = np.random.default_rng(1)
    long_clusters = set(int(c) for c in rng_perm.permutation(K)[:K // 2])
    queries = make_biased_queries(data, args.n_queries, K, long_clusters, centroids,
                                  long_query_fraction=args.long_query_fraction,
                                  seed=args.seed)
    print(f"[e39] long clusters: {sorted(long_clusters)}", flush=True)
    print(f"[e39] queries: {args.n_queries}, "
          f"long_query_fraction: {args.long_query_fraction}", flush=True)

    factory = lambda: HnswlibBackend(d, max_elements=n)
    buf_cap = batch * 50
    out_path = os.path.join(os.path.dirname(__file__),
                            f"output_sift_read_heavy_K{K}_q{args.n_queries}_200K.json")
    rows = []
    def write():
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    def run_variant(label, builder):
        print(f"\n========== {label}", flush=True)
        df, info = make_cluster_lifetime_pattern(
            data=data, K=K,
            long_mean_ops=args.long_mean_ops,
            short_mean_ops=args.short_mean_ops,
            workload_seed=args.seed)
        idx = builder()
        r = run_workload(idx, label, data, queries, init_n, batch, qstride, df,
                         use_gamma=isinstance(idx, ModularRouter),
                         has_mark_deleted=not isinstance(idx, ModularRouter),
                         seed=args.seed)
        extras = {}
        if isinstance(idx, ModularRouter):
            extras = {
                "rebuild_count": int(idx.rebuild_count),
                "admit_to_graph": int(idx.admit_to_graph_count),
                "admit_to_buffer": int(idx.admit_to_buffer_count),
            }
        rows.append({"variant": label, **extras, **r})
        write()
        a_g = extras.get("admit_to_graph", 0)
        print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} "
              f"qry_p95={r.get('query_latency_ms_p95', 0):.3f}ms admit_graph={a_g}", flush=True)

    # V4 baseline
    run_variant("V4_cluster_rebuild", lambda: ModularRouter(
        ClusterBuffer(d, n_clusters=K, init_capacity_per_cluster=max(buf_cap // K, 64), n_probe=4),
        factory(), d, maintain_strategy="rebuild",
        rebuild_threshold=args.rebuild_threshold, backend_factory=factory))

    # V10 (no fix — e37 baseline; admission never triggers)
    run_variant("V10_admission_no_fix", lambda: ModularRouter(
        ClusterBuffer(d, n_clusters=K, init_capacity_per_cluster=max(buf_cap // K, 64), n_probe=4),
        factory(), d, maintain_strategy="rebuild",
        rebuild_threshold=args.rebuild_threshold, backend_factory=factory,
        use_lifetime_admission=True, admission_horizon=args.admission_horizon))

    # V13 with both fixes
    run_variant("V13_admission_both_fixes", lambda: ModularRouter(
        ClusterBuffer(d, n_clusters=K, init_capacity_per_cluster=max(buf_cap // K, 64), n_probe=4),
        factory(), d, maintain_strategy="rebuild",
        rebuild_threshold=args.rebuild_threshold, backend_factory=factory,
        use_lifetime_admission=True, admission_horizon=args.admission_horizon,
        use_alive_age_for_admission=True, track_migrated_lifetimes=True))

    # V13_full_probe = V13 but search probes all K clusters
    # — isolates "is partial-probe the bottleneck for admission?"
    run_variant("V13_full_probe", lambda: ModularRouter(
        ClusterBuffer(d, n_clusters=K, init_capacity_per_cluster=max(buf_cap // K, 64), n_probe=K),
        factory(), d, maintain_strategy="rebuild",
        rebuild_threshold=args.rebuild_threshold, backend_factory=factory,
        use_lifetime_admission=True, admission_horizon=args.admission_horizon,
        use_alive_age_for_admission=True, track_migrated_lifetimes=True))

    # V4_full_probe baseline for comparison
    run_variant("V4_full_probe", lambda: ModularRouter(
        ClusterBuffer(d, n_clusters=K, init_capacity_per_cluster=max(buf_cap // K, 64), n_probe=K),
        factory(), d, maintain_strategy="rebuild",
        rebuild_threshold=args.rebuild_threshold, backend_factory=factory))

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
