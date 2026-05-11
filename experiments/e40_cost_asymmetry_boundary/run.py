"""e40 — cost-asymmetry boundary: at what backend cost does lifetime win?

The cost-asymmetry framework (COST_ANALYSIS.md) predicts that
lifetime admission wins iff:
    Q × time_in_buf × c_buf_search × p > c_g_admit

By varying HNSW M (which scales c_g_admit roughly linearly), we can
move along the cost asymmetry axis and find the BOUNDARY where
lifetime starts winning.

Hypothesis:
- M=32 (c_g_admit ≈ 1200μs): V13 << V4 (huge loss)
- M=16 (c_g_admit ≈ 600μs): V13 ≈ V4 (break-even, e39 result)
- M=8  (c_g_admit ≈ 250μs): V13 ≈ V4 or slight win
- M=4  (c_g_admit ≈ 100μs): V13 > V4 (clear win for lifetime)

If hypothesis confirmed → cost-asymmetry framework is PREDICTIVE.
Boundary identified: lifetime requires backend with c_g_admit < 200μs.

This is the difference between "we tried, it didn't work" and
"we have a predictive theory; here's when each design choice matters".
"""
import os, sys, json
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
    K = 16
    SEED = 42
    rebuild_threshold = 0.5
    long_mean_ops, short_mean_ops = 50_000, 5_000
    admission_horizon = 20_000
    slice_n, init_n, batch, qstride = 200_000, 20_000, 2_500, 10_000
    N_QUERIES = 10_000  # read-heavy from e39
    N_SEEDS = 3  # multi-seed to control variance

    data, _ = load_dataset("sift", n_queries=10, slice_n=slice_n)
    n, d = data.shape
    centroids = fit_workload_centroids(data, K, seed=1)
    rng = np.random.default_rng(1)
    long_clusters = set(int(c) for c in rng.permutation(K)[:K // 2])
    print(f"[e40] long clusters: {sorted(long_clusters)}", flush=True)
    print(f"[e40] N_QUERIES={N_QUERIES}, N_SEEDS={N_SEEDS}", flush=True)
    queries = make_biased_queries(data, N_QUERIES, K, long_clusters, centroids, seed=SEED)

    out_path = os.path.join(os.path.dirname(__file__), "output_boundary.json")
    rows = []
    def write():
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    buf_cap = batch * 50

    M_VALUES = [4, 8, 16, 32]

    for M in M_VALUES:
        # ef_construction scales with M to keep recall comparable
        ef_c = max(80, M * 8)
        print(f"\n========== HNSW M={M}, ef_c={ef_c}", flush=True)

        def factory(M=M, ef_c=ef_c):
            return HnswlibBackend(d, max_elements=n, M=M, ef_c=ef_c)

        for seed in range(SEED, SEED + N_SEEDS):
            # V4 baseline
            df, _ = make_cluster_lifetime_pattern(data=data, K=K,
                long_mean_ops=long_mean_ops, short_mean_ops=short_mean_ops,
                workload_seed=seed)
            buf = ClusterBuffer(d, n_clusters=K,
                                init_capacity_per_cluster=max(buf_cap // K, 64),
                                n_probe=4)
            g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                              rebuild_threshold=rebuild_threshold,
                              backend_factory=factory)
            r = run_workload(g, "V4", data, queries, init_n, batch, qstride, df,
                             use_gamma=True, has_mark_deleted=False, seed=seed)
            v4_total = r["total_s"]
            rows.append({"variant": "V4", "M": M, "ef_c": ef_c, "seed": seed,
                         "total_s": v4_total, "recall": r["recall"], "admit_graph": 0})
            write()
            print(f"  seed={seed} V4: {v4_total:.1f}s rec={r['recall']:.4f}", flush=True)

            # V13 lifetime + both fixes
            df, _ = make_cluster_lifetime_pattern(data=data, K=K,
                long_mean_ops=long_mean_ops, short_mean_ops=short_mean_ops,
                workload_seed=seed)
            buf = ClusterBuffer(d, n_clusters=K,
                                init_capacity_per_cluster=max(buf_cap // K, 64),
                                n_probe=4)
            g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                              rebuild_threshold=rebuild_threshold,
                              backend_factory=factory,
                              use_lifetime_admission=True,
                              admission_horizon=admission_horizon,
                              admission_min_observations=5,
                              use_alive_age_for_admission=True,
                              track_migrated_lifetimes=True)
            r = run_workload(g, "V13", data, queries, init_n, batch, qstride, df,
                             use_gamma=True, has_mark_deleted=False, seed=seed)
            delta = (r["total_s"] - v4_total) / v4_total * 100
            rows.append({"variant": "V13", "M": M, "ef_c": ef_c, "seed": seed,
                         "total_s": r["total_s"], "recall": r["recall"],
                         "admit_graph": int(g.admit_to_graph_count),
                         "delta_pct_vs_V4": delta})
            write()
            mark = "★" if delta < -3 else ("✓" if delta < 0 else "")
            print(f"  seed={seed} V13: {r['total_s']:.1f}s ({delta:+.1f}%) "
                  f"admit={g.admit_to_graph_count} rec={r['recall']:.4f} {mark}", flush=True)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
