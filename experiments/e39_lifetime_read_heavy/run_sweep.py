"""e39 sweep — find a parameter regime where lifetime admission wins.

The cost equation says lifetime admission wins iff direct-admit cost
(N × p × c_g_admit) is offset by query savings (Q × N × p × c_buf_search).

We sweep on three knobs to push toward a winning regime:
  1. admission_horizon — lower threshold → more direct admits (higher p)
  2. HNSW params (M=8 vs M=16) — lower M → cheaper direct admit (lower c_g_admit)
  3. n_queries per search — higher → more savings per direct admit (higher Q)

Cross-product: 3 horizons × 2 M values × 2 n_query rates = 12 V13 configs.
Each compared against V4 baseline (8 baselines, one per M × n_q config).
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
    seed = 42
    rebuild_threshold = 0.5
    long_mean_ops, short_mean_ops = 50_000, 5_000

    slice_n, init_n, batch, qstride = 200_000, 20_000, 2_500, 10_000
    data, _ = load_dataset("sift", n_queries=10, slice_n=slice_n)
    n, d = data.shape

    centroids = fit_workload_centroids(data, K, seed=1)
    rng_perm = np.random.default_rng(1)
    long_clusters = set(int(c) for c in rng_perm.permutation(K)[:K // 2])
    print(f"[e39_sweep] long clusters: {sorted(long_clusters)}", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), "output_sweep.json")
    rows = []
    def write():
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    buf_cap = batch * 50

    HNSW_CONFIGS = [
        {"M": 8, "ef_c": 100, "label": "M8"},   # cheaper direct admit
        {"M": 16, "ef_c": 200, "label": "M16"}, # default
    ]
    N_QUERIES_LIST = [5000, 10000]
    HORIZONS = [5000, 10000, 20000]

    for hnsw_cfg in HNSW_CONFIGS:
        for n_q in N_QUERIES_LIST:
            queries = make_biased_queries(data, n_q, K, long_clusters, centroids, seed=seed)
            print(f"\n========== HNSW {hnsw_cfg['label']}, n_queries={n_q}", flush=True)

            def factory(M=hnsw_cfg["M"], ef_c=hnsw_cfg["ef_c"]):
                return HnswlibBackend(d, max_elements=n, M=M, ef_c=ef_c)

            # V4 baseline
            print(f"  V4 baseline:", end=" ", flush=True)
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
            rows.append({"variant": "V4", "M": hnsw_cfg["M"], "n_queries": n_q,
                         "horizon": None, "total_s": v4_total, "recall": r["recall"],
                         "admit_graph": 0}); write()
            print(f"{v4_total:.1f}s rec={r['recall']:.4f}", flush=True)

            # V13 sweep on horizon
            for h in HORIZONS:
                print(f"  V13 horizon={h}:", end=" ", flush=True)
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
                                  admission_horizon=h, admission_min_observations=5,
                                  use_alive_age_for_admission=True,
                                  track_migrated_lifetimes=True)
                r = run_workload(g, f"V13_h{h}", data, queries, init_n, batch, qstride, df,
                                 use_gamma=True, has_mark_deleted=False, seed=seed)
                delta = (r["total_s"] - v4_total) / v4_total * 100
                rows.append({"variant": "V13", "M": hnsw_cfg["M"], "n_queries": n_q,
                             "horizon": h, "total_s": r["total_s"], "recall": r["recall"],
                             "admit_graph": int(g.admit_to_graph_count),
                             "delta_pct_vs_V4": delta}); write()
                print(f"{r['total_s']:.1f}s ({delta:+.1f}%) admit={g.admit_to_graph_count} rec={r['recall']:.4f}", flush=True)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
