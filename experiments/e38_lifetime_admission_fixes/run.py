"""e38 — Re-test lifetime admission with both EMA-fixes.

e37 finding: per-cluster lifetime EMA can't see long lifetimes
because it only observes absorbed deletes, and long-lived vecs get
migrated out before they die. Result: admission never triggered.

Two architectural fixes implemented in router_modular.py + cluster.py:
  Fix A: cluster_avg_lifetime_with_alive() — augment EMA with
    (current_op - insert_op) for currently-alive vecs as lower bounds.
  Fix B: track_migrated_lifetimes — when migrated vec dies in graph,
    contribute its lifetime to the original cluster's EMA.

Same workload as e37 (cluster-correlated lifetime, biased queries).

Variants:
  V4    cluster + rebuild                          (e33 champion baseline)
  V10   cluster + lifetime_admission + rebuild     (broken admission, e37)
  V11   V10 + Fix A
  V12   V10 + Fix B
  V13   V10 + Fix A + Fix B
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import _shared
from _shared import load_dataset, run_workload
from _shared.backends import HnswlibBackend
from _shared.buffers import FlatBuffer, ClusterBuffer
from _shared.router_modular import ModularRouter
from _shared.workloads_lifetime import (
    make_cluster_lifetime_pattern, fit_workload_centroids
)


def make_biased_queries(data, n_queries, K, long_clusters, centroids,
                        long_query_fraction=0.8, seed=42):
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
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    slice_n, init_n, batch, qstride, n_queries = 200_000, 20_000, 2_500, 10_000, 500
    data, _ = load_dataset("sift", n_queries=10, slice_n=slice_n)
    n, d = data.shape
    K = args.K

    centroids = fit_workload_centroids(data, K, seed=1)
    rng_perm = np.random.default_rng(1)
    long_clusters = set(int(c) for c in rng_perm.permutation(K)[:K // 2])
    queries = make_biased_queries(data, n_queries, K, long_clusters, centroids,
                                  seed=args.seed)
    print(f"[e38] long clusters: {sorted(long_clusters)}", flush=True)

    factory = lambda: HnswlibBackend(d, max_elements=n)
    buf_cap = batch * 50
    out_path = os.path.join(os.path.dirname(__file__),
                            f"output_sift_lifetime_fixes_K{K}_200K.json")
    rows = []
    def write():
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # V4 baseline
    print(f"\n========== V4 cluster + rebuild (baseline)", flush=True)
    df, info = make_cluster_lifetime_pattern(data=data, K=K,
        long_mean_ops=args.long_mean_ops, short_mean_ops=args.short_mean_ops,
        workload_seed=args.seed)
    buf = ClusterBuffer(d, n_clusters=K, init_capacity_per_cluster=max(buf_cap // K, 64), n_probe=4)
    g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                      rebuild_threshold=args.rebuild_threshold, backend_factory=factory)
    r = run_workload(g, "V4_cluster_rebuild", data, queries, init_n, batch, qstride, df,
                     use_gamma=True, has_mark_deleted=False, seed=args.seed)
    rows.append({"variant": "V4_cluster_rebuild", "rebuild_count": int(g.rebuild_count), **r}); write()
    print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} rebuilds={g.rebuild_count}", flush=True)

    # V10: lifetime admission, no fixes (will admit nothing per e37)
    print(f"\n========== V10 lifetime_admission + rebuild (no fixes — e37 baseline)", flush=True)
    df, info = make_cluster_lifetime_pattern(data=data, K=K,
        long_mean_ops=args.long_mean_ops, short_mean_ops=args.short_mean_ops,
        workload_seed=args.seed)
    buf = ClusterBuffer(d, n_clusters=K, init_capacity_per_cluster=max(buf_cap // K, 64), n_probe=4)
    g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                      rebuild_threshold=args.rebuild_threshold, backend_factory=factory,
                      use_lifetime_admission=True, admission_horizon=args.admission_horizon)
    r = run_workload(g, "V10_admission_no_fix", data, queries, init_n, batch, qstride, df,
                     use_gamma=True, has_mark_deleted=False, seed=args.seed)
    rows.append({"variant": "V10_admission_no_fix", "rebuild_count": int(g.rebuild_count),
                 "admit_to_graph": int(g.admit_to_graph_count),
                 "admit_to_buffer": int(g.admit_to_buffer_count), **r}); write()
    print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} admit_graph={g.admit_to_graph_count}", flush=True)

    # V11: + Fix A only
    print(f"\n========== V11 V10 + Fix A (alive-age in EMA)", flush=True)
    df, info = make_cluster_lifetime_pattern(data=data, K=K,
        long_mean_ops=args.long_mean_ops, short_mean_ops=args.short_mean_ops,
        workload_seed=args.seed)
    buf = ClusterBuffer(d, n_clusters=K, init_capacity_per_cluster=max(buf_cap // K, 64), n_probe=4)
    g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                      rebuild_threshold=args.rebuild_threshold, backend_factory=factory,
                      use_lifetime_admission=True, admission_horizon=args.admission_horizon,
                      use_alive_age_for_admission=True)
    r = run_workload(g, "V11_admission_fix_A", data, queries, init_n, batch, qstride, df,
                     use_gamma=True, has_mark_deleted=False, seed=args.seed)
    rows.append({"variant": "V11_admission_fix_A", "rebuild_count": int(g.rebuild_count),
                 "admit_to_graph": int(g.admit_to_graph_count),
                 "admit_to_buffer": int(g.admit_to_buffer_count), **r}); write()
    print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} admit_graph={g.admit_to_graph_count}", flush=True)

    # V12: + Fix B only
    print(f"\n========== V12 V10 + Fix B (track migrated lifetimes)", flush=True)
    df, info = make_cluster_lifetime_pattern(data=data, K=K,
        long_mean_ops=args.long_mean_ops, short_mean_ops=args.short_mean_ops,
        workload_seed=args.seed)
    buf = ClusterBuffer(d, n_clusters=K, init_capacity_per_cluster=max(buf_cap // K, 64), n_probe=4)
    g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                      rebuild_threshold=args.rebuild_threshold, backend_factory=factory,
                      use_lifetime_admission=True, admission_horizon=args.admission_horizon,
                      track_migrated_lifetimes=True)
    r = run_workload(g, "V12_admission_fix_B", data, queries, init_n, batch, qstride, df,
                     use_gamma=True, has_mark_deleted=False, seed=args.seed)
    rows.append({"variant": "V12_admission_fix_B", "rebuild_count": int(g.rebuild_count),
                 "admit_to_graph": int(g.admit_to_graph_count),
                 "admit_to_buffer": int(g.admit_to_buffer_count), **r}); write()
    print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} admit_graph={g.admit_to_graph_count}", flush=True)

    # V13: BOTH fixes
    print(f"\n========== V13 V10 + Fix A + Fix B (both)", flush=True)
    df, info = make_cluster_lifetime_pattern(data=data, K=K,
        long_mean_ops=args.long_mean_ops, short_mean_ops=args.short_mean_ops,
        workload_seed=args.seed)
    buf = ClusterBuffer(d, n_clusters=K, init_capacity_per_cluster=max(buf_cap // K, 64), n_probe=4)
    g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                      rebuild_threshold=args.rebuild_threshold, backend_factory=factory,
                      use_lifetime_admission=True, admission_horizon=args.admission_horizon,
                      use_alive_age_for_admission=True, track_migrated_lifetimes=True)
    r = run_workload(g, "V13_admission_both_fixes", data, queries, init_n, batch, qstride, df,
                     use_gamma=True, has_mark_deleted=False, seed=args.seed)
    rows.append({"variant": "V13_admission_both_fixes", "rebuild_count": int(g.rebuild_count),
                 "admit_to_graph": int(g.admit_to_graph_count),
                 "admit_to_buffer": int(g.admit_to_buffer_count), **r}); write()
    print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} admit_graph={g.admit_to_graph_count}", flush=True)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
