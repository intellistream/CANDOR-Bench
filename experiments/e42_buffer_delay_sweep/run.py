"""e42 — invert cost-asymmetry: make BUFFER admit expensive.

e41 confirmed: making graph admit MORE expensive doesn't let lifetime
admission win — V13 still pays MORE delay than V4 (direct admits in
small batches don't amortize).

This experiment tests the opposite: what if BUFFER admit becomes
expensive? Then V13's "skip buffer for long-lived vecs" should
genuinely save work.

Mechanism:
  V4: every vec → buffer.add (pay buf_delay) → migrate to graph (pay graph_admit)
  V13: long vec → backend.add directly (pay graph_admit)
       short vec → buffer.add (pay buf_delay) → absorbed (no migrate)

Per-vec cost difference (long-lived):
  V4: buf_delay + graph_admit_amortized
  V13: graph_admit_full
  Savings = buf_delay + graph_admit_amortized - graph_admit_full
         ≈ buf_delay - 540 μs  (since graph_admit_amortized ≈ 60 μs, full ≈ 600 μs)

So V13 wins when buf_delay > 540 μs. Sweep tests this:
  buf_delay ∈ {0, 200, 1000, 3000} μs

Real-world analogue: buffer backed by remote store / persistent storage
where admit is non-trivial.
"""
import os, sys, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import _shared
from _shared import load_dataset, run_workload
from _shared.backends import HnswlibBackend
from _shared.buffers import ClusterBuffer, Buffer
from _shared.router_modular import ModularRouter
from _shared.workloads_lifetime import (
    make_cluster_lifetime_pattern, fit_workload_centroids
)


class DelayedClusterBuffer(ClusterBuffer):
    """ClusterBuffer with busy-wait delay per add() vec."""
    def __init__(self, *args, buf_admit_delay_us=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.buf_admit_delay_us = float(buf_admit_delay_us)

    def _busy_wait(self, seconds):
        if seconds <= 0: return
        end = time.perf_counter() + seconds
        while time.perf_counter() < end: pass

    def add(self, ids, vecs):
        n = len(ids)
        if self.buf_admit_delay_us > 0:
            self._busy_wait(n * self.buf_admit_delay_us * 1e-6)
        return super().add(ids, vecs)


def make_biased_queries(data, n_queries, K, long_clusters, centroids, seed=42):
    rng = np.random.default_rng(seed)
    queries = np.empty((n_queries, data.shape[1]), dtype=np.float32)
    n_long = int(round(n_queries * 0.95))
    long_arr = sorted(long_clusters)
    for i in range(n_long):
        c = long_arr[rng.integers(len(long_arr))]
        queries[i] = centroids[c] + rng.standard_normal(data.shape[1]).astype(np.float32) * 5.0
    cold_idx = rng.choice(len(data), size=n_queries - n_long, replace=False)
    queries[n_long:] = data[cold_idx]
    rng.shuffle(queries)
    return queries


def main():
    K, SEED = 16, 42
    rebuild_threshold = 0.5
    long_mean_ops, short_mean_ops = 50_000, 5_000
    admission_horizon = 20_000
    slice_n, init_n, batch, qstride = 200_000, 20_000, 2_500, 10_000
    N_QUERIES = 10_000

    data, _ = load_dataset("sift", n_queries=10, slice_n=slice_n)
    n, d = data.shape
    centroids = fit_workload_centroids(data, K, seed=1)
    rng = np.random.default_rng(1)
    long_clusters = set(int(c) for c in rng.permutation(K)[:K // 2])
    queries = make_biased_queries(data, N_QUERIES, K, long_clusters, centroids, seed=SEED)
    print(f"[e42] long clusters: {sorted(long_clusters)}", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), "output_buffer_delay_sweep.json")
    rows = []
    def write():
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    factory = lambda: HnswlibBackend(d, max_elements=n, M=16, ef_c=200)
    buf_cap = batch * 50

    DELAYS = [0, 200, 1000, 3000]
    N_SEEDS = 2  # fewer seeds to fit in time

    for buf_delay in DELAYS:
        print(f"\n========== buffer_delay={buf_delay} μs", flush=True)
        for seed in range(SEED, SEED + N_SEEDS):
            # V4 baseline
            df, _ = make_cluster_lifetime_pattern(data=data, K=K,
                long_mean_ops=long_mean_ops, short_mean_ops=short_mean_ops,
                workload_seed=seed)
            buf = DelayedClusterBuffer(d, n_clusters=K,
                                init_capacity_per_cluster=max(buf_cap // K, 64),
                                n_probe=4, buf_admit_delay_us=buf_delay)
            g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                              rebuild_threshold=rebuild_threshold,
                              backend_factory=factory)
            r = run_workload(g, "V4", data, queries, init_n, batch, qstride, df,
                             use_gamma=True, has_mark_deleted=False, seed=seed)
            v4 = r["total_s"]
            rows.append({"variant": "V4", "buf_delay": buf_delay, "seed": seed,
                         "total_s": v4, "recall": r["recall"], "admit_graph": 0}); write()
            print(f"  seed={seed} V4: {v4:.1f}s rec={r['recall']:.4f}", flush=True)

            # V13 lifetime + both fixes
            df, _ = make_cluster_lifetime_pattern(data=data, K=K,
                long_mean_ops=long_mean_ops, short_mean_ops=short_mean_ops,
                workload_seed=seed)
            buf = DelayedClusterBuffer(d, n_clusters=K,
                                init_capacity_per_cluster=max(buf_cap // K, 64),
                                n_probe=4, buf_admit_delay_us=buf_delay)
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
            delta = (r["total_s"] - v4) / v4 * 100
            rows.append({"variant": "V13", "buf_delay": buf_delay, "seed": seed,
                         "total_s": r["total_s"], "recall": r["recall"],
                         "admit_graph": int(g.admit_to_graph_count),
                         "delta_pct_vs_V4": delta}); write()
            mark = "★" if delta < -3 else ("✓" if delta < 0 else "")
            print(f"  seed={seed} V13: {r['total_s']:.1f}s ({delta:+.1f}%) "
                  f"admit={g.admit_to_graph_count} rec={r['recall']:.4f} {mark}", flush=True)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
