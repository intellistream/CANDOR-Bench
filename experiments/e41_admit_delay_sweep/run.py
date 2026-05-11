"""e41 — true cost-asymmetry boundary via injected admit delay.

e40's M-sweep failed to identify a boundary because varying M changes
BOTH admit cost and search cost, leaving cost ratio approximately
constant.

This experiment uses a wrapper backend that adds a tunable delay to
each insert call (via busy-loop, not sleep — to maintain CPU
behavior). This isolates c_g_admit so we can sweep cost asymmetry
independently:
  - delay = 0 μs:    base HNSW M=16 (~600 μs admit)
  - delay = -300 μs: simulated cheaper backend (~300 μs effective) — N/A
  - delay = 500 μs:  ~1100 μs effective admit
  - delay = 1500 μs: ~2100 μs effective admit
  - delay = 5000 μs: ~5600 μs effective admit (DiskANN-style slow)

Critically: search latency UNCHANGED. So we move ALONG the cost-
asymmetry axis without confounds.

Hypothesis: V13 lifetime admission's penalty grows linearly with
extra delay. The cost-asymmetry framework predicts:
  V13_total - V4_total = N × p × delay - constant_savings
where N=200K, p≈0.19 (≈33K direct admits).

Each μs of extra delay → 33000 μs ≈ 33 ms extra V13 cost.
At delay=5000 μs → ~165 s extra V13 cost. Should massively swing
the balance.

Variants per delay level:
  V4 baseline (no admission)
  V13 lifetime + both fixes
"""
import os, sys, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import _shared
from _shared import load_dataset, run_workload
from _shared.backends import HnswlibBackend, GraphBackend
from _shared.buffers import ClusterBuffer
from _shared.router_modular import ModularRouter
from _shared.workloads_lifetime import (
    make_cluster_lifetime_pattern, fit_workload_centroids
)


class DelayedBackend(GraphBackend):
    """Wraps HnswlibBackend, adds busy-loop delay per insert call.
    Used to sweep effective insert cost independently of search cost.
    """
    def __init__(self, dim, max_elements, M=16, ef_c=200, ef_s=80,
                 num_threads=1, insert_delay_us=0.0):
        self.inner = HnswlibBackend(dim, max_elements, M=M, ef_c=ef_c,
                                    ef_s=ef_s, num_threads=num_threads)
        self.insert_delay_us = float(insert_delay_us)

    def _busy_wait(self, seconds: float):
        if seconds <= 0:
            return
        # Busy-loop avoids Python sleep granularity issues at <1ms
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            pass

    def add(self, vecs, ids):
        n = len(ids)
        # Charge per-vector delay
        if self.insert_delay_us > 0:
            self._busy_wait(n * self.insert_delay_us * 1e-6)
        return self.inner.add(vecs, ids)

    def mark_deleted(self, vid):
        return self.inner.mark_deleted(vid)

    def search(self, queries, k):
        return self.inner.search(queries, k)

    def size(self):
        return self.inner.size() if hasattr(self.inner, "size") else 0

    @property
    def n(self):
        return self.inner.n if hasattr(self.inner, "n") else 0


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
    print(f"[e41] long clusters: {sorted(long_clusters)}", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), "output_admit_delay_sweep.json")
    rows = []
    def write():
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    buf_cap = batch * 50

    # Sweep injected admit delay
    DELAY_VALUES_US = [0, 500, 1500, 3000, 6000]
    N_SEEDS = 3

    for delay in DELAY_VALUES_US:
        print(f"\n========== insert_delay={delay} μs (effective admit ≈ {600 + delay} μs)", flush=True)

        def factory(delay=delay):
            return DelayedBackend(d, max_elements=n, M=16, ef_c=200, insert_delay_us=delay)

        for seed in range(SEED, SEED + N_SEEDS):
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
            rows.append({"variant": "V4", "delay_us": delay, "seed": seed,
                         "total_s": v4_total, "recall": r["recall"], "admit_graph": 0})
            write()
            print(f"  seed={seed} V4: {v4_total:.1f}s rec={r['recall']:.4f}", flush=True)

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
            rows.append({"variant": "V13", "delay_us": delay, "seed": seed,
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
