"""e43 — multi-thread sanity test for V4 vs V2.

The whole paper uses single-thread for fair comparison (HNSW's
default to all-cores would silently mask algorithm differences).
Reviewers will ask: does the V4 cluster+rebuild advantage hold
under multi-thread?

Test: SIFT 200K, random pattern, n_threads ∈ {1, 4, 8, 16}.

Hypothesis: V4 vs V2 ratio is approximately invariant under
multi-thread, because both routers parallelize the same way
(per-batch). HNSW's internal parallelism may amplify ABSOLUTE
times but should not change the relative cost of V2 vs V4.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, run_workload, make_pattern
from _shared.backends import HnswlibBackend
from _shared.buffers import FlatBuffer, ClusterBuffer
from _shared.router_modular import ModularRouter


# Note: we explicitly DO NOT set OMP_NUM_THREADS — let hnswlib use
# its configured thread count. The HnswlibBackend constructor still
# defaults to num_threads=1; we override per-run.

def main():
    THREADS = [1, 4, 8, 16]
    SEED = 42
    slice_n, init_n, batch, qstride, n_queries = 200_000, 20_000, 2_500, 10_000, 500
    data, queries = load_dataset("sift", n_queries=n_queries, slice_n=slice_n)
    n, d = data.shape

    rows = []
    out_path = os.path.join(os.path.dirname(__file__), "output_multi_thread.json")
    def write():
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    for nt in THREADS:
        print(f"\n========== num_threads={nt}", flush=True)
        # Set thread env BEFORE building anything — these get cached
        # by NumPy/MKL at import. Hnswlib does respect set_num_threads().

        def factory(nt=nt):
            return HnswlibBackend(d, max_elements=n, M=16, ef_c=200,
                                  num_threads=nt)

        # V2 flat + rebuild
        delete_fn = make_pattern("random")
        buf = FlatBuffer(d, init_capacity=batch * 50)
        g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                          rebuild_threshold=0.5, backend_factory=factory)
        r = run_workload(g, "V2", data, queries, init_n, batch, qstride, delete_fn,
                         use_gamma=True, has_mark_deleted=False, seed=SEED)
        v2_t = r["total_s"]
        rows.append({"num_threads": nt, "variant": "V2_flat_rebuild",
                     "total_s": v2_t, "recall": r["recall"]}); write()
        print(f"  V2: {v2_t:.1f}s rec={r['recall']:.4f}", flush=True)

        # V4 cluster + rebuild
        delete_fn = make_pattern("random")
        buf = ClusterBuffer(d, n_clusters=16,
                            init_capacity_per_cluster=max(batch * 50 // 16, 64),
                            n_probe=4)
        g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                          rebuild_threshold=0.5, backend_factory=factory)
        r = run_workload(g, "V4", data, queries, init_n, batch, qstride, delete_fn,
                         use_gamma=True, has_mark_deleted=False, seed=SEED)
        v4_t = r["total_s"]
        delta = (v4_t - v2_t) / v2_t * 100
        rows.append({"num_threads": nt, "variant": "V4_cluster_rebuild",
                     "total_s": v4_t, "recall": r["recall"], "delta_vs_V2": delta}); write()
        print(f"  V4: {v4_t:.1f}s ({delta:+.1f}% vs V2) rec={r['recall']:.4f}", flush=True)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
