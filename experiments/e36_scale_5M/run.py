"""e36 — Scaling test: SIFT-style synthetic random 5M.

The cluster+rebuild win at 1M is −22 to −42% vs flat+rebuild on each
delete pattern. Does it hold at 5M?

We use synthetic random vectors (Normal(0, 1), L2-normalized) at the
same dim=128 as SIFT, because:
1. SIFT only ships 1M, and 100M experiments would take days.
2. The mechanism we test (rebuild cost grows with N; cluster-buffer
   amortizes per-step overhead) doesn't depend on data structure —
   it only depends on insert/delete/search arithmetic and graph
   topology.
3. Synthetic data gives us reproducible scaling without dataset
   licensing or download time.

Variants run (skip slow ones at scale):
  V0 hnswlib_direct  — reference
  V2 flat + rebuild   — strong baseline
  V4 cluster + rebuild — e33 champion
  V6 cluster + cluster_rebuild + refit centroids — e35 champion

(V1, V3, V5 omitted: V1/V3 are no-rebuild variants we already know
lose; V5 in_place is the slowest and we have its 1M result.)
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import _shared
from _shared import run_workload, make_pattern
from _shared.backends import HnswlibBackend
from _shared.buffers import FlatBuffer, ClusterBuffer
from _shared.router_modular import ModularRouter


def gen_synthetic(n: int, d: int = 128, seed: int = 1):
    """Synthetic random vectors, L2-normalized for cosine-like geometry."""
    rng = np.random.default_rng(seed)
    print(f"[gen] generating {n} × {d} synthetic vectors...", flush=True)
    t0 = time.time()
    data = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data /= np.maximum(norms, 1e-9)
    print(f"[gen] done in {time.time() - t0:.1f}s, dtype={data.dtype}, shape={data.shape}", flush=True)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5_000_000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--n_queries", type=int, default=200,
                   help="few queries to keep GT computation tractable at 5M")
    p.add_argument("--pattern", default="random",
                   choices=["sequential", "random", "cluster", "partial_reset"])
    p.add_argument("--n_clusters", type=int, default=16)
    p.add_argument("--rebuild_threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    n, d = args.n, args.dim
    init_n = max(50_000, n // 50)            # 1/50th initial bulk
    batch = max(20_000, n // 200)            # ~200 steps
    qstride = max(100_000, n // 20)
    K = args.n_clusters

    data = gen_synthetic(n, d, seed=1)
    queries = gen_synthetic(args.n_queries, d, seed=2)
    print(f"[setup] n={n} d={d} init_n={init_n} batch={batch} qstride={qstride} K={K}", flush=True)

    delete_fn = make_pattern(args.pattern)
    factory = lambda: HnswlibBackend(d, max_elements=n)
    buf_cap = batch * 50

    out_path = os.path.join(os.path.dirname(__file__),
                            f"output_random_{args.pattern}_K{K}_n{n}.json")
    rows = []

    def write():
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # --- V0 hnswlib direct ---
    print(f"\n========== n={n} {args.pattern} V0 hnswlib direct", flush=True)
    be = factory()
    r = run_workload(be, "V0_hnswlib_direct", data, queries, init_n, batch, qstride, delete_fn,
                     use_gamma=False, has_mark_deleted=True, seed=args.seed)
    rows.append({"variant": "V0_hnswlib_direct", "n": n, "pattern": args.pattern, "K": 0, **r})
    print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f}", flush=True)
    write()

    # --- V2 flat + rebuild ---
    print(f"\n========== n={n} {args.pattern} V2 flat + rebuild", flush=True)
    buf = FlatBuffer(d, init_capacity=buf_cap)
    g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                      rebuild_threshold=args.rebuild_threshold,
                      backend_factory=factory)
    r = run_workload(g, "V2_flat_rebuild", data, queries, init_n, batch, qstride, delete_fn,
                     use_gamma=True, has_mark_deleted=False, seed=args.seed)
    rows.append({"variant": "V2_flat_rebuild", "n": n, "pattern": args.pattern, "K": 0,
                 "rebuild_count": int(g.rebuild_count), **r})
    print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} rebuilds={g.rebuild_count}", flush=True)
    write()

    # --- V4 cluster + rebuild ---
    print(f"\n========== n={n} {args.pattern} V4 cluster K={K} + rebuild", flush=True)
    buf = ClusterBuffer(d, n_clusters=K,
                        init_capacity_per_cluster=max(buf_cap // K, 64),
                        n_probe=4)
    g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                      rebuild_threshold=args.rebuild_threshold,
                      backend_factory=factory)
    r = run_workload(g, "V4_cluster_rebuild", data, queries, init_n, batch, qstride, delete_fn,
                     use_gamma=True, has_mark_deleted=False, seed=args.seed)
    rows.append({"variant": "V4_cluster_rebuild", "n": n, "pattern": args.pattern, "K": K,
                 "rebuild_count": int(g.rebuild_count), **r})
    print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} rebuilds={g.rebuild_count}", flush=True)
    write()

    # --- V6 cluster + cluster_rebuild ---
    print(f"\n========== n={n} {args.pattern} V6 cluster K={K} + cluster_rebuild (refit)", flush=True)
    buf = ClusterBuffer(d, n_clusters=K,
                        init_capacity_per_cluster=max(buf_cap // K, 64),
                        n_probe=4)
    g = ModularRouter(buf, factory(), d, maintain_strategy="cluster_rebuild",
                      rebuild_threshold=args.rebuild_threshold,
                      backend_factory=factory)
    r = run_workload(g, "V6_cluster_full_rebuild", data, queries, init_n, batch, qstride, delete_fn,
                     use_gamma=True, has_mark_deleted=False, seed=args.seed)
    rows.append({"variant": "V6_cluster_full_rebuild", "n": n, "pattern": args.pattern, "K": K,
                 "rebuild_count": int(g.rebuild_count), **r})
    print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} rebuilds={g.rebuild_count}", flush=True)
    write()

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
