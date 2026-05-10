"""e34 — How sensitive is cluster+rebuild to K (number of cluster buckets)?

Hypothesis 1 (sanity): K=1 should approximately match flat+rebuild.
Hypothesis 2: There's a sweet spot for K in {16, 64} where buffer search
              latency drops (smaller probed set) but centroid quality
              is still meaningful.
Hypothesis 3: Very large K (256) hurts because (a) k-means init cost
              grows, (b) per-cluster overhead outweighs probe savings.

Variant:  cluster + rebuild on SIFT 200K random pattern, K ∈ K_LIST.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, run_workload, make_pattern
from _shared.backends import HnswlibBackend
from _shared.buffers import FlatBuffer, ClusterBuffer
from _shared.router_modular import ModularRouter


K_LIST = [1, 4, 16, 64, 256]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="200K", choices=["200K", "1M"])
    p.add_argument("--pattern", default="random",
                   choices=["sequential", "random", "cluster", "partial_reset"])
    p.add_argument("--rebuild_threshold", type=float, default=0.5)
    p.add_argument("--seeds", default="42")
    args = p.parse_args()

    if args.scale == "200K":
        slice_n, init_n, batch, qstride, n_queries = 200_000, 20_000, 2_500, 10_000, 500
    else:
        slice_n, init_n, batch, qstride, n_queries = 1_000_000, 100_000, 10_000, 50_000, 1000

    data, queries = load_dataset("sift", n_queries=n_queries, slice_n=slice_n)
    n, d = data.shape
    delete_fn = make_pattern(args.pattern)
    rows = []
    out_path = os.path.join(os.path.dirname(__file__),
                            f"output_sift_{args.pattern}_{args.scale}.json")
    seeds = [int(s) for s in args.seeds.split(",")]
    factory = lambda: HnswlibBackend(d, max_elements=n)
    buf_cap = batch * 50

    for seed in seeds:
        # Reference: flat + rebuild (the "K=0" point)
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} reference: flat + rebuild", flush=True)
        buf = FlatBuffer(d, init_capacity=buf_cap)
        g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                          rebuild_threshold=args.rebuild_threshold,
                          backend_factory=factory)
        r = run_workload(g, "flat_rebuild", data, queries, init_n, batch, qstride, delete_fn,
                         use_gamma=True, has_mark_deleted=False, seed=seed)
        rows.append({"K": 0, "n_probe": 0, "pattern": args.pattern, "scale": args.scale,
                     "seed": seed, "rebuild_count": int(g.rebuild_count), **r})
        print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} rebuilds={g.rebuild_count}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        for K in K_LIST:
            n_probe = min(K, max(1, K // 4 if K >= 4 else 1))
            print(f"\n========== {args.pattern}/{args.scale} seed={seed} cluster K={K} n_probe={n_probe}", flush=True)
            buf = ClusterBuffer(d, n_clusters=K,
                                init_capacity_per_cluster=max(buf_cap // K, 64),
                                n_probe=n_probe)
            g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                              rebuild_threshold=args.rebuild_threshold,
                              backend_factory=factory)
            r = run_workload(g, f"cluster_K{K}_rebuild", data, queries, init_n, batch, qstride, delete_fn,
                             use_gamma=True, has_mark_deleted=False, seed=seed)
            rows.append({"K": K, "n_probe": n_probe, "pattern": args.pattern, "scale": args.scale,
                         "seed": seed, "rebuild_count": int(g.rebuild_count), **r})
            print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} rebuilds={g.rebuild_count}", flush=True)
            with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
