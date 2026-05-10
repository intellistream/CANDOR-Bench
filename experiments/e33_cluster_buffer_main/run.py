"""e33 — Cluster-buffer vs flat-buffer, with maintain-strategy ablation.

Question: when the buffer itself is structured (IVF-style K-bucket),
does the periodic full-rebuild trigger (our F3 finding) still beat
in-place per-cluster migration (the C++ design)?

Variants
--------
V0  hnswlib_direct          : reference (no buffer, no rebuild)
V1  flat + lazy_flush        : current GammaRouter (router.py)
V2  flat + rebuild           : current GammaRouterWithRebuild (router_with_rebuild.py)
V3  cluster + lazy_flush     : structured buffer, no rebuild
V4  cluster + rebuild        : structured buffer + periodic full rebuild
V5  cluster + in_place_migr  : structured buffer + per-cluster migrate (C++ design)
V6  cluster + cluster_rebuild: structured buffer + periodic full rebuild + refit centroids

A clean 7-way comparison on 4 delete patterns at SIFT 200K.
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, run_workload, make_pattern
from _shared.backends import HnswlibBackend
from _shared.buffers import FlatBuffer, ClusterBuffer
from _shared.router_modular import ModularRouter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="200K", choices=["200K", "1M"])
    p.add_argument("--pattern", required=True,
                   choices=["sequential", "random", "cluster", "partial_reset"])
    p.add_argument("--n_clusters", type=int, default=16,
                   help="K for ClusterBuffer (default 16)")
    p.add_argument("--n_probe", type=int, default=4,
                   help="centroids probed per query (default 4)")
    p.add_argument("--rebuild_threshold", type=float, default=0.5)
    p.add_argument("--seeds", default="42")
    p.add_argument("--dataset", default="sift",
                   choices=["sift", "msong", "glove", "random-m"],
                   help="dataset name (default sift)")
    args = p.parse_args()

    if args.scale == "200K":
        slice_n, init_n, batch, qstride, n_queries = 200_000, 20_000, 2_500, 10_000, 500
    else:
        slice_n, init_n, batch, qstride, n_queries = 1_000_000, 100_000, 10_000, 50_000, 1000

    data, queries = load_dataset(args.dataset, n_queries=n_queries, slice_n=slice_n)
    n, d = data.shape
    delete_fn = make_pattern(args.pattern)
    rows = []
    out_path = os.path.join(os.path.dirname(__file__),
                            f"output_{args.dataset}_{args.pattern}_K{args.n_clusters}_{args.scale}.json")
    seeds = [int(s) for s in args.seeds.split(",")]

    factory = lambda: HnswlibBackend(d, max_elements=n)
    buf_cap = batch * 50  # match other experiments

    for seed in seeds:
        # --- V0 hnswlib direct ---
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} V0 hnswlib direct", flush=True)
        be = factory()
        r = run_workload(be, "V0_hnswlib_direct", data, queries, init_n, batch, qstride, delete_fn,
                         use_gamma=False, has_mark_deleted=True, seed=seed)
        rows.append({"variant": "V0_hnswlib_direct", "pattern": args.pattern, "scale": args.scale,
                     "seed": seed, "K": 0, **r})
        print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # --- V1 flat + lazy_flush ---
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} V1 flat + lazy_flush", flush=True)
        buf = FlatBuffer(d, init_capacity=buf_cap)
        g = ModularRouter(buf, factory(), d, maintain_strategy="lazy_flush")
        r = run_workload(g, "V1_flat_lazy", data, queries, init_n, batch, qstride, delete_fn,
                         use_gamma=True, has_mark_deleted=False, seed=seed)
        rows.append({"variant": "V1_flat_lazy", "pattern": args.pattern, "scale": args.scale,
                     "seed": seed, "K": 0, **r})
        print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # --- V2 flat + rebuild ---
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} V2 flat + rebuild(th={args.rebuild_threshold})", flush=True)
        buf = FlatBuffer(d, init_capacity=buf_cap)
        g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                          rebuild_threshold=args.rebuild_threshold,
                          backend_factory=factory)
        r = run_workload(g, "V2_flat_rebuild", data, queries, init_n, batch, qstride, delete_fn,
                         use_gamma=True, has_mark_deleted=False, seed=seed)
        rows.append({"variant": "V2_flat_rebuild", "pattern": args.pattern, "scale": args.scale,
                     "seed": seed, "K": 0, "rebuild_count": int(g.rebuild_count), **r})
        print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} rebuilds={g.rebuild_count}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # --- V3 cluster + lazy_flush ---
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} V3 cluster K={args.n_clusters} + lazy_flush", flush=True)
        buf = ClusterBuffer(d, n_clusters=args.n_clusters,
                            init_capacity_per_cluster=max(buf_cap // args.n_clusters, 64),
                            n_probe=args.n_probe)
        g = ModularRouter(buf, factory(), d, maintain_strategy="lazy_flush")
        r = run_workload(g, "V3_cluster_lazy", data, queries, init_n, batch, qstride, delete_fn,
                         use_gamma=True, has_mark_deleted=False, seed=seed)
        rows.append({"variant": "V3_cluster_lazy", "pattern": args.pattern, "scale": args.scale,
                     "seed": seed, "K": args.n_clusters, **r})
        print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # --- V4 cluster + rebuild ---
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} V4 cluster K={args.n_clusters} + rebuild", flush=True)
        buf = ClusterBuffer(d, n_clusters=args.n_clusters,
                            init_capacity_per_cluster=max(buf_cap // args.n_clusters, 64),
                            n_probe=args.n_probe)
        g = ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                          rebuild_threshold=args.rebuild_threshold,
                          backend_factory=factory)
        r = run_workload(g, "V4_cluster_rebuild", data, queries, init_n, batch, qstride, delete_fn,
                         use_gamma=True, has_mark_deleted=False, seed=seed)
        rows.append({"variant": "V4_cluster_rebuild", "pattern": args.pattern, "scale": args.scale,
                     "seed": seed, "K": args.n_clusters, "rebuild_count": int(g.rebuild_count), **r})
        print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} rebuilds={g.rebuild_count}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # --- V5 cluster + in_place_migrate ---
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} V5 cluster K={args.n_clusters} + in_place_migrate", flush=True)
        buf = ClusterBuffer(d, n_clusters=args.n_clusters,
                            init_capacity_per_cluster=max(buf_cap // args.n_clusters, 64),
                            n_probe=args.n_probe)
        g = ModularRouter(buf, factory(), d, maintain_strategy="in_place_migrate",
                          migrate_min_inserts=batch,                # ~1 batch worth
                          migrate_min_query_hits=2,
                          migrate_dead_drop_fraction=0.7)
        r = run_workload(g, "V5_cluster_inplace", data, queries, init_n, batch, qstride, delete_fn,
                         use_gamma=True, has_mark_deleted=False, seed=seed)
        rows.append({"variant": "V5_cluster_inplace", "pattern": args.pattern, "scale": args.scale,
                     "seed": seed, "K": args.n_clusters,
                     "cluster_migrate_count": int(g.cluster_migrate_count),
                     "cluster_drop_count": int(g.cluster_drop_count), **r})
        print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} migrate={g.cluster_migrate_count} drop={g.cluster_drop_count}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # --- V6 cluster + cluster_rebuild (refit centroids) ---
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} V6 cluster K={args.n_clusters} + cluster_rebuild", flush=True)
        buf = ClusterBuffer(d, n_clusters=args.n_clusters,
                            init_capacity_per_cluster=max(buf_cap // args.n_clusters, 64),
                            n_probe=args.n_probe)
        g = ModularRouter(buf, factory(), d, maintain_strategy="cluster_rebuild",
                          rebuild_threshold=args.rebuild_threshold,
                          backend_factory=factory)
        r = run_workload(g, "V6_cluster_full_rebuild", data, queries, init_n, batch, qstride, delete_fn,
                         use_gamma=True, has_mark_deleted=False, seed=seed)
        rows.append({"variant": "V6_cluster_full_rebuild", "pattern": args.pattern, "scale": args.scale,
                     "seed": seed, "K": args.n_clusters, "rebuild_count": int(g.rebuild_count), **r})
        print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} rebuilds={g.rebuild_count}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
