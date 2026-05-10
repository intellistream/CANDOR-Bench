"""e35 — Lifetime-aware migration on cluster-correlated lifetime workloads.

Hypothesis (from §user push-back, May 2026): the C++ team's
lifetime-aware admission idea may have been *workload-locked*, not
broken. Our prior negatives (e27, e30) used homogeneous lifetimes
where per-vector EMA had no structure to learn. With ClusterBuffer +
per-cluster lifetime EMA on a workload where cluster identity actually
correlates with lifetime, the controller may finally win.

Variants
--------
V0  hnswlib_direct
V2  flat + rebuild                  (the rebuild baseline)
V4  cluster + rebuild                (e33 cluster champion)
V5  cluster + in_place_migrate       (C++-style, no lifetime info)
V7  cluster + lifetime_aware_migrate           ★ new
V8  cluster + lifetime_aware_migrate + rebuild ★ new combined

Workloads
---------
1. cluster_lifetime_50K_5K  : K=16, 8 clusters live ~50K ops, 8 live ~5K ops
                              (10× lifetime asymmetry; the ideal scenario)
2-5: the 4 standard delete patterns (sequential, random, cluster,
     partial_reset) — to confirm V7/V8 don't *regress* when there is
     no lifetime structure to exploit.

If V7/V8 wins on (1) but not on (2-5) we have a workload-conditional
positive. If V7/V8 doesn't win even on (1), the negative result is
strong.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, run_workload, make_pattern
from _shared.backends import HnswlibBackend
from _shared.buffers import FlatBuffer, ClusterBuffer
from _shared.router_modular import ModularRouter
from _shared.workloads_lifetime import make_cluster_lifetime_pattern


def make_router(strategy: str, *, d, n, batch, K, factory, rebuild_th,
                lifetime_horizon=None):
    if strategy == "flat_rebuild":
        buf = FlatBuffer(d, init_capacity=batch * 50)
        return ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                             rebuild_threshold=rebuild_th,
                             backend_factory=factory)
    if strategy == "cluster_rebuild":
        buf = ClusterBuffer(d, n_clusters=K,
                            init_capacity_per_cluster=max(batch * 50 // K, 64),
                            n_probe=4)
        return ModularRouter(buf, factory(), d, maintain_strategy="rebuild",
                             rebuild_threshold=rebuild_th,
                             backend_factory=factory)
    if strategy == "cluster_inplace":
        buf = ClusterBuffer(d, n_clusters=K,
                            init_capacity_per_cluster=max(batch * 50 // K, 64),
                            n_probe=4)
        return ModularRouter(buf, factory(), d,
                             maintain_strategy="in_place_migrate",
                             migrate_min_inserts=batch,
                             migrate_min_query_hits=2)
    if strategy == "cluster_lifetime":
        buf = ClusterBuffer(d, n_clusters=K,
                            init_capacity_per_cluster=max(batch * 50 // K, 64),
                            n_probe=4)
        return ModularRouter(buf, factory(), d,
                             maintain_strategy="lifetime_aware_migrate",
                             lifetime_horizon=lifetime_horizon,
                             lifetime_min_observations=5,
                             lifetime_min_alive=batch // 2)
    if strategy == "cluster_lifetime_rebuild":
        buf = ClusterBuffer(d, n_clusters=K,
                            init_capacity_per_cluster=max(batch * 50 // K, 64),
                            n_probe=4)
        return ModularRouter(buf, factory(), d,
                             maintain_strategy="lifetime_aware_migrate_with_rebuild",
                             rebuild_threshold=rebuild_th,
                             backend_factory=factory,
                             lifetime_horizon=lifetime_horizon,
                             lifetime_min_observations=5,
                             lifetime_min_alive=batch // 2)
    raise ValueError(strategy)


def run_one(strategy, label, *, d, n, data, queries, init_n, batch, qstride,
            delete_fn, K, factory, rebuild_th, lifetime_horizon, seed):
    if strategy == "hnswlib_direct":
        be = factory()
        r = run_workload(be, label, data, queries, init_n, batch, qstride,
                         delete_fn, use_gamma=False, has_mark_deleted=True,
                         seed=seed)
        extras = {}
    else:
        g = make_router(strategy, d=d, n=n, batch=batch, K=K, factory=factory,
                        rebuild_th=rebuild_th, lifetime_horizon=lifetime_horizon)
        r = run_workload(g, label, data, queries, init_n, batch, qstride,
                         delete_fn, use_gamma=True, has_mark_deleted=False,
                         seed=seed)
        extras = {
            "rebuild_count": int(getattr(g, "rebuild_count", 0)),
            "cluster_migrate_count": int(getattr(g, "cluster_migrate_count", 0)),
            "cluster_drop_count": int(getattr(g, "cluster_drop_count", 0)),
            "lifetime_kept_in_buffer_count": int(getattr(g, "lifetime_kept_in_buffer_count", 0)),
        }
    return {**r, **extras}


VARIANTS = [
    ("V0_hnswlib_direct",          "hnswlib_direct"),
    ("V2_flat_rebuild",            "flat_rebuild"),
    ("V4_cluster_rebuild",         "cluster_rebuild"),
    ("V5_cluster_inplace",         "cluster_inplace"),
    ("V7_cluster_lifetime",        "cluster_lifetime"),
    ("V8_cluster_lifetime_rebuild","cluster_lifetime_rebuild"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="200K", choices=["200K", "1M"])
    p.add_argument("--workload", required=True,
                   choices=["lifetime", "sequential", "random", "cluster", "partial_reset"])
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--rebuild_threshold", type=float, default=0.5)
    p.add_argument("--lifetime_long_ops", type=float, default=50_000.0)
    p.add_argument("--lifetime_short_ops", type=float, default=5_000.0)
    p.add_argument("--lifetime_horizon", type=float, default=None,
                   help="default: midpoint(short, long)")
    p.add_argument("--seeds", default="42")
    args = p.parse_args()

    if args.scale == "200K":
        slice_n, init_n, batch, qstride, n_queries = 200_000, 20_000, 2_500, 10_000, 500
    else:
        slice_n, init_n, batch, qstride, n_queries = 1_000_000, 100_000, 10_000, 50_000, 1000

    data, queries = load_dataset("sift", n_queries=n_queries, slice_n=slice_n)
    n, d = data.shape
    K = args.K

    if args.lifetime_horizon is None:
        args.lifetime_horizon = (args.lifetime_long_ops + args.lifetime_short_ops) / 2.0

    out_path = os.path.join(
        os.path.dirname(__file__),
        f"output_sift_{args.workload}_K{K}_{args.scale}.json")
    rows = []
    seeds = [int(s) for s in args.seeds.split(",")]
    factory = lambda: HnswlibBackend(d, max_elements=n)

    def make_delete_fn(_seed):
        # Re-create per variant — the lifetime workload's delete_fn carries
        # internal heap/state across calls, so reusing across variants would
        # pollute the next variant's view of pending deaths.
        if args.workload == "lifetime":
            df, inf = make_cluster_lifetime_pattern(
                data=data, K=K,
                long_mean_ops=args.lifetime_long_ops,
                short_mean_ops=args.lifetime_short_ops,
                centroid_seed=1, workload_seed=_seed)
            return df, inf
        return make_pattern(args.workload), None

    for seed in seeds:
        # One-time printout of workload metadata
        if args.workload == "lifetime":
            _, info = make_delete_fn(seed)
            print(f"[lifetime workload seed={seed}] "
                  f"{info['cluster_is_long'].sum()} long / "
                  f"{(~info['cluster_is_long']).sum()} short clusters; "
                  f"long_mean={args.lifetime_long_ops:.0f}, "
                  f"short_mean={args.lifetime_short_ops:.0f}, "
                  f"horizon={args.lifetime_horizon:.0f}", flush=True)

        for label, strategy in VARIANTS:
            delete_fn, _ = make_delete_fn(seed)  # ★ fresh per variant
            print(f"\n========== {args.workload}/{args.scale} seed={seed} "
                  f"{label} ({strategy})", flush=True)
            r = run_one(strategy, label, d=d, n=n, data=data, queries=queries,
                        init_n=init_n, batch=batch, qstride=qstride,
                        delete_fn=delete_fn, K=K, factory=factory,
                        rebuild_th=args.rebuild_threshold,
                        lifetime_horizon=args.lifetime_horizon, seed=seed)
            row = {
                "variant": label,
                "strategy": strategy,
                "workload": args.workload,
                "scale": args.scale,
                "seed": seed,
                "K": K,
                "lifetime_horizon": args.lifetime_horizon,
                **r,
            }
            rows.append(row)
            extras = {k: r.get(k) for k in
                      ["rebuild_count", "cluster_migrate_count",
                       "cluster_drop_count", "lifetime_kept_in_buffer_count"]
                      if r.get(k)}
            extras_s = " ".join(f"{k}={v}" for k, v in extras.items() if v)
            print(f"  total={r['total_s']:.1f}s rec={r['recall']:.4f} "
                  f"qry_p95={r.get('query_latency_ms_p95', 0):.3f}ms "
                  f"n_alive={r['n_alive']} n_del={r['n_delete']} {extras_s}",
                  flush=True)
            with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
