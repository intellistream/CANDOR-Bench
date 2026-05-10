"""e30 — Rich modules on rich workloads.

Tests the gamma_py_rich modules (per-vector cost-admit, heat-aware hot
tier, γ-aware migration) on workloads designed to need them. The
peer-review critique says we couldn't show value for these modules
because our 4 patterns are stationary; this experiment uses
non-stationary workloads where adaptive decisions should win.

Workloads (from `_shared/workloads_rich.py`):
  mixed_lifetime   : 50% short-lived (≈2K ops) + 50% long-lived (≈50K ops)
  zipfian_queries  : queries Zipf-weighted from a Q pool (some queries
                      hit 100× more than others)

Variants (same on each workload):
  V0  hnswlib direct                  : reference, no router
  V1  gamma_v2 (always-buffer, no modules)
  V2  gamma_rich (modules off — sanity check ≈ V1)
  V3  gamma_rich + per_vector_cost    : per-vector lifetime EMA → admit
  V4  gamma_rich + hot_tier           : heat-tracked top-N pinned cache
  V5  gamma_rich + tombstone_rebuild  : per-partition rebuild (e25-style)
  V6  gamma_rich + ALL the above
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset
from _shared.backends import HnswlibBackend
from _shared.router import GammaRouter
from _shared.ablations.gamma_py_rich import GammaPyHybridRich
from _shared.workloads_rich import run_mixed_lifetime, run_zipfian_queries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="200K", choices=["200K", "1M"])
    p.add_argument("--workload", required=True,
                   choices=["mixed_lifetime", "zipfian_queries"])
    p.add_argument("--seeds", default="42")
    args = p.parse_args()

    if args.scale == "200K":
        slice_n, init_n, batch, qstride, n_queries = 200_000, 20_000, 2_500, 10_000, 500
    else:
        slice_n, init_n, batch, qstride, n_queries = 1_000_000, 100_000, 10_000, 50_000, 1000

    data, queries = load_dataset("sift", n_queries=n_queries, slice_n=slice_n)
    n, d = data.shape
    rows = []
    out_path = os.path.join(os.path.dirname(__file__),
                             f"output_sift_{args.workload}_{args.scale}.json")
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.workload == "mixed_lifetime":
        run_fn = lambda idx, name, **kw: run_mixed_lifetime(
            idx, name, data, queries, init_n, batch, qstride,
            short_lifetime_ops=2000, long_lifetime_ops=50000,
            short_fraction=0.5, **kw)
    else:
        run_fn = lambda idx, name, **kw: run_zipfian_queries(
            idx, name, data, queries, init_n, batch, qstride,
            zipf_alpha=1.2, delete_every_n_inserts=2, **kw)

    factory = lambda p: HnswlibBackend(d, max_elements=n)

    for seed in seeds:
        # V0: hnswlib direct
        print(f"\n========== {args.workload}/{args.scale} seed={seed} V0 hnswlib direct", flush=True)
        be = HnswlibBackend(d, max_elements=n)
        r = run_fn(be, "V0_hnswlib_direct", seed=seed,
                    use_gamma_rich=False, has_mark_deleted=True)
        rows.append({"workload": args.workload, "scale": args.scale, "seed": seed,
                     "variant": "V0_hnswlib_direct", **r})
        print(f"  {r['name']:35s} {r['total_s']:6.1f}s rec={r['recall']:.4f}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # V1: gamma_v2
        print(f"\n========== {args.workload}/{args.scale} seed={seed} V1 gamma_v2", flush=True)
        be1 = HnswlibBackend(d, max_elements=n)
        g1 = GammaRouter(be1, d, buf_capacity=batch * 50)
        r = run_fn(g1, "V1_gamma_v2", seed=seed, use_gamma_rich=False,
                    has_mark_deleted=False)
        rows.append({"workload": args.workload, "scale": args.scale, "seed": seed,
                     "variant": "V1_gamma_v2", **r})
        print(f"  {r['name']:35s} {r['total_s']:6.1f}s rec={r['recall']:.4f}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # V2: gamma_rich, all modules off
        print(f"\n========== {args.workload}/{args.scale} seed={seed} V2 gamma_rich (modules off)", flush=True)
        g2 = GammaPyHybridRich(factory, d, buf_capacity=batch * 50,
                                n_partitions=1, search_partitions=1)
        r = run_fn(g2, "V2_gamma_rich(off)", seed=seed,
                    use_gamma_rich=True, has_mark_deleted=False)
        rows.append({"workload": args.workload, "scale": args.scale, "seed": seed,
                     "variant": "V2_gamma_rich_off", **r})
        print(f"  {r['name']:35s} {r['total_s']:6.1f}s rec={r['recall']:.4f}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # V3: + per_vector_cost
        print(f"\n========== {args.workload}/{args.scale} seed={seed} V3 +per_vector_cost", flush=True)
        g3 = GammaPyHybridRich(factory, d, buf_capacity=batch * 50,
                                n_partitions=1, search_partitions=1,
                                use_per_vector_cost=True,
                                cost_admit_threshold_ops=20000)
        r = run_fn(g3, "V3_+per_vector_cost", seed=seed,
                    use_gamma_rich=True, has_mark_deleted=False)
        rows.append({"workload": args.workload, "scale": args.scale, "seed": seed,
                     "variant": "V3_per_vector_cost",
                     "n_admit_buffer": int(g3.admit_buffer_count),
                     "n_admit_direct": int(g3.admit_direct_count), **r})
        print(f"  {r['name']:35s} {r['total_s']:6.1f}s rec={r['recall']:.4f} buf={g3.admit_buffer_count} direct={g3.admit_direct_count}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # V4: + hot_tier
        print(f"\n========== {args.workload}/{args.scale} seed={seed} V4 +hot_tier", flush=True)
        g4 = GammaPyHybridRich(factory, d, buf_capacity=batch * 50,
                                n_partitions=1, search_partitions=1,
                                use_hot_tier=True,
                                hot_tier_size=500,
                                hot_tier_promote_threshold=3)
        r = run_fn(g4, "V4_+hot_tier", seed=seed,
                    use_gamma_rich=True, has_mark_deleted=False)
        rows.append({"workload": args.workload, "scale": args.scale, "seed": seed,
                     "variant": "V4_hot_tier",
                     "hot_promote_count": int(g4.hot_promote_count), **r})
        print(f"  {r['name']:35s} {r['total_s']:6.1f}s rec={r['recall']:.4f} hot_promote={g4.hot_promote_count}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # V5: + tombstone_rebuild
        print(f"\n========== {args.workload}/{args.scale} seed={seed} V5 +tombstone_rebuild", flush=True)
        g5 = GammaPyHybridRich(factory, d, buf_capacity=batch * 50,
                                n_partitions=1, search_partitions=1,
                                use_tombstone_rebuild=True,
                                rebuild_threshold=0.5)
        r = run_fn(g5, "V5_+tombstone_rebuild", seed=seed,
                    use_gamma_rich=True, has_mark_deleted=False)
        rows.append({"workload": args.workload, "scale": args.scale, "seed": seed,
                     "variant": "V5_tombstone_rebuild",
                     "rebuild_count": int(g5.rebuild_count), **r})
        print(f"  {r['name']:35s} {r['total_s']:6.1f}s rec={r['recall']:.4f} rebuilds={g5.rebuild_count}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # V6: ALL modules combined
        print(f"\n========== {args.workload}/{args.scale} seed={seed} V6 +ALL", flush=True)
        g6 = GammaPyHybridRich(factory, d, buf_capacity=batch * 50,
                                n_partitions=1, search_partitions=1,
                                use_per_vector_cost=True,
                                cost_admit_threshold_ops=20000,
                                use_hot_tier=True,
                                hot_tier_size=500,
                                hot_tier_promote_threshold=3,
                                use_tombstone_rebuild=True,
                                rebuild_threshold=0.5)
        r = run_fn(g6, "V6_+ALL", seed=seed,
                    use_gamma_rich=True, has_mark_deleted=False)
        rows.append({"workload": args.workload, "scale": args.scale, "seed": seed,
                     "variant": "V6_all",
                     "rebuild_count": int(g6.rebuild_count),
                     "hot_promote_count": int(g6.hot_promote_count),
                     "n_admit_buffer": int(g6.admit_buffer_count),
                     "n_admit_direct": int(g6.admit_direct_count), **r})
        print(f"  {r['name']:35s} {r['total_s']:6.1f}s rec={r['recall']:.4f} (rebuild={g6.rebuild_count} hot={g6.hot_promote_count} buf={g6.admit_buffer_count} direct={g6.admit_direct_count})", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
