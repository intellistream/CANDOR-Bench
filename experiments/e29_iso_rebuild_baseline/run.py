"""e29 — The missing baseline: hnswlib direct + periodic rebuild.

Critique from peer review of e25: the win could be entirely from the
rebuild trigger, not from the buffer/absorb router architecture. To
distinguish, this experiment runs at SIFT 200K + 1M:

  Variant A: hnswlib direct (no rebuild)        — original e15 baseline
  Variant B: hnswlib direct + periodic rebuild  — new isolated rebuild baseline
  Variant C: gamma_v2 (no rebuild)              — buffer router only
  Variant D: gamma_v2 + rebuild                 — combined (e25 winner)

If D > B and D > C strictly, the buffer+absorb is contributing
independently from the rebuild. If D ≈ B, the rebuild explains everything
and the buffer is overhead.

Same workload as e25 (per-pattern), threshold=0.5 for rebuild variants
matching e25's universal sweet spot.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, run_workload, make_pattern
from _shared.backends import HnswlibBackend
from _shared.router import GammaRouter
from _shared.router_with_rebuild import GammaRouterWithRebuild
from _shared.baseline_with_rebuild import HnswlibDirectWithRebuild


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="200K", choices=["200K", "1M"])
    p.add_argument("--pattern", default="cluster",
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
    rows = []
    out_path = os.path.join(os.path.dirname(__file__),
                             f"output_sift_{args.pattern}_{args.scale}.json")
    delete_fn = make_pattern(args.pattern)
    seeds = [int(s) for s in args.seeds.split(",")]
    rebuild_factory = lambda: HnswlibBackend(d, max_elements=n)

    for seed in seeds:
        # Variant A: hnswlib direct, no rebuild
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} (A) hnswlib direct (no rebuild)", flush=True)
        be = HnswlibBackend(d, max_elements=n)
        r = run_workload(be, "A_hnswlib_direct", data, queries, init_n,
                          batch, qstride, delete_fn,
                          use_gamma=False, has_mark_deleted=True, seed=seed)
        rows.append({"scale": args.scale, "pattern": args.pattern, "seed": seed,
                     "variant": "A_hnswlib_direct", **r})
        print(f"  {r['name']:30s} {r['total_s']:6.1f}s rec={r['recall']:.4f} qry={r['query_latency_ms_p95']:.3f}ms", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # Variant B: hnswlib direct + periodic rebuild (the missing baseline)
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} (B) hnswlib direct + rebuild(th={args.rebuild_threshold})", flush=True)
        be_b = HnswlibDirectWithRebuild(rebuild_factory,
                                          rebuild_threshold=args.rebuild_threshold)
        r = run_workload(be_b, "B_hnswlib_direct+rebuild", data, queries, init_n,
                          batch, qstride, delete_fn,
                          use_gamma=False, has_mark_deleted=True, seed=seed)
        rows.append({"scale": args.scale, "pattern": args.pattern, "seed": seed,
                     "variant": "B_hnswlib_direct+rebuild",
                     "rebuild_count": int(be_b.rebuild_count), **r})
        print(f"  {r['name']:30s} {r['total_s']:6.1f}s rec={r['recall']:.4f} qry={r['query_latency_ms_p95']:.3f}ms rebuilds={be_b.rebuild_count}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # Variant C: gamma_v2 (no rebuild)
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} (C) gamma_v2 only", flush=True)
        be_c = HnswlibBackend(d, max_elements=n)
        g = GammaRouter(be_c, d, buf_capacity=batch * 50)
        r = run_workload(g, "C_gamma_v2", data, queries, init_n,
                          batch, qstride, delete_fn,
                          use_gamma=True, has_mark_deleted=False, seed=seed)
        rows.append({"scale": args.scale, "pattern": args.pattern, "seed": seed,
                     "variant": "C_gamma_v2", **r})
        print(f"  {r['name']:30s} {r['total_s']:6.1f}s rec={r['recall']:.4f} qry={r['query_latency_ms_p95']:.3f}ms", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # Variant D: gamma_v2 + rebuild
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} (D) gamma_v2 + rebuild(th={args.rebuild_threshold})", flush=True)
        be_d = HnswlibBackend(d, max_elements=n)
        g = GammaRouterWithRebuild(be_d, rebuild_factory, d, buf_capacity=batch * 50,
                                   rebuild_threshold=args.rebuild_threshold)
        r = run_workload(g, "D_gamma_v2+rebuild", data, queries, init_n,
                          batch, qstride, delete_fn,
                          use_gamma=True, has_mark_deleted=False, seed=seed)
        rows.append({"scale": args.scale, "pattern": args.pattern, "seed": seed,
                     "variant": "D_gamma_v2+rebuild",
                     "rebuild_count": int(g.rebuild_count), **r})
        print(f"  {r['name']:30s} {r['total_s']:6.1f}s rec={r['recall']:.4f} qry={r['query_latency_ms_p95']:.3f}ms rebuilds={g.rebuild_count}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
