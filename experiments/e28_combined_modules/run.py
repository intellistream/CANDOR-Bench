"""e28 — Best-of-breed: gamma_v2 + tombstone-rebuild + cost-model admit.

Combines the two modules that empirically helped (e25 rebuild, e27
cost-admit) into one. Tests across 4 patterns at SIFT 200K to see if
the modules are orthogonal (combined > best-single) or substitutable
(combined ≈ best-single).

Variants:
  baseline           : gamma_v2 alone
  hnswlib_direct     : reference
  + rebuild only      : gamma_v2 + tombstone-rebuild (e25 best)
  + cost_admit only   : gamma_v2 + cost-model admit (e27 best)
  + both (combined)   : gamma_v2 + rebuild + cost-admit (this experiment)
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, run_workload, make_pattern
from _shared.gamma_py import HnswlibBackend
from _shared.gamma_py_v2 import GammaPyHybridV2
from _shared.gamma_py_rebuild import GammaPyHybridRebuild
from _shared.gamma_py_cost_admit import GammaPyHybridCostAdmit
from _shared.gamma_py_combined import GammaPyHybridCombined


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="200K", choices=["200K", "1M"])
    p.add_argument("--pattern", default="cluster",
                   choices=["sequential", "random", "cluster", "partial_reset"])
    p.add_argument("--admit_threshold", type=int, default=20000,
                   help="cost-admit threshold (e27 best at 20000 was always-direct on sequential)")
    p.add_argument("--rebuild_threshold", type=float, default=0.5,
                   help="rebuild threshold (e25 best=0.5)")
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

    # Reference 1: gamma_v2
    print(f"\n========== {args.pattern}/{args.scale} ref: gamma_v2", flush=True)
    be = HnswlibBackend(d, max_elements=n)
    g = GammaPyHybridV2(be, d, buf_capacity=batch * 50)
    r = run_workload(g, "gamma_v2", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=True, has_mark_deleted=False)
    rows.append({"scale": args.scale, "pattern": args.pattern, "config": "gamma_v2", **r})
    print(f"  {r['name']:30s} {r['total_s']:6.1f}s rec={r['recall']:.4f} qry={r['query_latency_ms_p95']:.3f}ms", flush=True)
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # Reference 2: hnswlib direct
    print(f"\n========== {args.pattern}/{args.scale} ref: hnswlib direct", flush=True)
    be_d = HnswlibBackend(d, max_elements=n)
    r = run_workload(be_d, "hnswlib direct", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=False, has_mark_deleted=True)
    rows.append({"scale": args.scale, "pattern": args.pattern, "config": "hnswlib_direct", **r})
    print(f"  {r['name']:30s} {r['total_s']:6.1f}s rec={r['recall']:.4f} qry={r['query_latency_ms_p95']:.3f}ms", flush=True)
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # +rebuild only
    print(f"\n========== {args.pattern}/{args.scale} +rebuild only (th={args.rebuild_threshold})", flush=True)
    be_r = HnswlibBackend(d, max_elements=n)
    rebuild_factory = lambda _d=d, _n=n: HnswlibBackend(_d, max_elements=_n)
    g = GammaPyHybridRebuild(be_r, rebuild_factory, d, buf_capacity=batch * 50,
                              rebuild_threshold=args.rebuild_threshold)
    r = run_workload(g, "gamma_v2+rebuild", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=True, has_mark_deleted=False)
    rows.append({"scale": args.scale, "pattern": args.pattern, "config": "rebuild_only",
                 "rebuild_count": int(g.rebuild_count), **r})
    print(f"  {r['name']:30s} {r['total_s']:6.1f}s rec={r['recall']:.4f} qry={r['query_latency_ms_p95']:.3f}ms rebuilds={g.rebuild_count}", flush=True)
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # +cost_admit only
    print(f"\n========== {args.pattern}/{args.scale} +cost_admit only (th={args.admit_threshold})", flush=True)
    be_c = HnswlibBackend(d, max_elements=n)
    g = GammaPyHybridCostAdmit(be_c, d, buf_capacity=batch * 50,
                                 admit_threshold_ops=args.admit_threshold)
    r = run_workload(g, "gamma_v2+cost_admit", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=True, has_mark_deleted=False)
    rows.append({"scale": args.scale, "pattern": args.pattern, "config": "cost_admit_only",
                 "n_admit_buffer": int(g.admit_to_buffer_count),
                 "n_admit_direct": int(g.admit_direct_count), **r})
    print(f"  {r['name']:30s} {r['total_s']:6.1f}s rec={r['recall']:.4f} qry={r['query_latency_ms_p95']:.3f}ms (buf={g.admit_to_buffer_count} direct={g.admit_direct_count})", flush=True)
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # +both combined
    print(f"\n========== {args.pattern}/{args.scale} +rebuild +cost_admit COMBINED", flush=True)
    be_b = HnswlibBackend(d, max_elements=n)
    g = GammaPyHybridCombined(be_b, rebuild_factory, d, buf_capacity=batch * 50,
                                rebuild_threshold=args.rebuild_threshold,
                                admit_threshold_ops=args.admit_threshold)
    r = run_workload(g, "gamma_v2+rebuild+cost_admit", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=True, has_mark_deleted=False)
    rows.append({"scale": args.scale, "pattern": args.pattern, "config": "combined",
                 "rebuild_count": int(g.rebuild_count),
                 "n_admit_buffer": int(g.admit_to_buffer_count),
                 "n_admit_direct": int(g.admit_direct_count), **r})
    print(f"  {r['name']:30s} {r['total_s']:6.1f}s rec={r['recall']:.4f} qry={r['query_latency_ms_p95']:.3f}ms (rebuilds={g.rebuild_count} buf={g.admit_to_buffer_count} direct={g.admit_direct_count})", flush=True)
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
