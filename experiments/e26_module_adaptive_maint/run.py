"""e26 — Adaptive maintenance scheduling module.

Adds an auto-trigger maint(): whenever buffer fullness ≥ threshold during
add(), maint runs immediately. Otherwise the caller-driven maint at
qstride is the only flush.

Sweep `auto_maint_fill_threshold` ∈ {0.1, 0.25, 0.5, 0.8, 1.1 (= no auto-maint)}
on each pattern at SIFT 200K. Compare against gamma_v2 and hnswlib direct.

Hypothesis:
  - Lower threshold = more frequent maint = lower buffer-scan latency at
    query time but higher maint overhead. Helps cluster pattern (high
    delete locality → fast cancel) less than partial_reset (where maint
    needs to run after every burst).
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, run_workload, make_pattern
from _shared.gamma_py import HnswlibBackend
from _shared.gamma_py_v2 import GammaPyHybridV2
from _shared.gamma_py_adaptive_maint import GammaPyHybridAdaptiveMaint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="200K", choices=["200K", "1M"])
    p.add_argument("--pattern", default="cluster",
                   choices=["sequential", "random", "cluster", "partial_reset"])
    p.add_argument("--thresholds", default="0.1,0.25,0.5,0.8,1.1")
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

    # 1. gamma_v2 reference (caller-driven maint at qstride)
    print(f"\n========== {args.pattern}/{args.scale} reference: gamma_v2", flush=True)
    be = HnswlibBackend(d, max_elements=n)
    g = GammaPyHybridV2(be, d, buf_capacity=batch * 50)
    r = run_workload(g, "gamma_v2 (qstride maint)", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=True, has_mark_deleted=False)
    rows.append({"scale": args.scale, "pattern": args.pattern,
                 "threshold": None, **r})
    print(f"  {r['name']:35s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # 2. hnswlib direct reference
    print(f"\n========== {args.pattern}/{args.scale} reference: hnswlib direct", flush=True)
    be_d = HnswlibBackend(d, max_elements=n)
    r = run_workload(be_d, "hnswlib direct", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=False, has_mark_deleted=True)
    rows.append({"scale": args.scale, "pattern": args.pattern,
                 "threshold": -1, **r})
    print(f"  {r['name']:35s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # 3. Sweep auto-maint threshold
    thresholds = [float(t) for t in args.thresholds.split(",")]
    for thresh in thresholds:
        print(f"\n========== {args.pattern}/{args.scale} adaptive_maint(threshold={thresh})", flush=True)
        be_g = HnswlibBackend(d, max_elements=n)
        g = GammaPyHybridAdaptiveMaint(be_g, d, buf_capacity=batch * 50,
                                         auto_maint_fill_threshold=thresh)
        tag = f"gamma_adaptive_maint(th={thresh})"
        r = run_workload(g, tag, data, queries, init_n, batch, qstride, delete_fn,
                          use_gamma=True, has_mark_deleted=False)
        rows.append({"scale": args.scale, "pattern": args.pattern,
                     "threshold": thresh,
                     "auto_maint_count": int(g.auto_maint_count), **r})
        print(f"  {r['name']:35s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms auto_maints={g.auto_maint_count}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
