"""e27 — Cost-model admit module.

Replaces the always-buffer admit policy of gamma_v2 with a learned
decision: based on the rolling-mean of observed delete lifetimes,
choose buffer (if recent vectors lived short) or direct-graph (if they
lived long).

Sweep `admit_threshold_ops` ∈ {0 (always direct), 5000, 20000, 100000,
1e9 (always buffer = v2)}. The "right" threshold should depend on
churn — at high churn, expect buffer-admit to win; at low churn,
direct-admit should pay off (or at least tie).

Run on each pattern at SIFT 200K vs gamma_v2 reference and hnswlib direct.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, run_workload, make_pattern
from _shared.backends import HnswlibBackend
from _shared.router import GammaRouter
from _shared.ablations.gamma_py_cost_admit import GammaPyHybridCostAdmit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="200K", choices=["200K", "1M"])
    p.add_argument("--pattern", default="cluster",
                   choices=["sequential", "random", "cluster", "partial_reset"])
    p.add_argument("--thresholds", default="0,5000,20000,100000,1000000000")
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

    # Reference: gamma_v2
    print(f"\n========== {args.pattern}/{args.scale} reference: gamma_v2", flush=True)
    be = HnswlibBackend(d, max_elements=n)
    g = GammaRouter(be, d, buf_capacity=batch * 50)
    r = run_workload(g, "gamma_v2", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=True, has_mark_deleted=False)
    rows.append({"scale": args.scale, "pattern": args.pattern,
                 "admit_threshold": None, **r})
    print(f"  {r['name']:35s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # Reference: hnswlib direct
    print(f"\n========== {args.pattern}/{args.scale} reference: hnswlib direct", flush=True)
    be_d = HnswlibBackend(d, max_elements=n)
    r = run_workload(be_d, "hnswlib direct", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=False, has_mark_deleted=True)
    rows.append({"scale": args.scale, "pattern": args.pattern,
                 "admit_threshold": -1, **r})
    print(f"  {r['name']:35s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # Sweep admit thresholds
    thresholds = [int(t) for t in args.thresholds.split(",")]
    for thresh in thresholds:
        print(f"\n========== {args.pattern}/{args.scale} cost_admit(threshold={thresh})", flush=True)
        be_g = HnswlibBackend(d, max_elements=n)
        g = GammaPyHybridCostAdmit(be_g, d, buf_capacity=batch * 50,
                                     admit_threshold_ops=thresh)
        tag = f"gamma_cost_admit(th={thresh})"
        r = run_workload(g, tag, data, queries, init_n, batch, qstride, delete_fn,
                          use_gamma=True, has_mark_deleted=False)
        rows.append({"scale": args.scale, "pattern": args.pattern,
                     "admit_threshold": thresh,
                     "n_admit_buffer": int(g.admit_to_buffer_count),
                     "n_admit_direct": int(g.admit_direct_count), **r})
        print(f"  {r['name']:35s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms (buf:{g.admit_to_buffer_count} direct:{g.admit_direct_count})", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
