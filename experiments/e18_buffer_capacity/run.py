"""e18 — Gamma buffer-capacity ablation on the cluster pattern.

Question: how sensitive is gamma's win to the buffer capacity multiplier?
- Too small → frequent maintenance flushes, high overhead.
- Too large → buffer scan dominates query latency.
Find the operating range where gamma's win is robust.

Methodology: SIFT 200K, cluster pattern, hnswlib backend (best case for gamma).
Sweep buf_capacity ∈ {batch * mult} for mult ∈ {5, 10, 25, 50, 100, 200, 400}.
Compare against hnswlib direct (constant, no buffer).
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, run_workload, make_pattern
from _shared.router import GammaRouter
from _shared.backends import HnswlibBackend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="200K", choices=["200K", "1M"])
    p.add_argument("--pattern", default="cluster",
                   choices=["sequential", "random", "cluster", "partial_reset"])
    p.add_argument("--mults", default="5,10,25,50,100,200,400",
                   help="comma-separated buffer-capacity multipliers (× batch)")
    args = p.parse_args()

    if args.scale == "200K":
        slice_n, init_n, batch, qstride, n_queries = 200_000, 20_000, 2_500, 10_000, 500
    else:
        slice_n, init_n, batch, qstride, n_queries = 1_000_000, 100_000, 10_000, 50_000, 1000

    data, queries = load_dataset("sift", n_queries=n_queries, slice_n=slice_n)
    n, d = data.shape
    rows = []
    out_path = os.path.join(os.path.dirname(__file__),
                             f"output_{args.pattern}_{args.scale}.json")
    delete_fn = make_pattern(args.pattern)
    mults = [int(x) for x in args.mults.split(",")]

    # Baseline: hnswlib direct (independent of buf_capacity)
    print(f"\n========== sift/{args.scale}/{args.pattern} hnswlib direct (baseline)", flush=True)
    be0 = HnswlibBackend(d, max_elements=n)
    r0 = run_workload(be0, "hnswlib direct", data, queries, init_n,
                       batch, qstride, delete_fn,
                       use_gamma=False, has_mark_deleted=True)
    rows.append({"scale": args.scale, "pattern": args.pattern, "buf_mult": None, **r0})
    print(f"  {r0['name']:25s} total={r0['total_s']:6.1f}s recall={r0['recall']:.4f} "
          f"qry_p95={r0['query_latency_ms_p95']:.3f}ms n_alive={r0['n_alive']}", flush=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    for mult in mults:
        print(f"\n========== buf_mult={mult} (capacity={batch*mult})", flush=True)
        be = HnswlibBackend(d, max_elements=n)
        g = GammaRouter(be, d, buf_capacity=batch * mult)
        r = run_workload(g, f"gamma_v2+hnswlib(buf={mult}x)", data, queries, init_n,
                          batch, qstride, delete_fn,
                          use_gamma=True, has_mark_deleted=False)
        rows.append({"scale": args.scale, "pattern": args.pattern, "buf_mult": mult, **r})
        print(f"  {r['name']:30s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} "
              f"qry_p95={r['query_latency_ms_p95']:.3f}ms n_alive={r['n_alive']}", flush=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
