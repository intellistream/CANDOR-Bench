"""e17 — Churn rate sweep on the cluster pattern (gamma's biggest win in e15).

Question: as deletes-per-batch ratio grows from light (0.5×) to heavy (3×),
does gamma's lead grow, shrink, or invert?

Methodology: cluster pattern, hnswlib backend (the relevant fair comparison),
SIFT 200K. Sweep `del_per_batch / insert_batch ∈ {0.25, 0.5, 1.0, 1.5, 2.0, 3.0}`.

For each ratio we run gamma_v2+hnswlib AND hnswlib-direct, then report
gamma's relative time/recall.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, run_workload, make_pattern
from _shared.gamma_py_v2 import GammaPyHybridV2
from _shared.gamma_py import HnswlibBackend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="200K", choices=["200K", "1M"])
    p.add_argument("--pattern", default="cluster",
                   choices=["sequential", "random", "cluster", "partial_reset"])
    p.add_argument("--ratios", default="0.25,0.5,1.0,1.5,2.0,3.0",
                   help="comma-separated del_per_batch / insert_batch ratios")
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
    ratios = [float(x) for x in args.ratios.split(",")]

    for ratio in ratios:
        n_del_per = max(1, int(round(batch * ratio)))
        print(f"\n========== sift/{args.scale}/{args.pattern} ratio={ratio} (del/batch={n_del_per})", flush=True)

        be = HnswlibBackend(d, max_elements=n)
        g = GammaPyHybridV2(be, d, buf_capacity=batch * 50)
        r = run_workload(g, f"gamma_v2+hnswlib", data, queries, init_n,
                          batch, qstride, delete_fn,
                          use_gamma=True, has_mark_deleted=False,
                          max_deletes_per_batch=n_del_per)
        rows.append({"scale": args.scale, "pattern": args.pattern, "ratio": ratio,
                     "n_del_per_batch": n_del_per, **r})
        print(f"  {r['name']:25s} total={r['total_s']:6.1f}s "
              f"recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms "
              f"n_alive={r['n_alive']}", flush=True)

        be2 = HnswlibBackend(d, max_elements=n)
        r = run_workload(be2, f"hnswlib direct", data, queries, init_n,
                          batch, qstride, delete_fn,
                          use_gamma=False, has_mark_deleted=True,
                          max_deletes_per_batch=n_del_per)
        rows.append({"scale": args.scale, "pattern": args.pattern, "ratio": ratio,
                     "n_del_per_batch": n_del_per, **r})
        print(f"  {r['name']:25s} total={r['total_s']:6.1f}s "
              f"recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms "
              f"n_alive={r['n_alive']}", flush=True)

        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
