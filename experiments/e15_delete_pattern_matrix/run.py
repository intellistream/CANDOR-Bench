"""e15 — Full delete-pattern × backend × architecture matrix."""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, run_workload, make_pattern
from _shared.router import GammaRouter
from _shared.backends import HnswlibBackend, FaissHnswBackend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["200K", "1M"], default="200K")
    p.add_argument("--patterns",
                   default="sequential,random,cluster,partial_reset",
                   help="comma-separated patterns to run (split for parallel runs)")
    args = p.parse_args()
    if args.scale == "200K":
        slice_n, init_n, batch, qstride, n_queries = 200_000, 20_000, 2_500, 10_000, 500
    else:
        slice_n, init_n, batch, qstride, n_queries = 1_000_000, 100_000, 10_000, 50_000, 1000
    data, queries = load_dataset("sift", n_queries=n_queries, slice_n=slice_n)
    n, d = data.shape
    rows = []

    patterns = args.patterns.split(",")
    backends = [
        ("hnswlib", lambda: HnswlibBackend(d, max_elements=n)),
        ("faiss_hnsw", lambda: FaissHnswBackend(d, max_elements=n)),
    ]

    for pattern in patterns:
        print(f"\n========== {args.scale} / pattern={pattern} ==========", flush=True)
        delete_fn = make_pattern(pattern)
        for be_name, be_factory in backends:
            be = be_factory()
            g = GammaRouter(be, d, buf_capacity=batch * 50)
            r = run_workload(g, f"gamma_v2+{be_name}", data, queries, init_n,
                              batch, qstride, delete_fn,
                              use_gamma=True, has_mark_deleted=False)
            rows.append({"scale": args.scale, "pattern": pattern, **r})
            print(f"  {r['name']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms n_alive={r['n_alive']}", flush=True)
            be2 = be_factory()
            r = run_workload(be2, f"{be_name} direct", data, queries, init_n,
                              batch, qstride, delete_fn,
                              use_gamma=False, has_mark_deleted=True)
            rows.append({"scale": args.scale, "pattern": pattern, **r})
            print(f"  {r['name']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms n_alive={r['n_alive']}", flush=True)
            tag = f"{args.scale}_{','.join(patterns)}" if len(patterns) < 4 else args.scale
            out = os.path.join(os.path.dirname(__file__), f"output_{tag}.json")
            with open(out, "w") as f:
                json.dump(rows, f, indent=2)
    print(f"\n✓ saved {out}")


if __name__ == "__main__":
    main()
