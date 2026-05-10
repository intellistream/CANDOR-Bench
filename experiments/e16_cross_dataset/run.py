"""e16 — Cross-dataset replication of e15.

Question: does the e15 finding (gamma wins random/cluster/partial_reset by 25-30%
on hnswlib backend) hold across datasets, or is it SIFT-specific?

Methodology: same 4-pattern × 2-backend × 2-architecture matrix as e15, but
cycle through datasets. SIFT is in e15 already, so default datasets here are
msong (high-dim audio), random-m (synthetic), and glove (text embeddings)
when available.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared  # sets OMP=1
from _shared import load_dataset, run_workload, make_pattern
from _shared.router import GammaRouter
from _shared.backends import HnswlibBackend, FaissHnswBackend


def slice_for(dataset: str, scale: str):
    """Returns (slice_n, init_n, batch, qstride, n_queries) per dataset/scale."""
    if scale == "200K":
        slice_n, init_n, batch, qstride = 200_000, 20_000, 2_500, 10_000
    elif scale == "100K":  # for cheaper datasets
        slice_n, init_n, batch, qstride = 100_000, 10_000, 1_250, 5_000
    elif scale == "1M":
        slice_n, init_n, batch, qstride = 1_000_000, 100_000, 10_000, 50_000
    else:
        raise ValueError(scale)
    n_queries = {
        "sift": 500, "msong": 200, "glove": 500, "random-m": 500,
    }.get(dataset, 200)
    return slice_n, init_n, batch, qstride, n_queries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["sift", "msong", "glove", "random-m"])
    p.add_argument("--scale", default="200K", choices=["100K", "200K", "1M"])
    p.add_argument("--patterns", default="sequential,random,cluster,partial_reset")
    args = p.parse_args()

    slice_n, init_n, batch, qstride, n_queries = slice_for(args.dataset, args.scale)
    data, queries = load_dataset(args.dataset, n_queries=n_queries, slice_n=slice_n)
    n, d = data.shape
    print(f"[e16] dataset={args.dataset} scale={args.scale} n={n} d={d} q={len(queries)}", flush=True)

    rows = []
    patterns = args.patterns.split(",")
    backends = [
        ("hnswlib", lambda: HnswlibBackend(d, max_elements=n)),
        ("faiss_hnsw", lambda: FaissHnswBackend(d, max_elements=n)),
    ]
    out_path = os.path.join(os.path.dirname(__file__),
                             f"output_{args.dataset}_{args.scale}.json")

    for pattern in patterns:
        print(f"\n========== {args.dataset}/{args.scale}/{pattern} ==========", flush=True)
        delete_fn = make_pattern(pattern)
        for be_name, be_factory in backends:
            be = be_factory()
            g = GammaRouter(be, d, buf_capacity=batch * 50)
            r = run_workload(g, f"gamma_v2+{be_name}", data, queries, init_n,
                              batch, qstride, delete_fn,
                              use_gamma=True, has_mark_deleted=False)
            rows.append({"dataset": args.dataset, "scale": args.scale,
                         "pattern": pattern, **r})
            print(f"  {r['name']:25s} total={r['total_s']:6.1f}s "
                  f"recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms "
                  f"n_alive={r['n_alive']}", flush=True)

            be2 = be_factory()
            r = run_workload(be2, f"{be_name} direct", data, queries, init_n,
                              batch, qstride, delete_fn,
                              use_gamma=False, has_mark_deleted=True)
            rows.append({"dataset": args.dataset, "scale": args.scale,
                         "pattern": pattern, **r})
            print(f"  {r['name']:25s} total={r['total_s']:6.1f}s "
                  f"recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms "
                  f"n_alive={r['n_alive']}", flush=True)
            with open(out_path, "w") as f:
                json.dump(rows, f, indent=2)
    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
