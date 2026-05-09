"""e24 — Add spatial routing module to Python POC, see what it buys.

This is the first in a series (e24-e2x) that incrementally adds C++-inspired
modules to the Python POC and measures their per-scenario value.

Spatial routing in the C++:
  - K partitions, each with its own backend graph.
  - New vectors routed to nearest-centroid partition.
  - Queries route to top-M partitions by centroid distance.

The hypothesis is that spatial routing helps:
  1. cluster delete pattern: deletes are spatially clustered, so they hit
     one partition's graph; the other K-1 partitions stay clean.
  2. query latency: only search M of K partitions.

Hypothesis is that it hurts:
  - sequential delete: deletions are random across partitions
  - cross-partition queries (recall loss for small M)

Sweep: K ∈ {1, 2, 4, 8, 16}, M = max(1, K // 2). Routing in {centroid,
round_robin}. SIFT 200K, 4 patterns.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, run_workload, make_pattern
from _shared.gamma_py import HnswlibBackend
from _shared.gamma_py_v2 import GammaPyHybridV2
from _shared.gamma_py_spatial import GammaPyHybridSpatial


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="200K", choices=["200K", "1M"])
    p.add_argument("--pattern", default="cluster",
                   choices=["sequential", "random", "cluster", "partial_reset"])
    p.add_argument("--ks", default="1,2,4,8,16")
    p.add_argument("--routing", default="centroid",
                   choices=["centroid", "round_robin", "both"])
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
    ks = [int(x) for x in args.ks.split(",")]
    routings = ["centroid", "round_robin"] if args.routing == "both" else [args.routing]

    # 1. Baseline: gamma_v2 (no spatial routing) for the side-by-side
    print(f"\n========== {args.pattern}/{args.scale} baseline gamma_v2 (k=1)", flush=True)
    be0 = HnswlibBackend(d, max_elements=n)
    g0 = GammaPyHybridV2(be0, d, buf_capacity=batch * 50)
    r = run_workload(g0, "gamma_v2 (k=1, baseline)", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=True, has_mark_deleted=False)
    rows.append({"scale": args.scale, "pattern": args.pattern,
                 "k_partitions": 1, "search_partitions": 1,
                 "routing": "(none)", **r})
    print(f"  {r['name']:35s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} "
          f"qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # 2. hnswlib direct for reference
    print(f"\n========== {args.pattern}/{args.scale} hnswlib direct (reference)", flush=True)
    be_ref = HnswlibBackend(d, max_elements=n)
    r = run_workload(be_ref, "hnswlib direct", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=False, has_mark_deleted=True)
    rows.append({"scale": args.scale, "pattern": args.pattern,
                 "k_partitions": 0, "search_partitions": 0,
                 "routing": "(none)", **r})
    print(f"  {r['name']:35s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} "
          f"qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # 3. Spatial variants
    for routing in routings:
        for k in ks:
            if k <= 1:
                continue  # already covered as gamma_v2 baseline
            m_search = max(1, k // 2)  # search half the partitions
            print(f"\n========== {args.pattern}/{args.scale} spatial K={k} M={m_search} routing={routing}", flush=True)
            be_factory = lambda p, _n=n, _k=k, _d=d: HnswlibBackend(_d, max_elements=max(8, _n // _k * 3))
            g = GammaPyHybridSpatial(be_factory, d, buf_capacity=batch * 50,
                                      k_partitions=k, search_partitions=m_search,
                                      routing=routing)
            tag = f"gamma_spatial(K={k}/M={m_search}/{routing})"
            r = run_workload(g, tag, data, queries, init_n, batch, qstride, delete_fn,
                              use_gamma=True, has_mark_deleted=False)
            rows.append({"scale": args.scale, "pattern": args.pattern,
                         "k_partitions": k, "search_partitions": m_search,
                         "routing": routing, **r})
            print(f"  {r['name']:35s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} "
                  f"qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
            with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
