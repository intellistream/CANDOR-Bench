"""e20 — Router-component ablation.

Tests which mechanism of the simplified hybrid router actually pays off:

  full       : all 4 mechanisms enabled (= gamma_v2 baseline)
  no_buffer_scan          : search() does graph-only
  no_buffer_inserts       : every add() goes straight to graph
  no_absorb_buffer_deletes: deletes always reach the graph as mark_deleted
  eager_maint             : maintenance after every add (no deferral)

For each ablation, run all 4 patterns at SIFT 200K vs hnswlib direct.
The comparison shows: when this mechanism is removed, does gamma's win
disappear, shrink, or stay?

Output: output_<pattern>_200K.json with one row per (variant, scenario).
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, run_workload, make_pattern
from _shared.ablations.gamma_py_v3 import GammaPyHybridV3
from _shared.backends import HnswlibBackend


VARIANTS = [
    ("full", dict()),
    ("no_buffer_scan",
     dict(use_buffer_scan=False)),
    ("no_buffer_inserts",
     dict(buffer_inserts=False)),
    ("no_absorb_deletes",
     dict(absorb_buffer_deletes=False)),
    ("eager_maint",
     dict(eager_maint=True)),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="200K", choices=["200K", "1M"])
    p.add_argument("--pattern", default="cluster",
                   choices=["sequential", "random", "cluster", "partial_reset"])
    p.add_argument("--dataset", default="sift",
                   choices=["sift", "msong", "glove", "random-m"])
    args = p.parse_args()

    if args.scale == "200K":
        slice_n, init_n, batch, qstride, n_queries = 200_000, 20_000, 2_500, 10_000, 500
    else:
        slice_n, init_n, batch, qstride, n_queries = 1_000_000, 100_000, 10_000, 50_000, 1000

    data, queries = load_dataset(args.dataset, n_queries=n_queries, slice_n=slice_n)
    n, d = data.shape
    rows = []
    out_path = os.path.join(os.path.dirname(__file__),
                             f"output_{args.dataset}_{args.pattern}_{args.scale}.json")
    delete_fn = make_pattern(args.pattern)

    # Baseline: hnswlib direct
    print(f"\n========== {args.dataset}/{args.scale}/{args.pattern} hnswlib direct (baseline)", flush=True)
    be0 = HnswlibBackend(d, max_elements=n)
    r = run_workload(be0, "hnswlib direct", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=False, has_mark_deleted=True)
    rows.append({"dataset": args.dataset, "scale": args.scale,
                 "pattern": args.pattern, "variant": "(hnswlib direct)", **r})
    print(f"  {r['name']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} "
          f"qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    for variant_name, kwargs in VARIANTS:
        print(f"\n========== {args.dataset}/{args.scale}/{args.pattern} variant={variant_name}", flush=True)
        be = HnswlibBackend(d, max_elements=n)
        g = GammaPyHybridV3(be, d, buf_capacity=batch * 50, **kwargs)
        r = run_workload(g, f"gamma_v3({variant_name})+hnswlib", data, queries,
                          init_n, batch, qstride, delete_fn,
                          use_gamma=True, has_mark_deleted=False)
        rows.append({"dataset": args.dataset, "scale": args.scale,
                     "pattern": args.pattern, "variant": variant_name,
                     "kwargs": {k: bool(v) for k, v in kwargs.items()}, **r})
        print(f"  {r['name']:35s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} "
              f"qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
