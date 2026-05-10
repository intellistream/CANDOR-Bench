"""e32 — Iso-recall Pareto: gamma+rebuild vs hnswlib direct efSearch sweep.

Peer review #5: "the recall trade-off is buried; gamma+rebuild trades 4
recall pts for -59% time on glove 1M sequential — that's a different
operating point, not a Pareto improvement." We need an iso-recall
comparison.

For each pattern at SIFT 200K, sweep hnswlib direct's efSearch ∈ {40, 60,
80, 120, 160, 240, 400} to produce a (recall, time) curve. Compare
against gamma_v2+rebuild at default config. Show whether gamma+rebuild
sits on or below the hnswlib Pareto frontier.

The fair claim is one of:
  (A) gamma+rebuild dominates: at every recall hnswlib direct can hit,
      gamma+rebuild is faster.
  (B) gamma+rebuild is on the Pareto frontier: at SOME recall regime
      gamma+rebuild wins, hnswlib wins elsewhere.
  (C) gamma+rebuild is dominated: hnswlib at appropriate efSearch
      always wins.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, run_workload, make_pattern
from _shared.gamma_py import HnswlibBackend
from _shared.gamma_py_v2 import GammaPyHybridV2
from _shared.gamma_py_rebuild import GammaPyHybridRebuild


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="200K", choices=["200K", "1M"])
    p.add_argument("--pattern", default="cluster",
                   choices=["sequential", "random", "cluster", "partial_reset"])
    p.add_argument("--ef_searches", default="40,60,80,120,160,240,400")
    p.add_argument("--seeds", default="1")
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
    ef_searches = [int(e) for e in args.ef_searches.split(",")]
    rebuild_factory = lambda: HnswlibBackend(d, max_elements=n)

    for seed in seeds:
        # Pareto curve: hnswlib direct at varying efSearch (and matching efC)
        for ef_s in ef_searches:
            ef_c = max(ef_s + 20, 120)  # efC must be >= efS for sane behavior
            print(f"\n========== {args.pattern}/{args.scale} seed={seed} hnswlib direct ef_s={ef_s} ef_c={ef_c}", flush=True)
            be = HnswlibBackend(d, max_elements=n, ef_c=ef_c, ef_s=ef_s)
            r = run_workload(be, f"hnswlib_direct(ef_s={ef_s})",
                              data, queries, init_n, batch, qstride, delete_fn,
                              use_gamma=False, has_mark_deleted=True, seed=seed)
            rows.append({"scale": args.scale, "pattern": args.pattern, "seed": seed,
                         "design": "hnswlib_direct", "ef_search": ef_s, "ef_construction": ef_c,
                         **r})
            print(f"  {r['name']:35s} {r['total_s']:6.1f}s rec={r['recall']:.4f}", flush=True)
            with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # Reference: gamma_v2 (no rebuild)
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} gamma_v2 (default ef)", flush=True)
        be = HnswlibBackend(d, max_elements=n)
        g = GammaPyHybridV2(be, d, buf_capacity=batch * 50)
        r = run_workload(g, "gamma_v2", data, queries, init_n, batch, qstride, delete_fn,
                          use_gamma=True, has_mark_deleted=False, seed=seed)
        rows.append({"scale": args.scale, "pattern": args.pattern, "seed": seed,
                     "design": "gamma_v2", "ef_search": 80, **r})
        print(f"  {r['name']:35s} {r['total_s']:6.1f}s rec={r['recall']:.4f}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # Reference: gamma_v2 + rebuild (recommended drop-in)
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} gamma_v2+rebuild (default ef)", flush=True)
        be_r = HnswlibBackend(d, max_elements=n)
        g = GammaPyHybridRebuild(be_r, rebuild_factory, d, buf_capacity=batch * 50,
                                   rebuild_threshold=0.5)
        r = run_workload(g, "gamma_v2+rebuild", data, queries, init_n, batch, qstride, delete_fn,
                          use_gamma=True, has_mark_deleted=False, seed=seed)
        rows.append({"scale": args.scale, "pattern": args.pattern, "seed": seed,
                     "design": "gamma_v2+rebuild", "ef_search": 80,
                     "rebuild_count": int(g.rebuild_count), **r})
        print(f"  {r['name']:35s} {r['total_s']:6.1f}s rec={r['recall']:.4f} rebuilds={g.rebuild_count}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
