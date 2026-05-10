"""e23 — Multi-seed variance estimation for the headline numbers.

VLDB reviewers will ask: "is your -23% gamma win on cluster a single-run
artifact?" This experiment runs the SIFT 200K cluster scenario with 5
different seeds for both gamma_v2+hnswlib and hnswlib direct, then reports
mean ± std.

Run on cluster + random patterns; SIFT 200K. ~10 runs total per pattern,
single-thread. Each run is ~2-3 min.
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
    p.add_argument("--seeds", default="1,2,3,4,5")
    args = p.parse_args()

    if args.scale == "200K":
        slice_n, init_n, batch, qstride, n_queries = 200_000, 20_000, 2_500, 10_000, 500
    else:
        slice_n, init_n, batch, qstride, n_queries = 1_000_000, 100_000, 10_000, 50_000, 1000

    data, queries = load_dataset("sift", n_queries=n_queries, slice_n=slice_n)
    n, d = data.shape
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    out_path = os.path.join(os.path.dirname(__file__),
                             f"output_sift_{args.pattern}_{args.scale}.json")
    delete_fn = make_pattern(args.pattern)

    for seed in seeds:
        print(f"\n========== seed={seed} (gamma_v2+hnswlib)", flush=True)
        be = HnswlibBackend(d, max_elements=n)
        g = GammaRouter(be, d, buf_capacity=batch * 50)
        r = run_workload(g, "gamma_v2+hnswlib", data, queries, init_n,
                          batch, qstride, delete_fn,
                          use_gamma=True, has_mark_deleted=False, seed=seed)
        rows.append({"scale": args.scale, "pattern": args.pattern,
                     "seed": seed, **r})
        print(f"  {r['name']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f}", flush=True)

        print(f"\n========== seed={seed} (hnswlib direct)", flush=True)
        be2 = HnswlibBackend(d, max_elements=n)
        r = run_workload(be2, "hnswlib direct", data, queries, init_n,
                          batch, qstride, delete_fn,
                          use_gamma=False, has_mark_deleted=True, seed=seed)
        rows.append({"scale": args.scale, "pattern": args.pattern,
                     "seed": seed, **r})
        print(f"  {r['name']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f}", flush=True)

        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)

    # Summary stats at end
    import numpy as np
    g_times = [r["total_s"] for r in rows if r["name"] == "gamma_v2+hnswlib"]
    d_times = [r["total_s"] for r in rows if r["name"] == "hnswlib direct"]
    g_recs = [r["recall"] for r in rows if r["name"] == "gamma_v2+hnswlib"]
    d_recs = [r["recall"] for r in rows if r["name"] == "hnswlib direct"]
    print(f"\n=== SUMMARY ({args.pattern} {args.scale}) ===")
    print(f"gamma_v2+hnswlib: total = {np.mean(g_times):.1f} ± {np.std(g_times):.1f} s, "
          f"recall = {np.mean(g_recs):.4f} ± {np.std(g_recs):.4f}")
    print(f"hnswlib direct  : total = {np.mean(d_times):.1f} ± {np.std(d_times):.1f} s, "
          f"recall = {np.mean(d_recs):.4f} ± {np.std(d_recs):.4f}")
    delta_pct = (np.mean(g_times) - np.mean(d_times)) / np.mean(d_times) * 100
    print(f"Δ% = {delta_pct:+.1f}%  (gamma vs direct)")
    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
