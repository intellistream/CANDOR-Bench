"""e31 — Per-op latency decomposition.

Critical user critique: "buffer's job is to make insert/delete fast — did
you actually measure that?" Answer: no, e15-e29 only measured `total_s`.

This experiment instruments the workload to record:
  - mean per-vector INSERT latency (μs)
  - mean per-vector DELETE latency (μs)
  - mean per-call MAINTAIN latency (ms) — gamma's amortized cost
  - mean per-query SEARCH latency (μs)

Run on 4 patterns at SIFT 200K, comparing 3 designs:
  hnswlib_direct           : reference
  gamma_v2                 : buffer + delete-absorb router
  gamma_v2 + rebuild(0.5)  : the recommended drop-in (e25 winner)

Goal: produce the table the paper actually needs. Rows are op type,
columns are designs.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, make_pattern
from _shared.gamma_py import HnswlibBackend
from _shared.gamma_py_v2 import GammaPyHybridV2
from _shared.gamma_py_rebuild import GammaPyHybridRebuild
from _shared.workloads_instrumented import run_instrumented


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="200K", choices=["200K", "1M"])
    p.add_argument("--pattern", default="cluster",
                   choices=["sequential", "random", "cluster", "partial_reset"])
    p.add_argument("--seeds", default="1,2,3")
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
    rebuild_factory = lambda: HnswlibBackend(d, max_elements=n)

    for seed in seeds:
        # 1. hnswlib direct
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} hnswlib direct", flush=True)
        be = HnswlibBackend(d, max_elements=n)
        r = run_instrumented(be, "hnswlib_direct", data, queries, init_n,
                              batch, qstride, delete_fn,
                              use_gamma=False, has_mark_deleted=True, seed=seed)
        rows.append({"scale": args.scale, "pattern": args.pattern, "seed": seed,
                     "design": "hnswlib_direct", **r})
        _print_row(r)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # 2. gamma_v2 (router only)
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} gamma_v2 (router only)", flush=True)
        be = HnswlibBackend(d, max_elements=n)
        g = GammaPyHybridV2(be, d, buf_capacity=batch * 50)
        r = run_instrumented(g, "gamma_v2", data, queries, init_n,
                              batch, qstride, delete_fn,
                              use_gamma=True, has_mark_deleted=False, seed=seed)
        rows.append({"scale": args.scale, "pattern": args.pattern, "seed": seed,
                     "design": "gamma_v2", **r})
        _print_row(r)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

        # 3. gamma_v2 + tombstone rebuild (recommended drop-in)
        print(f"\n========== {args.pattern}/{args.scale} seed={seed} gamma_v2+rebuild", flush=True)
        be = HnswlibBackend(d, max_elements=n)
        g = GammaPyHybridRebuild(be, rebuild_factory, d, buf_capacity=batch * 50,
                                   rebuild_threshold=0.5)
        r = run_instrumented(g, "gamma_v2+rebuild", data, queries, init_n,
                              batch, qstride, delete_fn,
                              use_gamma=True, has_mark_deleted=False, seed=seed)
        rows.append({"scale": args.scale, "pattern": args.pattern, "seed": seed,
                     "design": "gamma_v2+rebuild",
                     "rebuild_count": int(g.rebuild_count), **r})
        _print_row(r)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


def _print_row(r):
    print(f"  {r['name']:25s} total={r['total_s']:6.1f}s rec={r['recall']:.4f}")
    print(f"    insert/vec  : {r['mean_t_insert_per_vec_us']:8.2f} μs (cumulative {r['sum_t_insert_s']:.1f}s, n={r['n_inserts']})")
    print(f"    delete/vec  : {r['mean_t_delete_per_vec_us']:8.2f} μs (cumulative {r['sum_t_delete_s']:.1f}s, n={r['n_deletes']})")
    print(f"    maintain    : {r['mean_t_maintain_per_call_ms']:8.2f} ms/call x {r['n_maintain_calls']} calls = {r['sum_t_maintain_s']:.1f}s total")
    print(f"    search/q    : {r['mean_t_search_per_q_us']:8.2f} μs (cumulative {r['sum_t_search_s']:.1f}s, n={r['n_searches']})")


if __name__ == "__main__":
    main()
