"""e25 — Tombstone-rebuild trigger module on Python POC.

Adds a periodic full-rebuild policy to gamma_v2: when graph tombstone
fraction exceeds a threshold, drop the backend and rebuild from alive
vectors. Inspired by the C++ maintenance pass and hnswlib's
tombstone_rebuild_threshold.

Hypothesis:
  - Helps where gamma's per-batch maintenance can't keep up with hnswlib's
    tombstone bookkeeping (1M sequential, where gamma loses 39%+).
  - Hurts low-churn workloads (rebuild is expensive).
  - Trade-off knob: rebuild_threshold.

Sweep: rebuild_threshold ∈ {0.25, 0.5, 0.75, 1.1 (= no rebuild ≡ gamma_v2)}
on each of the 4 patterns at SIFT 200K. Compares against gamma_v2 (no
rebuild) and hnswlib direct.
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
    p.add_argument("--thresholds", default="0.25,0.5,0.75,1.1")
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

    # 1. Reference: gamma_v2 (no tombstone rebuild)
    print(f"\n========== {args.pattern}/{args.scale} reference: gamma_v2 (no rebuild)", flush=True)
    be = HnswlibBackend(d, max_elements=n)
    g = GammaPyHybridV2(be, d, buf_capacity=batch * 50)
    r = run_workload(g, "gamma_v2 (no rebuild)", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=True, has_mark_deleted=False)
    rows.append({"scale": args.scale, "pattern": args.pattern,
                 "rebuild_threshold": None, **r})
    print(f"  {r['name']:40s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} "
          f"qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # 2. Reference: hnswlib direct
    print(f"\n========== {args.pattern}/{args.scale} reference: hnswlib direct", flush=True)
    be_d = HnswlibBackend(d, max_elements=n)
    r = run_workload(be_d, "hnswlib direct", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=False, has_mark_deleted=True)
    rows.append({"scale": args.scale, "pattern": args.pattern,
                 "rebuild_threshold": -1, **r})
    print(f"  {r['name']:40s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} "
          f"qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # 3. Rebuild-threshold sweep
    thresholds = [float(t) for t in args.thresholds.split(",")]
    for thresh in thresholds:
        print(f"\n========== {args.pattern}/{args.scale} gamma_rebuild(threshold={thresh})", flush=True)
        be_g = HnswlibBackend(d, max_elements=n)
        rebuild_factory = lambda _d=d, _n=n: HnswlibBackend(_d, max_elements=_n)
        g = GammaPyHybridRebuild(be_g, rebuild_factory, d, buf_capacity=batch * 50,
                                  rebuild_threshold=thresh)
        tag = f"gamma_rebuild(th={thresh})"
        r = run_workload(g, tag, data, queries, init_n, batch, qstride, delete_fn,
                          use_gamma=True, has_mark_deleted=False)
        rows.append({"scale": args.scale, "pattern": args.pattern,
                     "rebuild_threshold": thresh,
                     "rebuild_count": int(g.rebuild_count), **r})
        print(f"  {r['name']:40s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} "
              f"qry_p95={r['query_latency_ms_p95']:.3f}ms rebuilds={g.rebuild_count}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
