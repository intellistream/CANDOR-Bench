"""e19 — hnswlib parameter sensitivity on the cluster pattern.

Question: can hnswlib close the gap to gamma on cluster delete by tuning
its hyperparameters (M, ef_construction, ef_search)? Or is the gap
architectural and not parameter-sensitive?

Methodology: SIFT 200K, cluster pattern.
- Sweep hnswlib (M, efC, efS) on a small grid: M ∈ {16, 32, 64},
  efC ∈ {120, 240}, efS ∈ {80, 160}. 12 configurations.
- Compare against gamma_v2+hnswlib at default M=32, efC=120, efS=80.
- Report time/recall/qry-latency for each.
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
    p.add_argument("--m_values", default="16,32,64")
    p.add_argument("--efc_values", default="120,240")
    p.add_argument("--efs_values", default="80,160")
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

    Ms = [int(x) for x in args.m_values.split(",")]
    efCs = [int(x) for x in args.efc_values.split(",")]
    efSs = [int(x) for x in args.efs_values.split(",")]

    # Reference gamma_v2+hnswlib at default params
    print(f"\n========== gamma_v2+hnswlib (default M=32 efC=120 efS=80)", flush=True)
    be_ref = HnswlibBackend(d, max_elements=n, M=32, ef_c=120, ef_s=80)
    g = GammaPyHybridV2(be_ref, d, buf_capacity=batch * 50)
    r = run_workload(g, "gamma_v2+hnswlib", data, queries, init_n, batch, qstride,
                      delete_fn, use_gamma=True, has_mark_deleted=False)
    rows.append({"scale": args.scale, "pattern": args.pattern,
                 "M": 32, "efC": 120, "efS": 80, "is_gamma": True, **r})
    print(f"  {r['name']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} "
          f"qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    # hnswlib direct grid sweep
    for M in Ms:
        for efC in efCs:
            for efS in efSs:
                tag = f"hnswlib(M={M}/efC={efC}/efS={efS})"
                print(f"\n========== {tag}", flush=True)
                be = HnswlibBackend(d, max_elements=n, M=M, ef_c=efC, ef_s=efS)
                r = run_workload(be, tag, data, queries, init_n, batch, qstride,
                                  delete_fn, use_gamma=False, has_mark_deleted=True)
                rows.append({"scale": args.scale, "pattern": args.pattern,
                             "M": M, "efC": efC, "efS": efS, "is_gamma": False, **r})
                print(f"  {r['name']:35s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} "
                      f"qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
                with open(out_path, "w") as f:
                    json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
