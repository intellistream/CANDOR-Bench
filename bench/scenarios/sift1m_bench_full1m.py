"""Run all v2 scenarios on the FULL SIFT 1M dataset (no 200K-slice).

Uses the same scenario implementations from sift1m_bench_v2 to avoid
duplication.  Larger init_n + full 1M data means each run takes
several minutes; total wall time ~60-90 min for the full 7-scenario
× 3-algorithm sweep.

Output: results/scenarios/sift1m_results_full1m.json
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, "/home/rprp/CANDOR-Bench")
sys.path.insert(0, "/home/rprp/CANDOR-Bench/algorithms_impl/GammaFresh/build/src/bindings/python")
sys.path.insert(0, "/home/rprp/CANDOR-Bench/algorithms_impl/GammaFresh/python")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sift1m_bench_v2 import (
    load_sift_1m, compute_gt, build,
    scenario_insert_only, scenario_insdel_stream, scenario_streaming,
    scenario_streaming_sliding, scenario_burst, scenario_drift,
    scenario_bulk_delete, scenario_churn,
)


ALGOS = ["gamma", "faiss", "ivf"]


def main():
    print("Loading SIFT 1M …")
    data, queries = load_sift_1m(n_queries=1000)

    gt_path = "/tmp/sift1m_gt10_q1000.npy"
    if os.path.exists(gt_path):
        gt = np.load(gt_path)
    else:
        print("Computing ground truth (1M × 1000)…")
        gt = compute_gt(data, queries, k=10)
        np.save(gt_path, gt)
    print(f"Data shape: {data.shape}  queries: {queries.shape}  GT: {gt.shape}")

    INIT_N = 100_000  # 10% of 1M for initial load
    runs = [
        ("insert_only/1M",                  scenario_insert_only,       dict(batch=20000)),
        ("insdel_stream(50%del)/1M",        scenario_insdel_stream,     dict(batch=10000, delete_ratio=0.5)),
        ("streaming(b=10000)/1M",           scenario_streaming,         dict(batch=10000, qstride=50000)),
        ("streaming_sliding(lag=2)/1M",     scenario_streaming_sliding, dict(batch=10000, qstride=50000, lag=2)),
        ("streaming_sliding(lag=4)/1M",     scenario_streaming_sliding, dict(batch=10000, qstride=50000, lag=4)),
        ("burst(b=50000)/1M",               scenario_burst,             dict(burst=50000)),
        ("drift(5cycles)/1M",               scenario_drift,             dict(drift_chunks=5)),
        ("bulk_delete(30%)/1M",             scenario_bulk_delete,       dict(delete_frac=0.3)),
        ("churn(10cyc,5%,m=10)/1M",         scenario_churn,             dict(cycles=10, churn_frac=0.05, maint_every=10)),
    ]

    rows = []
    sweep_t0 = time.perf_counter()
    for sc_name, sc_fn, kwargs in runs:
        print(f"\n========== {sc_name} ==========", flush=True)
        for algo in ALGOS:
            t0 = time.perf_counter()
            idx = build(algo, data.shape[1])
            try:
                m = sc_fn(idx, data, queries, gt, INIT_N, algo, **kwargs)
            except Exception as e:
                print(f"  {algo:8s}  ERROR: {type(e).__name__}: {e}", flush=True)
                continue
            wall = time.perf_counter() - t0
            row = {"scenario": sc_name, "algo": algo, "wall_s": wall, **m}
            rows.append(row)
            metric_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in m.items())
            print(f"  {algo:6s}  wall={wall:6.1f}s  {metric_str}", flush=True)
            # Save incrementally so partial results survive interrupt
            out = "/home/rprp/CANDOR-Bench/results/scenarios/sift1m_results_full1m.json"
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                json.dump(rows, f, indent=2)

    total = time.perf_counter() - sweep_t0
    print(f"\n✓ Done in {total/60:.1f} min — saved to results/scenarios/sift1m_results_full1m.json")


if __name__ == "__main__":
    main()
