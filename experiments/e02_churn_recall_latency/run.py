"""e02 — Sweep delete_frac across multiple datasets,
measure recall + query latency for each algo.

The thesis "buffer absorbs short-lived vectors" should hold across datasets
of different dim and distribution. Run on:
  - sift     (1M × 128d, image features)
  - msong    (992K × 420d, audio timbre — high-dim)
  - random-m (100K × 128d, synthetic uniform)

If gamma's recall stability + query latency invariance holds across all 3,
the thesis is dataset-independent.
"""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import build, load_dataset, compute_gt, recall_at_k, percentile_ms, maintain
import numpy as np


def run_churn(name, data, queries, init_n, batch, delete_frac, qstride):
    n = data.shape[0]
    n_total_del = int((n - init_n) * delete_frac)
    n_dels_per_round = max(0, n_total_del * batch // (n - init_n))
    idx = build(name, data.shape[1])
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    insert_lat, delete_lat, query_lat = [], [], []
    deleted = 0
    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        ti = time.perf_counter()
        idx.add(np.arange(lo, hi, dtype=np.uint64),
                np.ascontiguousarray(data[lo:hi]))
        insert_lat.append((time.perf_counter() - ti) / (hi - lo))
        n_del = min(n_dels_per_round, n_total_del - deleted)
        if n_del > 0:
            td = time.perf_counter()
            idx.delete(np.arange(deleted, deleted + n_del, dtype=np.uint64))
            delete_lat.append((time.perf_counter() - td) / n_del)
            deleted += n_del
        if (lo - init_n) % qstride == 0:
            maintain(idx, name)
            tq = time.perf_counter()
            idx.search(np.ascontiguousarray(queries), 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
    maintain(idx, name)
    sgt = (compute_gt(data[deleted:], queries, 10) + deleted).astype(np.uint32)
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    final_qlat = (time.perf_counter() - tq) / len(queries)
    total_s = time.perf_counter() - t0
    return {
        "algo": name,
        "delete_frac_target": delete_frac,
        "delete_frac_actual": deleted / (n - init_n),
        "n_insert": n - init_n, "n_delete": deleted,
        "total_s": total_s,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "delete_latency_us_avg": (float(np.mean(delete_lat)) * 1e6) if delete_lat else 0.0,
        "query_latency_ms_avg": float(np.mean(query_lat)) * 1000 if query_lat else 0.0,
        "query_latency_ms_p95": percentile_ms(query_lat, 95),
        "final_query_latency_ms": final_qlat * 1000,
        "recall": recall_at_k(nbrs.astype(np.uint32), sgt),
    }


DATASET_CONFIGS = {
    # name      slice_n  init_n  batch  qstride n_queries
    "sift":     (200_000, 20_000, 2_500, 10_000, 500),
    "msong":    (100_000, 10_000, 2_500, 10_000, 200),  # 420d -> smaller slice
    "random-m": (100_000, 10_000, 2_500, 10_000, 500),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["sift", "msong", "random-m"],
                   help="datasets to sweep")
    p.add_argument("--delete-fracs", nargs="+", type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 0.9])
    args = p.parse_args()

    rows = []
    for ds_name in args.datasets:
        slice_n, init_n, batch, qstride, n_queries = DATASET_CONFIGS[ds_name]
        print(f"\n##### dataset = {ds_name} (slice {slice_n}, init {init_n}) #####",
              flush=True)
        data, queries = load_dataset(ds_name, n_queries=n_queries, slice_n=slice_n)
        for df in args.delete_fracs:
            print(f"\n  delete_frac = {df:.2f}", flush=True)
            for algo in ["gamma", "faiss", "ivf"]:
                row = run_churn(algo, data, queries, init_n, batch, df, qstride)
                row["dataset"] = ds_name
                row["dim"] = int(data.shape[1])
                rows.append(row)
                print(f"    {algo:6s}  total={row['total_s']:6.1f}s  "
                      f"recall={row['recall']:.4f}  "
                      f"qry_avg={row['query_latency_ms_avg']:.3f}ms  "
                      f"qry_p95={row['query_latency_ms_p95']:.3f}ms", flush=True)
                out = os.path.join(os.path.dirname(__file__), "output.json")
                with open(out, "w") as f:
                    json.dump(rows, f, indent=2)
    print(f"\n✓ saved {out}")


if __name__ == "__main__":
    main()
