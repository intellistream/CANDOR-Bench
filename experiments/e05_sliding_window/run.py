"""e05 — Sliding-window streaming on SIFT 200K and 1M, lag sweep."""
import os, sys, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import build, load_sift, compute_gt, recall_at_k, percentile_ms, maintain
import numpy as np


def run_sliding(name, data, queries, init_n, batch, qstride, lag):
    n = data.shape[0]
    idx = build(name, data.shape[1])
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    insert_lat, delete_lat, query_lat = [], [], []
    deleted, batch_idx = 0, 0
    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        ti = time.perf_counter()
        idx.add(np.arange(lo, hi, dtype=np.uint64),
                np.ascontiguousarray(data[lo:hi]))
        insert_lat.append((time.perf_counter() - ti) / (hi - lo))
        if batch_idx >= lag:
            n_del = hi - lo
            if deleted + n_del <= init_n + (lo - init_n):
                td = time.perf_counter()
                idx.delete(np.arange(deleted, deleted + n_del, dtype=np.uint64))
                delete_lat.append((time.perf_counter() - td) / n_del)
                deleted += n_del
        if (lo - init_n) % qstride == 0:
            maintain(idx, name)
            tq = time.perf_counter()
            idx.search(np.ascontiguousarray(queries), 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
        batch_idx += 1
    maintain(idx, name)
    sgt = (compute_gt(data[deleted:], queries, 10) + deleted).astype(np.uint32)
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    final_qlat = (time.perf_counter() - tq) / len(queries)
    total_s = time.perf_counter() - t0
    return {
        "algo": name, "lag": lag,
        "n_insert": n - init_n, "n_delete": deleted,
        "total_s": total_s,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "delete_latency_us_avg": (float(np.mean(delete_lat)) * 1e6) if delete_lat else 0.0,
        "query_latency_ms_avg": float(np.mean(query_lat)) * 1000 if query_lat else 0.0,
        "query_latency_ms_p95": percentile_ms(query_lat, 95),
        "final_query_latency_ms": final_qlat * 1000,
        "recall": recall_at_k(nbrs.astype(np.uint32), sgt),
    }


def main():
    cases = [
        {"label": "200K", "slice_n": 200_000, "init_n": 20_000, "batch": 2_500, "qstride": 10_000},
        {"label": "1M",   "slice_n": 1_000_000, "init_n": 100_000, "batch": 10_000, "qstride": 50_000},
    ]
    lags = [2, 4, 8]
    rows = []
    for case in cases:
        data, queries = load_sift(n_queries=1000, slice_n=case["slice_n"])
        for lag in lags:
            print(f"\n========== sliding(lag={lag})/{case['label']} ==========", flush=True)
            for algo in ["gamma", "faiss", "ivf"]:
                row = run_sliding(algo, data, queries,
                                   case["init_n"], case["batch"], case["qstride"], lag)
                row["scale"] = case["label"]
                rows.append(row)
                print(f"  {algo:6s}  total={row['total_s']:6.1f}s  "
                      f"recall={row['recall']:.4f}  "
                      f"qry_avg={row['query_latency_ms_avg']:.3f}ms  "
                      f"qry_p95={row['query_latency_ms_p95']:.3f}ms", flush=True)
                out = os.path.join(os.path.dirname(__file__), "output.json")
                with open(out, "w") as f:
                    json.dump(rows, f, indent=2)
    print(f"\n✓ saved {out}")


if __name__ == "__main__":
    main()
