"""e06 — Bulk delete on SIFT 1M (gamma's strongest scenario)."""
import os, sys, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import build, load_sift, compute_gt, recall_at_k
import numpy as np


def run_bulk_delete(name, data, queries, init_n, delete_frac):
    n = data.shape[0]
    n_del = int(n * delete_frac)
    idx = build(name, data.shape[1])
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    insert_lat = []
    t0 = time.perf_counter()
    ti = time.perf_counter()
    idx.add(np.arange(init_n, n, dtype=np.uint64),
            np.ascontiguousarray(data[init_n:n]))
    insert_lat.append((time.perf_counter() - ti) / (n - init_n))
    td = time.perf_counter()
    idx.delete(np.arange(0, n_del, dtype=np.uint64))
    del_s = time.perf_counter() - td
    op_total = time.perf_counter() - t0
    sgt = (compute_gt(data[n_del:], queries, 10) + n_del).astype(np.uint32)
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    final_q_s = time.perf_counter() - tq
    return {
        "algo": name, "delete_frac": delete_frac,
        "n_insert": n - init_n, "n_delete": n_del,
        "op_total_s": op_total,
        "wall_total_s": op_total + final_q_s,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "delete_latency_us_avg": (del_s / n_del) * 1e6,
        "final_query_qps": len(queries) / final_q_s,
        "final_query_latency_ms": (final_q_s / len(queries)) * 1000,
        "recall": recall_at_k(nbrs.astype(np.uint32), sgt),
    }


def main():
    data, queries = load_sift(n_queries=1000, slice_n=1_000_000)
    init_n, delete_frac = 100_000, 0.3
    rows = []
    print(f"========== bulk_delete({delete_frac:.0%})/1M ==========", flush=True)
    for algo in ["gamma", "faiss", "ivf"]:
        row = run_bulk_delete(algo, data, queries, init_n, delete_frac)
        rows.append(row)
        print(f"  {algo:6s}  op={row['op_total_s']:6.1f}s  "
              f"final_qry={row['final_query_latency_ms']:6.2f}ms  "
              f"recall={row['recall']:.4f}", flush=True)
        out = os.path.join(os.path.dirname(__file__), "output.json")
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)
    print(f"\n✓ saved {out}")


if __name__ == "__main__":
    main()
