"""e01 — Pure-insert stream throughput. gamma vs faiss vs ivf."""
import os, sys, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import build, load_sift, cached_gt, recall_at_k, maintain
import numpy as np


def run_one(name, data, queries, gt, init_n, batch):
    idx = build(name, data.shape[1])
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    n = data.shape[0]
    insert_lat = []
    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        ti = time.perf_counter()
        idx.add(np.arange(lo, hi, dtype=np.uint64),
                np.ascontiguousarray(data[lo:hi]))
        insert_lat.append((time.perf_counter() - ti) / (hi - lo))
    stream_s = time.perf_counter() - t0
    # Final maint (separate timing) + recall query
    tm = time.perf_counter()
    maintain(idx, name)
    maint_s = time.perf_counter() - tm
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    qry_s = time.perf_counter() - tq
    recall = recall_at_k(nbrs.astype(np.uint32), gt)
    return {
        "algo": name,
        "n_insert": n - init_n,
        "stream_s": stream_s,
        "stream_throughput_per_sec": (n - init_n) / stream_s,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "insert_latency_us_p99": float(np.percentile(np.asarray(insert_lat) * 1e6, 99)),
        "maint_s": maint_s,
        "final_query_latency_ms": (qry_s / len(queries)) * 1000,
        "recall": recall,
    }


def main():
    # Sweep two scales: 200K (fast iteration) and 1M (publication scale)
    cases = [
        {"label": "200K", "slice_n": 200_000, "init_n": 20_000, "batch": 2_500},
        {"label": "1M",   "slice_n": 1_000_000, "init_n": 100_000, "batch": 10_000},
    ]
    rows = []
    for case in cases:
        print(f"\n========== insert_only/{case['label']} ==========", flush=True)
        data, queries = load_sift(n_queries=1000, slice_n=case["slice_n"])
        gt = cached_gt(data, queries, k=10, tag=f"slice{case['slice_n']}_q1000")
        for algo in ["gamma", "faiss", "ivf"]:
            row = run_one(algo, data, queries, gt, case["init_n"], case["batch"])
            row["scale"] = case["label"]
            rows.append(row)
            print(f"  {algo:6s}  stream={row['stream_s']:6.1f}s  "
                  f"throughput={row['stream_throughput_per_sec']:8.0f} ops/s  "
                  f"per-insert={row['insert_latency_us_avg']:6.1f}µs  "
                  f"maint={row['maint_s']:6.1f}s  "
                  f"recall={row['recall']:.4f}", flush=True)
            out = os.path.join(os.path.dirname(__file__), "output.json")
            with open(out, "w") as f:
                json.dump(rows, f, indent=2)
    print(f"\n✓ saved {out}")


if __name__ == "__main__":
    main()
