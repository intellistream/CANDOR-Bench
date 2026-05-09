"""Churn sweep: vary delete fraction, measure recall / query_latency / total_time.

For each delete fraction f in {0, 0.25, 0.5, 0.75, 0.9}:
  - Init load 20K
  - Stream 180K inserts in batches of 2500
  - During stream, also delete oldest f * 180K vectors interleaved
  - Periodic queries (qstride=10000) measure query latency under accumulating
    churn state
  - Final maint, final query for recall

This produces three plots' worth of data:
  1. recall vs delete_frac (HNSW expected to collapse, gamma expected stable)
  2. query_latency vs delete_frac (HNSW grows with tombstones, gamma flat)
  3. total_time vs delete_frac (gamma's relative advantage grows with churn)

Output: results/scenarios/churn_sweep.json
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, "/home/rprp/CANDOR-Bench")
sys.path.insert(0, "/home/rprp/CANDOR-Bench/algorithms_impl/GammaFresh/build/src/bindings/python")
sys.path.insert(0, "/home/rprp/CANDOR-Bench/algorithms_impl/GammaFresh/python")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sift1m_bench_v2 import build, load_sift_1m, compute_gt


def maintain(idx, name, hnsw_force_rebuild=False):
    if name == "gamma":
        idx.maintain(0, 0, False)
    elif name == "faiss" and hnsw_force_rebuild:
        idx.maintain(0, 0, True)


def run_churn(idx, name, data, queries, init_n, n, batch, delete_frac,
              qstride, hnsw_rebuild_every=0):
    n_total_del = int((n - init_n) * delete_frac)
    del_cap = init_n + (n - init_n)  # don't exceed available range
    n_total_del = min(n_total_del, del_cap)

    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    insert_lat, delete_lat, query_lat = [], [], []
    deleted = 0
    n_writes_per_round = batch  # one insert batch per round
    n_dels_per_round = max(0, n_total_del * batch // (n - init_n))
    batch_idx = 0

    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        ti = time.perf_counter()
        idx.add(np.arange(lo, hi, dtype=np.uint64),
                np.ascontiguousarray(data[lo:hi]))
        insert_lat.append((time.perf_counter() - ti) / (hi - lo))
        # delete oldest
        n_del = min(n_dels_per_round, n_total_del - deleted)
        if n_del > 0 and deleted + n_del <= del_cap:
            td = time.perf_counter()
            idx.delete(np.arange(deleted, deleted + n_del, dtype=np.uint64))
            delete_lat.append((time.perf_counter() - td) / n_del)
            deleted += n_del
        # periodic maint + query
        if (lo - init_n) % qstride == 0:
            maintain(idx, name)
            if hnsw_rebuild_every > 0 and name == "faiss" and \
               batch_idx % hnsw_rebuild_every == 0:
                idx.maintain(0, 0, True)
            tq = time.perf_counter()
            idx.search(np.ascontiguousarray(queries), 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
        batch_idx += 1

    # Final maint + recall query
    maintain(idx, name)
    if hnsw_rebuild_every > 0 and name == "faiss":
        idx.maintain(0, 0, True)
    surviving = data[deleted:n]
    surviving_gt = (compute_gt(surviving, queries, 10) + deleted).astype(np.uint32)
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    final_qlat = (time.perf_counter() - tq) / len(queries)
    total = time.perf_counter() - t0

    recall = float(np.mean([len(set(p)&set(g))/10 for p, g in
                             zip(nbrs.astype(np.uint32), surviving_gt)]))
    return {
        "total_s": total,
        "n_insert": n - init_n,
        "n_delete": deleted,
        "delete_frac_actual": deleted / (n - init_n),
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "delete_latency_us_avg": (float(np.mean(delete_lat)) * 1e6) if delete_lat else 0.0,
        "query_latency_ms_avg": float(np.mean(query_lat)) * 1000 if query_lat else 0.0,
        "query_latency_ms_p95": float(np.percentile(np.asarray(query_lat) * 1000, 95)) if query_lat else 0.0,
        "final_query_latency_ms": final_qlat * 1000,
        "recall": recall,
    }


def main():
    print("Loading SIFT 200K …")
    data, queries = load_sift_1m(500)
    data = data[:200000]
    init_n, batch, qstride = 20000, 2500, 10000
    n = data.shape[0]

    delete_fracs = [0.0, 0.25, 0.5, 0.75, 0.9]
    rows = []
    for df in delete_fracs:
        print(f"\n========== delete_frac = {df:.2f} ==========")
        for algo, kwargs in [("gamma", {}), ("faiss", {}), ("ivf", {}),
                              ("faiss", {"hnsw_rebuild_every": 16})]:
            label = algo + ("+rebuild16" if kwargs else "")
            try:
                idx = build(algo, data.shape[1])
                m = run_churn(idx, algo, data, queries, init_n, n, batch, df,
                              qstride, **kwargs)
                row = {"delete_frac": df, "algo": label, **m}
                rows.append(row)
                print(f"  {label:15s} total={m['total_s']:6.1f}s "
                      f"recall={m['recall']:.4f} "
                      f"qry_avg={m['query_latency_ms_avg']:.3f}ms "
                      f"qry_p95={m['query_latency_ms_p95']:.3f}ms "
                      f"final_qry={m['final_query_latency_ms']:.3f}ms",
                      flush=True)
                # incremental save so partial results survive Ctrl-C
                out = "/home/rprp/CANDOR-Bench/results/scenarios/churn_sweep.json"
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w") as f:
                    json.dump(rows, f, indent=2)
            except Exception as e:
                print(f"  {label:15s} ERROR: {type(e).__name__}: {e}")

    out = "/home/rprp/CANDOR-Bench/results/scenarios/churn_sweep.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n✓ Saved: {out}")


if __name__ == "__main__":
    main()
