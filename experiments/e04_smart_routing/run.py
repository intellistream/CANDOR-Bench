"""e04 — Smart vs DUMB partition routing A/B.
Spawns subprocesses with GAMMA_DUMB_ROUTING={0,1} so the env var is read at
candy-import time."""
import os, sys, json, subprocess


CHILD_CODE = r"""
import os, sys, time, json, numpy as np
sys.path.insert(0, os.environ['EXP_DIR'])
import _shared
from _shared import load_sift, cached_gt, recall_at_k
from candy import Index

data, queries = load_sift(n_queries=2000, slice_n=200_000)
gt = cached_gt(data, queries, k=10, tag='slice200000_q2000')

# Tight candidate budget = none; rely on geometric pruning
cfg = {'global': {'dim': 128},
       'indexes': {'gamma_fresh': {'backend': 'FaissHNSW',
                                    'split_factor': 4.0,
                                    'gamma_split_threshold': 0.4,
                                    'candidate_reserve_size': 999,
                                    'maintenance_interval': 4,
                                    'maintenance_query_interval': 4,
                                    'maintenance_query_threshold_scale': 0.5},
                   'faiss_hnsw': {'m': 32, 'ef_construction': 120,
                                   'ef_search': 80, 'tombstone_rebuild_threshold': 512}}}
idx = Index('GammaFresh', 128, config=cfg)
idx.initial_load(np.arange(5000, dtype=np.uint64), data[:5000])
ti = time.perf_counter()
idx.add(np.arange(5000, 200000, dtype=np.uint64),
        np.ascontiguousarray(data[5000:200000]))
t_ins = time.perf_counter() - ti
# NO MAINT -- everything stays in buffer; query relies on partition pruning
tq = time.perf_counter()
nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
t_qry = time.perf_counter() - tq
recall = recall_at_k(nbrs.astype(np.uint32), gt)
mode = 'DUMB' if os.environ.get('GAMMA_DUMB_ROUTING') == '1' else 'SMART'
print('JSON:' + json.dumps({'mode': mode, 'insert_s': t_ins,
                             'query_s': t_qry, 'recall': recall}))
"""


def run_subprocess(dumb: bool):
    env = os.environ.copy()
    env["GAMMA_DUMB_ROUTING"] = "1" if dumb else "0"
    env["EXP_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    result = subprocess.run([sys.executable, "-c", CHILD_CODE], env=env,
                             capture_output=True, text=True, timeout=900)
    for line in result.stdout.splitlines():
        if line.startswith("JSON:"):
            return json.loads(line[5:])
    print(result.stdout); print(result.stderr, file=sys.stderr)
    raise RuntimeError(f"run dumb={dumb} failed")


def main():
    print("e04 — smart routing A/B (buffer-resident query, candidate_reserve=999)\n")
    rows = []
    for dumb in (False, True):
        r = run_subprocess(dumb)
        r["total_s"] = r["insert_s"] + r["query_s"]
        rows.append(r)
        print(f"  {r['mode']:6s}  insert={r['insert_s']:6.2f}s  "
              f"query={r['query_s']:6.2f}s  total={r['total_s']:6.2f}s  "
              f"recall={r['recall']:.4f}")
    out = os.path.join(os.path.dirname(__file__), "output.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    smart = next(r for r in rows if r["mode"] == "SMART")
    dumb = next(r for r in rows if r["mode"] == "DUMB")
    qry_speedup = dumb["query_s"] / smart["query_s"]
    delta_pct = (1.0 - smart["total_s"] / dumb["total_s"]) * 100.0
    print(f"\nSMART query speedup at same recall: {qry_speedup:.2f}×")
    print(f"SMART total advantage: {delta_pct:+.1f}%")
    print(f"\n✓ saved {out}")


if __name__ == "__main__":
    main()
