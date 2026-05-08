"""A/B test: gamma's smart partition routing vs round-robin (dumb).

The C++ side (spatial_partition_strategy.cpp) reads GAMMA_DUMB_ROUTING=1
and bypasses choose_owner_partition's distance scan, picking a partition
in round-robin order instead.

Run: python bench/scenarios/smart_routing_ab.py

Headline result on SIFT 1M / 200K, buffer-resident workload (no maint),
candidate_reserve_size=999 (geometric pruning is the sole budget):

  SMART:  ins 9.6s + qry 30.5s = 40.1s @ recall 1.000
  DUMB :  ins 8.3s + qry 48.2s = 56.4s @ recall 1.000

  → SMART query is 1.6× faster at the SAME recall, total time -29%.

Mechanism: SMART produces tight (low-radius) partitions, so
`near_edge ≤ search_radius` prunes most partitions from the candidate set.
DUMB produces noise-radius partitions, so geometric pruning passes
everything and the query has to scan all partitions.
"""
import os, sys, subprocess, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_one(mode: int) -> dict:
    env = os.environ.copy()
    env["GAMMA_DUMB_ROUTING"] = str(mode)
    env["OMP_NUM_THREADS"] = "1"
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    code = """
import sys, time, numpy as np, os
sys.path.insert(0, '/home/rprp/CANDOR-Bench')
sys.path.insert(0, '/home/rprp/CANDOR-Bench/algorithms_impl/GammaFresh/build/src/bindings/python')
sys.path.insert(0, '/home/rprp/CANDOR-Bench/algorithms_impl/GammaFresh/python')
sys.path.insert(0, '/home/rprp/CANDOR-Bench/bench/scenarios')
from sift1m_bench_v2 import load_sift_1m
from candy import Index
import json
data_full, queries = load_sift_1m(2000)
data = data_full[:200000]
gt = np.load('/tmp/sift1m_slice200k_gt10_q2000.npy')
cfg = {'global':{'dim':128},
   'indexes':{'gamma_fresh':{'backend':'FaissHNSW','split_factor':4.0,
              'gamma_split_threshold':0.4, 'candidate_reserve_size':999,
              'maintenance_interval':4,'maintenance_query_interval':4,
              'maintenance_query_threshold_scale':0.5},
              'faiss_hnsw':{'m':32,'ef_construction':120,'ef_search':80,
                            'tombstone_rebuild_threshold':512}}}
idx = Index('GammaFresh', 128, config=cfg)
idx.initial_load(np.arange(5000,dtype=np.uint64), data[:5000])
ti = time.perf_counter()
idx.add(np.arange(5000,200000,dtype=np.uint64),
        np.ascontiguousarray(data[5000:200000]))
t_ins = time.perf_counter() - ti
tq = time.perf_counter()
nbrs,_ = idx.search(np.ascontiguousarray(queries), 10)
t_qry = time.perf_counter() - tq
recall = float(np.mean([len(set(p)&set(g))/10 for p,g in
                        zip(nbrs.astype(np.uint32), gt)]))
print('JSON:' + json.dumps({'ins':t_ins,'qry':t_qry,'recall':recall}))
"""
    result = subprocess.run([sys.executable, "-c", code], env=env,
                             capture_output=True, text=True, timeout=600)
    for line in result.stdout.splitlines():
        if line.startswith("JSON:"):
            return json.loads(line[5:])
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    raise RuntimeError(f"mode={mode} failed")


def main():
    print("Running buffer-resident query A/B (SIFT 1M / 200K, candidate_reserve=999)…\n")
    smart = run_one(0)
    dumb = run_one(1)

    print(f"{'mode':6s}  {'insert':>8s}  {'query':>8s}  {'total':>8s}  {'recall':>6s}")
    for label, r in [("SMART", smart), ("DUMB", dumb)]:
        tot = r["ins"] + r["qry"]
        print(f"{label:6s}  {r['ins']:>7.2f}s  {r['qry']:>7.2f}s  {tot:>7.2f}s  {r['recall']:>6.4f}")

    qry_speedup = dumb["qry"] / smart["qry"]
    tot_smart = smart["ins"] + smart["qry"]
    tot_dumb = dumb["ins"] + dumb["qry"]
    delta_pct = (1.0 - tot_smart / tot_dumb) * 100.0
    print(f"\nSMART query speedup: {qry_speedup:.2f}× at same recall")
    print(f"SMART total time:    {tot_smart:.2f}s vs DUMB {tot_dumb:.2f}s ({delta_pct:+.1f}% for SMART)")


if __name__ == "__main__":
    main()
