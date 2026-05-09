"""e15 — Full delete-pattern × backend × architecture matrix."""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, compute_gt, recall_at_k
from _shared.gamma_py_v2 import GammaPyHybridV2
from _shared.gamma_py import HnswlibBackend, FaissHnswBackend
import numpy as np


def deletes_sequential(alive, n_to_del, rng, data, deleted_so_far, **kw):
    return np.arange(deleted_so_far, deleted_so_far + n_to_del, dtype=np.int64)


def deletes_random(alive, n_to_del, rng, data, deleted_so_far, **kw):
    arr = np.fromiter(alive, dtype=np.int64, count=len(alive))
    if len(arr) < n_to_del:
        return arr
    return rng.choice(arr, size=n_to_del, replace=False)


def deletes_cluster(alive, n_to_del, rng, data, deleted_so_far, **kw):
    arr = np.fromiter(alive, dtype=np.int64, count=len(alive))
    if len(arr) <= n_to_del:
        return arr
    pivot = data[int(rng.choice(arr))]
    sample_size = min(len(arr), 5000)  # speedup: sample for distance compute
    sampled = rng.choice(arr, size=sample_size, replace=False)
    diffs = data[sampled] - pivot
    dists = (diffs * diffs).sum(axis=1)
    return sampled[np.argpartition(dists, min(n_to_del, len(sampled)-1))[:n_to_del]]


def deletes_partial_reset(alive, n_to_del, rng, data, deleted_so_far, batch_idx, **kw):
    """Every 8 batches do a BIG delete of n_to_del * 8."""
    if batch_idx % 8 == 0:
        big = n_to_del * 8
        arr = np.fromiter(alive, dtype=np.int64, count=len(alive))
        if len(arr) <= big:
            return arr
        return rng.choice(arr, size=big, replace=False)
    return np.array([], dtype=np.int64)


def make_pattern(kind):
    return {
        "sequential": deletes_sequential,
        "random": deletes_random,
        "cluster": deletes_cluster,
        "partial_reset": deletes_partial_reset,
    }[kind]


def run_workload(idx, name, data, queries, init_n, batch, qstride,
                  delete_fn, use_gamma, has_mark_deleted, max_deletes_per_batch=None):
    n = data.shape[0]
    rng = np.random.default_rng(42)
    if use_gamma:
        idx.initial_load(np.arange(init_n, dtype=np.int64),
                          np.ascontiguousarray(data[:init_n]))
    else:
        idx.add(np.ascontiguousarray(data[:init_n]),
                np.arange(init_n, dtype=np.int64))
    alive_set = set(range(init_n))
    deleted_so_far = 0
    batch_idx = 0
    query_lat = []
    n_del_per = max_deletes_per_batch if max_deletes_per_batch else batch
    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        new_ids = np.arange(lo, hi, dtype=np.int64)
        new_vecs = np.ascontiguousarray(data[lo:hi])
        if use_gamma:
            idx.add(new_ids, new_vecs)
        else:
            idx.add(new_vecs, new_ids)
        alive_set.update(new_ids.tolist())
        if batch_idx >= 1 and len(alive_set) > n_del_per:
            del_ids = delete_fn(alive_set, n_del_per, rng, data, deleted_so_far,
                                batch_idx=batch_idx)
            del_ids = np.asarray(del_ids, dtype=np.int64)
            if len(del_ids) > 0:
                if use_gamma:
                    idx.delete(del_ids)
                elif has_mark_deleted:
                    for did in del_ids:
                        idx.mark_deleted(int(did))
                for did in del_ids:
                    alive_set.discard(int(did))
                deleted_so_far += len(del_ids)
        if (lo - init_n) % qstride == 0:
            if use_gamma:
                idx.maintain()
            tq = time.perf_counter()
            idx.search(queries, 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
        batch_idx += 1
    if use_gamma:
        idx.maintain()
    alive_arr = sorted(alive_set)
    sgt = compute_gt(data[alive_arr], queries, 10)
    alive_arr_np = np.array(alive_arr, dtype=np.int64)
    sgt_orig = alive_arr_np[sgt]
    nbrs, _ = idx.search(queries, 10)
    return {
        "name": name,
        "total_s": time.perf_counter() - t0,
        "n_delete": deleted_so_far,
        "n_alive": len(alive_set),
        "query_latency_ms_p95": float(np.percentile(np.asarray(query_lat) * 1000, 95)) if query_lat else 0.0,
        "recall": recall_at_k(nbrs.astype(np.int64), sgt_orig.astype(np.int64)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["200K", "1M"], default="200K")
    args = p.parse_args()
    if args.scale == "200K":
        slice_n, init_n, batch, qstride, n_queries = 200_000, 20_000, 2_500, 10_000, 500
    else:
        slice_n, init_n, batch, qstride, n_queries = 1_000_000, 100_000, 10_000, 50_000, 1000
    data, queries = load_dataset("sift", n_queries=n_queries, slice_n=slice_n)
    n, d = data.shape
    rows = []

    patterns = ["sequential", "random", "cluster", "partial_reset"]
    backends = [
        ("hnswlib", lambda: HnswlibBackend(d, max_elements=n)),
        ("faiss_hnsw", lambda: FaissHnswBackend(d, max_elements=n)),
    ]

    for pattern in patterns:
        print(f"\n========== {args.scale} / pattern={pattern} ==========", flush=True)
        delete_fn = make_pattern(pattern)
        for be_name, be_factory in backends:
            be = be_factory()
            g = GammaPyHybridV2(be, d, buf_capacity=batch * 50)
            r = run_workload(g, f"gamma_v2+{be_name}", data, queries, init_n,
                              batch, qstride, delete_fn,
                              use_gamma=True, has_mark_deleted=False)
            rows.append({"scale": args.scale, "pattern": pattern, **r})
            print(f"  {r['name']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms n_alive={r['n_alive']}", flush=True)
            be2 = be_factory()
            r = run_workload(be2, f"{be_name} direct", data, queries, init_n,
                              batch, qstride, delete_fn,
                              use_gamma=False, has_mark_deleted=True)
            rows.append({"scale": args.scale, "pattern": pattern, **r})
            print(f"  {r['name']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms n_alive={r['n_alive']}", flush=True)
            out = os.path.join(os.path.dirname(__file__), f"output_{args.scale}.json")
            with open(out, "w") as f:
                json.dump(rows, f, indent=2)
    print(f"\n✓ saved {out}")


if __name__ == "__main__":
    main()
