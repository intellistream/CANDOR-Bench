"""e22 — Can the C++ index be saved by tuning?

e21 showed the C++ default config (split_factor=1.5, gamma_split_threshold=2.0,
min_size_for_split=4) is severely broken on partial_reset (recall 0.006) and
slow on every pattern (~2.6× slower than no-spatial). This experiment tests
whether configurations more aligned with the workload size restore good
behavior — and benchmarks against gamma_v2+FaissHNSW (the simple Python POC
with the same backend).

Variants tested:
  default_cpp        : same as e21 default, for reference
  no_spatial         : split_factor=1e9, single partition (e21 winner)
  e04_tuned          : the original e04 config (split_factor=4, γ=0.4)
  large_split        : split_factor=16, γ=1.0 (mirrors gamma_py defaults)
  python_poc_v2      : gamma_v2+FaissHNSW for one-to-one comparison
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, compute_gt, recall_at_k, make_pattern, run_workload
from _shared.gamma_py import FaissHnswBackend
from _shared.gamma_py_v2 import GammaPyHybridV2
import numpy as np


CPP_VARIANTS = {
    "default_cpp": {},
    "no_spatial": {"split_factor": 1e9},
    "e04_tuned": {"split_factor": 4.0, "gamma_split_threshold": 0.4,
                   "min_size_for_split": 64,
                   "maintenance_interval": 4, "maintenance_query_interval": 4},
    "large_split": {"split_factor": 16.0, "gamma_split_threshold": 1.0,
                     "min_size_for_split": 512,
                     "maintenance_interval": 4, "maintenance_query_interval": 4},
}


def make_gamma_config(d, **overrides):
    placement_cfg = {
        "ema_beta": 0.2,
        "newcomer_epsilon_ms": 0.0,
        "migration_epsilon_ms": 0.0,
        "default_expected_lifetime_ops": 100.0,
    }
    pc_overrides = {k.replace("placement_controller_", ""): v
                     for k, v in overrides.items()
                     if k.startswith("placement_controller_")}
    placement_cfg.update(pc_overrides)
    gamma_cfg = {
        "backend": "FaissHNSW",
        "log_interval": 0,
        "split_factor": 1.5,
        "gamma_split_threshold": 2.0,
        "maintenance_interval": 100,
        "maintenance_query_interval": 100,
        "maintenance_query_threshold_scale": 0.5,
        "min_size_for_split": 4,
        "candidate_reserve_size": 8,
        "ema_beta": 0.2,
        "radius_beta": 0.1,
        "bic_min_variance": 1e-12,
        "bic_gaussian_log_coeff": -0.5,
        "bic_log_likelihood_weight": -2.0,
        "bic_min_improvement": 2.0,
        "bic_min_points": 2,
        "slab_size": 2097152,
        "internal_logs_dir": "/tmp/gamma_logs",
        "placement_controller": placement_cfg,
    }
    gamma_cfg.update({k: v for k, v in overrides.items()
                      if not k.startswith("placement_controller_")})
    faiss_cfg = {"m": 32, "ef_construction": 120, "ef_search": 80,
                 "tombstone_rebuild_threshold": 512}
    return {"global": {"dim": d},
            "indexes": {"gamma_fresh": gamma_cfg, "faiss_hnsw": faiss_cfg}}


def run_native(idx, name, data, queries, init_n, batch, qstride, delete_fn):
    n = data.shape[0]
    rng = np.random.default_rng(42)
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                      np.ascontiguousarray(data[:init_n]))
    alive_set = set(range(init_n))
    deleted_so_far = 0
    batch_idx = 0
    query_lat = []
    n_del_per = batch
    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        new_ids = np.arange(lo, hi, dtype=np.uint64)
        new_vecs = np.ascontiguousarray(data[lo:hi])
        idx.add(new_ids, new_vecs)
        alive_set.update(new_ids.tolist())
        if batch_idx >= 1 and len(alive_set) > n_del_per:
            del_ids = delete_fn(alive_set, n_del_per, rng, data, deleted_so_far,
                                batch_idx=batch_idx)
            del_ids = np.asarray(del_ids, dtype=np.uint64)
            if len(del_ids) > 0:
                idx.delete(del_ids)
                for did in del_ids:
                    alive_set.discard(int(did))
                deleted_so_far += len(del_ids)
        if (lo - init_n) % qstride == 0:
            idx.maintain(0, 0, False)
            tq = time.perf_counter()
            idx.search(np.ascontiguousarray(queries), 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
        batch_idx += 1
    idx.maintain(0, 0, False)
    alive_arr = sorted(alive_set)
    sgt = compute_gt(data[alive_arr], queries, 10)
    sgt_orig = np.array(alive_arr, dtype=np.int64)[sgt]
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
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
    p.add_argument("--scale", default="200K", choices=["200K", "1M"])
    p.add_argument("--pattern", default="cluster",
                   choices=["sequential", "random", "cluster", "partial_reset"])
    args = p.parse_args()

    from candy import Index

    if args.scale == "200K":
        slice_n, init_n, batch, qstride, n_queries = 200_000, 20_000, 2_500, 10_000, 500
    else:
        slice_n, init_n, batch, qstride, n_queries = 1_000_000, 100_000, 10_000, 50_000, 1000

    data, queries = load_dataset("sift", n_queries=n_queries, slice_n=slice_n)
    n, d = data.shape
    rows = []
    out_path = os.path.join(os.path.dirname(__file__),
                             f"output_sift_{args.pattern}_{args.scale}.json")
    delete_fn = make_pattern(args.pattern)

    # Python POC reference (gamma_v2 + FaissHNSW backend)
    print(f"\n========== {args.pattern}/{args.scale} python_poc_v2 (gamma_v2+FaissHNSW)", flush=True)
    be = FaissHnswBackend(d, max_elements=n)
    g = GammaPyHybridV2(be, d, buf_capacity=batch * 50)
    r = run_workload(g, "python_poc_v2+FaissHNSW", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=True, has_mark_deleted=False)
    rows.append({"scale": args.scale, "pattern": args.pattern,
                 "variant": "python_poc_v2", **r})
    print(f"  {r['name']:35s} total={r['total_s']:6.1f}s recall={r['recall']:.4f}", flush=True)
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # FaissHNSW direct baseline (no gamma)
    print(f"\n========== {args.pattern}/{args.scale} FaissHNSW direct (baseline)", flush=True)
    be2 = FaissHnswBackend(d, max_elements=n)
    r = run_workload(be2, "FaissHNSW direct", data, queries, init_n,
                      batch, qstride, delete_fn,
                      use_gamma=False, has_mark_deleted=True)
    rows.append({"scale": args.scale, "pattern": args.pattern,
                 "variant": "(faiss_hnsw direct)", **r})
    print(f"  {r['name']:35s} total={r['total_s']:6.1f}s recall={r['recall']:.4f}", flush=True)
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    # Native C++ variants
    for variant_name, overrides in CPP_VARIANTS.items():
        print(f"\n========== {args.pattern}/{args.scale} cpp variant={variant_name} {overrides}", flush=True)
        cfg = make_gamma_config(d, **overrides)
        idx = Index("GammaFresh", d, config=cfg)
        r = run_native(idx, f"gamma_cpp[{variant_name}]+FaissHNSW",
                        data, queries, init_n, batch, qstride, delete_fn)
        rows.append({"scale": args.scale, "pattern": args.pattern,
                     "variant": variant_name, "overrides": overrides, **r})
        print(f"  {r['name']:35s} total={r['total_s']:6.1f}s recall={r['recall']:.4f}", flush=True)
        with open(out_path, "w") as f: json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
