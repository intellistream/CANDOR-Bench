"""e21 — C++ GammaFresh component ablation.

The native C++ index has more sophisticated machinery than the Python POC:
  - spatial routing (multiple partitions, choose-owner via summary)
  - adaptive split (gamma_split_threshold, BIC, min_size_for_split)
  - placement controller (cost-model: newcomer/migration epsilon_ms)
  - maintenance scheduling (interval + query-side maint)

This experiment isolates each by ablating it via config:

  Variant            | Disabled by                        | Tests
  -------------------+------------------------------------+--------------------
  default            | nothing                            | reference
  no_spatial         | split_factor=1e9 (one partition)   | spatial routing value
  no_split           | gamma_split_threshold=1e9          | adaptive split value
  cost_off_admit     | newcomer_epsilon_ms=0              | cost-model admit ⇒ graph
  cost_off_buffer    | newcomer_epsilon_ms=1e9            | cost-model admit ⇒ buffer
  eager_maint        | maintenance_interval=1             | deferred maint value
  lazy_maint         | maintenance_interval=1e6           | scheduled-maint value

Each is run on each pattern (sequential / random / cluster / partial_reset)
at SIFT 200K, compared against the same workload on FaissHNSW direct.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, compute_gt, recall_at_k, make_pattern
import numpy as np


VARIANTS = {
    "default": {},
    "no_spatial": {"split_factor": 1e9},
    "no_split": {"gamma_split_threshold": 1e9, "min_size_for_split": int(1e9)},
    "cost_off_admit": {"placement_controller_newcomer_epsilon_ms": 0.0},
    "cost_off_buffer": {"placement_controller_newcomer_epsilon_ms": 1e9},
    "eager_maint": {"maintenance_interval": 1, "maintenance_query_interval": 1},
    "lazy_maint": {"maintenance_interval": int(1e6), "maintenance_query_interval": int(1e6)},
}


def make_gamma_config(d, **overrides):
    """Build a candy GammaFresh config with field overrides flattened."""
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
    # Defaults match algorithms_impl/GammaFresh/config/defaults.json so the
    # native index validates without "missing field" errors.
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


def run_native_workload(idx, name, data, queries, init_n, batch, qstride,
                         delete_fn):
    """Like _shared.run_workload but for the candy native interface."""
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
    alive_arr_np = np.array(alive_arr, dtype=np.int64)
    sgt_orig = alive_arr_np[sgt]
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

    # Late import — fails if candy C++ isn't built
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

    for variant_name, overrides in VARIANTS.items():
        print(f"\n========== {args.pattern}/{args.scale} variant={variant_name} {overrides}", flush=True)
        cfg = make_gamma_config(d, **overrides)
        idx = Index("GammaFresh", d, config=cfg)
        r = run_native_workload(idx, f"gamma_cpp[{variant_name}]",
                                 data, queries, init_n, batch, qstride, delete_fn)
        rows.append({"scale": args.scale, "pattern": args.pattern,
                     "variant": variant_name, "overrides": overrides, **r})
        print(f"  {r['name']:30s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} "
              f"qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out_path}")


if __name__ == "__main__":
    main()
