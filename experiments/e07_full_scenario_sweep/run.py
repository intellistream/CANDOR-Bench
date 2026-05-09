"""e07 — Run all 9 scenarios × 3 algorithms at chosen scale(s)."""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import build, load_sift, cached_gt
from scenarios import (
    scenario_insert_only, scenario_insdel_stream, scenario_streaming,
    scenario_streaming_sliding, scenario_burst, scenario_drift,
    scenario_bulk_delete, scenario_churn,
)


CASES = {
    "200K": dict(slice_n=200_000, init_n=20_000,
                 batch=2_500, qstride=10_000, burst=10_000,
                 dataset_label="SIFT 200K"),
    "1M":   dict(slice_n=1_000_000, init_n=100_000,
                 batch=10_000, qstride=50_000, burst=50_000,
                 dataset_label="SIFT 1M"),
}


def runs_for(case):
    return [
        (scenario_insert_only,        dict(batch=case["batch"]*4)),
        (scenario_insdel_stream,      dict(batch=case["batch"], delete_ratio=0.5)),
        (scenario_streaming,          dict(batch=case["batch"], qstride=case["qstride"])),
        (scenario_streaming_sliding,  dict(batch=case["batch"], qstride=case["qstride"], lag=2)),
        (scenario_streaming_sliding,  dict(batch=case["batch"], qstride=case["qstride"], lag=4)),
        (scenario_burst,              dict(burst=case["burst"])),
        (scenario_drift,              dict(drift_chunks=5)),
        (scenario_bulk_delete,        dict(delete_frac=0.3)),
        (scenario_churn,              dict(cycles=10, churn_frac=0.05, maint_every=10)),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["200K", "1M", "both"], default="200K")
    args = p.parse_args()
    scales = ["200K", "1M"] if args.scale == "both" else [args.scale]

    for scale in scales:
        case = CASES[scale]
        print(f"\n##### scale = {scale} #####", flush=True)
        data, queries = load_sift(n_queries=1000, slice_n=case["slice_n"])
        gt_full = cached_gt(data, queries, k=10, tag=f"slice{case['slice_n']}_q1000")
        rows = []
        for sc_fn, kwargs in runs_for(case):
            sc_label = sc_fn.__name__.replace("scenario_", "")
            extras = ",".join(f"{k}={v}" for k, v in kwargs.items() if k != "batch")
            print(f"\n========== {sc_label}({extras})/{scale} ==========", flush=True)
            for algo in ["gamma", "faiss", "ivf"]:
                t_start = time.perf_counter()
                idx = build(algo, data.shape[1])
                m = sc_fn(idx, algo, data, queries, gt_full, case["init_n"], **kwargs)
                wall_runner = time.perf_counter() - t_start
                row = {"scale": scale, "algo": algo, "wall_runner_s": wall_runner, **m}
                rows.append(row)
                print(f"  {algo:6s}  wall={m['wall_s']:6.1f}s  recall={m['recall']:.4f}", flush=True)
                out = os.path.join(os.path.dirname(__file__), f"output_{scale}.json")
                with open(out, "w") as f:
                    json.dump(rows, f, indent=2)
        print(f"\n✓ saved output_{scale}.json")


if __name__ == "__main__":
    main()
