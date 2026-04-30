#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml

from datasets.registry import get_dataset


def _write_gt100(path: Path, ids: np.ndarray, dists: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.asarray([ids.shape[0]], dtype=np.uint32).tofile(f)
        np.asarray([ids.shape[1]], dtype=np.uint32).tofile(f)
        ids.astype(np.uint32, copy=False).tofile(f)
        dists.astype(np.float32, copy=False).tofile(f)


def _topk_l2(base: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    q_norm = np.sum(queries * queries, axis=1, keepdims=True)
    b_norm = np.sum(base * base, axis=1, dtype=np.float32)[None, :]
    dists = q_norm + b_norm - 2.0 * (queries @ base.T)
    dists = np.maximum(dists, 0.0)
    idx = np.argpartition(dists, kth=min(k, dists.shape[1] - 1), axis=1)[:, :k]
    top_d = np.take_along_axis(dists, idx, axis=1)
    order = np.argsort(top_d, axis=1)
    ids = np.take_along_axis(idx, order, axis=1).astype(np.uint32, copy=False)
    top_d = np.take_along_axis(top_d, order, axis=1).astype(np.float32, copy=False)
    return ids, top_d


def _resolve_runbook(runbook: str) -> Path:
    path = Path(runbook)
    if path.exists():
        return path
    root = Path(__file__).resolve().parents[1] / "runbooks"
    candidate = root / f"{runbook}.yaml"
    if candidate.exists():
        return candidate
    for subdir in root.iterdir():
        if subdir.is_dir():
            candidate = subdir / f"{runbook}.yaml"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Runbook not found: {runbook}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate dynamic gt100 files for small streaming runbooks via brute force")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runbook", required=True)
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    ds.prepare(skip_data=False)
    data = np.asarray(ds.get_dataset(), dtype=np.float32)
    queries = np.asarray(ds.get_queries(), dtype=np.float32)
    runbook_path = _resolve_runbook(args.runbook)
    config = yaml.safe_load(runbook_path.read_text())
    if args.dataset not in config:
        raise ValueError(f"{args.dataset} not found in {runbook_path}")

    dataset_cfg = config[args.dataset]
    gt_dir = Path(ds.basedir) / str(ds.nb) / runbook_path.name
    gt_dir.mkdir(parents=True, exist_ok=True)

    active_end = 0
    batch_insert_idx = 0
    for step in sorted(k for k in dataset_cfg if isinstance(k, int)):
        entry = dataset_cfg[step]
        op = entry["operation"]
        if op in {"initial", "initial_load"}:
            active_end = entry["end"]
            continue
        if op not in {"batch_insert", "batch_insert_delete"}:
            continue

        start = entry["start"]
        end = entry["end"]
        batch_size = entry.get("batchSize", 2500)
        total_data = end - start
        interval = max(1, total_data // 100)
        counter = 0
        batch_idx = 0

        for batch_start in range(start, end, batch_size):
            batch_end = min(batch_start + batch_size, end)
            active_end = max(active_end, batch_end)
            counter += batch_end - batch_start
            if counter < interval:
                batch_idx += 1
                continue

            base = data[:active_end]
            ids, dists = _topk_l2(base, queries, args.k)
            out = gt_dir / f"batch{batch_insert_idx}_{batch_idx}.gt100"
            _write_gt100(out, ids, dists)
            print(out)
            counter = 0
            batch_idx += 1

        batch_insert_idx += 1


if __name__ == "__main__":
    main()
