#!/usr/bin/env python3
"""Run SIFT10M streamseed_hybrid core/off ablation tests.

This script runs four benchmark jobs:
  1. streamseed_mode=streamseed_core -> results/.../test1
  2. streamseed_mode=streamseed_core -> results/.../test2
  3. streamseed_mode=off             -> results/.../test1-off
  4. streamseed_mode=off             -> results/.../test2-off

It updates both the `sift` and `sift10M` streamseed_hybrid config blocks because
this benchmark command uses `--dataset sift10M`, while some configs are often
copied from the `sift` block.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "bench/algorithms/streamseed_hybrid/config.yaml"
RESULT_ROOT = REPO / "results/sift10M/streamseed_hybrid"
RUNBOOK = "runbooks/general_experiment/general_experiment.yaml"
DATASETS_TO_EDIT = ("sift", "sift10M")
ALGORITHM = "streamseed_hybrid"


def find_block_end(text: str, start: int, indent: int) -> int:
    """Return the next line offset whose indentation is <= indent."""
    pattern = re.compile(rf"^ {{{0},{indent}}}\S", re.MULTILINE)
    match = pattern.search(text, start + 1)
    return match.start() if match else len(text)


def find_yaml_key(text: str, key: str, start: int = 0, end: int | None = None, indent: int = 0) -> re.Match[str]:
    end = len(text) if end is None else end
    pattern = re.compile(rf"^ {{{indent}}}{re.escape(key)}:\s*$", re.MULTILINE)
    match = pattern.search(text, start, end)
    if not match:
        raise RuntimeError(f"Cannot find YAML key {key!r} at indent {indent}")
    return match


def set_streamseed_mode(mode: str) -> None:
    if mode not in {"streamseed_core", "off"}:
        raise ValueError(f"unsupported streamseed_mode: {mode}")

    text = CONFIG.read_text()
    updated = text

    # Work from right to left so offsets remain valid after replacement.
    replacements: list[tuple[int, int, str]] = []
    for dataset in DATASETS_TO_EDIT:
        dataset_match = find_yaml_key(updated, dataset, indent=0)
        dataset_start = dataset_match.end()
        dataset_end = find_block_end(updated, dataset_match.start(), indent=0)

        algo_match = find_yaml_key(updated, ALGORITHM, start=dataset_start, end=dataset_end, indent=2)
        algo_start = algo_match.end()
        algo_end = find_block_end(updated, algo_match.start(), indent=2)
        algo_block = updated[algo_start:algo_end]

        mode_pattern = re.compile(r'"streamseed_mode"\s*:\s*"[^"]+"')
        mode_match = mode_pattern.search(algo_block)
        if not mode_match:
            raise RuntimeError(f"Cannot find streamseed_mode under {dataset}.{ALGORITHM}")

        abs_start = algo_start + mode_match.start()
        abs_end = algo_start + mode_match.end()
        replacements.append((abs_start, abs_end, f'"streamseed_mode": "{mode}"'))

    for start, end, replacement in sorted(replacements, reverse=True):
        updated = updated[:start] + replacement + updated[end:]

    if updated != text:
        CONFIG.write_text(updated)
    print(f"[config] set {', '.join(DATASETS_TO_EDIT)}.{ALGORITHM} streamseed_mode={mode}")


def snapshot_result_dirs() -> dict[str, float]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    return {
        child.name: child.stat().st_mtime
        for child in RESULT_ROOT.iterdir()
        if child.is_dir()
    }


def pick_touched_dir(before: dict[str, float]) -> Path:
    candidates: list[tuple[float, Path]] = []
    for child in RESULT_ROOT.iterdir():
        if not child.is_dir():
            continue
        old_mtime = before.get(child.name)
        new_mtime = child.stat().st_mtime
        if old_mtime is None or new_mtime > old_mtime + 1e-6:
            candidates.append((new_mtime, child))

    if not candidates:
        raise RuntimeError(f"No new or updated result directory found under {RESULT_ROOT}")

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def run_benchmark_once(target_name: str, overwrite: bool) -> None:
    target = RESULT_ROOT / target_name
    if target.exists():
        if not overwrite:
            raise RuntimeError(f"Target result directory already exists: {target}\nUse --overwrite to replace it.")
        shutil.rmtree(target)

    before = snapshot_result_dirs()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    cmd = [
        sys.executable,
        "run_benchmark.py",
        "--algorithm",
        ALGORITHM,
        "--dataset",
        "sift10M",
        "--runbook",
        RUNBOOK,
        "--enable-cache-profiling",
    ]

    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO, env=env, check=True)

    touched = pick_touched_dir(before)
    # Give the filesystem a tiny breath so subsequent mtime comparisons are clear.
    time.sleep(0.2)
    touched.rename(target)
    print(f"[rename] {touched.name} -> {target.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SIFT10M streamseed_hybrid core/off ablation tests")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="remove existing test1/test2/test1-off/test2-off result directories before renaming",
    )
    args = parser.parse_args()

    if not CONFIG.exists():
        raise RuntimeError(f"Missing config file: {CONFIG}")

    set_streamseed_mode("streamseed_core")
    run_benchmark_once("test1", overwrite=args.overwrite)
    run_benchmark_once("test2", overwrite=args.overwrite)

    set_streamseed_mode("off")
    run_benchmark_once("test1-off", overwrite=args.overwrite)
    run_benchmark_once("test2-off", overwrite=args.overwrite)

    print("[done] all four runs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
