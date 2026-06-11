#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


DATASETS = ("msong",)
EFS = (40, 80, 120)
DEGREE_BOUNDS = (32,64,96)


def parse_int_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_str_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def normalize_str_list(values) -> list[str]:
    if isinstance(values, str):
        return parse_str_list(values)
    return list(values)


def section(text: str, start_pattern: str, end_pattern: str | None) -> tuple[int, int]:
    start_match = re.search(start_pattern, text, flags=re.MULTILINE)
    if start_match is None:
        raise RuntimeError(f"Could not find section start: {start_pattern}")
    start = start_match.start()
    if end_pattern is None:
        return start, len(text)
    end_match = re.search(end_pattern, text[start_match.end() :], flags=re.MULTILINE)
    end = len(text) if end_match is None else start_match.end() + end_match.start()
    return start, end


def replace_once(pattern: str, repl: str, text: str, label: str) -> str:
    new_text, count = re.subn(pattern, repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not update {label}; pattern matched {count} times")
    return new_text


def update_config(config_path: Path, dataset: str, degree_bound: int, ef: int) -> None:
    text = config_path.read_text()
    dataset_start, dataset_end = section(text, rf"^{re.escape(dataset)}:\s*$", r"^\S")
    dataset_block = text[dataset_start:dataset_end]

    algo_start, algo_end = section(
        dataset_block,
        r"^  streamseed_hybrid_symphonyqg:\s*$",
        r"^  \S",
    )
    algo_block = dataset_block[algo_start:algo_end]
    algo_block = replace_once(
        r'"degree_bound"\s*:\s*\d+',
        f'"degree_bound": {int(degree_bound)}',
        algo_block,
        f"{dataset} streamseed_hybrid_symphonyqg degree_bound",
    )
    algo_block = replace_once(
        r'"ef"\s*:\s*\d+',
        f'"ef": {int(ef)}',
        algo_block,
        f"{dataset} streamseed_hybrid_symphonyqg ef",
    )

    dataset_block = dataset_block[:algo_start] + algo_block + dataset_block[algo_end:]
    text = text[:dataset_start] + dataset_block + text[dataset_end:]
    config_path.write_text(text)


def snapshot_dirs(result_base: Path) -> set[Path]:
    if not result_base.exists():
        return set()
    return {path.resolve() for path in result_base.iterdir() if path.is_dir()}


def archive_result_dir(
    result_base: Path,
    before_dirs: set[Path],
    degree_bound: int,
    ef: int,
    overwrite: bool,
) -> Path:
    target = result_base / f"ef-{int(ef)}-degree-{int(degree_bound)}"
    if target.exists():
        if not overwrite:
            raise RuntimeError(f"Target result directory already exists: {target}")
        shutil.rmtree(target)

    after_dirs = snapshot_dirs(result_base)
    new_dirs = sorted(after_dirs - before_dirs, key=lambda path: path.stat().st_mtime, reverse=True)
    raw_candidates = [
        result_base / "ef-120",
        result_base / f"ef-{int(ef)}",
    ]

    src: Path | None = None
    for candidate in new_dirs:
        if candidate.name.startswith("ef-") and "-degree-" in candidate.name:
            continue
        src = candidate
        break
    if src is None:
        for candidate in raw_candidates:
            if candidate.exists() and candidate.resolve() not in before_dirs:
                src = candidate
                break
    if src is None:
        for candidate in raw_candidates:
            if candidate.exists():
                src = candidate
                break

    if src is None:
        raise RuntimeError(f"Could not find generated result directory under {result_base}")

    src.rename(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run streamseed_hybrid_symphonyqg degree_bound/ef grid for msong."
    )
    parser.add_argument("--config", default="bench/algorithms/streamseed_hybrid/config.yaml")
    parser.add_argument("--runbook", default="runbooks/general_experiment/general_experiment.yaml")
    parser.add_argument("--algorithm", default="streamseed_hybrid_symphonyqg")
    parser.add_argument("--datasets", type=parse_str_list, default=list(DATASETS))
    parser.add_argument("--efs", type=parse_int_list, default=list(EFS))
    parser.add_argument("--degree-bounds", type=parse_int_list, default=list(DEGREE_BOUNDS))
    parser.add_argument("--omp-num-threads", type=int, default=16)
    parser.add_argument("--python-cmd", default="python")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path.cwd()
    config_path = repo / args.config
    runbook_path = repo / args.runbook
    original_config = config_path.read_text()

    try:
        for dataset in normalize_str_list(args.datasets):
            for degree_bound in args.degree_bounds:
                for ef in args.efs:
                    result_base = repo / "results" / dataset / args.algorithm
                    print(f"[grid] dataset={dataset} degree_bound={degree_bound} ef={ef}", flush=True)
                    if args.dry_run:
                        print(
                            f"OMP_NUM_THREADS={args.omp_num_threads} "
                            f"{args.python_cmd} run_benchmark.py --algorithm {args.algorithm} "
                            f"--dataset {dataset} --runbook {runbook_path} --enable-cache-profiling",
                            flush=True,
                        )
                        print(
                            f"archive: {result_base}/ef-120 -> "
                            f"{result_base}/ef-{ef}-degree-{degree_bound}",
                            flush=True,
                        )
                        continue

                    update_config(config_path, dataset, degree_bound, ef)
                    result_base.mkdir(parents=True, exist_ok=True)
                    before_dirs = snapshot_dirs(result_base)

                    env = os.environ.copy()
                    env["OMP_NUM_THREADS"] = str(args.omp_num_threads)
                    cmd = [
                        args.python_cmd,
                        "run_benchmark.py",
                        "--algorithm",
                        args.algorithm,
                        "--dataset",
                        dataset,
                        "--runbook",
                        str(runbook_path),
                        "--enable-cache-profiling",
                    ]
                    subprocess.run(cmd, cwd=repo, env=env, check=True)
                    archived = archive_result_dir(result_base, before_dirs, degree_bound, ef, args.overwrite)
                    print(f"[grid] archived {archived}", flush=True)
    finally:
        if not args.dry_run:
            config_path.write_text(original_config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
