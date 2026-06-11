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
INDEXKEYS = ("HNSWIncremental32", "HNSWIncremental16","HNSWIncremental24", "HNSWIncremental40")


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


def update_config(config_path: Path, dataset: str, indexkey: str, ef: int) -> None:
    text = config_path.read_text()
    dataset_start, dataset_end = section(text, rf"^{re.escape(dataset)}:\s*$", r"^\S")
    dataset_block = text[dataset_start:dataset_end]

    algo_start, algo_end = section(
        dataset_block,
        r"^  streamseed_hybrid:\s*$",
        r"^  \S",
    )
    algo_block = dataset_block[algo_start:algo_end]
    algo_block = replace_once(
        r'"indexkey"\s*:\s*"[^"]+"',
        f'"indexkey": "{indexkey}"',
        algo_block,
        f"{dataset} streamseed_hybrid indexkey",
    )
    algo_block = replace_once(
        r'"ef"\s*:\s*\d+',
        f'"ef": {int(ef)}',
        algo_block,
        f"{dataset} streamseed_hybrid ef",
    )

    dataset_block = dataset_block[:algo_start] + algo_block + dataset_block[algo_end:]
    text = text[:dataset_start] + dataset_block + text[dataset_end:]
    config_path.write_text(text)


def indexkey_suffix(indexkey: str) -> str:
    match = re.search(r"(\d+)$", indexkey)
    if match is None:
        return re.sub(r"\W+", "", indexkey).lower()
    return match.group(1)


def snapshot_dirs(result_base: Path) -> set[Path]:
    if not result_base.exists():
        return set()
    return {path.resolve() for path in result_base.iterdir() if path.is_dir()}


def archive_result_dir(
    result_base: Path,
    before_dirs: set[Path],
    indexkey: str,
    ef: int,
    overwrite: bool,
) -> Path:
    target = result_base / f"indexkey{indexkey_suffix(indexkey)}-ef{int(ef)}"
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
        if candidate.name.startswith("indexkey"):
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
        description="Run streamseed_hybrid HNSW indexkey/ef grid for sift, msong, and glove."
    )
    parser.add_argument("--config", default="bench/algorithms/streamseed_hybrid/config.yaml")
    parser.add_argument("--runbook", default="runbooks/general_experiment/general_experiment.yaml")
    parser.add_argument("--algorithm", default="streamseed_hybrid")
    parser.add_argument("--datasets", type=parse_str_list, default=list(DATASETS))
    parser.add_argument("--efs", type=parse_int_list, default=list(EFS))
    parser.add_argument("--indexkeys", type=parse_str_list, default=list(INDEXKEYS))
    parser.add_argument("--omp-num-threads", type=int, default=4)
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
            for indexkey in normalize_str_list(args.indexkeys):
                for ef in args.efs:
                    result_base = repo / "results" / dataset / args.algorithm
                    print(f"[grid] dataset={dataset} indexkey={indexkey} ef={ef}", flush=True)
                    if args.dry_run:
                        print(
                            f"OMP_NUM_THREADS={args.omp_num_threads} "
                            f"{args.python_cmd} run_benchmark.py --algorithm {args.algorithm} "
                            f"--dataset {dataset} --runbook {runbook_path} --enable-cache-profiling",
                            flush=True,
                        )
                        print(
                            f"archive: {result_base}/ef-120 -> "
                            f"{result_base}/indexkey{indexkey_suffix(indexkey)}-ef{ef}",
                            flush=True,
                        )
                        continue

                    update_config(config_path, dataset, indexkey, ef)
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
                    archived = archive_result_dir(result_base, before_dirs, indexkey, ef, args.overwrite)
                    print(f"[grid] archived {archived}", flush=True)
    finally:
        if not args.dry_run:
            config_path.write_text(original_config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
