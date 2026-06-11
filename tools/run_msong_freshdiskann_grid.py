#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_RS = (16, 32, 48)
DEFAULT_EFS = (40, 80, 120)


def _replace_one(pattern: str, repl: str, text: str, label: str) -> str:
    new_text, count = re.subn(pattern, repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not update {label}; pattern did not match exactly once")
    return new_text


def _section(text: str, start_pattern: str, end_pattern: str | None) -> tuple[int, int]:
    start_match = re.search(start_pattern, text, flags=re.MULTILINE)
    if start_match is None:
        raise RuntimeError(f"Could not find section start: {start_pattern}")
    start = start_match.start()
    if end_pattern is None:
        return start, len(text)
    end_match = re.search(end_pattern, text[start_match.end() :], flags=re.MULTILINE)
    if end_match is None:
        return start, len(text)
    return start, start_match.end() + end_match.start()


def update_config(config_path: Path, r_value: int, ef_value: int) -> int:
    text = config_path.read_text()

    msong_start, msong_end = _section(text, r"^msong:\s*$", r"^\S")
    msong = text[msong_start:msong_end]

    algo_start, algo_end = _section(
        msong,
        r"^  streamseed_hybrid_freshdiskann:\s*$",
        r"^  \S",
    )
    algo = msong[algo_start:algo_end]

    old_l_match = re.search(r'"L"\s*:\s*(\d+)', algo)
    if old_l_match is None:
        raise RuntimeError('Could not find "L" in msong streamseed_hybrid_freshdiskann args')
    l_value = int(old_l_match.group(1))

    algo = _replace_one(r'"R"\s*:\s*\d+', f'"R": {r_value}', algo, "R")
    algo = _replace_one(r'"ef"\s*:\s*\d+', f'"ef": {ef_value}', algo, "ef")

    msong = msong[:algo_start] + algo + msong[algo_end:]
    text = text[:msong_start] + msong + text[msong_end:]
    config_path.write_text(text)
    return l_value


def rename_result_dir(results_root: Path, l_value: int, r_value: int, ef_value: int, overwrite: bool) -> None:
    src = results_root / f"L-{l_value}_R-{r_value}"
    dst = results_root / f"L-{l_value}_R-{r_value}_ef_{ef_value}"
    if not src.exists():
        raise RuntimeError(f"Expected result directory does not exist: {src}")
    if dst.exists():
        if not overwrite:
            raise RuntimeError(f"Target result directory already exists: {dst}")
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    src.rename(dst)


def parse_int_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the msong streamseed_hybrid_freshdiskann R/ef grid and rename result directories."
    )
    parser.add_argument("--config", default="bench/algorithms/streamseed_hybrid/config.yaml")
    parser.add_argument("--runbook", default="runbooks/general_experiment/general_experiment.yaml")
    parser.add_argument("--algorithm", default="streamseed_hybrid_freshdiskann")
    parser.add_argument("--dataset", default="msong")
    parser.add_argument("--rs", type=parse_int_list, default=list(DEFAULT_RS), help="comma list, default: 16,32,48")
    parser.add_argument("--efs", type=parse_int_list, default=list(DEFAULT_EFS), help="comma list, default: 40,80,120")
    parser.add_argument("--omp-num-threads", type=int, default=16)
    parser.add_argument("--results-root", default="results/msong/streamseed_hybrid_freshdiskann")
    parser.add_argument("--overwrite", action="store_true", help="replace renamed result directories if they exist")
    parser.add_argument("--dry-run", action="store_true", help="print planned commands without changing config or running")
    args = parser.parse_args()

    repo = Path.cwd()
    config_path = repo / args.config
    runbook_path = repo / args.runbook
    results_root = repo / args.results_root
    original_config = config_path.read_text()

    try:
        for r_value in args.rs:
            for ef_value in args.efs:
                print(f"[grid] R={r_value} ef={ef_value}", flush=True)
                if args.dry_run:
                    print(
                        "OMP_NUM_THREADS="
                        f"{args.omp_num_threads} python run_benchmark.py --algorithm {args.algorithm} "
                        f"--dataset {args.dataset} --runbook {runbook_path} --enable-cache-profiling",
                        flush=True,
                    )
                    continue

                l_value = update_config(config_path, r_value, ef_value)
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(args.omp_num_threads)
                cmd = [
                    sys.executable,
                    "run_benchmark.py",
                    "--algorithm",
                    args.algorithm,
                    "--dataset",
                    args.dataset,
                    "--runbook",
                    str(runbook_path),
                    "--enable-cache-profiling",
                ]
                subprocess.run(cmd, cwd=repo, env=env, check=True)
                rename_result_dir(results_root, l_value, r_value, ef_value, args.overwrite)
    finally:
        if not args.dry_run:
            config_path.write_text(original_config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
