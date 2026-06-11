#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


DATASET = "sift"
THREADS = (4, 8, 16, 32, 64)
ALGORITHMS = (
    "streamseed_hybrid",
    "streamseed_hybrid_symphonyqg",
    "streamseed_hybrid_freshdiskann",
)


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


def snapshot_dirs(result_base: Path) -> set[Path]:
    if not result_base.exists():
        return set()
    return {path.resolve() for path in result_base.iterdir() if path.is_dir()}


def raw_result_candidates(result_base: Path, algorithm: str) -> list[Path]:
    candidates = [
        result_base / "ef-120",
        *sorted(result_base.glob("ef-*"), key=lambda path: path.stat().st_mtime, reverse=True),
    ]
    if algorithm == "streamseed_hybrid_freshdiskann":
        candidates = [
            *sorted(result_base.glob("L-*_R-*"), key=lambda path: path.stat().st_mtime, reverse=True),
            *candidates,
        ]
    return candidates


def archive_result_dir(
    result_base: Path,
    before_dirs: set[Path],
    algorithm: str,
    thread_count: int,
    overwrite: bool,
) -> Path:
    target = result_base / f"thread{int(thread_count)}"
    if target.exists():
        if not overwrite:
            raise RuntimeError(f"Target result directory already exists: {target}")
        shutil.rmtree(target)

    after_dirs = snapshot_dirs(result_base)
    new_dirs = sorted(after_dirs - before_dirs, key=lambda path: path.stat().st_mtime, reverse=True)

    src: Path | None = None
    for candidate in new_dirs:
        if candidate.name.startswith("thread"):
            continue
        src = candidate
        break

    if src is None:
        for candidate in raw_result_candidates(result_base, algorithm):
            if candidate.exists() and candidate.resolve() not in before_dirs:
                src = candidate
                break

    if src is None:
        for candidate in raw_result_candidates(result_base, algorithm):
            if candidate.exists():
                src = candidate
                break

    if src is None:
        raise RuntimeError(f"Could not find generated result directory under {result_base}")

    src.rename(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run streamseed hybrid algorithms on sift with multiple OMP_NUM_THREADS values."
    )
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--algorithms", type=parse_str_list, default=list(ALGORITHMS))
    parser.add_argument("--threads", type=parse_int_list, default=list(THREADS))
    parser.add_argument("--runbook", default="runbooks/general_experiment/general_experiment.yaml")
    parser.add_argument("--python-cmd", default="python")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path.cwd()
    runbook_path = repo / args.runbook

    for algorithm in normalize_str_list(args.algorithms):
        result_base = repo / "results" / args.dataset / algorithm
        for thread_count in args.threads:
            target = result_base / f"thread{int(thread_count)}"
            print(f"[threads] algorithm={algorithm} dataset={args.dataset} threads={thread_count}", flush=True)

            if target.exists() and not args.overwrite:
                if args.skip_existing:
                    print(f"[threads] skip existing {target}", flush=True)
                    continue
                raise RuntimeError(f"Target result directory already exists: {target}")

            if args.dry_run:
                print(
                    f"OMP_NUM_THREADS={thread_count} {args.python_cmd} run_benchmark.py "
                    f"--algorithm {algorithm} --dataset {args.dataset} "
                    f"--runbook {runbook_path} --enable-cache-profiling",
                    flush=True,
                )
                print(f"archive: {result_base}/<generated> -> {target}", flush=True)
                continue

            result_base.mkdir(parents=True, exist_ok=True)
            before_dirs = snapshot_dirs(result_base)

            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(thread_count)
            cmd = [
                args.python_cmd,
                "run_benchmark.py",
                "--algorithm",
                algorithm,
                "--dataset",
                args.dataset,
                "--runbook",
                str(runbook_path),
                "--enable-cache-profiling",
            ]
            subprocess.run(cmd, cwd=repo, env=env, check=True)
            archived = archive_result_dir(result_base, before_dirs, algorithm, thread_count, args.overwrite)
            print(f"[threads] archived {archived}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
