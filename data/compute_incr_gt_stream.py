from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DATA_ROOT.parent
WORK_ROOT = REPO_ROOT  # alias to make intent clearer

DATASETS = []

for ds_name, threads, dim, prefix in [
    ("sift", 64, 128, "sift"),
    # ("glove1.2m", 64, 100, "glove1.2m"),
    # ("glove2.2m", 64, 100, "glove2.2m"),
    # ("gist", 64, 960, "gist"),
    # ("msong", 64, 420, "msong"),
    # ("deep1M", 64, 256, "deep1M"),
]:
    for inc in [20, 100, 200]:
        k = 20
        DATASETS.append(
            {
                "name": ds_name,
                "base_bin": f"{ds_name}/{prefix}_base.bin",
                "qstream_bin": f"{ds_name}/{prefix}_query_stream.bin",
                "incremental_gt": f"{ds_name}/{prefix}_stream_i{inc}.gt{k}",
                "k": k,
                "increment": inc,
                "noise_std": 1e-3,
                "noise_seed": 42,
                "threads": threads,
            }
        )

# FreewayML datasets
for pattern in ["G", "A1", "A2", "B1", "B2", "B3", "B4", "C1", "C2"]:
    for inc in [20, 100, 200]:
        k = 20
        DATASETS.append(
            {
                "name": f"FreewayML_{pattern}",
                "base_bin": f"FreewayML/{pattern}/base.bin",
                "qstream_bin": f"FreewayML/{pattern}/query_stream.bin",
                "incremental_gt": f"FreewayML/{pattern}/stream_i{inc}.gt{k}",
                "k": k,
                "increment": inc,
                "noise_std": 1e-3,
                "noise_seed": 42,
                "threads": 64,
            }
        )


UTILS_BUILD = Path(__file__).resolve().parent.parent / "tools" / "build"


def ensure_exists(path: Path) -> None:
    if not path.exists():
        print(f"Required file not found: {path}", file=sys.stderr)
        sys.exit(1)


def load_bin(path: Path) -> tuple[np.ndarray, int, int]:
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"{path} missing header")
        npts, dim = struct.unpack("ii", header)
        data = np.fromfile(f, dtype=np.float32)
    if data.size != npts * dim:
        raise ValueError(f"{path} size mismatch")
    return data.reshape(npts, dim), npts, dim


def write_bin(path: Path, data: np.ndarray) -> None:
    npts, dim = data.shape
    with path.open("wb") as f:
        f.write(struct.pack("ii", npts, dim))
        data.astype(np.float32).tofile(f)


def resolve_dataset_paths(cfg: dict) -> tuple[Path, Path, Path]:
    prefix = cfg.get("prefix", cfg["name"])
    dataset_dir = Path(cfg["dir"]) if "dir" in cfg else None

    base_filename = cfg.get("base_bin", f"{prefix}_base.bin")

    if "qstream_bin" in cfg:
        query_filename = cfg["qstream_bin"]
    elif "query_bin" in cfg:
        query_filename = cfg["query_bin"]
    else:
        query_suffix = cfg.get("query_suffix", "query_stream")
        query_filename = f"{prefix}_{query_suffix}.bin"

    incr_filename = cfg.get(
        "incremental_gt",
        f"{prefix}_qhot_i{cfg['increment']}.gt{cfg['k']}",
    )

    def to_path(filename: str) -> Path:
        if dataset_dir is None:
            return DATA_ROOT / filename
        return DATA_ROOT / dataset_dir / filename

    return to_path(base_filename), to_path(query_filename), to_path(incr_filename)


def generate_noisy_queries(
    base: Path, query: Path, noise_std: float, seed: int | None
) -> None:
    data, _, _ = load_bin(base)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, size=data.shape).astype(np.float32)
    query.parent.mkdir(parents=True, exist_ok=True)
    write_bin(query, (data + noise).astype(np.float32))


def run(cmd: list[str]) -> None:
    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    compute_incr_gt = UTILS_BUILD / "compute_incr_gt"
    ensure_exists(compute_incr_gt)

    for cfg in DATASETS:
        name = cfg["name"]
        base, query, incr_gt = resolve_dataset_paths(cfg)

        ensure_exists(base)

        print(f"[{name}] Generating noisy query data: {query}")
        generate_noisy_queries(
            base,
            query,
            cfg.get("noise_std", 1e-3),
            cfg.get("noise_seed"),
        )

        incr_gt.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] Computing incremental ground truth: {incr_gt}")

        run(
            [
                str(compute_incr_gt),
                "--base_path",
                str(base),
                "--query_path",
                str(query),
                "--batch_gt_path",
                str(incr_gt),
                "--k",
                str(cfg["k"]),
                "--inc",
                str(cfg["increment"]),
                "--threads",
                str(cfg.get("threads", 64)),
                "--stream",
            ]
        )

        print(
            f"[{name}] Incremental GT generation completed -> {incr_gt}_offset_*.gt{cfg['k']}"
        )


if __name__ == "__main__":
    main()
