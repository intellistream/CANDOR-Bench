from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DATASETS = [
    # {
    #     "name": "2cluster_d128",
    #     "base_bin": "2cluster/2cluster_d128_base.bin",
    #     "query_bin": "2cluster/2cluster_d128_query.bin",
    #     "incremental_gt": "2cluster/2cluster_d128_base_i20.gt20",
    #     "k": 20,
    #     "increment": 20,
    #     "threads": 16,
    # },
    # {
    #     "name": "3cluster_d128",
    #     "base_bin": "3cluster/3cluster_d128_base.bin",
    #     "query_bin": "3cluster/3cluster_d128_query.bin",
    #     "incremental_gt": "3cluster/3cluster_d128_base_i20.gt20",
    #     "k": 20,
    #     "increment": 20,
    #     "threads": 16,
    # },
    # {
    #     "name": "4cluster_d128",
    #     "base_bin": "4cluster/4cluster_d128_base.bin",
    #     "query_bin": "4cluster/4cluster_d128_query.bin",
    #     "incremental_gt": "4cluster/4cluster_d128_base_i20.gt20",
    #     "k": 20,
    #     "increment": 20,
    #     "threads": 16,
    # },
    # {
    #     "name": "gist", # dim 960
    #     "base_bin": "gist/gist_base.bin",
    #     "query_bin": "gist/gist_query.bin",
    #     "incremental_gt": "gist/gist_base_i20.gt20",
    #     "k": 20,
    #     "increment": 20,
    #     "threads": 64,
    # },
    # {
    #     "name": "msong", # dim 420
    #     "base_bin": "msong/msong_base.bin",
    #     "query_bin": "msong/msong_query.bin",
    #     "incremental_gt": "msong/msong_base_i20.gt20",
    #     "k": 20,
    #     "increment": 20,
    #     "threads": 16,
    # },
    # {
    #     "name": "deep1M", # dim 256
    #     "base_bin": "deep1M/deep1M_base.bin",
    #     "query_bin": "deep1M/deep1M_query.bin",
    #     "incremental_gt": "deep1M/deep1M_base_i20.gt20",
    #     "k": 20,
    #     "increment": 20,
    #     "threads": 16,
    # },
    # {
    #     "name": "glove1.2m", # dim 200
    #     "base_bin": "glove1.2m/glove1.2m_base.bin",
    #     "query_bin": "glove1.2m/glove1.2m_query.bin",
    #     "incremental_gt": "glove1.2m/glove1.2m_base_i20.gt20",
    #     "k": 20,
    #     "increment": 20,
    #     "threads": 16,
    # },
    # {
    #     "name": "glove2.2m", # dim 300
    #     "base_bin": "glove2.2m/glove2.2m_base.bin",
    #     "query_bin": "glove2.2m/glove2.2m_query.bin",
    #     "incremental_gt": "glove2.2m/glove2.2m_base_i20.gt20",
    #     "k": 20,
    #     "increment": 20,
    #     "threads": 16,
    # },
    # {
    #     "name": "sift", # dim 128
    #     "base_bin": "sift/sift_base.bin",
    #     "query_bin": "sift/sift_query.bin",
    #     "incremental_gt": "sift/sift_base_i20.gt20",
    #     "k": 20,
    #     "increment": 20,
    #     "threads": 16,
    # },
    # {
    #     "name": "deep10M",  # dim 96
    #     "base_bin": "deep10M/base.10M.fbin",
    #     "query_bin": "deep10M/query_1k.bin",
    #     "incremental_gt": "deep10M/deep10M_base_i20.gt20",
    #     "k": 20,
    #     "increment": 20,
    #     "threads": 16,
    # },
    # FreewayML datasets
    {
        "name": "FreewayML_G",
        "base_bin": "FreewayML/G/base.bin",
        "query_bin": "FreewayML/G/query.bin",
        "incremental_gt": "FreewayML/G/base_i20.gt20",
        "k": 20,
        "increment": 20,
        "threads": 64,
    },
    {
        "name": "FreewayML_A1",
        "base_bin": "FreewayML/A1/base.bin",
        "query_bin": "FreewayML/A1/query.bin",
        "incremental_gt": "FreewayML/A1/base_i20.gt20",
        "k": 20,
        "increment": 20,
        "threads": 64,
    },
    {
        "name": "FreewayML_A2",
        "base_bin": "FreewayML/A2/base.bin",
        "query_bin": "FreewayML/A2/query.bin",
        "incremental_gt": "FreewayML/A2/base_i20.gt20",
        "k": 20,
        "increment": 20,
        "threads": 64,
    },
    {
        "name": "FreewayML_B1",
        "base_bin": "FreewayML/B1/base.bin",
        "query_bin": "FreewayML/B1/query.bin",
        "incremental_gt": "FreewayML/B1/base_i20.gt20",
        "k": 20,
        "increment": 20,
        "threads": 64,
    },
    {
        "name": "FreewayML_B2",
        "base_bin": "FreewayML/B2/base.bin",
        "query_bin": "FreewayML/B2/query.bin",
        "incremental_gt": "FreewayML/B2/base_i20.gt20",
        "k": 20,
        "increment": 20,
        "threads": 64,
    },
    {
        "name": "FreewayML_B3",
        "base_bin": "FreewayML/B3/base.bin",
        "query_bin": "FreewayML/B3/query.bin",
        "incremental_gt": "FreewayML/B3/base_i20.gt20",
        "k": 20,
        "increment": 20,
        "threads": 64,
    },
    {
        "name": "FreewayML_B4",
        "base_bin": "FreewayML/B4/base.bin",
        "query_bin": "FreewayML/B4/query.bin",
        "incremental_gt": "FreewayML/B4/base_i20.gt20",
        "k": 20,
        "increment": 20,
        "threads": 64,
    },
    {
        "name": "FreewayML_C1",
        "base_bin": "FreewayML/C1/base.bin",
        "query_bin": "FreewayML/C1/query.bin",
        "incremental_gt": "FreewayML/C1/base_i20.gt20",
        "k": 20,
        "increment": 20,
        "threads": 64,
    },
    {
        "name": "FreewayML_C2",
        "base_bin": "FreewayML/C2/base.bin",
        "query_bin": "FreewayML/C2/query.bin",
        "incremental_gt": "FreewayML/C2/base_i20.gt20",
        "k": 20,
        "increment": 20,
        "threads": 64,
    },
]

UTILS_BUILD = Path(__file__).resolve().parent.parent / "utils" / "build"


def run(cmd: list[str]) -> None:
    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_exists(path: Path) -> None:
    if not path.exists():
        print(f"Required file not found: {path}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    compute_incr_gt = UTILS_BUILD / "compute_incr_gt"
    ensure_exists(compute_incr_gt)

    for cfg in DATASETS:
        name = cfg["name"]
        base = Path("data") / cfg["base_bin"]
        query = Path("data") / cfg["query_bin"]
        incr_gt = Path("data") / cfg["incremental_gt"]

        if not base.exists() or not query.exists():
            missing = []
            if not base.exists():
                missing.append(str(base))
            if not query.exists():
                missing.append(str(query))
            print(f"[{name}] skip: missing {' & '.join(missing)}", file=sys.stderr)
            continue

        incr_gt.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
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
            # "--gpu",
        ]
        run(cmd)

        print(f"[{name}] incremental GT -> {incr_gt}_offset_*.gt{cfg['k']}")


if __name__ == "__main__":
    main()
