import os
import subprocess
import sys
from pathlib import Path

DATASETS = [
    # {
    #     "name": "deep1M",
    #     "base": "deep1M/deep1M_base.bin",
    #     "query": "deep1M/deep1M_query.bin",
    #     "overall_gt": "deep1M/deep1M_base.gt20",
    #     "k": 20,
    # },
    # {
    #     "name": "gist",
    #     "base": "gist/gist_base.bin",
    #     "query": "gist/gist_query.bin",
    #     "overall_gt": "gist/gist_base.gt20",
    #     "k": 20,
    # },
    # {
    #     "name": "glove1.2m",
    #     "base": "glove1.2m/glove1.2m_base.bin",
    #     "query": "glove1.2m/glove1.2m_query.bin",
    #     "overall_gt": "glove1.2m/glove1.2m_base.gt20",
    #     "k": 20,
    # },
    # {
    #     "name": "msong",
    #     "base": "msong/msong_base.bin",
    #     "query": "msong/msong_query.bin",
    #     "overall_gt": "msong/msong_base.gt20",
    #     "k": 20,
    # },
    # {
    #     "name": "sift",
    #     "base": "sift/sift_base.bin",
    #     "query": "sift/sift_query.bin",
    #     "overall_gt": "sift/sift_base.gt20",
    #     "k": 20,
    # },
    # {
    #     "name": "2cluster_d128",
    #     "base": "2cluster/2cluster_d128_base.bin",
    #     "query": "2cluster/2cluster_d128_query.bin",
    #     "overall_gt": "2cluster/2cluster_d128_base.gt20",
    #     "k": 20,
    # },
    # {
    #     "name": "2cluster_d256",
    #     "base": "2cluster/2cluster_d256_base.bin",
    #     "query": "2cluster/2cluster_d256_query.bin",
    #     "overall_gt": "2cluster/2cluster_d256_base.gt20",
    #     "k": 20,
    # },
    # {
    #     "name": "3cluster_d128",
    #     "base": "3cluster/3cluster_d128_base.bin",
    #     "query": "3cluster/3cluster_d128_query.bin",
    #     "overall_gt": "3cluster/3cluster_d128_base.gt20",
    #     "k": 20,
    # },
    # {
    #     "name": "3cluster_d256",
    #     "base": "3cluster/3cluster_d256_base.bin",
    #     "query": "3cluster/3cluster_d256_query.bin",
    #     "overall_gt": "3cluster/3cluster_d256_base.gt20",
    #     "k": 20,
    # },
    # {
    #     "name": "4cluster_d128",
    #     "base": "4cluster/4cluster_d128_base.bin",
    #     "query": "4cluster/4cluster_d128_query.bin",
    #     "overall_gt": "4cluster/4cluster_d128_base.gt20",
    #     "k": 20,
    # },
    # {
    #     "name": "4cluster_d256",
    #     "base": "4cluster/4cluster_d256_base.bin",
    #     "query": "4cluster/4cluster_d256_query.bin",
    #     "overall_gt": "4cluster/4cluster_d256_base.gt20",
    #     "k": 20,
    # },
    # FreewayML datasets
    {
        "name": "FreewayML_G",
        "base": "FreewayML/G/base.bin",
        "query": "FreewayML/G/query.bin",
        "overall_gt": "FreewayML/G/base.gt20",
        "k": 20,
    },
    {
        "name": "FreewayML_A1",
        "base": "FreewayML/A1/base.bin",
        "query": "FreewayML/A1/query.bin",
        "overall_gt": "FreewayML/A1/base.gt20",
        "k": 20,
    },
    {
        "name": "FreewayML_A2",
        "base": "FreewayML/A2/base.bin",
        "query": "FreewayML/A2/query.bin",
        "overall_gt": "FreewayML/A2/base.gt20",
        "k": 20,
    },
    {
        "name": "FreewayML_B1",
        "base": "FreewayML/B1/base.bin",
        "query": "FreewayML/B1/query.bin",
        "overall_gt": "FreewayML/B1/base.gt20",
        "k": 20,
    },
    {
        "name": "FreewayML_B2",
        "base": "FreewayML/B2/base.bin",
        "query": "FreewayML/B2/query.bin",
        "overall_gt": "FreewayML/B2/base.gt20",
        "k": 20,
    },
    {
        "name": "FreewayML_B3",
        "base": "FreewayML/B3/base.bin",
        "query": "FreewayML/B3/query.bin",
        "overall_gt": "FreewayML/B3/base.gt20",
        "k": 20,
    },
    {
        "name": "FreewayML_B4",
        "base": "FreewayML/B4/base.bin",
        "query": "FreewayML/B4/query.bin",
        "overall_gt": "FreewayML/B4/base.gt20",
        "k": 20,
    },
    {
        "name": "FreewayML_C1",
        "base": "FreewayML/C1/base.bin",
        "query": "FreewayML/C1/query.bin",
        "overall_gt": "FreewayML/C1/base.gt20",
        "k": 20,
    },
    {
        "name": "FreewayML_C2",
        "base": "FreewayML/C2/base.bin",
        "query": "FreewayML/C2/query.bin",
        "overall_gt": "FreewayML/C2/base.gt20",
        "k": 20,
    },
]


def run_command(cmd: list[str]) -> None:
    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_exists(path: Path) -> None:
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)


def resolve_vector_bin(path_str: str) -> Path:
    path = Path(path_str)
    ensure_exists(path)
    if path.suffix != ".bin":
        print(f"Expected .bin file, got {path}", file=sys.stderr)
        sys.exit(1)
    return path


def generate_gt() -> None:
    for dataset in DATASETS:
        print(f"=== {dataset['name']} ===")
        base_bin = resolve_vector_bin(dataset["base"])
        query_bin = resolve_vector_bin(dataset["query"])

        cmd = [
            "../utils/build/compute_gt",
            "--base_file",
            str(base_bin),
            "--query_file",
            str(query_bin),
            "--gt_file",
            dataset["overall_gt"],
            "--k",
            str(dataset["k"]),
        ]
        run_command(cmd)
        print(f"Overall GT written to: {dataset['overall_gt']}\n")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    generate_gt()
