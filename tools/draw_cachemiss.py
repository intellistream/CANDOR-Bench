import csv
from pathlib import Path

import matplotlib.pyplot as plt

CSV_PATHS = [
    Path("/home/ghr/candor-bench/SAGE-DB-Bench/results/sift/faiss_HNSW/ef-120-test-goodfordraw/ef-120_query_cache_miss.csv"),
    Path("/home/ghr/candor-bench/SAGE-DB-Bench/results/sift/faiss_HNSW_Optimized/ef-120/ef-120_query_cache_miss.csv"),
]
OUTPUT_PATH = Path("/home/ghr/candor-bench/SAGE-DB-Bench/tools/cache_miss_rate_vs_query_idx.png")


def load_cache_miss_rate(csv_path: Path):
    query_idx = []
    miss_rate = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_idx.append(int(row["query_idx"]))
            miss_rate.append(float(row["cache_miss_rate"]))
    return query_idx, miss_rate


def main():
    plt.figure(figsize=(8, 4.5), dpi=150)

    for csv_path in CSV_PATHS:
        query_idx, miss_rate = load_cache_miss_rate(csv_path)
        label = csv_path.parent.parent.name
        plt.plot(query_idx, miss_rate, marker="o", linewidth=1.5, markersize=3, label=label)

    plt.title("Cache Miss Rate vs Query Index")
    plt.xlabel("Query Index")
    plt.ylabel("Cache Miss Rate")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
