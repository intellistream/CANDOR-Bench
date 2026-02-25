import csv
from pathlib import Path
import matplotlib.pyplot as plt

CSV_PATHS = [
    Path("/home/ghr/candor-bench/SAGE-DB-Bench/results/sift/faiss_HNSW/ef-120-test-goodfordraw/ef-120_batch_query_qps.csv"),
    Path("/home/ghr/candor-bench/SAGE-DB-Bench/results/sift/faiss_HNSW_Optimized/ef-120/ef-120_batch_query_qps.csv"),
]
OUTPUT_PATH = Path("/home/ghr/candor-bench/SAGE-DB-Bench/tools/qps_vs_batch_idx.png")


def load_qps(csv_path: Path):
    batch_idx = []
    qps_values = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch_idx.append(int(row["batch_idx"]))
            qps_values.append(float(row["query_qps"]))
    return batch_idx, qps_values


def main():
    plt.figure(figsize=(10, 5), dpi=150)

    for csv_path in CSV_PATHS:
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping")
            continue
        batch_idx, qps_values = load_qps(csv_path)
        # use parent parent dir name as label (e.g., faiss_HNSW or faiss_HNSW_Optimized)
        label = csv_path.parent.parent.name
        plt.plot(batch_idx, qps_values, marker="o", linewidth=1.5, markersize=4, label=label)

    plt.title("Query QPS vs Batch Index")
    plt.xlabel("Batch Index")
    plt.ylabel("Query QPS")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    print(str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
