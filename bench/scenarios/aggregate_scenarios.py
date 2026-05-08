"""Aggregate scenario_bench results into a clean markdown comparison + CSV."""
import json, os, sys

SRC = "/tmp/scenario_results.json"
OUT_DIR = "/home/rprp/CANDOR-Bench/results/scenarios"
os.makedirs(OUT_DIR, exist_ok=True)

with open(SRC) as f:
    rows = json.load(f)

# Group by (dataset, scenario)
grouped = {}
for r in rows:
    k = (r["dataset"], r["scenario"])
    grouped.setdefault(k, []).append(r)

md_lines = ["# GammaFresh vs FaissHNSW vs AdaIVF — Multi-scenario benchmark",
            "",
            "Single-thread (`OMP_NUM_THREADS=1`), apples-to-apples via candy bindings (same C++ compute engine).",
            "",
            "## Datasets",
            "- `sift-small` — 10K × 128d (real SIFT)",
            "- `random-100k` — 100K × 128d Gaussian noise (high-difficulty for ANN)",
            "",
            "## Scenarios",
            "- `streaming(b=100)` — incremental stream, 100/batch insert + 100 queries every batch",
            "- `burst(b=2500)` — large-batch insert, single final query",
            "- `drift(4chunk)` — alternating insert/delete-oldest cycles to simulate concept drift",
            "- `bulk_delete(30%)` — insert all + bulk-delete oldest 30% + query (HNSW tombstone stress)",
            "- `churn(20cyc, 5%)` / `churn(50cyc, 2%)` — sustained insert/delete/query cycles (key GammaFresh use case)",
            "",
            "## Results",
            ""]

for (ds, sc), entries in grouped.items():
    md_lines.append(f"### {ds} | {sc}")
    md_lines.append("")
    md_lines.append("| algo | time (ms) | ops/s | recall@10 | notes |")
    md_lines.append("|---|---:|---:|---:|---|")
    # Sort by time ascending
    entries_sorted = sorted(entries, key=lambda x: x["time_ms"])
    for e in entries_sorted:
        notes = ""
        if e["recall"] >= 0.99:
            notes = "✅ high recall"
        elif e["recall"] >= 0.5:
            notes = ""
        else:
            notes = "❌ recall too low to be useful"
        md_lines.append(f"| {e['algo']} | {e['time_ms']:.0f} | {e['qps']:,.0f} | {e['recall']:.4f} | {notes} |")
    md_lines.append("")

with open(os.path.join(OUT_DIR, "comparison.md"), "w") as f:
    f.write("\n".join(md_lines))

# CSV
import csv
with open(os.path.join(OUT_DIR, "comparison.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "scenario", "algo", "time_ms", "ops_per_sec", "recall_at_10"])
    for r in rows:
        writer.writerow([r["dataset"], r["scenario"], r["algo"],
                         f"{r['time_ms']:.2f}", f"{r['qps']:.2f}", f"{r['recall']:.4f}"])

print(f"Wrote: {OUT_DIR}/comparison.md")
print(f"Wrote: {OUT_DIR}/comparison.csv")
