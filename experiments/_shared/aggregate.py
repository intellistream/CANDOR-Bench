"""Aggregate output_*.json across all experiments → markdown tables.

Produces:
- ALL_RESULTS_TABLE.md  : grand table sorted by (experiment, dataset, pattern, name)
- per-experiment summaries (gamma vs hnswlib_direct deltas)

Run from repo root:
    python experiments/_shared/aggregate.py
"""
import os, json, glob
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
EXP_DIR = REPO / "experiments"


def load_all():
    rows = []
    for path in sorted(EXP_DIR.glob("e*/output*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        exp = path.parent.name
        for r in data:
            r2 = dict(r)
            r2["__exp"] = exp
            r2["__file"] = path.name
            rows.append(r2)
    return rows


def fmt_row(r):
    name = r.get("name", "?")
    total = r.get("total_s", 0.0)
    rec = r.get("recall", 0.0)
    qp95 = r.get("query_latency_ms_p95", 0.0)
    return f"{name:30s} total={total:7.1f}s recall={rec:.4f} qry_p95={qp95:.3f}ms"


def write_grand_table(rows, out_path):
    with open(out_path, "w") as f:
        f.write("# All experiment results (auto-generated)\n\n")
        f.write(f"Auto-generated from `experiments/_shared/aggregate.py`. Total rows: **{len(rows)}**.\n\n")

        by_exp = defaultdict(list)
        for r in rows:
            by_exp[r["__exp"]].append(r)

        for exp in sorted(by_exp):
            f.write(f"## {exp}\n\n")
            cols = ["name", "scale", "dataset", "pattern", "ratio",
                    "buf_mult", "M", "efC", "efS", "total_s", "recall",
                    "query_latency_ms_p95", "n_alive", "n_delete"]
            present = [c for c in cols if any(c in r for r in by_exp[exp])]
            f.write("| " + " | ".join(present) + " |\n")
            f.write("|" + "|".join(["---"] * len(present)) + "|\n")
            for r in by_exp[exp]:
                cells = []
                for c in present:
                    v = r.get(c, "")
                    if isinstance(v, float):
                        if c == "total_s":
                            cells.append(f"{v:.1f}")
                        elif c in ("recall",):
                            cells.append(f"{v:.4f}")
                        elif c == "query_latency_ms_p95":
                            cells.append(f"{v:.3f}")
                        else:
                            cells.append(f"{v}")
                    else:
                        cells.append(str(v))
                f.write("| " + " | ".join(cells) + " |\n")
            f.write("\n")


def gamma_deltas(rows):
    """For each (experiment, dataset, pattern, scale), find gamma_v2+X and X-direct
    rows that match on configuration, then report Δtime%."""
    by_key = defaultdict(list)
    for r in rows:
        key = (r["__exp"], r.get("dataset", ""), r.get("pattern", ""),
               r.get("scale", ""), r.get("ratio", ""), r.get("buf_mult", ""),
               r.get("M", ""), r.get("efC", ""), r.get("efS", ""))
        by_key[key].append(r)

    deltas = []
    for key, group in by_key.items():
        # Find a gamma row and a matching direct row.
        # Match the v2 (e15-e19) and v3 (e20) router naming as well as the
        # native C++ ablation (e21).
        gammas = [r for r in group
                  if "gamma_v2+" in r.get("name", "")
                  or "gamma_v3" in r.get("name", "")
                  or "gamma_cpp" in r.get("name", "")]
        directs = [r for r in group if " direct" in r.get("name", "")
                   and "gamma" not in r.get("name", "")]
        for g in gammas:
            be_in_g = g["name"].split("+")[1] if "+" in g["name"] else ""
            be_in_g = be_in_g.split("(")[0]  # strip e19 (M=...)
            for d in directs:
                if be_in_g and not d["name"].startswith(be_in_g):
                    continue
                gt, dt = g.get("total_s", 0), d.get("total_s", 0)
                if dt > 0:
                    pct = (gt - dt) / dt * 100.0
                    deltas.append({
                        "exp": g["__exp"], "dataset": g.get("dataset", ""),
                        "pattern": g.get("pattern", ""), "scale": g.get("scale", ""),
                        "ratio": g.get("ratio", ""), "buf_mult": g.get("buf_mult", ""),
                        "gamma": g["name"], "direct": d["name"],
                        "gamma_s": gt, "direct_s": dt, "delta_pct": pct,
                        "gamma_recall": g.get("recall", 0),
                        "direct_recall": d.get("recall", 0),
                    })
    return deltas


def write_deltas(deltas, out_path):
    deltas = sorted(deltas, key=lambda x: x["delta_pct"])
    with open(out_path, "w") as f:
        f.write("# Gamma vs direct-backend deltas (auto)\n\n")
        f.write("Δ% < 0 means gamma is FASTER than direct.\n\n")
        f.write("| exp | dataset | scale | pattern | knob | gamma_s | direct_s | Δ% | gamma_rec | direct_rec |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for d in deltas:
            knob = ""
            if d["ratio"] != "":
                knob = f"ratio={d['ratio']}"
            elif d["buf_mult"] != "":
                knob = f"buf={d['buf_mult']}x"
            f.write(f"| {d['exp']} | {d['dataset']} | {d['scale']} | "
                    f"{d['pattern']} | {knob} | "
                    f"{d['gamma_s']:.1f} | {d['direct_s']:.1f} | "
                    f"{d['delta_pct']:+.1f}% | "
                    f"{d['gamma_recall']:.4f} | {d['direct_recall']:.4f} |\n")


def main():
    rows = load_all()
    print(f"loaded {len(rows)} rows from {len(set(r['__file'] for r in rows))} files")
    out_table = EXP_DIR / "ALL_RESULTS_TABLE.md"
    write_grand_table(rows, out_table)
    print(f"wrote {out_table}")
    deltas = gamma_deltas(rows)
    out_deltas = EXP_DIR / "ALL_DELTAS_TABLE.md"
    write_deltas(deltas, out_deltas)
    print(f"wrote {out_deltas} ({len(deltas)} pairs)")


if __name__ == "__main__":
    main()
