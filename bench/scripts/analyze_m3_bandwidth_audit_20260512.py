#!/usr/bin/env python3
"""Aggregate M3 bandwidth audit results into summary CSVs and key tables."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path


CACHE_LINE_BYTES = 64
# Unit handling: perf -x, prints numeric value in column 1 and unit in column 2.
# For uncore_imc/cas_count_*: unit is "MiB" and value already encodes bytes
#   (one CAS == 64 B). So we convert MiB→bytes (×1024²).
# For LLC-*/cycles/instructions: unit is "" and value is a raw event count.
#   To get bytes we multiply LLC events by the cache line size (64 B).
UNIT_TO_BYTES = {
    "": 1.0,
    "MiB": 1024 * 1024,
    "MB": 1_000_000,
    "GiB": 1024 ** 3,
    "GB": 1_000_000_000,
    "KiB": 1024,
}


def parse_perf_csv(path: Path) -> dict[str, float]:
    """Parse perf stat -x, output. Returns metric->count + metric+'__unit'
    + metric+'__run_ns' + metric+'__run_pct'.
    """
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # CSV format: count,unit,event,run_ns,run_pct[,metric,metric_unit]
        parts = s.split(",")
        if len(parts) < 3:
            continue
        cnt_str, unit, event = parts[0], parts[1], parts[2]
        if cnt_str in {"<not counted>", "<not supported>"}:
            out[event] = float("nan")
            continue
        try:
            cnt = float(cnt_str)
        except ValueError:
            continue
        out[event] = cnt
        out[f"{event}__unit"] = unit  # type: ignore[assignment]
        if len(parts) > 3 and parts[3]:
            try:
                out[f"{event}__run_ns"] = float(parts[3])
            except ValueError:
                pass
        if len(parts) > 4 and parts[4]:
            try:
                out[f"{event}__run_pct"] = float(parts[4])
            except ValueError:
                pass
    return out


def perf_value_bytes(perf: dict, event: str) -> float:
    """Convert a perf value to bytes. uncore_imc/cas_count_* values are in MiB
    already (1 CAS == 64 B), other events are raw counts × cache line."""
    cnt = perf.get(event, float("nan"))
    if isinstance(cnt, float) and math.isnan(cnt):
        return float("nan")
    unit = perf.get(f"{event}__unit", "")
    if isinstance(unit, str) and unit in UNIT_TO_BYTES:
        if unit == "":
            return float(cnt) * CACHE_LINE_BYTES
        return float(cnt) * UNIT_TO_BYTES[unit]
    return float(cnt) * CACHE_LINE_BYTES


def perf_elapsed_seconds(path: Path) -> float | None:
    """Try to parse `# X.XXX seconds time elapsed`. With `perf stat -x,` perf
    omits this line, so this typically returns None and the caller falls back
    to sample_ms/start_unix/end_unix."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        m = re.search(r"(\d+(?:\.\d+)?)\s+seconds time elapsed", line)
        if m:
            return float(m.group(1))
    return None


def cell_perf_window_seconds(cell: Path) -> float | None:
    """Resolve the perf measurement window in seconds.

    Priority:
      1. cell.meta key `sample_ms` (authoritative for the runner).
      2. (perf.end_unix.txt - perf.start_unix.txt) read from wall clock files.
    """
    meta = load_meta(cell / "cell.meta")
    if "sample_ms" in meta:
        try:
            return float(meta["sample_ms"]) / 1000.0
        except ValueError:
            pass
    try:
        s = float((cell / "perf.start_unix.txt").read_text().strip())
        e = float((cell / "perf.end_unix.txt").read_text().strip())
        if e > s:
            return e - s
    except (FileNotFoundError, ValueError):
        pass
    return None


def load_uncore_freq(path: Path) -> tuple[float, float, float, float] | None:
    if not path.exists():
        return None
    f0_vals: list[float] = []
    f1_vals: list[float] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        try:
            f0 = float(parts[1]); f1 = float(parts[2])
        except ValueError:
            continue
        if not math.isnan(f0): f0_vals.append(f0)
        if not math.isnan(f1): f1_vals.append(f1)
    if not f0_vals: return None
    return (
        statistics.mean(f0_vals)/1000.0,  # MHz
        statistics.mean(f1_vals)/1000.0,
        max(f0_vals)/1000.0,
        min(f0_vals)/1000.0,
    )


def load_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().split():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def load_bench_metrics(cell: Path) -> dict[str, float]:
    """Pull key metrics from benchmark_results.csv."""
    csv_path = cell / "benchmark_results.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    r = rows[-1]
    def fnum(k: str) -> float:
        try:
            return float(r.get(k, "nan"))
        except (TypeError, ValueError):
            return float("nan")
    return {
        "search_qps": fnum("search_qps (per-point)"),
        "search_mean_ms": fnum("search_mean_op_latency_ms (per-batch)"),
        "search_p95_ms": fnum("search_p95_op_latency_ms (per-batch)"),
        "search_p99_ms": fnum("search_p99_op_latency_ms (per-batch)"),
        "insert_qps": fnum("insert_qps (per-point)"),
        "insert_mean_ms": fnum("insert_mean_op_latency_ms (per-batch)"),
        "insert_p99_ms": fnum("insert_p99_op_latency_ms (per-batch)"),
        "recall": fnum("recall"),
    }


def summarize_cell(cell: Path) -> dict[str, float | str | None]:
    meta = load_meta(cell / "cell.meta")
    bench = load_bench_metrics(cell)
    pr = parse_perf_csv(cell / "perf_reader.txt")
    pc = parse_perf_csv(cell / "perf_competing.txt")
    pu = parse_perf_csv(cell / "perf_uncore.txt")
    wnd = cell_perf_window_seconds(cell) or float("nan")
    el_r = perf_elapsed_seconds(cell / "perf_reader.txt") or wnd
    el_c = perf_elapsed_seconds(cell / "perf_competing.txt") or wnd
    el_u = perf_elapsed_seconds(cell / "perf_uncore.txt") or wnd
    uf = load_uncore_freq(cell / "uncore_freq.csv")

    def llc_bytes_per_sec(perf: dict[str, float], elapsed: float) -> tuple[float, float, float, float]:
        if not perf or not elapsed:
            return (float("nan"),) * 4
        bpsl = perf_value_bytes(perf, "LLC-loads") / elapsed
        bpslm = perf_value_bytes(perf, "LLC-load-misses") / elapsed
        bpss = perf_value_bytes(perf, "LLC-stores") / elapsed
        bpssm = perf_value_bytes(perf, "LLC-store-misses") / elapsed
        return bpsl, bpslm, bpss, bpssm

    r_load_bps, r_loadmiss_bps, r_store_bps, r_storemiss_bps = llc_bytes_per_sec(pr, el_r)
    c_load_bps, c_loadmiss_bps, c_store_bps, c_storemiss_bps = llc_bytes_per_sec(pc, el_c)

    # uncore_imc reports MiB already. perf_value_bytes converts. Sum read/write across IMC PMUs.
    dram_read_bytes = perf_value_bytes(pu, "uncore_imc/cas_count_read/")
    dram_write_bytes = perf_value_bytes(pu, "uncore_imc/cas_count_write/")
    dram_read_bps = (dram_read_bytes / el_u) if el_u and not math.isnan(dram_read_bytes) else float("nan")
    dram_write_bps = (dram_write_bytes / el_u) if el_u and not math.isnan(dram_write_bytes) else float("nan")

    n_compete_str = meta.get("n_competing", "0")
    try:
        n_compete = int(n_compete_str)
    except ValueError:
        n_compete = 0

    # Per-thread bandwidth for competing role
    if n_compete > 0 and not math.isnan(c_loadmiss_bps):
        per_thread_competing_LLC_miss_bps = c_loadmiss_bps / n_compete
        per_thread_competing_LLC_load_bps = c_load_bps / n_compete
        per_thread_competing_LLC_storemiss_bps = c_storemiss_bps / n_compete if not math.isnan(c_storemiss_bps) else float("nan")
        per_thread_competing_LLC_store_bps = c_store_bps / n_compete if not math.isnan(c_store_bps) else float("nan")
    else:
        per_thread_competing_LLC_miss_bps = float("nan")
        per_thread_competing_LLC_load_bps = float("nan")
        per_thread_competing_LLC_storemiss_bps = float("nan")
        per_thread_competing_LLC_store_bps = float("nan")

    # Per-thread bandwidth for reader role (8 readers)
    per_thread_reader_LLC_load_bps = r_load_bps / 8 if not math.isnan(r_load_bps) else float("nan")
    per_thread_reader_LLC_miss_bps = r_loadmiss_bps / 8 if not math.isnan(r_loadmiss_bps) else float("nan")
    per_thread_reader_LLC_storemiss_bps = r_storemiss_bps / 8 if not math.isnan(r_storemiss_bps) else float("nan")

    return {
        "cell": cell.name,
        "mode": meta.get("mode", meta.get("variant", "")),
        "n_competing": n_compete,
        "competing_cpus": meta.get("competing_cpus", ""),
        "perf_window_s_reader": el_r,
        "perf_window_s_competing": el_c,
        "perf_window_s_uncore": el_u,
        # reader side
        "reader_LLC_loads_per_s": pr.get("LLC-loads", float("nan")) / el_r if el_r else float("nan"),
        "reader_LLC_load_misses_per_s": pr.get("LLC-load-misses", float("nan")) / el_r if el_r else float("nan"),
        "reader_LLC_stores_per_s": pr.get("LLC-stores", float("nan")) / el_r if el_r else float("nan"),
        "reader_LLC_store_misses_per_s": pr.get("LLC-store-misses", float("nan")) / el_r if el_r else float("nan"),
        "reader_cycles_per_s": pr.get("cycles", float("nan")) / el_r if el_r else float("nan"),
        "reader_IPC": pr.get("instructions", float("nan")) / pr.get("cycles", float("nan")) if pr.get("cycles", 0) else float("nan"),
        "reader_LLC_load_bytes_per_s": r_load_bps,
        "reader_LLC_loadmiss_bytes_per_s": r_loadmiss_bps,
        "reader_LLC_store_bytes_per_s": r_store_bps,
        "reader_LLC_storemiss_bytes_per_s": r_storemiss_bps,
        "reader_per_thread_LLC_load_bytes_per_s": per_thread_reader_LLC_load_bps,
        "reader_per_thread_LLC_loadmiss_bytes_per_s": per_thread_reader_LLC_miss_bps,
        "reader_per_thread_LLC_storemiss_bytes_per_s": per_thread_reader_LLC_storemiss_bps,
        "reader_LLC_run_pct": pr.get("LLC-loads__run_pct", float("nan")),
        # competing side
        "competing_LLC_loads_per_s": pc.get("LLC-loads", float("nan")) / el_c if el_c else float("nan"),
        "competing_LLC_load_misses_per_s": pc.get("LLC-load-misses", float("nan")) / el_c if el_c else float("nan"),
        "competing_LLC_stores_per_s": pc.get("LLC-stores", float("nan")) / el_c if el_c else float("nan"),
        "competing_LLC_store_misses_per_s": pc.get("LLC-store-misses", float("nan")) / el_c if el_c else float("nan"),
        "competing_cycles_per_s": pc.get("cycles", float("nan")) / el_c if el_c else float("nan"),
        "competing_IPC": pc.get("instructions", float("nan")) / pc.get("cycles", float("nan")) if pc.get("cycles", 0) else float("nan"),
        "competing_LLC_load_bytes_per_s": c_load_bps,
        "competing_LLC_loadmiss_bytes_per_s": c_loadmiss_bps,
        "competing_LLC_store_bytes_per_s": c_store_bps,
        "competing_LLC_storemiss_bytes_per_s": c_storemiss_bps,
        "competing_per_thread_LLC_load_bytes_per_s": per_thread_competing_LLC_load_bps,
        "competing_per_thread_LLC_loadmiss_bytes_per_s": per_thread_competing_LLC_miss_bps,
        "competing_per_thread_LLC_storemiss_bytes_per_s": per_thread_competing_LLC_storemiss_bps,
        "competing_LLC_run_pct": pc.get("LLC-loads__run_pct", float("nan")),
        # uncore (whole-system DRAM)
        "socket_DRAM_read_bytes_per_s": dram_read_bps,
        "socket_DRAM_write_bytes_per_s": dram_write_bps,
        "socket_DRAM_total_bytes_per_s": (dram_read_bps + dram_write_bps) if not (math.isnan(dram_read_bps) or math.isnan(dram_write_bps)) else float("nan"),
        # uncore freq
        "uncore_freq_pkg0_MHz_mean": uf[0] if uf else float("nan"),
        "uncore_freq_pkg1_MHz_mean": uf[1] if uf else float("nan"),
        "uncore_freq_pkg0_MHz_max": uf[2] if uf else float("nan"),
        "uncore_freq_pkg0_MHz_min": uf[3] if uf else float("nan"),
        # bench metrics
        **{f"bench_{k}": v for k, v in bench.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    root: Path = args.root
    rows: list[dict] = []
    for sub in ("phase1_bandwidth", "phase3_hotcold"):
        d = root / sub
        if not d.exists(): continue
        for cell in sorted(d.iterdir()):
            if cell.is_dir() and (cell / "cell.meta").exists():
                row = summarize_cell(cell)
                row["phase"] = sub
                rows.append(row)
    if not rows:
        print("no cells found", file=sys.stderr)
        return 1
    keys = list(rows[0].keys())
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out} with {len(rows)} rows")

    # Brief table to stdout
    print("\n=== Phase 1 summary (mode, n, reader-side LLC load bytes/s, competing per-thread LLC miss bytes/s, socket DRAM bytes/s, uncore MHz, reader P99 ms) ===")
    for r in rows:
        if r.get("phase") != "phase1_bandwidth": continue
        print(f"{r['cell']:>14s}  mode={r['mode']:<8s} N={r['n_competing']:<3d}  "
              f"rdLLCmiss/s={r['reader_LLC_loadmiss_bytes_per_s']/1e6:8.1f}MB  "
              f"cmptPT-LLCmiss/s={r['competing_per_thread_LLC_loadmiss_bytes_per_s']/1e6 if not math.isnan(r['competing_per_thread_LLC_loadmiss_bytes_per_s']) else float('nan'):8.1f}MB  "
              f"DRAM/s={r['socket_DRAM_total_bytes_per_s']/1e9:6.2f}GB  "
              f"unc0={r['uncore_freq_pkg0_MHz_mean']:6.1f}MHz  "
              f"rdP99={r.get('bench_search_p99_ms', float('nan')):5.3f}ms")

    print("\n=== Phase 3 (hot/cold) summary ===")
    for r in rows:
        if r.get("phase") != "phase3_hotcold": continue
        print(f"{r['cell']:>16s}  mode={r['mode']:<6s}  "
              f"cmptPT-LLCstrmiss/s={r['competing_per_thread_LLC_loadmiss_bytes_per_s']/1e6 if not math.isnan(r['competing_per_thread_LLC_loadmiss_bytes_per_s']) else float('nan'):8.1f}MB  "
              f"DRAMwr/s={r['socket_DRAM_write_bytes_per_s']/1e9:6.2f}GB  "
              f"rdP99={r.get('bench_search_p99_ms', float('nan')):5.3f}ms  rdP999~mean+P99? n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
