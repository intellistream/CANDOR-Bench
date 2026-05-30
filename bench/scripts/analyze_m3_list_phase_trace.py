#!/usr/bin/env python3
"""Compatibility entrypoint for period-based insert phase trace binning."""

from __future__ import annotations

import sys
from pathlib import Path

from bin_m3_insert_phase_trace import main as bin_insert_phase_trace


def main() -> int:
    if len(sys.argv) >= 2 and Path(sys.argv[1]).is_file():
        return bin_insert_phase_trace()
    print(
        "analyze_m3_list_phase_trace.py no longer reads insert_wide cumulative deltas.\n"
        "Use bin_m3_insert_phase_trace.py on ANNCHOR_INSERT_PHASE_TRACE output, for example:\n"
        "  python3 scripts/bin_m3_insert_phase_trace.py insert_phase_trace.csv "
        "--search-wide raw_latency/search_wide.csv --meta raw_latency/meta.csv --bin-ms 5 --out periods.csv",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
