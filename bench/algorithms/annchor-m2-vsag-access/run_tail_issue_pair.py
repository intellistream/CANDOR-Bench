#!/usr/bin/env python3
import argparse
import fcntl
import os
import statistics
import subprocess
import sys
import time


TAIL_ENV_KEYS = {
    "ANNCHOR_TAIL_ISSUE_ENABLE",
    "ANNCHOR_TAIL_ISSUE_READER_PREFETCH",
    "ANNCHOR_TAIL_ISSUE_READER_WINDOW_EXTRA",
    "ANNCHOR_TAIL_ISSUE_READER_EF_LIMIT",
    "ANNCHOR_TAIL_ISSUE_READER_EF_LIMIT2",
    "ANNCHOR_TAIL_ISSUE_READER_BUDGET_LINES",
    "ANNCHOR_TAIL_ISSUE_READER_LIMIT_MODE",
    "ANNCHOR_TAIL_ISSUE_READER_EXPANSION_BUDGET",
    "ANNCHOR_TAIL_ISSUE_READER_STOP_ALPHA_PERMILLE",
    "ANNCHOR_TAIL_ISSUE_READER_CONF_MARGIN_PERMILLE",
    "ANNCHOR_TAIL_ISSUE_READER_BATCH_SIZE",
    "ANNCHOR_TAIL_ISSUE_READER_LOOKAHEAD_MODE",
    "ANNCHOR_TAIL_ISSUE_READER_LOOKAHEAD_BUDGET",
    "ANNCHOR_TAIL_ISSUE_READER_STAG_CLAMP_HOPS",
    "ANNCHOR_TAIL_ISSUE_READER_STAG_CLAMP_MIN_EXPANSIONS",
    "ANNCHOR_TAIL_ISSUE_READER_FRONTIER_EF",
    "ANNCHOR_TAIL_ISSUE_READER_FRONTIER_STAG_HOPS",
    "ANNCHOR_TAIL_ISSUE_READER_FRONTIER_MIN_EXPANSIONS",
    "ANNCHOR_TAIL_ISSUE_READER_FRONTIER_CONTINUE_BUDGET",
    "ANNCHOR_TAIL_ISSUE_READER_RETRY_ENABLE",
    "ANNCHOR_TAIL_ISSUE_READER_RETRY_MARGIN_PERMILLE",
    "ANNCHOR_TAIL_ISSUE_READER_CONTINUE_ENABLE",
    "ANNCHOR_TAIL_ISSUE_READER_CONTINUE_MARGIN_PERMILLE",
    "ANNCHOR_TAIL_ISSUE_READER_BOUNDED_DISTANCE",
    "ANNCHOR_TAIL_ISSUE_READER_PREFIX_GUARD",
    "ANNCHOR_TAIL_ISSUE_READER_PREFIX_GUARD_LINES",
    "ANNCHOR_TAIL_ISSUE_READER_PREFIX_ESTIMATE",
    "ANNCHOR_TAIL_ISSUE_READER_PREFIX_ESTIMATE_LINES",
    "ANNCHOR_TAIL_ISSUE_READER_PREFIX_ESTIMATE_MARGIN_PERMILLE",
    "ANNCHOR_TAIL_ISSUE_READER_CURRENT_LINES",
    "ANNCHOR_TAIL_ISSUE_READER_REFINE_SEEDS",
    "ANNCHOR_TAIL_ISSUE_READER_REFINE_EDGES",
    "ANNCHOR_TAIL_ISSUE_WRITER_EF_BUDGET",
    "ANNCHOR_TAIL_ISSUE_WRITER_EF_BUDGET_CAP",
    "ANNCHOR_TAIL_ISSUE_WRITER_EF_LIMIT",
    "ANNCHOR_TAIL_ISSUE_WRITER_REPAIR_DEFER_BUDGET",
    "ANNCHOR_TAIL_ISSUE_WRITER_REPAIR_DEFER_CAP",
    "ANNCHOR_TAIL_ISSUE_WRITER_REPAIR_DRAIN_BUDGET",
    "ANNCHOR_TAIL_ISSUE_WRITER_REPAIR_NO_DRAIN",
    "ANNCHOR_TAIL_ISSUE_WRITER_REPAIR_QUEUE_CAP",
    "ANNCHOR_TAIL_ISSUE_WRITER_LOWVALUE_SKIP_BUDGET",
    "ANNCHOR_TAIL_ISSUE_WRITER_LOWVALUE_SKIP_CAP",
    "ANNCHOR_TAIL_ISSUE_WRITER_LOWVALUE_MARGIN_PERMILLE",
    "ANNCHOR_TAIL_ISSUE_WRITER_LEVEL0_BUDGET",
    "ANNCHOR_TAIL_ISSUE_WRITER_LEVEL0_CAP",
    "ANNCHOR_TAIL_ISSUE_READER_GUARD_NS",
    "ANNCHOR_TAIL_ISSUE_READER_GUARD2_NS",
    "ANNCHOR_TAIL_ISSUE_LEASE_NS",
    "ANNCHOR_TAIL_ISSUE_WRITER_LINES",
    "ANNCHOR_TAIL_ISSUE_WRITER_ACTION",
    "ANNCHOR_TAIL_ISSUE_CHECK_PERIOD",
    "ANNCHOR_TAIL_ISSUE_WRITER_BUDGET_LINES",
    "ANNCHOR_TAIL_ISSUE_WRITER_DIST_MODE",
    "ANNCHOR_TAIL_ISSUE_WRITER_DIST_BUDGET",
    "ANNCHOR_TAIL_ISSUE_WRITER_DIST_BUDGET_CAP",
    "ANNCHOR_TAIL_ISSUE_WRITER_DIST_PREFIX_GUARD_LINES",
    "ANNCHOR_TAIL_ISSUE_WRITER_DIST_PREFIX_ESTIMATE_LINES",
    "ANNCHOR_TAIL_ISSUE_WRITER_DIST_PREFIX_ESTIMATE_MARGIN_PERMILLE",
    "ANNCHOR_TAIL_ISSUE_WRITER_LEAD_SKIP_BUDGET",
    "ANNCHOR_TAIL_ISSUE_WRITER_LEAD_SKIP_CAP",
    "ANNCHOR_TAIL_ISSUE_WRITER_LEAD_HINT_BUDGET",
    "ANNCHOR_TAIL_ISSUE_WRITER_LEAD_HINT_CAP",
    "ANNCHOR_TAIL_ISSUE_WRITER_LEAD_HINT_ACTION",
    "ANNCHOR_TAIL_ISSUE_WRITER_FRONTIER_BUDGET",
    "ANNCHOR_TAIL_ISSUE_WRITER_FRONTIER_CAP",
    "ANNCHOR_TAIL_ISSUE_WRITER_FRONTIER_MARGIN_PERMILLE",
    "ANNCHOR_TAIL_ISSUE_WRITER_SHARED_SCAN_BUDGET",
    "ANNCHOR_TAIL_ISSUE_WRITER_SHARED_SCAN_CAP",
    "ANNCHOR_TAIL_ISSUE_WRITER_SHARED_SCAN_ALWAYS",
    "ANNCHOR_TAIL_ISSUE_WRITER_DEMOTE_BUDGET",
    "ANNCHOR_TAIL_ISSUE_WRITER_DEMOTE_CAP",
    "ANNCHOR_TAIL_ISSUE_WRITER_DEMOTE_LINES",
    "ANNCHOR_TAIL_ISSUE_WRITER_OFFLOCK_PRUNE_BUDGET",
    "ANNCHOR_TAIL_ISSUE_WRITER_OFFLOCK_PRUNE_CAP",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default="/tmp/annchor-tail-issue-shallow-build/bench_tail_issue")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--initial", type=int, default=10000)
    parser.add_argument("--inserts", type=int, default=20000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--search-threads", type=int, default=4)
    parser.add_argument("--insert-threads", type=int, default=1)
    parser.add_argument("--duration-ms", type=int, default=5000)
    parser.add_argument("--ef", type=int, default=64)
    parser.add_argument("--guard-ns", type=int, default=100000)
    parser.add_argument("--guard2-ns", type=int, default=0)
    parser.add_argument("--lease-ns", type=int, default=50000)
    parser.add_argument("--check-period", type=int, default=32)
    parser.add_argument("--reader-prefetch", type=int, default=0)
    parser.add_argument("--reader-budget-lines", type=int, default=0)
    parser.add_argument("--reader-window-extra", type=int, default=0)
    parser.add_argument("--reader-ef-limit", type=int, default=0)
    parser.add_argument("--reader-ef-limit2", type=int, default=0)
    parser.add_argument("--reader-limit-mode", type=int, default=0)
    parser.add_argument("--reader-expansion-budget", type=int, default=0)
    parser.add_argument("--reader-stop-alpha-permille", type=int, default=1000)
    parser.add_argument("--reader-conf-margin-permille", type=int, default=0)
    parser.add_argument("--reader-batch-size", type=int, default=0)
    parser.add_argument("--reader-lookahead-mode", type=int, default=0)
    parser.add_argument("--reader-lookahead-budget", type=int, default=0)
    parser.add_argument("--reader-stag-clamp-hops", type=int, default=0)
    parser.add_argument("--reader-stag-clamp-min-expansions", type=int, default=0)
    parser.add_argument("--reader-frontier-ef", type=int, default=0)
    parser.add_argument("--reader-frontier-stag-hops", type=int, default=0)
    parser.add_argument("--reader-frontier-min-expansions", type=int, default=0)
    parser.add_argument("--reader-frontier-continue-budget", type=int, default=0)
    parser.add_argument("--reader-retry-enable", type=int, default=0)
    parser.add_argument("--reader-retry-margin-permille", type=int, default=1100)
    parser.add_argument("--reader-continue-enable", type=int, default=0)
    parser.add_argument("--reader-continue-margin-permille", type=int, default=1050)
    parser.add_argument("--reader-bounded-distance", type=int, default=0)
    parser.add_argument("--reader-prefix-guard", type=int, default=0)
    parser.add_argument("--reader-prefix-guard-lines", type=int, default=1)
    parser.add_argument("--reader-prefix-estimate", type=int, default=0)
    parser.add_argument("--reader-prefix-estimate-lines", type=int, default=1)
    parser.add_argument("--reader-prefix-estimate-margin-permille", type=int, default=1200)
    parser.add_argument("--reader-current-lines", type=int, default=0)
    parser.add_argument("--reader-refine-seeds", type=int, default=0)
    parser.add_argument("--reader-refine-edges", type=int, default=0)
    parser.add_argument("--writer-ef-budget", type=int, default=0)
    parser.add_argument("--writer-ef-cap", type=int, default=0)
    parser.add_argument("--writer-ef-limit", type=int, default=0)
    parser.add_argument("--repair-defer-budget", type=int, default=0)
    parser.add_argument("--repair-defer-cap", type=int, default=0)
    parser.add_argument("--repair-drain-budget", type=int, default=0)
    parser.add_argument("--repair-no-drain", type=int, default=0)
    parser.add_argument("--repair-queue-cap", type=int, default=4096)
    parser.add_argument("--lowvalue-skip-budget", type=int, default=0)
    parser.add_argument("--lowvalue-skip-cap", type=int, default=0)
    parser.add_argument("--lowvalue-margin-permille", type=int, default=1100)
    parser.add_argument("--writer-level0-budget", type=int, default=0)
    parser.add_argument("--writer-level0-cap", type=int, default=0)
    parser.add_argument("--writer-lines", type=int, default=0)
    parser.add_argument("--writer-action", type=int, default=0)
    parser.add_argument("--writer-budget-lines", type=int, default=0)
    parser.add_argument("--dist-mode", type=int, default=2)
    parser.add_argument("--dist-budget", type=int, default=0)
    parser.add_argument("--dist-cap", type=int, default=0)
    parser.add_argument("--dist-prefix-lines", type=int, default=1)
    parser.add_argument("--dist-prefix-estimate-lines", type=int, default=1)
    parser.add_argument("--dist-prefix-estimate-margin-permille", type=int, default=1200)
    parser.add_argument("--lead-skip-budget", type=int, default=0)
    parser.add_argument("--lead-skip-cap", type=int, default=0)
    parser.add_argument("--lead-hint-budget", type=int, default=0)
    parser.add_argument("--lead-hint-cap", type=int, default=0)
    parser.add_argument("--lead-hint-action", type=int, default=0)
    parser.add_argument("--writer-frontier-budget", type=int, default=0)
    parser.add_argument("--writer-frontier-cap", type=int, default=0)
    parser.add_argument("--writer-frontier-margin-permille", type=int, default=990)
    parser.add_argument("--writer-shared-scan-budget", type=int, default=0)
    parser.add_argument("--writer-shared-scan-cap", type=int, default=0)
    parser.add_argument("--writer-shared-scan-always", type=int, default=0)
    parser.add_argument("--writer-demote-budget", type=int, default=0)
    parser.add_argument("--writer-demote-cap", type=int, default=0)
    parser.add_argument("--writer-demote-lines", type=int, default=1)
    parser.add_argument("--writer-offlock-prune-budget", type=int, default=0)
    parser.add_argument("--writer-offlock-prune-cap", type=int, default=0)
    parser.add_argument("--search-cpus", default="")
    parser.add_argument("--insert-cpus", default="")
    parser.add_argument("--out", default=None)
    parser.add_argument("--lock", default="/tmp/annchor_tail_issue_pair.lock")
    return parser.parse_args()


def metric_row(stdout, pair, mode, returncode):
    row = {"pair": pair, "mode": mode, "returncode": returncode}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            row[key] = float(value)
        except ValueError:
            row[key] = value
    return row


def median(rows, key):
    return statistics.median(row[key] for row in rows)


def run_once(cmd, env):
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode, proc.stdout


def main():
    args = parse_args()
    if args.pairs < 1:
        raise SystemExit("--pairs must be positive")
    if not os.path.exists(args.bin):
        raise SystemExit(f"missing bench binary: {args.bin}")

    out = args.out or f"/tmp/annchor_tail_single_valid_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    bench_cmd = [
        args.bin,
        str(args.initial),
        str(args.inserts),
        str(args.dim),
        str(args.search_threads),
        str(args.insert_threads),
        str(args.duration_ms),
        str(args.ef),
    ]

    base_env = os.environ.copy()
    for key in TAIL_ENV_KEYS:
        base_env.pop(key, None)
    if args.search_cpus:
        base_env["ANNCHOR_TAIL_ISSUE_SEARCH_CPUS"] = args.search_cpus
    if args.insert_cpus:
        base_env["ANNCHOR_TAIL_ISSUE_INSERT_CPUS"] = args.insert_cpus

    policy_env = base_env.copy()
    policy_env.update(
        {
            "ANNCHOR_TAIL_ISSUE_ENABLE": "1",
            "ANNCHOR_TAIL_ISSUE_READER_PREFETCH": str(args.reader_prefetch),
            "ANNCHOR_TAIL_ISSUE_READER_WINDOW_EXTRA": str(args.reader_window_extra),
            "ANNCHOR_TAIL_ISSUE_READER_EF_LIMIT": str(args.reader_ef_limit),
            "ANNCHOR_TAIL_ISSUE_READER_EF_LIMIT2": str(args.reader_ef_limit2),
            "ANNCHOR_TAIL_ISSUE_READER_BUDGET_LINES": str(args.reader_budget_lines),
            "ANNCHOR_TAIL_ISSUE_READER_LIMIT_MODE": str(args.reader_limit_mode),
            "ANNCHOR_TAIL_ISSUE_READER_EXPANSION_BUDGET": str(args.reader_expansion_budget),
            "ANNCHOR_TAIL_ISSUE_READER_STOP_ALPHA_PERMILLE": str(args.reader_stop_alpha_permille),
            "ANNCHOR_TAIL_ISSUE_READER_CONF_MARGIN_PERMILLE": str(args.reader_conf_margin_permille),
            "ANNCHOR_TAIL_ISSUE_READER_BATCH_SIZE": str(args.reader_batch_size),
            "ANNCHOR_TAIL_ISSUE_READER_LOOKAHEAD_MODE": str(args.reader_lookahead_mode),
            "ANNCHOR_TAIL_ISSUE_READER_LOOKAHEAD_BUDGET": str(args.reader_lookahead_budget),
            "ANNCHOR_TAIL_ISSUE_READER_STAG_CLAMP_HOPS": str(args.reader_stag_clamp_hops),
            "ANNCHOR_TAIL_ISSUE_READER_STAG_CLAMP_MIN_EXPANSIONS": str(args.reader_stag_clamp_min_expansions),
            "ANNCHOR_TAIL_ISSUE_READER_FRONTIER_EF": str(args.reader_frontier_ef),
            "ANNCHOR_TAIL_ISSUE_READER_FRONTIER_STAG_HOPS": str(args.reader_frontier_stag_hops),
            "ANNCHOR_TAIL_ISSUE_READER_FRONTIER_MIN_EXPANSIONS": str(args.reader_frontier_min_expansions),
            "ANNCHOR_TAIL_ISSUE_READER_FRONTIER_CONTINUE_BUDGET": str(args.reader_frontier_continue_budget),
            "ANNCHOR_TAIL_ISSUE_READER_RETRY_ENABLE": str(args.reader_retry_enable),
            "ANNCHOR_TAIL_ISSUE_READER_RETRY_MARGIN_PERMILLE": str(args.reader_retry_margin_permille),
            "ANNCHOR_TAIL_ISSUE_READER_CONTINUE_ENABLE": str(args.reader_continue_enable),
            "ANNCHOR_TAIL_ISSUE_READER_CONTINUE_MARGIN_PERMILLE": str(args.reader_continue_margin_permille),
            "ANNCHOR_TAIL_ISSUE_READER_BOUNDED_DISTANCE": str(args.reader_bounded_distance),
            "ANNCHOR_TAIL_ISSUE_READER_PREFIX_GUARD": str(args.reader_prefix_guard),
            "ANNCHOR_TAIL_ISSUE_READER_PREFIX_GUARD_LINES": str(args.reader_prefix_guard_lines),
            "ANNCHOR_TAIL_ISSUE_READER_PREFIX_ESTIMATE": str(args.reader_prefix_estimate),
            "ANNCHOR_TAIL_ISSUE_READER_PREFIX_ESTIMATE_LINES": str(args.reader_prefix_estimate_lines),
            "ANNCHOR_TAIL_ISSUE_READER_PREFIX_ESTIMATE_MARGIN_PERMILLE": str(args.reader_prefix_estimate_margin_permille),
            "ANNCHOR_TAIL_ISSUE_READER_CURRENT_LINES": str(args.reader_current_lines),
            "ANNCHOR_TAIL_ISSUE_READER_REFINE_SEEDS": str(args.reader_refine_seeds),
            "ANNCHOR_TAIL_ISSUE_READER_REFINE_EDGES": str(args.reader_refine_edges),
            "ANNCHOR_TAIL_ISSUE_WRITER_EF_BUDGET": str(args.writer_ef_budget),
            "ANNCHOR_TAIL_ISSUE_WRITER_EF_BUDGET_CAP": str(args.writer_ef_cap),
            "ANNCHOR_TAIL_ISSUE_WRITER_EF_LIMIT": str(args.writer_ef_limit),
            "ANNCHOR_TAIL_ISSUE_WRITER_REPAIR_DEFER_BUDGET": str(args.repair_defer_budget),
            "ANNCHOR_TAIL_ISSUE_WRITER_REPAIR_DEFER_CAP": str(args.repair_defer_cap),
            "ANNCHOR_TAIL_ISSUE_WRITER_REPAIR_DRAIN_BUDGET": str(args.repair_drain_budget),
            "ANNCHOR_TAIL_ISSUE_WRITER_REPAIR_NO_DRAIN": str(args.repair_no_drain),
            "ANNCHOR_TAIL_ISSUE_WRITER_REPAIR_QUEUE_CAP": str(args.repair_queue_cap),
            "ANNCHOR_TAIL_ISSUE_WRITER_LOWVALUE_SKIP_BUDGET": str(args.lowvalue_skip_budget),
            "ANNCHOR_TAIL_ISSUE_WRITER_LOWVALUE_SKIP_CAP": str(args.lowvalue_skip_cap),
            "ANNCHOR_TAIL_ISSUE_WRITER_LOWVALUE_MARGIN_PERMILLE": str(args.lowvalue_margin_permille),
            "ANNCHOR_TAIL_ISSUE_WRITER_LEVEL0_BUDGET": str(args.writer_level0_budget),
            "ANNCHOR_TAIL_ISSUE_WRITER_LEVEL0_CAP": str(args.writer_level0_cap),
            "ANNCHOR_TAIL_ISSUE_READER_GUARD_NS": str(args.guard_ns),
            "ANNCHOR_TAIL_ISSUE_READER_GUARD2_NS": str(args.guard2_ns),
            "ANNCHOR_TAIL_ISSUE_LEASE_NS": str(args.lease_ns),
            "ANNCHOR_TAIL_ISSUE_WRITER_LINES": str(args.writer_lines),
            "ANNCHOR_TAIL_ISSUE_WRITER_ACTION": str(args.writer_action),
            "ANNCHOR_TAIL_ISSUE_CHECK_PERIOD": str(args.check_period),
            "ANNCHOR_TAIL_ISSUE_WRITER_BUDGET_LINES": str(args.writer_budget_lines),
            "ANNCHOR_TAIL_ISSUE_WRITER_DIST_MODE": str(args.dist_mode),
            "ANNCHOR_TAIL_ISSUE_WRITER_DIST_BUDGET": str(args.dist_budget),
            "ANNCHOR_TAIL_ISSUE_WRITER_DIST_BUDGET_CAP": str(args.dist_cap),
            "ANNCHOR_TAIL_ISSUE_WRITER_DIST_PREFIX_GUARD_LINES": str(args.dist_prefix_lines),
            "ANNCHOR_TAIL_ISSUE_WRITER_DIST_PREFIX_ESTIMATE_LINES": str(args.dist_prefix_estimate_lines),
            "ANNCHOR_TAIL_ISSUE_WRITER_DIST_PREFIX_ESTIMATE_MARGIN_PERMILLE": str(args.dist_prefix_estimate_margin_permille),
            "ANNCHOR_TAIL_ISSUE_WRITER_LEAD_SKIP_BUDGET": str(args.lead_skip_budget),
            "ANNCHOR_TAIL_ISSUE_WRITER_LEAD_SKIP_CAP": str(args.lead_skip_cap),
            "ANNCHOR_TAIL_ISSUE_WRITER_LEAD_HINT_BUDGET": str(args.lead_hint_budget),
            "ANNCHOR_TAIL_ISSUE_WRITER_LEAD_HINT_CAP": str(args.lead_hint_cap),
            "ANNCHOR_TAIL_ISSUE_WRITER_LEAD_HINT_ACTION": str(args.lead_hint_action),
            "ANNCHOR_TAIL_ISSUE_WRITER_FRONTIER_BUDGET": str(args.writer_frontier_budget),
            "ANNCHOR_TAIL_ISSUE_WRITER_FRONTIER_CAP": str(args.writer_frontier_cap),
            "ANNCHOR_TAIL_ISSUE_WRITER_FRONTIER_MARGIN_PERMILLE": str(args.writer_frontier_margin_permille),
            "ANNCHOR_TAIL_ISSUE_WRITER_SHARED_SCAN_BUDGET": str(args.writer_shared_scan_budget),
            "ANNCHOR_TAIL_ISSUE_WRITER_SHARED_SCAN_CAP": str(args.writer_shared_scan_cap),
            "ANNCHOR_TAIL_ISSUE_WRITER_SHARED_SCAN_ALWAYS": str(args.writer_shared_scan_always),
            "ANNCHOR_TAIL_ISSUE_WRITER_DEMOTE_BUDGET": str(args.writer_demote_budget),
            "ANNCHOR_TAIL_ISSUE_WRITER_DEMOTE_CAP": str(args.writer_demote_cap),
            "ANNCHOR_TAIL_ISSUE_WRITER_DEMOTE_LINES": str(args.writer_demote_lines),
            "ANNCHOR_TAIL_ISSUE_WRITER_OFFLOCK_PRUNE_BUDGET": str(args.writer_offlock_prune_budget),
            "ANNCHOR_TAIL_ISSUE_WRITER_OFFLOCK_PRUNE_CAP": str(args.writer_offlock_prune_cap),
            }
        )

    with open(args.lock, "w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another tail-issue runner holds {args.lock}")

        rows = []
        with open(out, "w") as out_file:
            header = (
                f"RUN mode={args.mode} pairs={args.pairs} initial={args.initial} "
                f"inserts={args.inserts} dim={args.dim} search_threads={args.search_threads} "
                f"insert_threads={args.insert_threads} duration_ms={args.duration_ms} ef={args.ef} "
                f"guard_ns={args.guard_ns} lease_ns={args.lease_ns} check_period={args.check_period} "
                f"guard2_ns={args.guard2_ns} "
                f"reader_prefetch={args.reader_prefetch} "
                f"reader_budget_lines={args.reader_budget_lines} "
                f"reader_window_extra={args.reader_window_extra} "
                f"reader_ef_limit={args.reader_ef_limit} "
                f"reader_ef_limit2={args.reader_ef_limit2} "
                f"reader_limit_mode={args.reader_limit_mode} "
                f"reader_expansion_budget={args.reader_expansion_budget} "
                f"reader_stop_alpha_permille={args.reader_stop_alpha_permille} "
                f"reader_conf_margin_permille={args.reader_conf_margin_permille} "
                f"reader_batch_size={args.reader_batch_size} "
                f"reader_lookahead_mode={args.reader_lookahead_mode} "
                f"reader_lookahead_budget={args.reader_lookahead_budget} "
                f"reader_stag_clamp_hops={args.reader_stag_clamp_hops} "
                f"reader_stag_clamp_min_expansions={args.reader_stag_clamp_min_expansions} "
                f"reader_frontier_ef={args.reader_frontier_ef} "
                f"reader_frontier_stag_hops={args.reader_frontier_stag_hops} "
                f"reader_frontier_min_expansions={args.reader_frontier_min_expansions} "
                f"reader_frontier_continue_budget={args.reader_frontier_continue_budget} "
                f"reader_retry_enable={args.reader_retry_enable} "
                f"reader_retry_margin_permille={args.reader_retry_margin_permille} "
                f"reader_continue_enable={args.reader_continue_enable} "
                f"reader_continue_margin_permille={args.reader_continue_margin_permille} "
                f"reader_bounded_distance={args.reader_bounded_distance} "
                f"reader_prefix_guard={args.reader_prefix_guard} "
                f"reader_prefix_guard_lines={args.reader_prefix_guard_lines} "
                f"reader_prefix_estimate={args.reader_prefix_estimate} "
                f"reader_prefix_estimate_lines={args.reader_prefix_estimate_lines} "
                f"reader_prefix_estimate_margin_permille={args.reader_prefix_estimate_margin_permille} "
                f"reader_current_lines={args.reader_current_lines} "
                f"reader_refine_seeds={args.reader_refine_seeds} "
                f"reader_refine_edges={args.reader_refine_edges} "
                f"writer_ef_budget={args.writer_ef_budget} "
                f"writer_ef_cap={args.writer_ef_cap} "
                f"writer_ef_limit={args.writer_ef_limit} "
                f"repair_defer_budget={args.repair_defer_budget} "
                f"repair_defer_cap={args.repair_defer_cap} "
                f"repair_drain_budget={args.repair_drain_budget} "
                f"repair_no_drain={args.repair_no_drain} "
                f"repair_queue_cap={args.repair_queue_cap} "
                f"lowvalue_skip_budget={args.lowvalue_skip_budget} "
                f"lowvalue_skip_cap={args.lowvalue_skip_cap} "
                f"lowvalue_margin_permille={args.lowvalue_margin_permille} "
                f"writer_level0_budget={args.writer_level0_budget} "
                f"writer_level0_cap={args.writer_level0_cap} "
                f"writer_lines={args.writer_lines} "
                f"writer_action={args.writer_action} "
                f"writer_budget_lines={args.writer_budget_lines} "
                f"dist_mode={args.dist_mode} dist_budget={args.dist_budget} "
                f"dist_cap={args.dist_cap} dist_prefix_lines={args.dist_prefix_lines} "
                f"dist_prefix_estimate_lines={args.dist_prefix_estimate_lines} "
                f"dist_prefix_estimate_margin_permille={args.dist_prefix_estimate_margin_permille} "
                f"lead_skip_budget={args.lead_skip_budget} lead_skip_cap={args.lead_skip_cap} "
                f"lead_hint_budget={args.lead_hint_budget} lead_hint_cap={args.lead_hint_cap} "
                f"lead_hint_action={args.lead_hint_action} "
                f"writer_frontier_budget={args.writer_frontier_budget} "
                f"writer_frontier_cap={args.writer_frontier_cap} "
                f"writer_frontier_margin_permille={args.writer_frontier_margin_permille} "
                f"writer_shared_scan_budget={args.writer_shared_scan_budget} "
                f"writer_shared_scan_cap={args.writer_shared_scan_cap} "
                f"writer_shared_scan_always={args.writer_shared_scan_always} "
                f"writer_demote_budget={args.writer_demote_budget} "
                f"writer_demote_cap={args.writer_demote_cap} "
                f"writer_demote_lines={args.writer_demote_lines} "
                f"writer_offlock_prune_budget={args.writer_offlock_prune_budget} "
                f"writer_offlock_prune_cap={args.writer_offlock_prune_cap} "
                f"search_cpus={args.search_cpus or 'none'} insert_cpus={args.insert_cpus or 'none'}\n"
            )
            print(header, end="", flush=True)
            out_file.write(header)
            out_file.flush()

            for pair in range(1, args.pairs + 1):
                for mode, env in (("baseline", base_env), (args.mode, policy_env)):
                    line = f"pair={pair} mode={mode}\n"
                    print(line, end="", flush=True)
                    out_file.write(line)
                    out_file.flush()
                    returncode, stdout = run_once(bench_cmd, env)
                    print(stdout, end="", flush=True)
                    out_file.write(stdout)
                    out_file.flush()
                    rows.append(metric_row(stdout, pair, mode, returncode))

            by_mode = {}
            for row in rows:
                by_mode.setdefault(row["mode"], []).append(row)

            for mode, mode_rows in by_mode.items():
                line = (
                    f"SUMMARY {mode} n {len(mode_rows)} "
                    f"p50_med {median(mode_rows, 'p50_ns')} "
                    f"p95_med {median(mode_rows, 'p95_ns')} "
                    f"p99_med {median(mode_rows, 'p99_ns')} "
                    f"p999_med {median(mode_rows, 'p999_ns')} "
                    f"insert_qps_med {median(mode_rows, 'insert_qps')} "
                    f"search_qps_med {median(mode_rows, 'search_qps')} "
                    f"insert_ops_med {median(mode_rows, 'insert_ops')} "
                    f"urgent_med {median(mode_rows, 'tail_reader_urgent_ops')} "
                    f"beam_clamps_med {median(mode_rows, 'tail_reader_beam_clamps')} "
                    f"beam_trimmed_med {median(mode_rows, 'tail_reader_beam_trimmed')} "
                    f"beam_clamps2_med {median(mode_rows, 'tail_reader_beam_clamps2')} "
                    f"beam_trimmed2_med {median(mode_rows, 'tail_reader_beam_trimmed2')} "
                    f"expansion_stops_med {median(mode_rows, 'tail_reader_expansion_stops')} "
                    f"margin_stops_med {median(mode_rows, 'tail_reader_margin_stops')} "
                    f"confidence_checks_med {median(mode_rows, 'tail_reader_confidence_checks')} "
                    f"confidence_grants_med {median(mode_rows, 'tail_reader_confidence_grants')} "
                    f"confidence_rejects_med {median(mode_rows, 'tail_reader_confidence_rejects')} "
                    f"batch_prefetches_med {median(mode_rows, 'tail_reader_batch_prefetches')} "
                    f"lookahead_deferred_med {median(mode_rows, 'tail_reader_lookahead_deferred')} "
                    f"lookahead_skipped_med {median(mode_rows, 'tail_reader_lookahead_skipped')} "
                    f"frontier_checks_med {median(mode_rows, 'tail_reader_frontier_checks')} "
                    f"frontier_eligible_med {median(mode_rows, 'tail_reader_frontier_eligible')} "
                    f"frontier_stops_med {median(mode_rows, 'tail_reader_frontier_stops')} "
                    f"frontier_continue_checks_med {median(mode_rows, 'tail_reader_frontier_continue_checks')} "
                    f"frontier_continue_grants_med {median(mode_rows, 'tail_reader_frontier_continue_grants')} "
                    f"retry_checks_med {median(mode_rows, 'tail_reader_retry_checks')} "
                    f"retry_runs_med {median(mode_rows, 'tail_reader_retry_runs')} "
                    f"retry_confident_skips_med {median(mode_rows, 'tail_reader_retry_confident_skips')} "
                    f"continue_checks_med {median(mode_rows, 'tail_reader_continue_checks')} "
                    f"continue_runs_med {median(mode_rows, 'tail_reader_continue_runs')} "
                    f"continue_confident_stops_med {median(mode_rows, 'tail_reader_continue_confident_stops')} "
                    f"reader_bounded_ops_med {median(mode_rows, 'tail_reader_bounded_distance_ops')} "
                    f"reader_bounded_early_med {median(mode_rows, 'tail_reader_bounded_distance_early')} "
                    f"reader_prefix_ops_med {median(mode_rows, 'tail_reader_prefix_guard_ops')} "
                    f"reader_prefix_early_med {median(mode_rows, 'tail_reader_prefix_guard_early')} "
                    f"reader_prefix_full_med {median(mode_rows, 'tail_reader_prefix_guard_full')} "
                    f"reader_prefix_estimate_ops_med {median(mode_rows, 'tail_reader_prefix_estimate_ops')} "
                    f"reader_prefix_estimate_skips_med {median(mode_rows, 'tail_reader_prefix_estimate_skips')} "
                    f"reader_prefix_estimate_full_med {median(mode_rows, 'tail_reader_prefix_estimate_full')} "
                    f"reader_current_prefetch_lines_med {median(mode_rows, 'tail_reader_current_prefetch_lines')} "
                    f"reader_refine_runs_med {median(mode_rows, 'tail_reader_refine_runs')} "
                    f"reader_refine_edges_med {median(mode_rows, 'tail_reader_refine_edges')} "
                    f"reader_refine_inserted_med {median(mode_rows, 'tail_reader_refine_inserted')} "
                    f"writer_ef_granted_med {median(mode_rows, 'tail_writer_ef_budget_granted')} "
                    f"writer_ef_limited_med {median(mode_rows, 'tail_writer_ef_limited_searches')} "
                    f"writer_ef_trimmed_med {median(mode_rows, 'tail_writer_ef_trimmed')} "
                    f"repair_defer_granted_med {median(mode_rows, 'tail_writer_repair_defer_granted')} "
                    f"repair_deferred_points_med {median(mode_rows, 'tail_writer_repair_deferred_points')} "
                    f"repair_deferred_edges_med {median(mode_rows, 'tail_writer_repair_deferred_edges')} "
                    f"repair_drained_med {median(mode_rows, 'tail_writer_repair_drained')} "
                    f"repair_queue_max_med {median(mode_rows, 'tail_writer_repair_queue_max')} "
                    f"lowvalue_skip_granted_med {median(mode_rows, 'tail_writer_lowvalue_skip_granted')} "
                    f"lowvalue_skip_checks_med {median(mode_rows, 'tail_writer_lowvalue_skip_checks')} "
                    f"lowvalue_skipped_med {median(mode_rows, 'tail_writer_lowvalue_skipped')} "
                    f"level0_granted_med {median(mode_rows, 'tail_writer_level0_granted')} "
                    f"level0_forced_med {median(mode_rows, 'tail_writer_level0_forced')} "
                    f"shape_med {median(mode_rows, 'tail_writer_dist_shaped_ops')} "
                    f"bounded_early_med {median(mode_rows, 'tail_writer_dist_bounded_early')} "
                    f"prefix_full_med {median(mode_rows, 'tail_writer_dist_prefix_full')} "
                    f"prefix_estimate_skips_med {median(mode_rows, 'tail_writer_dist_prefix_estimate_skips')} "
                    f"prefix_estimate_full_med {median(mode_rows, 'tail_writer_dist_prefix_estimate_full')} "
                    f"admission_skips_med {median(mode_rows, 'tail_writer_dist_admission_skips')} "
                    f"lead_skip_granted_med {median(mode_rows, 'tail_writer_lead_skip_granted')} "
                    f"lead_prefetch_skipped_med {median(mode_rows, 'tail_writer_lead_prefetch_skipped')} "
                    f"lead_hint_granted_med {median(mode_rows, 'tail_writer_lead_hint_granted')} "
                    f"lead_prefetch_hinted_med {median(mode_rows, 'tail_writer_lead_prefetch_hinted')} "
                    f"writer_frontier_granted_med {median(mode_rows, 'tail_writer_frontier_granted')} "
                    f"writer_frontier_checks_med {median(mode_rows, 'tail_writer_frontier_checks')} "
                    f"writer_frontier_skipped_med {median(mode_rows, 'tail_writer_frontier_skipped')} "
                    f"writer_shared_scan_granted_med {median(mode_rows, 'tail_writer_shared_scan_granted')} "
                    f"writer_shared_scan_used_med {median(mode_rows, 'tail_writer_shared_scan_used')} "
                    f"writer_demote_granted_med {median(mode_rows, 'tail_writer_demote_granted')} "
                    f"writer_demote_used_med {median(mode_rows, 'tail_writer_demote_used')} "
                    f"writer_demote_lines_med {median(mode_rows, 'tail_writer_demote_lines')} "
                    f"writer_offlock_prune_granted_med {median(mode_rows, 'tail_writer_offlock_prune_granted')} "
                    f"writer_offlock_prune_attempts_med {median(mode_rows, 'tail_writer_offlock_prune_attempts')} "
                    f"writer_offlock_prune_success_med {median(mode_rows, 'tail_writer_offlock_prune_success')} "
                    f"writer_offlock_prune_validate_fail_med {median(mode_rows, 'tail_writer_offlock_prune_validate_fail')} "
                    f"writer_offlock_prune_skipped_med {median(mode_rows, 'tail_writer_offlock_prune_skipped')}\n"
                )
                print(line, end="")
                out_file.write(line)

            base = {row["pair"]: row for row in by_mode.get("baseline", [])}
            policy = {row["pair"]: row for row in by_mode.get(args.mode, [])}
            print("PAIRED_DELTAS")
            out_file.write("PAIRED_DELTAS\n")
            paired = []
            for pair in sorted(base.keys() & policy.keys()):
                b = base[pair]
                p = policy[pair]
                vals = {
                    "p99_pct": (p["p99_ns"] / b["p99_ns"] - 1.0) * 100.0,
                    "p999_pct": (p["p999_ns"] / b["p999_ns"] - 1.0) * 100.0,
                    "insert_pct": (p["insert_qps"] / b["insert_qps"] - 1.0) * 100.0,
                    "search_pct": (p["search_qps"] / b["search_qps"] - 1.0) * 100.0,
                }
                paired.append(vals)
                line = (
                    f"PAIR {pair} p99_pct {vals['p99_pct']:.3f} "
                    f"p999_pct {vals['p999_pct']:.3f} "
                    f"insert_pct {vals['insert_pct']:.3f} "
                    f"search_pct {vals['search_pct']:.3f} "
                    f"urgent {int(p.get('tail_reader_urgent_ops', 0))} "
                    f"beam_clamps {int(p.get('tail_reader_beam_clamps', 0))} "
                    f"beam_trimmed {int(p.get('tail_reader_beam_trimmed', 0))} "
                    f"beam_clamps2 {int(p.get('tail_reader_beam_clamps2', 0))} "
                    f"beam_trimmed2 {int(p.get('tail_reader_beam_trimmed2', 0))} "
                    f"expansion_stops {int(p.get('tail_reader_expansion_stops', 0))} "
                    f"margin_stops {int(p.get('tail_reader_margin_stops', 0))} "
                    f"confidence_checks {int(p.get('tail_reader_confidence_checks', 0))} "
                    f"confidence_grants {int(p.get('tail_reader_confidence_grants', 0))} "
                    f"confidence_rejects {int(p.get('tail_reader_confidence_rejects', 0))} "
                    f"batch_prefetches {int(p.get('tail_reader_batch_prefetches', 0))} "
                    f"lookahead_deferred {int(p.get('tail_reader_lookahead_deferred', 0))} "
                    f"lookahead_skipped {int(p.get('tail_reader_lookahead_skipped', 0))} "
                    f"frontier_checks {int(p.get('tail_reader_frontier_checks', 0))} "
                    f"frontier_eligible {int(p.get('tail_reader_frontier_eligible', 0))} "
                    f"frontier_stops {int(p.get('tail_reader_frontier_stops', 0))} "
                    f"frontier_continue_checks {int(p.get('tail_reader_frontier_continue_checks', 0))} "
                    f"frontier_continue_grants {int(p.get('tail_reader_frontier_continue_grants', 0))} "
                    f"retry_checks {int(p.get('tail_reader_retry_checks', 0))} "
                    f"retry_runs {int(p.get('tail_reader_retry_runs', 0))} "
                    f"retry_confident_skips {int(p.get('tail_reader_retry_confident_skips', 0))} "
                    f"continue_checks {int(p.get('tail_reader_continue_checks', 0))} "
                    f"continue_runs {int(p.get('tail_reader_continue_runs', 0))} "
                    f"continue_confident_stops {int(p.get('tail_reader_continue_confident_stops', 0))} "
                    f"reader_bounded_ops {int(p.get('tail_reader_bounded_distance_ops', 0))} "
                    f"reader_bounded_early {int(p.get('tail_reader_bounded_distance_early', 0))} "
                    f"reader_prefix_ops {int(p.get('tail_reader_prefix_guard_ops', 0))} "
                    f"reader_prefix_early {int(p.get('tail_reader_prefix_guard_early', 0))} "
                    f"reader_prefix_full {int(p.get('tail_reader_prefix_guard_full', 0))} "
                    f"reader_prefix_estimate_ops {int(p.get('tail_reader_prefix_estimate_ops', 0))} "
                    f"reader_prefix_estimate_skips {int(p.get('tail_reader_prefix_estimate_skips', 0))} "
                    f"reader_prefix_estimate_full {int(p.get('tail_reader_prefix_estimate_full', 0))} "
                    f"reader_current_prefetch_lines {int(p.get('tail_reader_current_prefetch_lines', 0))} "
                    f"reader_refine_runs {int(p.get('tail_reader_refine_runs', 0))} "
                    f"reader_refine_edges {int(p.get('tail_reader_refine_edges', 0))} "
                    f"reader_refine_inserted {int(p.get('tail_reader_refine_inserted', 0))} "
                    f"writer_ef_limited {int(p.get('tail_writer_ef_limited_searches', 0))} "
                    f"writer_ef_trimmed {int(p.get('tail_writer_ef_trimmed', 0))} "
                    f"repair_deferred_points {int(p.get('tail_writer_repair_deferred_points', 0))} "
                    f"repair_drained {int(p.get('tail_writer_repair_drained', 0))} "
                    f"repair_queue_max {int(p.get('tail_writer_repair_queue_max', 0))} "
                    f"lowvalue_skip_granted {int(p.get('tail_writer_lowvalue_skip_granted', 0))} "
                    f"lowvalue_skip_checks {int(p.get('tail_writer_lowvalue_skip_checks', 0))} "
                    f"lowvalue_skipped {int(p.get('tail_writer_lowvalue_skipped', 0))} "
                    f"level0_granted {int(p.get('tail_writer_level0_granted', 0))} "
                    f"level0_forced {int(p.get('tail_writer_level0_forced', 0))} "
                    f"shape {int(p.get('tail_writer_dist_shaped_ops', 0))} "
                    f"bounded_early {int(p.get('tail_writer_dist_bounded_early', 0))} "
                    f"prefix_full {int(p.get('tail_writer_dist_prefix_full', 0))} "
                    f"prefix_estimate_skips {int(p.get('tail_writer_dist_prefix_estimate_skips', 0))} "
                    f"prefix_estimate_full {int(p.get('tail_writer_dist_prefix_estimate_full', 0))} "
                    f"admission_skips {int(p.get('tail_writer_dist_admission_skips', 0))} "
                    f"lead_skip {int(p.get('tail_writer_lead_prefetch_skipped', 0))} "
                    f"lead_hint_granted {int(p.get('tail_writer_lead_hint_granted', 0))} "
                    f"lead_prefetch_hinted {int(p.get('tail_writer_lead_prefetch_hinted', 0))} "
                    f"writer_frontier_granted {int(p.get('tail_writer_frontier_granted', 0))} "
                    f"writer_frontier_checks {int(p.get('tail_writer_frontier_checks', 0))} "
                    f"writer_frontier_skipped {int(p.get('tail_writer_frontier_skipped', 0))} "
                    f"writer_shared_scan_granted {int(p.get('tail_writer_shared_scan_granted', 0))} "
                    f"writer_shared_scan_used {int(p.get('tail_writer_shared_scan_used', 0))} "
                    f"writer_demote_granted {int(p.get('tail_writer_demote_granted', 0))} "
                    f"writer_demote_used {int(p.get('tail_writer_demote_used', 0))} "
                    f"writer_demote_lines {int(p.get('tail_writer_demote_lines', 0))} "
                    f"writer_offlock_prune_granted {int(p.get('tail_writer_offlock_prune_granted', 0))} "
                    f"writer_offlock_prune_attempts {int(p.get('tail_writer_offlock_prune_attempts', 0))} "
                    f"writer_offlock_prune_success {int(p.get('tail_writer_offlock_prune_success', 0))} "
                    f"writer_offlock_prune_validate_fail {int(p.get('tail_writer_offlock_prune_validate_fail', 0))} "
                    f"writer_offlock_prune_skipped {int(p.get('tail_writer_offlock_prune_skipped', 0))}\n"
                )
                print(line, end="")
                out_file.write(line)

            for key in ("p99_pct", "p999_pct", "insert_pct", "search_pct"):
                vals = [row[key] for row in paired]
                line = (
                    f"PAIRED_SUMMARY {key} median {statistics.median(vals)} "
                    f"mean {statistics.mean(vals)} wins_lt0 {sum(v < 0 for v in vals)} "
                    f"n {len(vals)}\n"
                )
                print(line, end="")
                out_file.write(line)

            line = f"OUT={out}\n"
            print(line, end="")
            out_file.write(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
