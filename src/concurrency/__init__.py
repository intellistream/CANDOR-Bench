"""Python orchestration for the native concurrent benchmark driver.

The hot path (concurrent insert+search, rate limiting, latency capture)
lives in the C++ `concurrency_native` extension; this package holds the
offline pieces: incremental ground truth,
recall evaluation, config handling.
"""

from .incremental_gt import (
    IncrementalGTProvider,
    compute_partial_diff_distribution,
    compute_recall_against_gt,
    evaluate_records,
)

__all__ = [
    "IncrementalGTProvider",
    "compute_partial_diff_distribution",
    "compute_recall_against_gt",
    "evaluate_records",
]
