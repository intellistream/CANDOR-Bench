"""Unit test for the incremental ground truth loader.

Synthesizes an index + split file in the binary format the Go tooling
emits, then checks top_k lookup and record evaluation. Run directly:
``python3 test_incremental_gt.py``
"""

import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
)

from concurrency.incremental_gt import (  # noqa: E402
    IncrementalGTProvider,
    compute_recall_against_gt,
    evaluate_records,
)


def write_split(path, offsets, n_per_batch, k, with_query_tags):
    """Write one split file; GT for (offset, tag) is [tag, tag+1, ...] * 10."""
    with open(path, "wb") as f:
        f.write(struct.pack("<iii", n_per_batch, k, len(offsets)))
        for off in offsets:
            f.write(struct.pack("<Q", off))
            tags = np.arange(n_per_batch, dtype="<u4")
            if with_query_tags:
                f.write(tags.tobytes())
            dists = np.zeros((n_per_batch, k), dtype="<f4")
            f.write(dists.tobytes())
            indices = (
                tags[:, None] * 10 + np.arange(k, dtype="<u4")[None, :]
            ).astype("<u4")
            f.write(indices.tobytes())


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        n, k = 8, 5
        write_split(os.path.join(tmp, "gt_0.bin"), [100, 200], n, k, True)
        write_split(os.path.join(tmp, "gt_1.bin"), [300], n, k, False)
        index_path = os.path.join(tmp, "gt_index.txt")
        with open(index_path, "w") as f:
            f.write("# comment line\n")
            f.write("gt_0.bin 100 200\n")
            f.write("gt_1.bin 300 300\n")

        provider = IncrementalGTProvider(index_path, recall_at=3)

        gt = provider.top_k(100, 2)
        if gt is None or gt.tolist() != [20, 21, 22]:
            failures.append(f"top_k(100,2) = {gt}, want [20, 21, 22]")
        gt = provider.top_k(300, 7)  # second file, implicit query tags
        if gt is None or gt.tolist() != [70, 71, 72]:
            failures.append(f"top_k(300,7) = {gt}, want [70, 71, 72]")
        if provider.top_k(999, 0) is not None:
            failures.append("top_k(999,...) should be None (no split)")
        gt = provider.top_k(200, 1)  # switch back: reload of first file
        if gt is None or gt.tolist() != [10, 11, 12]:
            failures.append(f"top_k(200,1) = {gt}, want [10, 11, 12]")

        r = compute_recall_against_gt(
            np.array([10, 11, 12]), np.array([10, 12, 99]), 3
        )
        if abs(r - 2 / 3) > 1e-9:
            failures.append(f"recall = {r}, want 2/3")

        # evaluate_records: 2 perfect hits, 1 half(ish), 1 missing GT.
        records = {
            "insert_offsets": np.array([100, 100, 300, 999], dtype=np.uint64),
            "query_tags": np.array([2, 3, 7, 0], dtype=np.uint32),
            "result_tags": np.array(
                [
                    [20, 21, 22],
                    [30, 31, 32],
                    [70, 99, 98],
                    [0, 1, 2],
                ],
                dtype=np.uint32,
            ),
            "k": 3,
        }
        out = evaluate_records(records, provider)
        if out["scored"] != 3 or out["missing_gt"] != 1:
            failures.append(f"scored/missing = {out['scored']}/{out['missing_gt']}")
        want = (1.0 + 1.0 + 1 / 3) / 3
        if abs(out["recall"] - want) > 1e-9:
            failures.append(f"overall recall = {out['recall']}, want {want}")
        if abs(out["per_stage"].get(300, -1) - 1 / 3) > 1e-9:
            failures.append(f"per_stage[300] = {out['per_stage'].get(300)}")

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
