# Incremental ground truth

When the index changes while queries run, "recall" against the final
dataset is the wrong question. A search that executed when only 40k of
100k points were inserted should be judged against the true neighbors
among those 40k. Incremental ground truth is exactly that: one
ground-truth table per committed insert offset.

## Generating it

Build the tools once:

    cmake -B tools/build -S tools -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
    cmake --build tools/build -j

`-DENABLE_CUDA=ON` (the default) picks up a CUDA toolkit if present and
speeds up the brute-force pass considerably on large datasets.

Then, with base and query sets in aligned-bin format (int32 npts, int32
dim, float32 data — `data/fvecs_to_bin.py` converts fvecs):

    tools/build/compute_incr_gt \
        --base_path  sift_base.bin \
        --query_path sift_query.bin \
        --batch_gt_path sift_incr.gt \
        --k 20 --inc 500 --threads 16

This writes slice files plus an index file named like
`sift_incr_offset_index.txt`. The index file is what the benchmark
config points at. One slice holds many offsets; the loader keeps a
single slice in memory at a time. File-level details and how to sanity
check sizes: `tools/README_GT_SLICES.md`.

Picking `--inc`: the benchmark commits inserts at multiples of
`workload.batch_size` starting from `data.begin_num`. Every committed
offset a search can observe needs a ground-truth entry, so use
`--inc` equal to `batch_size`, or a divisor of it. If they don't line
up the run still works, but the affected records are skipped and
reported as `missing GT` — a nonzero count there is the first thing to
check when recall looks off.

`--k` must be at least `search.recall_at`. `--stream` additionally
stores query ids per offset and restricts each offset's queries to the
most recent increment window; without it all queries get an entry at
every offset, which is what the round_robin workload wants.

## Consuming it

In the run config:

    data:
      incr_query_path: sift_query.bin     # defaults to query_path
      incr_gt_path: sift_incr_offset_index.txt

The runner prints one line per variant:

    [sift] incremental recall: 0.9420 (40000 scored, 0 missing GT)

and writes the value to the `incr_recall` column of
`benchmark_results.csv`. Per-stage means are available from
`concurrency.incremental_gt.evaluate_records` if you need the curve
rather than the average.

What the number means: each search record carries the committed offset
it ran at. Its results are compared with the true top-k among the
points committed at that moment. Results pointing at points that were
inserted but not yet committed count as misses — the gap between
incremental and full-visibility recall is the freshness cost, and it
shrinks as the snapshot lag (printed in the progress line) shrinks.

## Cross-checking

The C++ side can score the same records independently:

    tools/build/calc_incr_recall \
        --res_path  results/.../files/<name>_incr.res \
        --gt_path   sift_incr_offset_index.txt \
        --recall_path out.rc --k 20

The weighted average over its offset buckets matches the runner's
number; we keep both paths alive precisely so they can check each
other.
