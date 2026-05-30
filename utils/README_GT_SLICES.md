# Incremental GT Slices: What Is Normal

This note explains how `compute_incr_gt` writes `*_slices` files, what sizes are expected, and how to detect bad outputs quickly.

## 1) Output layout

Given:

- `--batch_gt_path data/<ds>/<name>.gt20`

The tool writes:

- slices directory: `data/<ds>/<name>_slices/`
- index file: `data/<ds>/<name>_offset_index.txt`

Each slice file stores:

- header (12 bytes): `n`, `k`, `batches_in_file` (3 x `uint32`)
- for each batch:
  - `offset` (`uint64`, 8 bytes)
  - optional `query_ids` (`uint32 * n`, only in `--stream`)
  - distances (`float * n * k`)
  - ids (`int32 * n * k`)

Relevant implementation:

- query selection (`--stream` vs non-stream): `utils/compute_incr_gt.cpp` lines ~317-330
- payload writing: `utils/compute_incr_gt.cpp` lines ~428-450
- default `max_batches_per_file=1000`: `utils/compute_incr_gt.cpp` line ~121

## 2) Size formulas

Let:

- `n = header query count`
- `k = top-k`
- `b = batches in this slice file`

Then:

- non-stream batch bytes: `8 + n * k * 8`
- stream batch bytes: `8 + n * 4 + n * k * 8`
- file size:
  - non-stream: `12 + b * (8 + n*k*8)`
  - stream: `12 + b * (8 + n*4 + n*k*8)`

## 3) What is "normal" for common modes

### A) Base incremental GT (non-stream)

Expected `n` in header:

- `n = number_of_queries_in_<query_path>`

If `n` is very large (for example 1,000,000 or 10,000,000), files become huge by design.

### B) Stream incremental GT (`--stream`)

Expected `n` in header:

- `n = inc` (e.g. 20, 100, 200), because each batch evaluates only the latest stream window

So stream slice files should be much smaller.

## 4) Mapping to benchmark query modes

This is the part that is easy to misread.

The online benchmark always draws **workload queries** from `data.incr_query_path`.
`overall_query_path` is only for final overall recall, not for incremental recall.

That means the incremental GT must match the **same workload query dataset** that the benchmark uses.

### A) `query_mode: round_robin`

Online behavior:

- the benchmark cycles through `incr_query_path`
- query tags are the workload-query indices `0..Q-1`
- the query set itself is fixed across stages

Matching GT:

- use the same file as `incr_query_path`
- run `compute_incr_gt` **without** `--stream`
- use `--inc = workload.batch_size`

Important consequence:

- if `incr_query_path` is a large `*_query_stream.bin` file, exact round-robin incremental GT will also be huge, because each snapshot stores results for **all** workload queries

### B) `query_mode: chasing`

Online behavior:

- the benchmark uses the latest stage window `[offset-batch_size, offset)` from `incr_query_path`
- query tags are the global positions inside that workload stream

Matching GT:

- use the same file as `incr_query_path`
- run `compute_incr_gt` **with** `--stream`
- use `--inc = workload.batch_size`

### C) `overall_gt_path`

This is separate from incremental GT.
It is only for final overall recall on `overall_query_path`.

## 5) Current SIFT example in this workspace

Observed:

- `data/sift/sift_stream_i20_slices`: header sample `n=20,k=20,b=1000`, total ~157M (normal)
- `data/sift/sift_stream_i100_slices`: header sample `n=100,k=20,b=1000`, total ~157M (normal)
- `data/sift/sift_stream_i200_slices`: header sample `n=200,k=20,b=1000`, total ~157M (normal)
- historical `sift_base_i20_slices`: header sample `n=1000000,k=20,b=1000`, total ~459G (abnormally large for practical use)

Interpretation:

- stream sets look healthy
- the old `sift_base_i20` shape exploded because non-stream GT stores all workload queries for every offset; if the workload query set is very large, the output becomes impractical very quickly

## 6) Quick sanity checks

### Check slice directory sizes

```bash
du -h --max-depth=1 data/<dataset>
```

### Check one file header (`n k batches`)

```bash
od -An -t u4 -N 12 data/<dataset>/<name>_slices/<one_file>.gt20
```

### Compare expected vs actual size

```bash
# non-stream
expected=$((12 + b*(8 + n*k*8)))

# stream
expected=$((12 + b*(8 + n*4 + n*k*8)))
```

If `actual != expected`, the file is likely incomplete/corrupted.

## 7) Practical guidance

- For incremental recall, always start from the file actually used as `incr_query_path` in the benchmark config.
- For `round_robin`, use the same workload query file but do **not** pass `--stream`.
- For `chasing`, use the same workload query file and do pass `--stream`.
- If a large run is interrupted, remove partial files before re-generating, otherwise index and payload can disagree.
- `compute_incr_gt --threads` is query-parallel. If `--stream --inc 20`, using more than about `20` threads has little value because each batch only contains `20` queries.

## 8) Ready-to-run commands

### A) Remove abnormal old artifacts

```bash
rm -rf data/sift/sift_base_i20_slices data/sift/sift_base_i20_offset_index.txt
```

### B) Full SIFT round-robin incremental GT (expensive)

This matches `round_robin` semantics for a fixed workload query set:

```bash
utils/build/compute_incr_gt \
  --base_path data/sift/sift_base.bin \
  --query_path data/sift/sift_query_stream.bin \
  --batch_gt_path data/sift/sift_rr_i20.gt20 \
  --k 20 \
  --inc 20 \
  --threads 20
```

### C) Full SIFT chasing incremental GT

This matches `chasing` semantics for the same workload query stream:

```bash
utils/build/compute_incr_gt \
  --base_path data/sift/sift_base.bin \
  --query_path data/sift/sift_query_stream.bin \
  --batch_gt_path data/sift/sift_stream_i20.gt20 \
  --k 20 \
  --inc 20 \
  --threads 20 \
  --stream
```

### D) Lightweight sanity run (fast)

Use this to verify correctness of pipeline/header before launching expensive full runs:

```bash
utils/build/compute_incr_gt \
  --base_path data/sift/sift_base_200k.bin \
  --query_path data/sift/sift_query_1k.bin \
  --batch_gt_path data/sift/sift_base200k_q1k_i200k.gt20 \
  --k 20 \
  --inc 200000 \
  --threads 32
```

Expected sample header for this lightweight run:

- `n=1000, k=20, batches=1`

## 9) Current status after cleanup (2026-03-22)

- Abnormal artifacts removed:
  - `data/sift/sift_base_i20_slices`
  - `data/sift/sift_base_i20_offset_index.txt`
- `data/sift` disk usage dropped from ~461G to ~2.1G.
- Existing stream slices remain unchanged and valid.
