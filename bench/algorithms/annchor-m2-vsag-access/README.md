# ANNchor M2 VSAG-Access Prototype

This is a C++-only fork of `annchor-m2` for testing a VSAG-style access
pattern before adding tail-priority budget logic. It is intentionally not wired
into the Go benchmark factory.

Current change:

```text
graph expansion keeps the original ANNchor control flow
  -> prefetch candidate vector payloads with compile-time stride/depth knobs
  -> include an opt-in code-size-aware depth mode for full-vector access tests
  -> compute full-vector distances with the original distance function
```

No quantization, no central queue, no reader/writer priority policy, and no
cross-thread load scheduler. The first rough version that gathered neighbors
into `std::vector` before distance computation was discarded because it broke
ANNchor's existing prefetch/compute overlap and was not a faithful VSAG port.

Initial standalone smoke on random 128D vectors:

```text
compile-time depth=1 stride=1:
  close to original, preserves baseline shape

compile-time depth=2:
  fewer cache misses in a 5k perf smoke, but higher instructions/cycles and
  slower insert wall time

compile-time depth=4 stride=2:
  no stable single-thread search win, slower insert construction

code-aware depth=0 cap=8:
  functional smoke passes, but 128D raw float vectors issue up to 8 prefetch
  lines per candidate and should not be treated as an always-on writer setting

Review smoke after unrolling the helper:
  20k random 128D, 3 serial runs:
    original search avg ~= 0.638s, total avg ~= 13.89s
    depth=1 search avg ~= 0.656s, total avg ~= 14.12s
    code-aware search avg ~= 0.614s, total avg ~= 14.91s
  20k perf sample:
    code-aware reduces L2 demand-data misses, but increases software prefetch
    traffic, instructions, cycles, and total insert/build time.

VSAG-like auto stride review:
  ANNCHOR_VSAG_PREFETCH_STRIDE=0 derives the lookahead as
  max(1, data_size / 128 - 1), matching VSAG HNSWlib's raw-vector prefetch
  distance rule. On 20k random 128D, code-aware auto-stride search avg was
  ~= 0.644s versus original ~= 0.673s in that run, but total avg was
  ~= 15.57s versus original ~= 14.21s. This confirms the partial mechanism can
  move demand misses earlier, but it is not a complete VSAG port.

Tail-issue prototype review:
  Added a no-pause/no-sleep tail lease prototype controlled by
  ANNCHOR_TAIL_ISSUE_ENABLE. Reader operations can publish a short tail lease
  after a guard time. No correctness path is skipped. The implementation is
  intentionally not a pause/sleep/yield mechanism and does not use a central
  cross-thread load queue.

  Negative routes tested on the 10k+3k, 4-search/1-insert, 2s microbench:
    - writer prefetch suppression: sometimes lowers p99, but insert QPS drops
      by about 10%+, too high for the M3 constraint;
    - writer T1/T2/NTA prefetch hints: no stable p99 win and still hurts insert;
    - reader urgent extra prefetch and reader local prefetch window: increases
      search-side prefetch traffic and usually worsens p99;
    - scalar or uncapped narrow writer distance compute: lowers writer issue
      width but destroys insert throughput.

  Important implementation correction:
    enabled-but-no-policy must be near baseline. Early versions paid
    per-search/per-insert scope and per-prefetch checks even when no policy
    fired. The current code gates the hot path so ANNCHOR_TAIL_ISSUE_ENABLE=1
    with all policy budgets at zero is effectively a no-policy control.

  Clean rejection after paired validation:
    tail-triggered, globally capped writer distance-load shaping, using a
    writer-only SSE4 distance path for a small number of future construction
    distance computations. In one 5-run microbench this looked promising:
      baseline median p99=188578 ns, insert_qps=1104.54
      sse48cap48_g80 median p99=170079 ns, insert_qps=1071.17
      delta: p99 -9.81%, insert_qps -3.02%
    A cleaner 4-pair run with a larger insert pool rejected it:
      artifact: /tmp/annchor_tail_single_valid_sse48_20260528_044319.log
      p99 median delta: -0.43%
      p999 median delta: +0.89%
      insert_qps median delta: -5.75%
    Decision: sse48cap48_g80 is not a valid candidate. The p99 movement is
    too small, p999 does not improve, and insert loss exceeds the target.

    A lighter 4-pair version is also rejected:
      artifact: /tmp/annchor_tail_single_valid_sse24cap24_g100_20260528_044959.log
      p99 median delta: +1.80%
      p999 median delta: +4.22%
      insert_qps median delta: -5.50%
    Decision: narrower writer distance-load shaping is not the next route.
    It still costs insert throughput and does not produce stable search-tail
    improvement.

    Writer lookahead prefetch suppression is also rejected at smoke level:
      artifact: /tmp/annchor_tail_single_valid_smoke_leadskip64_g80_20260528_045625.log
      lead_prefetch_skipped=16840
      p99 delta: +13.35%, insert_qps delta: -6.87%
      artifact: /tmp/annchor_tail_single_valid_smoke_leadskip8_g120_20260528_045702.log
      lead_prefetch_skipped=80
      p99 delta: +5.04%, insert_qps delta: -9.23%
    Decision: suppressing writer future prefetch is not a viable priority lane;
    it slows or perturbs construction without giving a stable reader-tail win.
```

Experiment runner:

```bash
python3 bench/algorithms/annchor-m2-vsag-access/run_tail_issue_pair.py \
  --mode sse24cap24_g100 \
  --dist-budget 24 \
  --dist-cap 24 \
  --guard-ns 100000 \
  --pairs 4
```

The runner uses `/tmp/annchor_tail_issue_pair.lock` to prevent overlapping
paired runs. Overlapping logs are contaminated and must be discarded.

Additional rejected writer-side routes:

```text
writer lookahead prefetch suppression:
  /tmp/annchor_tail_single_valid_smoke_leadskip64_g80_20260528_045625.log
    lead_prefetch_skipped=16840
    p99 +13.35%, insert_qps -6.87%
  /tmp/annchor_tail_single_valid_smoke_leadskip8_g120_20260528_045702.log
    lead_prefetch_skipped=80
    p99 +5.04%, insert_qps -9.23%

writer bounded-distance early exit:
  /tmp/annchor_tail_single_valid_smoke_bound64_g80_20260528_050054.log
    bounded_early=30357
    p99 +41.95%, insert_qps -11.27%
  /tmp/annchor_tail_single_valid_smoke_bound8_g120_20260528_050121.log
    bounded_early=29
    p99 -1.89%, insert_qps -11.77%
```

Current conclusion:

```text
Do not keep optimizing by directly slowing writer vector issue or writer
distance computation in this fork. The tested writer-side routes all hurt
insert throughput and do not produce stable search-tail improvement.
```

Writer lookahead suppression knobs:

```text
ANNCHOR_TAIL_ISSUE_WRITER_LEAD_SKIP_BUDGET
ANNCHOR_TAIL_ISSUE_WRITER_LEAD_SKIP_CAP
```

These knobs are currently for negative evidence only.

The current takeaway is not that VSAG's access design is ineffective. VSAG's
reported benefit depends on compact code layout, batch distance computation,
and tuned code-size-aware prefetch. This fork now includes the code-size-aware
part for raw ANNchor vectors as an explicit mode: with
`ANNCHOR_VSAG_PREFETCH_DEPTH_LINES=0`, prefetch depth is derived from
`data_size_` in 64-byte cache-line units and capped by
`ANNCHOR_VSAG_PREFETCH_MAX_CODE_LINES`. It is intentionally not the default
because `d=128` raw float vectors already span 8 cache lines, and always-on
full-vector prefetch can increase insert-side memory pressure. ANNchor's raw
float-vector layout still does not get VSAG's full benefit from prefetch-only
changes. For M3, this fork should be used to add tail-active role policy, not
as an always-on writer prefetch optimization.

Knobs:

```text
ANNCHOR_VSAG_PREFETCH_DEPTH_LINES=1
  default: shallow fixed-depth prefetch, capped by the vector payload size

ANNCHOR_VSAG_PREFETCH_MAX_CODE_LINES=8
  cap for the code-size-aware depth; the current helper issues at most 8 lines

ANNCHOR_VSAG_PREFETCH_DEPTH_LINES=0
  code-size-aware mode; use ceil(data_size_ / 64) cache lines, then cap it

ANNCHOR_VSAG_PREFETCH_STRIDE=1
  how far ahead the graph-expansion loop prefetches candidate payloads

ANNCHOR_VSAG_PREFETCH_STRIDE=0
  VSAG-like auto lookahead: max(1, data_size / 128 - 1)

CMAKE_CXX_FLAGS="-DANNCHOR_VSAG_PREFETCH_DEPTH_LINES=2 -DANNCHOR_VSAG_PREFETCH_STRIDE=1"
CMAKE_CXX_FLAGS="-DANNCHOR_VSAG_PREFETCH_DEPTH_LINES=0 -DANNCHOR_VSAG_PREFETCH_MAX_CODE_LINES=8"
CMAKE_CXX_FLAGS="-DANNCHOR_VSAG_PREFETCH_DEPTH_LINES=0 -DANNCHOR_VSAG_PREFETCH_STRIDE=0"
```

Tail-issue knobs:

```text
ANNCHOR_TAIL_ISSUE_ENABLE=1
ANNCHOR_TAIL_ISSUE_READER_GUARD_NS=500000
ANNCHOR_TAIL_ISSUE_LEASE_NS=50000
ANNCHOR_TAIL_ISSUE_READER_BUDGET_LINES=16
ANNCHOR_TAIL_ISSUE_READER_PREFETCH=0|1
ANNCHOR_TAIL_ISSUE_READER_WINDOW_EXTRA=0|2|4|8
ANNCHOR_TAIL_ISSUE_WRITER_LINES=0|1
ANNCHOR_TAIL_ISSUE_WRITER_ACTION=0|1|2|3
ANNCHOR_TAIL_ISSUE_WRITER_BUDGET_LINES=0
ANNCHOR_TAIL_ISSUE_WRITER_DIST_MODE=0|1|2
ANNCHOR_TAIL_ISSUE_WRITER_DIST_BUDGET=0
ANNCHOR_TAIL_ISSUE_WRITER_DIST_BUDGET_CAP=0
ANNCHOR_TAIL_ISSUE_CHECK_PERIOD=512
```

Standalone build check:

```bash
cmake -S bench/algorithms/annchor-m2-vsag-access/ANNchor \
  -B /tmp/annchor-m2-vsag-access-build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/annchor-m2-vsag-access-build --target bench -j 8
```

# Original ANNchor M2 Notes

This directory is a clean M1 fork plus the current M2 fresh-validation path.
M1 supplies the committed graph view. M2 handles the in-flight fresh horizon
with online graph-region rendezvous and exact admission.

The hot path is:

```text
committed query footprint regions
  join with
fresh insert-side graph regions
  through a temporary validation sidecar
```

The graph does not store live query lists or live insert lists. Graph nodes only
carry lightweight region metadata:

```text
node -> primary graph region + small overlap region set
```

The region metadata is an online graph-region cover, not a disjoint partition.
Regions may overlap. Dense graph neighborhoods open more regions only after
their evidence-supported local regions hit `ANNCHOR_M2_REGION_CAP`; sparse
neighborhoods keep reusing under-filled regions. The maximum region budget grows
with the committed graph size and defaults to a 65536 cap for compact postings.

Integrated path:

```text
1. M1 searches the committed, pruning-aware graph prefix.
2. M2 lazily builds graph-region metadata from committed base-layer edges.
3. Each committed node owns one primary region and a few overlap regions.
4. Each region keeps a few short edge anchors `(outside -> inside)`.
5. Query regions come from the committed competitive footprint.
6. Fresh regions come from inserted-node graph neighbors when available.
7. Large fresh windows build a temporary `region -> fresh rows` posting table.
8. Query regions union matching fresh postings into candidates.
9. Candidates are exact-scored before admission.
```

The region evidence is intentionally local. If `x` is closer to the inside
endpoint of a graph edge than to the outside endpoint, that edge supports the
inside endpoint's region for `x`. This follows the same practical signal used by
graph ANN search: useful graph steps move closer to the target. M2 does not try
to prove a full partition theorem; it uses the signal only to reduce candidate
fresh rows before exact scoring.

The implementation is wired through:

```text
annchor_m2.hpp::fresh_merge_graph_region
annchor_m2_cgo.cpp::annchor_m2_fresh_merge_float
```

Small fresh windows still use exact validation because region lookup can cost
more than it saves. Larger windows use region rendezvous, with optional exact
fallback if a candidate floor is configured. If a fresh vector has not published
graph-region metadata yet, M2 exact-scores that vector instead of using a
representative-point fallback.

The hot intersection path is layered:

```text
primary path: region posting table, no full fresh scan
fallback:     small-set bit test over inline region bitsets
SIMD path:    AVX2 word-wise bitset test when sets are dense enough
```

Default knobs:

```text
ANNCHOR_M2_REGION_CAP=512
ANNCHOR_M2_REGION_OVERLAP=1        # 2 when dim > 512
ANNCHOR_M2_EDGE_ANCHOR_K=2
ANNCHOR_M2_REGION_MAX unset        # adaptive: min 4096, capped at 65536
ANNCHOR_M2_POSTING_REGION_LIMIT=65536
```

Correctness contract:

```text
region hit -> candidate only
exact distance remains final
region miss -> allowed only under bounded-FN policy or covered by exact fallback
```
