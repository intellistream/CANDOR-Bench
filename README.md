# CANDOR-Bench: Benchmarking In-Memory Continuous ANNS under Dynamic Open-World Streams [SIGMOD'2026]

CANDOR-Bench (Continuous Approximate Nearest Neighbor Search under Dynamic Open-woRld Streams) is a benchmarking framework for evaluating **in-memory ANN systems under dynamic, non-static workloads**.

This repository now brings together two closely related benchmark stacks:

- **CANDOR benchmark stack** — the original stream-oriented ANN evaluation framework
- **Concurrency benchmark stack** — focused on read/write concurrency experiments

---

> **🚨 Strong recommendation: use Docker**
>
> This project pulls together multiple ANN implementations and third-party dependencies. Local setup can be tedious and fragile. If you want the most reproducible path, use the provided Dockerfile.
>
> **Note:** building the image may take **15–30 minutes** depending on your machine and network.

---

## Table of Contents

- [Overview](#overview)
- [Track Overview](#track-overview)
- [Original CANDOR Benchmark Tracks](#original-candor-benchmark-tracks)
- [Datasets and Algorithms](#datasets-and-algorithms)
  - [Summary of Datasets](#summary-of-datasets)
  - [Summary of Algorithms](#summary-of-algorithms)
- [Quick Start](#quick-start)
- [Concurrency Track](#concurrency-track)
- [CANDOR Track Usage (big-ann-benchmarks)](#candy-track-usage-big-ann-benchmarks)
- [Output Layout](#output-layout)
- [Repository Map](#repository-map)
- [Common Pitfalls](#common-pitfalls)
- [Citation](#citation)
- [License](#license)

## Overview

CANDOR-Bench targets ANN workloads where **data keeps changing while queries keep arriving**.

It is designed for scenarios such as:

- continuous inserts while search is running
- read/write contention and lock-strategy trade-offs
- workload shifts such as `round_robin`, `chasing`, `peeking`, and `zipfian`
- concurrency-control features such as MVCC and search sharing

At a high level, the repository now has **two primary user-facing tracks**:

- **Track A: CANDOR Track** — core C++ / Python-oriented benchmark stack, with paper-facing scripts living under `big-ann-benchmarks/`
- **Track B: Concurrency Track** — concurrency-control benchmark runner and configs

## Track Overview

| Track | Purpose | Main entry |
| --- | --- | --- |
| CANDOR Track | Stream-oriented ANN indexing library + paper-facing benchmark scripts | `src/index/`, `include/index/`, `big-ann-benchmarks/scripts/` |
| Concurrency Track | Read/write concurrency, MVCC, search-sharing, workload-mode studies | `src/concurrency/`, `src/index/concurrent/`, `configs/` |
| big-ann-benchmarks (submodule) | Additional benchmark framework integration and original benchmark scripts | `big-ann-benchmarks/` |

**Notes**

- `big-ann-benchmarks/` is a git submodule and appears empty until submodules are initialized.
- Concurrency-track pieces by function: concurrent index engine in `src/index/concurrent/`, python harness in `src/concurrency/`, run configs in `configs/`, ground-truth tools in `tools/`, dataset scripts in `data/`.
- The original CANDOR concurrent prototype is retained as `LegacyConcurrentIndex` and is **not** the default path.

## Original CANDOR Benchmark Tracks

Besides the repo-level split above, the original CANDOR benchmark stack also contains multiple paper-facing evaluation tracks under `big-ann-benchmarks/`.

| Benchmark track | Purpose | Typical entry |
| --- | --- | --- |
| General | Main dynamic ANNS evaluation | `big-ann-benchmarks/scripts/run_general.sh` |
| Congestion | Read/write congestion under dynamic streams | `big-ann-benchmarks/scripts/run_congestion.sh` |
| Concurrent | Concurrent-track evaluation in the benchmark stack | `big-ann-benchmarks/scripts/run_concurrent.sh` |
| OOD | Out-of-distribution evaluation | `big-ann-benchmarks/scripts/run_ood.sh` |
| Sparse | Sparse-vector benchmark | `big-ann-benchmarks/scripts/run_sparse.sh` |
| Streaming | Streaming scenario evaluation | `big-ann-benchmarks/scripts/run_streaming.sh` |

Under `big-ann-benchmarks/neurips23/`, you will also see benchmark assets for tracks such as `concurrent/`, `congestion/`, `filter/`, `ood/`, `sparse/`, and `streaming/`, together with the corresponding runbooks.

## Datasets and Algorithms

CANDOR-Bench evaluates a mix of real-world and synthetic datasets, together with a broad set of ANN baselines and dynamic variants.

### Summary of Datasets

| Category | Name | Description | Dimension | Data Size | Query Size | Code Identifier |
|:--------:|:----:|:-----------:|:---------:|:---------:|:----------:|:---------------:|
| **Real-world** | SIFT | Image | 128 | 1M | 10K | `sift` |
|  | OpenImagesStreaming | Image | 512 | 1M | 10K | — |
|  | Sun | Image | 512 | 79K | 200 | `sun` |
|  | SIFT100M | Image | 128 | 100M | 10K | `sift100M` |
|  | Msong | Audio | 420 | 990K | 200 | `msong` |
|  | COCO | Multi-Modal | 768 | 100K | 500 | `coco` |
|  | Glove | Text | 100 | 1.192M | 200 | `glove` |
|  | MSTuring | Text | 100 | 30M | 10K | `msturing` |
| **Synthetic** | Gaussian | i.i.d values | Adjustable | 500K | 1000 | — |
|  | Blob | Gaussian Blobs | 768 | 500K | 1000 | — |
|  | WTE | Text | 768 | 100K | 100 | — |
|  | FreewayML | Constructed | 128 | 100K | 1K | — |

### Summary of Algorithms

| Category | Algorithm | Description | Code Identifier |
|:--------:|:---------:|:------------|:---------------:|
| **Tree-based** | SPTAG | Space-partitioning tree structure for efficient data segmentation | `candy_sptag` |
| **LSH-based** | LSH | Data-independent hashing for approximate nearest neighbors | `faiss_lsh` |
|  | LSHAPG | LSH-driven optimization using LSB-Tree to differentiate graph regions | `candy_lshapg` |
|  | PLSH | Parallel LSH optimized for high-throughput similarity search on data streams | `plsh` |
| **Clustering-based** | PQ | Product quantization for efficient clustering into compact subspaces | `faiss_pq` |
|  | IVFPQ | Inverted index with product quantization for hierarchical clustering | `faiss_IVFPQ` |
|  | OnlinePQ | Incremental updates of centroids in product quantization for streaming data | `faiss_onlinepq` |
|  | Puck | Non-orthogonal inverted indexes with multiple quantization | `puck` |
|  | SCANN | Small-bit quantization to improve register utilization | `faiss_fast_scan` |
| **Graph-based** | NSW | Navigable Small World graph for fast nearest neighbor search | `faiss_NSW` |
|  | HNSW | Hierarchical Navigable Small World for scalable search | `faiss_HNSW` |
|  | FreshDiskANN | Streaming graph construction with refined robust edge pruning | `diskann` |
|  | MNRU | Enhances HNSW with efficient updates to prevent unreachable points | `candy_mnru` |
|  | Cufe | Enhances FreshDiskANN with batched neighbor expansion | `cufe` |
|  | Pyanns | Enhances FreshDiskANN with fix-sized huge pages | `pyanns` |
|  | IPDiskANN | Efficient in-place deletions for FreshDiskANN | `ipdiskann` |
|  | GTI | Hybrid tree-graph indexing for dynamic high-dimensional search | `gti` |
|  | PARLAY_HNSW | Parallel, deterministic HNSW for improved scalability | `parlay_hnsw` |
|  | PARLAY_VAMANA | Parallel, deterministic Vamana-based graph construction | `parlay_vamana` |

## Quick Start

### 1. Initialize submodules

```bash
git submodule update --init --recursive
```

### 2. Choose a build path

#### Option A: Docker (recommended)

```bash
docker build -t candor-bench .
docker run -it candor-bench
```

#### Option B: Local CPU build helper

```bash
bash build_cpu_only.sh
```

#### Option C: Local CUDA build helper

```bash
bash build_with_cuda.sh
```

### 3. Choose a track

#### Track A: CANDOR Track

Build the core library / Python bindings:

```bash
# CPU
bash build_cpu_only.sh

# or CUDA
bash build_with_cuda.sh
```

#### Track B: Concurrency Track

Build the native module (needs pybind11: `pip install pybind11`), then run
a config:

```bash
cd src/index/concurrent
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
make -j$(nproc) concurrency_native

cd ../../../..
PYTHONPATH=src python3 -m concurrency.runner --config configs/round_robin_baseline_sift.yaml
```

## Concurrency Track

Architecture notes: [docs/architecture.md](docs/architecture.md). Usage guides: [docs/concurrent_benchmark.md](docs/concurrent_benchmark.md) and
[docs/incremental_groundtruth.md](docs/incremental_groundtruth.md).

The track measures index behavior under concurrent inserts and searches:
rate-limited producers feed insert/search worker pools, searches run
against the committed snapshot, and recall is scored against incremental
ground truth captured per committed offset.

Layout:

- `src/index/concurrent/` — index backends (hnswlib, ParlayANN,
  DiskANN) plus the benchmark driver in `driver/`. The driver owns the
  whole hot path: producers, queues, token-bucket rate limiters, worker
  pools, latency capture. It runs entirely in C++ threads; the Python
  layer is not involved while a benchmark is running (the GIL is released
  for the whole call — `test/concurrency/test_gil_isolation.py` checks this).
- `src/index/concurrent/python/` — the `concurrency_native` module:
  index construction, build/insert/search, snapshot/restore, and
  `run_benchmark`, which executes a full concurrent run and returns stats
  plus per-query search records.
- `src/concurrency/*.py` — everything offline, in Python: config loading and
  variant expansion, recall evaluation, incremental ground truth, result
  files, and the sweep / compare / simulate pipelines.
- `configs/` — YAML configs. The format is unchanged from the
  previous releases; existing configs run as-is.
- `tools/` — ground-truth tooling: `compute_gt`
  (full-visibility ground truth), `compute_incr_gt` (per-offset
  incremental ground truth splits + index file), `calc_incr_recall` /
  `calc_recall` (recall from `.res` files), plus format debuggers.
  Build with `cmake -B build && cmake --build build` inside that
  directory (`-DENABLE_CUDA=OFF` for CPU-only); the dataset scripts in
  `data/` invoke these binaries.
- `data/` — dataset download / conversion / ground-truth scripts.
  Dataset payloads stay out of git: place them (or a symlink) at
  `data/data/`.

Run modes, all selected from the config file:

| Mode | Config switch | What it does |
| --- | --- | --- |
| benchmark | (default) | concurrent inserts + searches, stats CSV + `.res` files |
| throughput sweep | `throughput_sweep.enabled` | log-spaced search rates, one fresh run per step, latency curve CSV |
| compare | `compare.enabled` | run A records its search schedule, run B replays it at identical committed offsets (insert gating, shared snapshot baseline) — used for e.g. external-RW-lock cost |
| simulate | `compare.simulate.enabled` | sequential lag-timeline: a lead index and a snapshot-replaying lag index, with an optional partial-insert trial scored against incremental ground truth |

Tests live in `test/concurrency/`: one smoke test per index type, replay,
GIL isolation, ground-truth loader, runner end-to-end and
sweep/compare/simulate.

### Example configs

- `configs/round_robin_baseline_{sift,gist}.yaml`
- `configs/round_robin_search_sharing_{sift,gist}.yaml`
- `configs/chasing_warm_start_{sift,gist}.yaml`

### Config cheat sheet

Example config: `configs/round_robin_baseline_sift.yaml`

| Section | Key fields | Meaning |
| --- | --- | --- |
| `data` | `data_path`, `query_path`, `incr_query_path`, `overall_query_path`, `max_elements`, `begin_num` | Dataset locations and active data window |
| `index` | `index_type`, `m`, `ef_construction`, `alpha`, `visit_limit` | Index type and build/search structure parameters |
| `search` | `ef_search`, `recall_at`, `enable_search_sharing`, `enable_warm_start`, `use_node_lock` | Runtime search behavior |
| `workload` | `batch_size`, `num_threads`, `rate_groups(r/w)`, `query_mode` | Concurrency load pattern |
| `throughput_sweep` | `enabled`, `start_rate`, `end_rate`, `steps` | Auto-generated throughput / latency curve |
| `result` | `output_dir` | Result export root |

Current documented `index_type` values in this merged track:

- `hnsw`
- `vamana`
- `parlayhnsw`
- `parlayvamana`
- `segmented` (sealed-segment index; `index.sealed_type` picks the
  sealed backend, `index.seal_threshold` the segment size)

Notes: `parlayhnsw`/`parlayvamana` need explicit `search.beam_width` and
`search.visit_limit` (plus `index.level_m` for parlayhnsw); `hnsw` is the
only type supporting snapshots, so compare and simulate require it.
`workload.queue_size: 0` means an unbuffered handoff between producers
and workers, matching the previous runner. The hnswlib fork under
`src/index/concurrent/hnsw/hnswlib/` is vendored (not a submodule),
so the search-feature flags (`search.enable_s3`,
`enable_search_sharing`, `enable_path_skip`,
`enable_candidate_injection`) work out of the box on the `hnsw` type.

Current supported `query_mode` values in code:

- `round_robin`
- `chasing`
- `peeking`
- `zipfian`

## CANDOR Track Usage (big-ann-benchmarks)

All commands below are meant to be run inside `big-ann-benchmarks/` unless stated otherwise.

### Preparing datasets

```bash
cd big-ann-benchmarks
bash scripts/compute_general.sh
```

To create a small sample dataset:

```bash
python3 create_dataset.py --dataset random-xs
```

### Running algorithms on the congestion track

```bash
python3 run.py \
  --neurips23track congestion \
  --algorithm "$ALGO" \
  --nodocker \
  --rebuild \
  --runbook_path "$PATH" \
  --dataset "$DS"
```

- `$ALGO`: algorithm code identifier (see the Summary of Algorithms table)
- `$DS`: dataset name
- `$PATH`: path to the runbook YAML, e.g. `neurips23/runbooks/congestion/general_experiment/general_experiment.yaml`

### Computing ground truth for runbooks

```bash
python3 benchmark/congestion/compute_gt.py \
  --runbook "$PATH" \
  --dataset "$DS" \
  --gt_cmdline_tool ./thirdparty/DiskANN/build/apps/utils/compute_groundtruth
```

### Exporting results

```bash
chmod 777 -R results/
python3 data_export.py --out "$OUT" --track congestion
```

Common `--out` values include:

- `gen`
- `batch`
- `event`
- `conceptDrift`
- `randomContamination`
- `randomDrop`
- `wordContamination`
- `bulkDeletion`
- `batchDeletion`
- `multiModal`

## Output Layout

### Concurrency Track outputs

Concurrency-track runs write results under `result.output_dir` from the YAML config:

- summary CSV: `benchmark_results.csv` (one row per variant, appended)
- per-experiment files: `result.output_dir/<index>/<dataset>/<query>/files/*.res`
  (binary search records; simulate adds `*_incr_lag.res` / `*_incr_partial.res`)
- throughput sweep curve: `throughput_latency_curve.csv`
- compare summary: `compare_stats.csv`
- partial-insert recall diff: `<index>_partial_diff.csv`

Recall (overall and incremental) is computed by the runner itself and
lands in the summary CSV; the separate `.rc` recall-tool files are no
longer produced.

### CANDOR Track outputs

Legacy CANDOR scripts also generate outputs under locations such as:

- `figures/benchmark/`
- per-script `perfLists/` folders

## Repository Map

```text
CANDOR-Bench/
├── src/
│   ├── index/                      # all ANN index implementations
│   │   ├── ...                     # torch-based dynamic families (AbstractIndex)
│   │   └── concurrent/             # cc-capable families + native bench engine
│   ├── concurrency/                # concurrency benchmark harness (python package)
│   ├── data_loader/                # streaming data loaders
│   ├── utils/  cl/                 # shared utilities, OpenCL helpers
│   └── bindings/                   # PyCANDOR python bindings, torch adapters
├── docs/                           # usage guides
├── test/                           # all tests: index/ (C++ catch), concurrency/ (python)
├── figures/                        # paper figures (incl. figures/benchmark/)
├── configs/                        # benchmark YAML configs
├── data/                           # dataset prep scripts (payloads stay local)
├── tools/                          # ground-truth C++ tools
├── scripts/                        # frozen legacy experiment scripts (paper assets)
├── big-ann-benchmarks/             # Original benchmark framework (submodule)
│   ├── scripts/                    # Paper-facing scripts
│   └── neurips23/                  # Benchmark assets and runbooks
├── thirdparty/                     # external algorithm libraries
│   ├── DiskANN/  GTI/  IP-DiskANN/  PLSH/  SPTAG/  puck/  faiss/ ...
└── Dockerfile
```

## Common Pitfalls

- Datasets are not bundled. Config paths must point to real `.bin` files and ground-truth files on your machine.
- Many scripts assume Linux and native build tools such as `cmake`, `g++`, `go`, OpenMP, TBB, and Python packages.
- If `big-ann-benchmarks/` is empty, initialize submodules first.
- `big-ann-benchmarks/scripts` (CANDOR track) and `src/concurrency/` (concurrency track) belong to different tracks. Pick one track first and follow it end to end.

## Citation

```bibtex
@inproceedings{candorbench2026,
  title     = {CANDOR-Bench: Benchmarking In-Memory Continuous ANNS under Dynamic Open-World Streams},
  booktitle = {SIGMOD},
  year      = {2026}
}
```

## License

See [LICENSE](LICENSE).
