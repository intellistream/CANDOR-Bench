# CANDOR-Bench

Benchmarking in-memory continuous ANNS under dynamic open-world streams (SIGMOD 2026).

This repository merges two benchmark stacks:

- CANDY benchmark stack
- ccANN-Bench concurrency-control benchmark stack

## What This Project Does

CANDOR-Bench is a research benchmark suite for ANN systems where data and workloads are dynamic, not static.

It focuses on scenarios such as:

- continuous inserts while search is running
- read/write contention and lock strategy trade-offs
- workload shifts (for example round-robin, chasing, peeking, zipfian query modes)
- concurrency-control features such as MVCC and search sharing

This repo has two primary tracks:

- Track A: CANDY Track (core C++/Py bindings + legacy benchmark scripts)
- Track B: ccANN Track (ccANN-Bench integration for concurrency-control experiments)

## Track Overview

| Track | Purpose | Main entry |
| --- | --- | --- |
| CANDY Track | Stream-oriented ANN indexing library + legacy experiment scripts | `src/CANDY/`, `include/CANDY/`, `benchmark/scripts/` |
| ccANN Track | Read/write concurrency, MVCC, search-sharing, workload-mode studies | `src/CANDY/ConcurrentIndex/`, `include/CANDY/ConcurrentIndex/`, `benchmark/concurrent/`, `benchmark/configs/concurrent/` |
| big-ann-benchmarks (submodule) | Additional benchmark framework integration | `big-ann-benchmarks/` |

Note: `big-ann-benchmarks/` is a git submodule. It appears empty until submodules are initialized.
Fusion layout: ccANN algorithms are merged under `src/CANDY/ConcurrentIndex/`; runner stays in `benchmark/concurrent/`.
Legacy note: original CANDY concurrent prototype is retained as `LegacyConcurrentIndex` and not used in default build.

## Quick Start

### 1. Initialize submodules

```bash
git submodule update --init --recursive
```

### 2. Build options

Option A: Docker (full environment)

```bash
docker build -t candor-bench .
docker run -it candor-bench
```

Option B: Local CPU build helper

```bash
bash buildCPUOnly.sh
```

Option C: Local CUDA build helper

```bash
bash buildWithCuda.sh
```

### 3. Choose a track

Track A: CANDY Track (build core library / Python bindings)

```bash
# CPU
bash buildCPUOnly.sh

# or CUDA
bash buildWithCuda.sh
```

Track B: ccANN Track (merged ccANN-Bench runner)

```bash
cd benchmark/concurrent
./build.sh
./cc-bench -config ../configs/concurrent/dir_a_baseline_sift.yaml
```

## ccANN Track (Merged ccANN-Bench)

This track combines:

- C++ index backends in `src/CANDY/ConcurrentIndex/`
- a Go runner in `benchmark/concurrent`
- YAML sweep/variant configs in `benchmark/configs/concurrent`

### Config Cheat Sheet

Example config: `benchmark/configs/concurrent/dir_a_baseline_sift.yaml`

| Section | Key fields | Meaning |
| --- | --- | --- |
| `data` | `data_path`, `query_path`, `max_elements`, `begin_num` | Dataset locations and active data window |
| `index` | `index_type`, `m`, `ef_construction`, `alpha`, `visit_limit` | Index type and build/search structure params |
| `search` | `ef_search`, `recall_at`, `enable_mvcc`, `enable_search_sharing` | Runtime search behavior |
| `workload` | `batch_size`, `num_threads`, `rate_groups(r/w)`, `query_mode` | Concurrency load pattern |
| `throughput_sweep` | `enabled`, `start_rate`, `end_rate`, `steps` | Auto-generate throughput/latency curve |
| `result` | `output_dir` | Result export root |

Current supported `index_type` values in code:

- `hnsw`
- `vamana`
- `parlayhnsw`
- `parlayvamana`
- `segmented`

Current supported `query_mode` values in code:

- `round_robin`
- `chasing`
- `peeking`
- `zipfian`

## Output Layout

ccANN run outputs are written under `result.output_dir` (from YAML), including:

- summary CSV: `benchmark_results.csv`
- per-experiment files: `result.output_dir/<index>/<dataset>/<query>/files/*.res`
- optional recall files: `*.rc`
- optional throughput sweep curve: `throughput_latency_curve.csv`

Legacy CANDY scripts also generate outputs under `benchmark/figures/` and per-script `perfLists/` folders.

## Repository Map

```text
CANDOR-Bench/
├── benchmark/
│   ├── concurrent/            # Go runner (merged ccANN-Bench track)
│   ├── configs/concurrent/    # YAML configs for CC experiments
│   ├── scripts/               # CANDY legacy benchmark scripts and plotting utilities
│   └── figures/               # Generated/archived figures
├── src/
│   └── CANDY/
│       ├── ...                # CANDY algorithm implementations
│       ├── ConcurrentIndex/   # Merged ccANN algorithm implementations
│       └── LegacyConcurrentIndex.cpp  # Old prototype impl (not used by default)
├── include/
│   └── CANDY/
│       ├── ConcurrentIndex/   # Merged ccANN index interfaces
│       └── LegacyConcurrentIndex.h # Old prototype header
├── thirdparty/                # Third-party dependencies
├── GTI/                       # Submodule
├── DiskANN/                   # Submodule
├── IP-DiskANN/                # Submodule
├── PLSH/                      # Submodule
├── big-ann-benchmarks/        # Submodule (init required)
└── Dockerfile
```

## Common Pitfalls

- Datasets are not bundled. Paths in YAML configs must point to real `.bin`/ground-truth files on your machine.
- Many scripts assume Linux and native build tools (`cmake`, `g++`, `go`, OpenMP/TBB, Python packages).
- If `big-ann-benchmarks/` is empty, run submodule initialization first.
- `benchmark/scripts` and `benchmark/concurrent` belong to different tracks. Pick one track first, then follow that track end-to-end.

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
