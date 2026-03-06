# CANDOR-Bench: Benchmarking In-Memory Continuous ANNS under Dynamic Open-World Streams [SIGMOD'2026]

CANDOR-Bench (Continuous Approximate Nearest neighbor search under Dynamic Open-woRld Streams) is a benchmarking framework designed to evaluate in-memory ANNS algorithms under realistic, dynamic data stream conditions.

This repository merges two benchmark stacks:

- **CANDY benchmark stack** — the original stream-oriented ANN evaluation framework
- **ccANN-Bench concurrency-control benchmark stack** — read/write concurrency experiments

---

> **🚨🚨 Strong Recommendation: Use Docker! 🚨🚨**
>
> We strongly recommend using Docker to build and run this project.
>
> There are many algorithm libraries with complex dependencies. Setting up the environment locally can be difficult and error-prone.
> Docker provides a consistent and reproducible environment, saving you time and avoiding compatibility issues.
>
> **Note:** Building the Docker image may take **15–30 minutes** depending on your network and hardware, please be patient.

---

## What This Project Does

CANDOR-Bench is a research benchmark suite for ANN systems where data and workloads are dynamic, not static.

It focuses on scenarios such as:

- continuous inserts while search is running
- read/write contention and lock strategy trade-offs
- workload shifts (for example round-robin, chasing, peeking, zipfian query modes)
- concurrency-control features such as MVCC and search sharing

This repo has two primary tracks:

- Track A: CANDY Track (core C++/Py bindings + benchmark scripts via big-ann-benchmarks)
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

Our evaluation involves the following datasets and algorithms.

### Summary of Datasets

| Category | Name | Description | Dimension | Data Size | Query Size | Code Identifier |
|:--------:|:----:|:-----------:|:---------:|:---------:|:----------:|:---------------:|
| **Real-world** | SIFT | Image | 128 | 1M | 10K | `sift` |
| | OpenImagesStreaming | Image | 512 | 1M | 10K | — |
| | Sun | Image | 512 | 79K | 200 | `sun` |
| | SIFT100M | Image | 128 | 100M | 10K | `sift100M` |
| | Msong | Audio | 420 | 990K | 200 | `msong` |
| | COCO | Multi-Modal | 768 | 100K | 500 | `coco` |
| | Glove | Text | 100 | 1.192M | 200 | `glove` |
| | MSTuring | Text | 100 | 30M | 10K | `msturing` |
| **Synthetic** | Gaussian | i.i.d values | Adjustable | 500K | 1000 | — |
| | Blob | Gaussian Blobs | 768 | 500K | 1000 | — |
| | WTE | Text | 768 | 100K | 100 | — |
| | FreewayML | Constructed | 128 | 100K | 1K | — |

### Summary of Algorithms

| Category | Algorithm | Description | Code Identifier |
|:--------:|:---------:|:------------|:---------------:|
| **Tree-based** | SPTAG | Space-partitioning tree structure for efficient data segmentation | `candy_sptag` |
| **LSH-based** | LSH | Data-independent hashing for approximate nearest neighbors | `faiss_lsh` |
| | LSHAPG | LSH-driven optimization using LSB-Tree to differentiate graph regions | `candy_lshapg` |
| | PLSH | Parallel LSH optimized for high-throughput similarity search on data streams | `plsh` |
| **Clustering-based** | PQ | Product quantization for efficient clustering into compact subspaces | `faiss_pq` |
| | IVFPQ | Inverted index with product quantization for hierarchical clustering | `faiss_IVFPQ` |
| | OnlinePQ | Incremental updates of centroids in product quantization for streaming data | `faiss_onlinepq` |
| | Puck | Non-orthogonal inverted indexes with multiple quantization | `puck` |
| | SCANN | Small-bit quantization to improve register utilization | `faiss_fast_scan` |
| **Graph-based** | NSW | Navigable Small World graph for fast nearest neighbor search | `faiss_NSW` |
| | HNSW | Hierarchical Navigable Small World for scalable search | `faiss_HNSW` |
| | FreshDiskANN | Streaming graph construction with refined robust edge pruning | `diskann` |
| | MNRU | Enhances HNSW with efficient updates to prevent unreachable points | `candy_mnru` |
| | Cufe | Enhances FreshDiskANN with batched neighbor expansion | `cufe` |
| | Pyanns | Enhances FreshDiskANN with fix-sized huge pages | `pyanns` |
| | IPDiskANN | Efficient in-place deletions for FreshDiskANN | `ipdiskann` |
| | GTI | Hybrid tree-graph indexing for dynamic high-dimensional search | `gti` |
| | PARLAY_HNSW | Parallel, deterministic HNSW for improved scalability | `parlay_hnsw` |
| | PARLAY_VAMANA | Parallel, deterministic Vamana-based graph construction | `parlay_vamana` |

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
./cc-bench -config ../configs/concurrent/round_robin_baseline_sift.yaml
```

## ccANN Track (Merged ccANN-Bench)

This track combines:

- C++ index backends in `src/CANDY/ConcurrentIndex/`
- a Go runner in `benchmark/concurrent`
- YAML sweep/variant configs in `benchmark/configs/concurrent`

### Config Cheat Sheet

Example config: `benchmark/configs/concurrent/round_robin_baseline_sift.yaml`

Included example configs:

- `benchmark/configs/concurrent/round_robin_baseline_{sift,gist}.yaml`
- `benchmark/configs/concurrent/round_robin_search_sharing_{sift,gist}.yaml`
- `benchmark/configs/concurrent/chasing_warm_start_{sift,gist}.yaml`

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

## CANDY Track Usage (big-ann-benchmarks)

All the following operations are performed inside the `big-ann-benchmarks/` directory.

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

- `$ALGO`: Algorithm code identifier (see Summary of Algorithms table)
- `$DS`: Dataset name
- `$PATH`: Path to runbook YAML, e.g. `neurips23/runbooks/congestion/general_experiment/general_experiment.yaml`

### Computing ground truth for runbooks

```bash
python3 benchmark/congestion/compute_gt.py \
  --runbook "$PATH" \
  --dataset "$DS" \
  --gt_cmdline_tool ./DiskANN/build/apps/utils/compute_groundtruth
```

### Exporting results

```bash
chmod 777 -R results/
python3 data_export.py --out "$OUT" --track congestion
```

Common `--out` values: `gen`, `batch`, `event`, `conceptDrift`, `randomContamination`, `randomDrop`, `wordContamination`, `bulkDeletion`, `batchDeletion`, `multiModal`, etc.

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
