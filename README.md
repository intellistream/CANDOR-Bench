# CANDOR-Bench CC Track

A benchmark harness for **concurrent** approximate-nearest-neighbor (ANN) vector
indexes. It drives mixed insert + search workloads against graph-based ANN
indexes and measures **throughput, recall, and tail latency (P50/P99) under
concurrency**, with a focus on comparing different concurrency-control
strategies (per-node locking, MVCC, coarse RW-locking, and sealed-segment
isolation).

The harness is a config-driven Go program; the index implementations are C++
and are linked into the harness through cgo.

> Research code. Interfaces, configs, and algorithm variants change frequently.

## Features

- Mixed read/write workloads with independent **insert** and **search** event
  rates (open-loop arrival via `rate_groups(r/w)`), optional burst schedules.
- Per-query latency capture with **P50/P99** tail reporting.
- **Overall** recall (against a static ground truth) and **incremental** recall
  (against a streaming ground truth that tracks inserts).
- Multiple query distributions: `round_robin`, `chasing`, `peeking`, `zipfian`.
- Built-in **parameter sweeps**: any YAML scalar written as a list expands into
  the cartesian product of variants, so one config file can fan out a whole
  matrix (thread counts, ef, lock mode, rate groups, …).
- Concurrency-control comparison: per-node locks (`use_node_lock`), MVCC
  visibility (`enable_mvcc`), external RW lock (`with_external_rw_lock`), and
  segmented/sealed indexes (`index_type: segmented`).

## Repository layout

```
.
├── bench/                     # Go benchmark harness (module ANN-CC-bench/bench)
│   ├── main.go                # entrypoint: ./bench -config <file.yaml>
│   ├── internal/              # workload driver, config, recall, output
│   ├── algorithms/            # C++ index implementations (linked via cgo)
│   │   ├── index_cgo.{cpp,hpp}, index.hpp   # cgo bridge
│   │   ├── hnsw/              # HNSW (hnswlib submodule)
│   │   ├── parlayann/         # ParlayANN submodule (HNSW / Vamana)
│   │   ├── vamana/            # DiskANN/Vamana submodule
│   │   └── annchor*/          # ANNchor and its variants (m1/m2/m3, trim, preempt, …)
│   ├── cmd/                   # auxiliary Go commands
│   ├── scripts/              # Python analysis scripts
│   ├── configs/, configs_v2/ # example workload configs
│   └── run_*.sh              # experiment driver scripts
├── utils/                     # C++/CUDA recall & ground-truth tools
│                              #   (calc_recall, calc_incr_recall, compute_gt, …)
├── data/                      # dataset prep scripts only (datasets NOT included)
└── REPRODUCE.md
```

## Supported indexes

Set `index.index_type` in the config to one of:

| `index_type`        | Notes |
|---------------------|-------|
| `hnsw`              | HNSW (hnswlib); baseline for the ANNchor variants |
| `hnsw-visible`      | HNSW with visibility-mode filtering |
| `parlayhnsw`        | ParlayANN HNSW |
| `parlayvamana`      | ParlayANN Vamana |
| `vamana`            | Vamana / DiskANN |
| `annchor`           | ANNchor (MVCC / undo-recovery / S3 computation-sharing knobs) |
| `annchor-dev`, `annchor-m1`, `annchor-m2`, `annchor-m3` | ANNchor research variants |
| `annchor-trim`, `annchor-preempt` | trim-recovery / search-preemption variants |
| `segmented`         | Sealed-segment isolation over a base index (`sealed_type`) |

## Requirements

- Linux, x86-64
- Go **1.24+**
- A C++17 compiler (GCC 9+/Clang 10+), CMake **3.16+**
- OpenMP; BLAS (e.g. OpenBLAS) for some algorithms
- Python 3.8+ (only for dataset preparation / analysis)

## Build

```bash
# 1. Fetch the third-party index libraries (hnswlib, ParlayANN, DiskANN)
git submodule update --init --recursive

# 2. Build the recall / ground-truth tools used by the harness
#    (the harness invokes ../utils/build/calc_recall and calc_incr_recall)
cmake -S utils -B utils/build -DCMAKE_BUILD_TYPE=Release
cmake --build utils/build -j

# 3. Build the harness (cgo compiles/links the C++ index under bench/algorithms)
cd bench
go build -o bench .
```

## Datasets

Datasets are **not** included. The harness reads raw vectors in a flat binary
`.bin` layout and standard `.fvecs/.ivecs` files; ground truth is a `.gt`/`.txt`
neighbor list. Use the scripts in `data/` to download and convert public ANN
datasets (SIFT, GIST, Deep, MSong, …) and to compute ground truth:

```bash
python data/download.py          # fetch a dataset
python data/fvecs_to_bin.py ...  # convert .fvecs -> .bin
python data/compute_gt_faiss.py ...        # overall ground truth
python data/compute_incr_gt_stream.py ...  # incremental (streaming) ground truth
```

See `data/gen.md` for the dataset/ground-truth generation notes. Configs expect
datasets under `../data/<name>/` relative to `bench/`.

## Configuration

A config is a YAML file with `data`, `index`, `search`, `workload`, and
`result` sections. Minimal example (`bench/config_sigmod_hnsw_sift_ch_full.yaml`):

```yaml
data:
  dataset_name: sift
  max_elements: 1000000
  begin_num: 50000                     # elements pre-built before the workload starts
  data_type: float
  data_path: ../data/sift/sift_base.bin
  incr_query_path: ../data/sift/sift_query_stream.bin
  incr_gt_path: ../data/sift/sift_stream_i20_offset_index.txt
  overall_query_path: ../data/sift/sift_query.bin
  overall_gt_path: ../data/sift/sift_base.gt20

index:
  index_type: hnsw
  m: 16
  ef_construction: 200

search:
  recall_at: 10
  ef_search: 35
  use_node_lock: [true, false]         # list -> sweep both lock modes

workload:
  batch_size: 20
  num_threads: [8, 16, 32, 64]         # list -> sweep
  query_mode: chasing
  rate_groups(r/w):                    # each pair -> one (search, insert) rate variant
    - [20000, 20000]                   # balanced
    - [4000, 36000]                    # read-heavy
    - [36000, 4000]                    # write-heavy

result:
  output_dir: ./result/run1
```

Any scalar written as a YAML list (e.g. `num_threads`, `ef_search`,
`use_node_lock`) is swept; the run produces one labelled variant per
combination. `rate_groups(r/w)` expands into per-rate variants.

## Running

```bash
cd bench
./bench -config configs/<your-config>.yaml
```

Per-variant results (throughput, recall, latency percentiles) are written under
the config's `result.output_dir`. The `run_*.sh` scripts wrap common experiment
matrices.

## License

No license file is currently included; add one before any public release.
