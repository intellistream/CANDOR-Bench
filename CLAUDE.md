# CLAUDE.md — Project context for future Claude sessions

This file is the entry-point for Claude (or any new contributor) working on
the GammaFresh experiments in this repository. Read this first.

## What this project is

We're benchmarking **GammaFresh** — a hybrid ANN index that combines an
IVF-style buffer with an HNSW-style graph backend, with explicit
caller-driven maintenance. The goal is to support **streaming workloads
with concurrent insert / delete / query** while maintaining high recall
and low query latency.

Target paper: SIGMOD/VLDB 2026 (or EDBT/CIKM as fallback).

## Repo layout

```
CANDOR-Bench/
├── algorithms_impl/
│   ├── GammaFresh/         # Submodule (DynaGraph repo, our index impl)
│   │   ├── src/index/gamma_fresh/    # C++ source
│   │   ├── build/                    # CMake build artifacts
│   │   └── python/                   # Python bindings
│   ├── faiss/              # Faiss source (used as backend)
│   └── ...                 # Other algorithm impls (DiskANN, SPTAG, etc.)
│
├── bench/
│   └── algorithms/         # Pre-existing algorithm wrappers (less used now)
│
├── experiments/            # 🌟 OUR WORK LIVES HERE 🌟
│   ├── _shared/            # Shared utilities (data loaders, builders, gamma_py)
│   ├── e01_..._/run.py     # Each experiment is one folder
│   ├── INTENTS.md          # What each experiment proves
│   └── ALL_RESULTS_TABLE.md  # Latest aggregated results
│
├── datasets/               # SIFT/MSong/GloVe/Random loaders
├── raw_data/              # Downloaded datasets (~3GB)
├── setup_new_server.sh    # One-shot setup for fresh machine
└── CLAUDE.md              # This file
```

## Current state of the thesis

**v1 thesis (now refined):** "Gamma's middle-routing architecture provides
graceful degradation under churn vs HNSW."

**v2 thesis (after hnswlib comparison):** This held only against Faiss HNSW
(which has bad tombstone handling). hnswlib's mark_deleted is much better
and beats gamma in most scenarios.

**v3 thesis (CURRENT, after e14/e15):** "Gamma's hybrid architecture
provides **robustness across delete patterns**. hnswlib handles sequential
(FIFO/LRU) delete well but degrades on random / cluster / bursty patterns,
where gamma wins by 26-47% in time at same recall."

This is the defensible claim now.

## Key technical knowledge

### Backend matters HUGELY
- **Faiss HNSW**: 188µs/insert, naive tombstone (filter-after-search), recall
  collapses under churn
- **hnswlib**: 16µs/insert (12× faster than Faiss HNSW), filter-during-search
  tombstones, recall preserved under sequential churn

When comparing gamma vs algorithm X, **MUST swap gamma's backend to also be
X** for fair comparison. e13/e14/e15 use this methodology via
`experiments/_shared/gamma_py_v2.py` (Python POC of gamma + arbitrary backend).

### Delete pattern matters
- **sequential / FIFO / LRU**: hnswlib's best case, gamma loses
- **random uniform**: gamma wins (-27%) — most realistic for many workloads
- **cluster (hot-spot)**: gamma wins big (-47%) — real for category-takedown,
  user account churn
- **partial_reset (periodic burst)**: gamma wins (-28%) — TTL cleanup pattern

### Critical commits (in chronological order)
1. `718e8c4` (DynaGraph) — `perf(maint): rebuild graph BEFORE migration`. Fixed
   tombstone-causing-2.4×-overhead in maint. -46% on insdel_stream/1M.
2. `b7be797` (DynaGraph) — `perf(maint): skip redundant graph.contains in migration`.
   -6% on insert_only.
3. e14/e15 in CANDOR-Bench — discovered hnswlib's weakness on non-sequential
   delete patterns.

## Common pitfalls (we hit these)

| Pitfall | Solution |
|---|---|
| `import torch` fails build via CUDA detection | Use `torch==2.5.1` not 2.11+ |
| `nvcc` not found, CUDA build fails | Set `CMAKE_CUDA_ARCHITECTURES=70` or skip CUDA |
| `diskannpy` no Python 3.12 wheel | Use hnswlib instead |
| Datasets download to wrong subdirectory | `cp raw_data/glove/Glove/* raw_data/glove/` |
| Glove file count mismatch (1192514 vs 1183514) | `ln -sf data_1192514_100 raw_data/glove/data_1183514_100` |
| `_shared` not importable | Use `sys.path.insert(0, '<repo>/experiments')` |

## How to set up on a new machine

```bash
git clone --recursive -b feiyu/gamma-bench-fixes git@github.com:intellistream/CANDOR-Bench.git
cd CANDOR-Bench
bash setup_new_server.sh         # Full setup (~30 min)
# OR
bash setup_new_server.sh --skip-system  # If you already have build tools
```

After setup, smoke test:
```bash
OMP_NUM_THREADS=1 uv run python experiments/e15_delete_pattern_matrix/run.py --scale 200K
```

## How to read the experiments

1. Start with `experiments/INTENTS.md` — overview of all 15 experiments
2. Pick a specific experiment, read `eXX/README.md` for setup detail
3. Check `eXX/output*.json` for raw data
4. Some have `eXX/RESULTS.md` for write-up

For the BIG picture: read `experiments/HONEST_ASSESSMENT.md` (what we
discovered as the thesis evolved).

## Branch strategy

Two repos, two branches each:

| Repo | Branch | Purpose |
|---|---|---|
| **CANDOR-Bench** | `feiyu/gamma-bench-fixes` | All experiments + bench infra |
| **DynaGraph** (submodule) | `feiyu/gammafresh-fixes-and-explicit-maint` | Gamma C++ improvements |

Both are pushed to `intellistream/...` — don't overwrite original author's branches.

## Single-thread by design

All benchmarks use `OMP_NUM_THREADS=1` for fair comparison. Multi-threaded
benchmarks would mix algorithm quality with concurrency efficiency.

## Datasets used

- **SIFT** (1M × 128d, image features, INRIA Texmex) — primary
- **MSong** (992K × 420d, music timbre) — high-dim secondary
- **Random-m** (100K × 128d, synthetic uniform) — sanity / hard case
- **GloVe** (1.18M × 100d, word embeddings) — alternative

Supported datasets verified working — see `experiments/_shared/data.py`.

## Conventions

- All experiments go in `experiments/eXX_descriptive_name/`
- Each has `README.md` (intent), `run.py` (script), `output*.json` (data)
- `RESULTS.md` is optional, write when finding is significant
- Use `_shared/builders.py` for index construction (one source of truth for configs)
- Use `_shared/data.py` for dataset loading (handles caching)

## Open work

- **e15 1M scale** running in background (~40-60 min)
- Need to test: extreme churn rate (95%+) — would gamma's lead grow?
- Need to test: ID re-insertion patterns
- Need to consider: multi-threaded benchmarks (out of scope but reviewer may ask)

## Contact / blame

Last major work: feiyu (intellistream) + this Claude session (May 2026).
See `git log` for detailed history.
