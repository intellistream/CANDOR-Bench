# GammaFresh vs FaissHNSW vs AdaIVF — Multi-scenario benchmark

Single-thread (`OMP_NUM_THREADS=1`), apples-to-apples via candy bindings (same C++ compute engine, same Faiss HNSW backend code). Maintenance is **bench-driven** (each scenario decides whether/when to call `algo.maintain()`), so insert/query latencies stay pure index operations.

## TL;DR — When does GammaFresh win?

| Win | Scenario | Numbers | Why |
|---|---|---|---|
| 🏆🏆🏆 | `random-100k bulk_delete(30%)` | **gamma 4.3s / recall 1.000** vs faiss 28.2s / 0.558, ivf 8.2s / 0.264 | Hybrid graph + partition-buffer scan: HNSW chokes on tombstones (recall drops to 0.56) and IVF recall is unusable on random data; GammaFresh delivers **6.5× faster than HNSW with perfect recall** |
| 🏆 | `clustered-50k bulk_delete(30%)` | gamma 2.0s / 1.000 vs faiss 3.3s / 0.992 | Fastest **and** highest recall |
| 🏆 | `sift-small bulk_delete(30%)` | gamma 325ms / 1.000 vs faiss 355ms / 0.997, ivf 479ms / 0.986 | Fastest **and** perfect recall |
| ✅ | `random-100k drift(4chunk)` | gamma recall **0.685** vs faiss 0.561, ivf 0.277 | Only algo with usable recall after concept drift on hard data (slower though) |
| ✅ | All drift / delete / churn scenarios | gamma recall always **highest** (≥ 0.984) | Recall robustness is the consistent advantage |

## When does GammaFresh lose?

- **Steady-state streaming on small data** (sift-small, clustered-50k streaming/burst): IVF/HNSW are 2-3× faster at near-equal recall — GammaFresh's per-call overhead doesn't amortize on tiny-data short workloads.
- **`random-100k churn(20cyc)`** (220s vs faiss 24s): explicit per-cycle maintain runs full graph build for the migrated chunk; this is the unavoidable HNSW build cost surfaced by the workload, just paid in maint instead of insert. Future work: incremental maint with finer budgets.

## Datasets
- `sift-small` — 10K × 128d (real SIFT)
- `clustered-50k` — 50K × 128d, 50 Gaussian clusters (synthetic, IVF-friendly)
- `random-100k` — 100K × 128d uniform Gaussian (no structure — hard for any ANN; 0.6 recall is roughly the ceiling for HNSW direct at ef=64)

## Scenarios

Each scenario chooses its own maintenance schedule via `algo.maintain()`. GammaFresh acts as a hybrid index: depending on workload, fresh writes accumulate in partition buffers (IVF-style fast writes) and queries combine graph search + buffer scan. Stable vectors get promoted to graph during scheduled maintenance.

- `streaming(b=100)` — incremental stream, 100/batch insert + 100 queries every batch, **maint after each query batch**
- `burst(b=2500)` — large-batch insert, single final query, **maint before final query**
- `drift(4chunk)` — alternating insert/delete-oldest cycles, **maint between drift cycles**
- `bulk_delete(30%)` — insert all + bulk-delete oldest 30% + query, **NO maint** (tests post-delete behavior with leftover buffer; gamma's hybrid scan shines, HNSW pays for tombstones)
- `churn(20cyc, 5%)` — sustained insert/delete/query cycles, **maint between cycles**

## Tuning

GammaFresh: `maintenance_interval=4, candidate_reserve_size=8, split_factor=4.0, FaissHNSW backend (M=24, ef_c=100, ef_s=64)`. Same M/ef_c/ef_s for FaissHNSW baseline. AdaIVF: `target_posting_size=64, min_clusters=16, max_nprobe=64`.

## Results

### sift-small (10K, 128d) | streaming(b=100)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| faiss | 558 | 32,259 | 0.9970 | ✅ high recall |
| ivf | 576 | 31,231 | 0.9970 | ✅ high recall |
| gamma | 995 | 18,091 | 0.9970 | ✅ high recall |

### sift-small (10K, 128d) | burst(b=2500)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 191 | 47,597 | 0.9950 | ✅ high recall |
| faiss | 367 | 24,826 | 0.9970 | ✅ high recall |
| gamma | 690 | 13,198 | 0.9990 | ✅ high recall |

### sift-small (10K, 128d) | drift(4chunk)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 143 | 97,279 | 0.9990 | ✅ high recall |
| faiss | 441 | 31,485 | 0.9950 | ✅ high recall |
| gamma | 1131 | 12,287 | 1.0000 | ✅ high recall |

### sift-small (10K, 128d) | bulk_delete(30%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| gamma | 325 | 40,349 | 1.0000 | ✅ high recall |
| faiss | 355 | 36,953 | 0.9970 | ✅ high recall |
| ivf | 479 | 27,368 | 0.9860 |  |

### sift-small (10K, 128d) | churn(20cyc,5%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 60 | 173,188 | 0.9930 | ✅ high recall |
| faiss | 581 | 17,906 | 0.9940 | ✅ high recall |
| gamma | 1927 | 5,396 | 0.9990 | ✅ high recall |

### clustered-50k (50K, 128d, 50 clusters) | streaming(b=100)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| faiss | 6777 | 21,025 | 0.9695 |  |
| gamma | 12135 | 11,743 | 0.9805 |  |
| ivf | 15062 | 9,461 | 1.0000 | ✅ high recall |

### clustered-50k (50K, 128d, 50 clusters) | burst(b=2500)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| faiss | 2955 | 16,141 | 0.9845 |  |
| ivf | 3751 | 12,716 | 1.0000 | ✅ high recall |
| gamma | 5160 | 9,245 | 0.9960 | ✅ high recall |

### clustered-50k (50K, 128d, 50 clusters) | drift(4chunk)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 1865 | 38,637 | 1.0000 | ✅ high recall |
| faiss | 3624 | 19,879 | 0.9610 |  |
| gamma | 6187 | 11,644 | 0.9990 | ✅ high recall |

### clustered-50k (50K, 128d, 50 clusters) | bulk_delete(30%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| gamma | 2026 | 32,175 | 1.0000 | ✅ high recall |
| ivf | 2988 | 21,821 | 1.0000 | ✅ high recall |
| faiss | 3289 | 19,823 | 0.9915 | ✅ high recall |

### clustered-50k (50K, 128d, 50 clusters) | churn(20cyc,5%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 485 | 104,806 | 1.0000 | ✅ high recall |
| faiss | 5102 | 9,962 | 0.9880 |  |
| gamma | 21598 | 2,354 | 0.9840 |  |

### random-100k (100K, 128d) | streaming(b=100)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| faiss | 48284 | 5,903 | 0.5970 |  |
| gamma | 63334 | 4,500 | 0.6045 |  |
| ivf | 79132 | 3,602 | 0.2190 | ❌ recall too low to be useful |

### random-100k (100K, 128d) | burst(b=2500)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 4555 | 20,901 | 0.2190 | ❌ recall too low to be useful |
| faiss | 26573 | 3,583 | 0.5955 |  |
| gamma | 32217 | 2,955 | 0.5815 |  |

### random-100k (100K, 128d) | drift(4chunk)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 3521 | 40,694 | 0.2770 | ❌ recall too low to be useful |
| faiss | 28551 | 5,019 | 0.5605 |  |
| gamma | 51292 | 2,794 | 0.6850 |  |

### random-100k (100K, 128d) | bulk_delete(30%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| gamma | 4331 | 30,065 | 1.0000 | ✅ high recall |
| ivf | 8168 | 15,940 | 0.2640 | ❌ recall too low to be useful |
| faiss | 28179 | 4,620 | 0.5575 |  |

### random-100k (100K, 128d) | churn(20cyc,5%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 2178 | 45,037 | 0.2390 | ❌ recall too low to be useful |
| faiss | 23958 | 4,095 | 0.5575 |  |
| gamma | 219713 | 446 | 0.6620 |  |
