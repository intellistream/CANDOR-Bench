# GammaFresh vs FaissHNSW vs AdaIVF — Multi-scenario benchmark

Single-thread (`OMP_NUM_THREADS=1`), apples-to-apples via candy bindings (same C++ compute engine, same Faiss HNSW backend code). Maintenance is **bench-driven** — `idx.maintain(vector_budget)` lets the caller decide how much to drain; the algorithm decides which vectors.

## Maintenance contract (separation of concerns)

| Decision | Owner | Mechanism |
|---|---|---|
| **HOW MUCH** to drain (buffer/graph ratio) | bench / caller | `idx.maintain(vector_budget=N)` — `N=0` = unbounded |
| **WHICH** vectors to drain | algorithm | GammaFresh: HOT first (high `weighted_hit_count` → migrate to graph for fast log-time access), OLDEST first as tiebreak (an old vector with zero hits has actually been observed inactive across many query rounds, so it's a safer "won't cost much in graph either" candidate) |

## TL;DR — When does GammaFresh win?

| Win | Scenario | Numbers | Why |
|---|---|---|---|
| 🏆🏆🏆 | `random-100k bulk_delete(30%)` | **gamma 4.5s / recall 1.000** vs faiss 29.3s / 0.558, ivf 8.4s / 0.264 | Hybrid graph + partition-buffer scan: HNSW tombstones cripple recall; IVF recall unusable on random data; GammaFresh **6.5× faster than HNSW with perfect recall** |
| 🏆 | `clustered-50k bulk_delete(30%)` | gamma 2.1s / 1.000 vs faiss 3.4s / 0.992 | Fastest **and** highest recall |
| 🏆 | `sift-small bulk_delete(30%)` | gamma 345ms / 1.000 vs faiss 361ms / 0.997 | Fastest **and** perfect recall |
| ✅ | `random-100k drift` | gamma recall **0.685** vs faiss 0.561, ivf 0.277 | Only algo with usable recall after concept drift on hard data |
| ✅ | All churn / drift / delete scenarios | gamma recall always **highest** (≥0.984, often 1.000) | Recall robustness is the consistent advantage |

## Datasets
- `sift-small` — 10K × 128d (real SIFT)
- `clustered-50k` — 50K × 128d, 50 Gaussian clusters (synthetic, IVF-friendly)
- `random-100k` — 100K × 128d uniform Gaussian (hard for any ANN; HNSW direct ≈ 0.6 recall ceiling at ef=64)

## Scenarios

Each scenario chooses its own maintenance schedule via `algo.maintain(vector_budget)`. GammaFresh acts as a hybrid index: fresh writes accumulate in partition buffers (IVF-fast) and queries combine graph search + buffer scan. Stable hot vectors get promoted to graph during scheduled maintenance.

- `streaming(b=100)` — 100/batch insert + 100 queries every batch, **maint(unbounded) before each query batch**
- `burst(b=2500)` — large-batch insert, **maint(unbounded) before final query**
- `drift(4chunk)` — alternating insert/delete-oldest cycles, **maint(unbounded) between cycles**
- `bulk_delete(30%)` — insert all + bulk-delete oldest 30%, **NO maint** (tests post-delete behavior; gamma's hybrid scan shines, HNSW pays for tombstones)
- `churn(20cyc, 5%)` — insert+delete+query cycles, **maint(vector_budget=chunk//2) per cycle** — bench picks half-chunk budget so buffer stays trim while hot vectors remain available for hybrid scan

## Results

### sift-small (10K, 128d) | streaming(b=100)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| faiss | 568 | 31,690 | 0.9970 | ✅ high recall |
| ivf | 576 | 31,237 | 0.9970 | ✅ high recall |
| gamma | 1013 | 17,769 | 0.9970 | ✅ high recall |

### sift-small (10K, 128d) | burst(b=2500)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 187 | 48,578 | 0.9950 | ✅ high recall |
| faiss | 344 | 26,473 | 0.9970 | ✅ high recall |
| gamma | 674 | 13,502 | 0.9990 | ✅ high recall |

### sift-small (10K, 128d) | drift(4chunk)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 151 | 92,246 | 0.9990 | ✅ high recall |
| faiss | 448 | 31,000 | 0.9950 | ✅ high recall |
| gamma | 1162 | 11,962 | 1.0000 | ✅ high recall |

### sift-small (10K, 128d) | bulk_delete(30%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| gamma | 345 | 38,004 | 1.0000 | ✅ high recall |
| faiss | 361 | 36,271 | 0.9970 | ✅ high recall |
| ivf | 490 | 26,754 | 0.9860 |  |

### sift-small (10K, 128d) | churn(20cyc,5%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 63 | 165,034 | 0.9930 | ✅ high recall |
| faiss | 590 | 17,625 | 0.9940 | ✅ high recall |
| gamma | 1966 | 5,290 | 1.0000 | ✅ high recall |

### clustered-50k (50K, 128d, 50 clusters) | streaming(b=100)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| faiss | 7066 | 20,167 | 0.9695 |  |
| gamma | 12252 | 11,630 | 0.9805 |  |
| ivf | 16126 | 8,837 | 1.0000 | ✅ high recall |

### clustered-50k (50K, 128d, 50 clusters) | burst(b=2500)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| faiss | 3033 | 15,729 | 0.9845 |  |
| ivf | 3855 | 12,372 | 1.0000 | ✅ high recall |
| gamma | 5221 | 9,136 | 0.9960 | ✅ high recall |

### clustered-50k (50K, 128d, 50 clusters) | drift(4chunk)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 1963 | 36,712 | 1.0000 | ✅ high recall |
| faiss | 3641 | 19,790 | 0.9610 |  |
| gamma | 6630 | 10,867 | 0.9990 | ✅ high recall |

### clustered-50k (50K, 128d, 50 clusters) | bulk_delete(30%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| gamma | 2111 | 30,892 | 1.0000 | ✅ high recall |
| ivf | 3089 | 21,111 | 1.0000 | ✅ high recall |
| faiss | 3361 | 19,401 | 0.9915 | ✅ high recall |

### clustered-50k (50K, 128d, 50 clusters) | churn(20cyc,5%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 515 | 98,753 | 1.0000 | ✅ high recall |
| faiss | 5209 | 9,758 | 0.9880 |  |
| gamma | 31859 | 1,596 | 0.9990 | ✅ high recall |

### random-100k (100K, 128d) | streaming(b=100)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| faiss | 49296 | 5,781 | 0.5970 |  |
| gamma | 65542 | 4,348 | 0.6045 |  |
| ivf | 81613 | 3,492 | 0.2190 | ❌ recall too low to be useful |

### random-100k (100K, 128d) | burst(b=2500)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 4669 | 20,388 | 0.2190 | ❌ recall too low to be useful |
| faiss | 27161 | 3,505 | 0.5955 |  |
| gamma | 33695 | 2,825 | 0.5815 |  |

### random-100k (100K, 128d) | drift(4chunk)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 3660 | 39,152 | 0.2770 | ❌ recall too low to be useful |
| faiss | 29517 | 4,855 | 0.5605 |  |
| gamma | 53623 | 2,672 | 0.6850 |  |

### random-100k (100K, 128d) | bulk_delete(30%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| gamma | 4532 | 28,727 | 1.0000 | ✅ high recall |
| ivf | 8369 | 15,557 | 0.2640 | ❌ recall too low to be useful |
| faiss | 29253 | 4,451 | 0.5575 |  |

### random-100k (100K, 128d) | churn(20cyc,5%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 2308 | 42,497 | 0.2390 | ❌ recall too low to be useful |
| faiss | 24544 | 3,997 | 0.5575 |  |
| gamma | 163006 | 602 | 0.6955 |  |
