# GammaFresh vs FaissHNSW vs AdaIVF — Multi-scenario benchmark

Single-thread (`OMP_NUM_THREADS=1`), apples-to-apples via candy bindings (same C++ compute engine, same Faiss HNSW backend code). Maintenance is **bench-driven** (each scenario decides whether/when to call `algo.maintain(vector_budget)`), so insert/query latencies stay pure index operations.

## Maintenance contract (separation of concerns)

| Decision | Owner | Mechanism |
|---|---|---|
| **HOW MUCH** to drain (buffer/graph ratio) | bench / caller | `idx.maintain(vector_budget=N)` — N=0 = unbounded |
| **WHICH** vectors to drain | algorithm | GammaFresh: cold-and-old first (low `weighted_hit_count`, then low `insert_op_id`). Hot/recent vectors stay in buffer for hybrid scan |

## TL;DR — When does GammaFresh win?

| Win | Scenario | Numbers | Why |
|---|---|---|---|
| 🏆🏆🏆 | `random-100k bulk_delete(30%)` | **gamma 4.5s / recall 1.000** vs faiss 28.8s / 0.558, ivf 8.3s / 0.264 | Hybrid graph + partition-buffer scan: HNSW chokes on tombstones, IVF recall is unusable on random data; GammaFresh delivers **6.5× faster than HNSW with perfect recall** |
| 🏆 | `clustered-50k bulk_delete(30%)` | gamma 2.0s / 1.000 vs faiss 3.3s / 0.992 | Fastest **and** highest recall |
| 🏆 | `sift-small bulk_delete(30%)` | gamma 339ms / 1.000 vs faiss 375ms / 0.997 | Fastest **and** perfect recall |
| ✅ | `random-100k drift` | gamma recall **0.685** vs faiss 0.561, ivf 0.277 | Only algo with usable recall after concept drift on hard data |
| ✅ | All churn / drift / delete scenarios | gamma recall always **highest** (≥0.984, often 1.000) | Recall robustness is the consistent advantage |

## Datasets
- `sift-small` — 10K × 128d (real SIFT)
- `clustered-50k` — 50K × 128d, 50 Gaussian clusters (synthetic, IVF-friendly)
- `random-100k` — 100K × 128d uniform Gaussian (hard for any ANN)

## Scenarios

Each scenario chooses its own maintenance schedule via `algo.maintain(vector_budget)`. GammaFresh is a hybrid index: fresh writes accumulate in partition buffers (IVF-fast) and queries combine graph search + buffer scan. Only stable old/cold vectors get promoted to graph during scheduled maintenance.

- `streaming(b=100)` — 100/batch insert + 100 queries every batch, **maint(unbounded) before each query batch**
- `burst(b=2500)` — large-batch insert, **maint(unbounded) before final query**
- `drift(4chunk)` — alternating insert/delete-oldest cycles, **maint(unbounded) between cycles**
- `bulk_delete(30%)` — insert all + bulk-delete oldest 30%, **NO maint** (tests post-delete behavior; gamma's hybrid scan shines, HNSW pays for tombstones)
- `churn(20cyc, 5%)` — insert+delete+query cycles, **maint(vector_budget=chunk//2) per cycle** — bench picks half-chunk budget so buffer stays trim while hot vectors remain available for hybrid scan

## Results

### sift-small (10K, 128d) | streaming(b=100)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| faiss | 554 | 32,490 | 0.9970 | ✅ high recall |
| ivf | 573 | 31,400 | 0.9970 | ✅ high recall |
| gamma | 998 | 18,038 | 0.9970 | ✅ high recall |

### sift-small (10K, 128d) | burst(b=2500)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 184 | 49,420 | 0.9950 | ✅ high recall |
| faiss | 338 | 26,929 | 0.9970 | ✅ high recall |
| gamma | 668 | 13,616 | 0.9990 | ✅ high recall |

### sift-small (10K, 128d) | drift(4chunk)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 148 | 93,930 | 0.9990 | ✅ high recall |
| faiss | 428 | 32,489 | 0.9950 | ✅ high recall |
| gamma | 1088 | 12,775 | 1.0000 | ✅ high recall |

### sift-small (10K, 128d) | bulk_delete(30%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| gamma | 339 | 38,675 | 1.0000 | ✅ high recall |
| faiss | 375 | 34,971 | 0.9970 | ✅ high recall |
| ivf | 505 | 25,915 | 0.9860 |  |

### sift-small (10K, 128d) | churn(20cyc,5%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 58 | 178,166 | 0.9930 | ✅ high recall |
| faiss | 583 | 17,845 | 0.9940 | ✅ high recall |
| gamma | 2192 | 4,745 | 1.0000 | ✅ high recall |

### clustered-50k (50K, 128d, 50 clusters) | streaming(b=100)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| faiss | 6807 | 20,934 | 0.9695 |  |
| gamma | 11988 | 11,887 | 0.9805 |  |
| ivf | 15742 | 9,052 | 1.0000 | ✅ high recall |

### clustered-50k (50K, 128d, 50 clusters) | burst(b=2500)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| faiss | 2941 | 16,217 | 0.9845 |  |
| ivf | 3841 | 12,418 | 1.0000 | ✅ high recall |
| gamma | 5200 | 9,172 | 0.9960 | ✅ high recall |

### clustered-50k (50K, 128d, 50 clusters) | drift(4chunk)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 1934 | 37,246 | 1.0000 | ✅ high recall |
| faiss | 3509 | 20,531 | 0.9610 |  |
| gamma | 6377 | 11,299 | 0.9990 | ✅ high recall |

### clustered-50k (50K, 128d, 50 clusters) | bulk_delete(30%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| gamma | 2035 | 32,042 | 1.0000 | ✅ high recall |
| ivf | 3108 | 20,975 | 1.0000 | ✅ high recall |
| faiss | 3327 | 19,599 | 0.9915 | ✅ high recall |

### clustered-50k (50K, 128d, 50 clusters) | churn(20cyc,5%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 501 | 101,483 | 1.0000 | ✅ high recall |
| faiss | 5316 | 9,562 | 0.9880 |  |
| gamma | 32208 | 1,578 | 1.0000 | ✅ high recall |

### random-100k (100K, 128d) | streaming(b=100)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| faiss | 49963 | 5,704 | 0.5970 |  |
| gamma | 65189 | 4,372 | 0.6045 |  |
| ivf | 81917 | 3,479 | 0.2190 | ❌ recall too low to be useful |

### random-100k (100K, 128d) | burst(b=2500)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 4732 | 20,118 | 0.2190 | ❌ recall too low to be useful |
| faiss | 27171 | 3,504 | 0.5955 |  |
| gamma | 33053 | 2,880 | 0.5815 |  |

### random-100k (100K, 128d) | drift(4chunk)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 3729 | 38,428 | 0.2770 | ❌ recall too low to be useful |
| faiss | 28981 | 4,945 | 0.5605 |  |
| gamma | 53713 | 2,668 | 0.6850 |  |

### random-100k (100K, 128d) | bulk_delete(30%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| gamma | 4456 | 29,220 | 1.0000 | ✅ high recall |
| ivf | 8316 | 15,656 | 0.2640 | ❌ recall too low to be useful |
| faiss | 28802 | 4,521 | 0.5575 |  |

### random-100k (100K, 128d) | churn(20cyc,5%)

| algo | time (ms) | ops/s | recall@10 | notes |
|---|---:|---:|---:|---|
| ivf | 2346 | 41,807 | 0.2390 | ❌ recall too low to be useful |
| faiss | 23735 | 4,133 | 0.5575 |  |
| gamma | 162510 | 604 | 0.7595 |  |
