# e09 — Comparison against Faiss-native baselines (NSG, IVF, IVFPQ)

**Question:** Does gamma's advantage hold against MORE algorithm types (not just HNSW)?

**Answer:** Conditional. Two distinct regimes emerge:

1. **Offline rebuild allowed** (batch ETL, scheduled maintenance) → IVF + rebuild and HNSW + rebuild are competitive or better than gamma
2. **No offline rebuild** (streaming, 24/7 SLA) → gamma's hybrid architecture is the only one that maintains recall AND query latency

## Setup

- SIFT 1M, init_n=100K, bulk_delete 30%
- Gamma: insert + delete + maint, then query
- Faiss-native: insert + delete simulated by **rebuild from surviving slice** (most-favorable case for them)
- All using fair search-time params: HNSW efSearch=80, NSG search_L=80, IVF nprobe=16

## Results — SIFT 1M, bulk_delete(30%)

| algo | op_total (s) | query (ms) | recall |
|---|---:|---:|---:|
| **gamma** | 226 | **0.23** | 0.984 |
| **faiss HNSW + rebuild** | **159** 🏆 | 0.17 | 0.983 |
| **faiss NSG + rebuild** | 1164 ⚠️ | **0.11** 🏆 | 0.980 |
| **faiss IVF + rebuild** | **1.1** ⚡ | **3.43** ⚠️ | **0.999** 🏆 |
| **faiss IVFPQ + rebuild** | 8.1 | 0.56 | 0.314 ⚠️ |

## Results — SIFT 200K, bulk_delete(30%)

| algo | op_total (s) | query (ms) | recall |
|---|---:|---:|---:|
| gamma | 29 | **0.15** | 0.991 |
| faiss HNSW + rebuild | **22** 🏆 | 0.11 | 0.994 |
| faiss NSG + rebuild | 197 ⚠️ | **0.08** 🏆 | 0.990 |
| faiss IVF + rebuild | **0.3** ⚡ | 0.58 | **0.999** 🏆 |
| faiss IVFPQ + rebuild | 2.4 | 0.12 | 0.365 ⚠️ |

## Two distinct regimes

### Regime A: Offline rebuild allowed (e.g., daily ETL → online query)

| algo | strength | weakness |
|---|---|---|
| **IVF + rebuild** | 1-2 orders of magnitude faster build | query 3-15× slower than HNSW/gamma |
| **HNSW + rebuild** | balanced (159s build + 0.17ms query) | rebuild blocks service |
| **NSG + rebuild** | fastest query (0.11ms) | 1164s build (7× HNSW) |
| **gamma** | similar to HNSW+rebuild | NO clear advantage in this regime |
| **IVFPQ** | small memory | recall destroyed (0.31) — not viable |

**Recommendation for this regime:** IVF if rebuild speed matters most; HNSW+rebuild if query speed matters most. **Gamma offers no compelling reason here.**

### Regime B: Streaming / no rebuild downtime

This is the workload covered by e02 / e05 / e08:

| algo | recall under churn | query p95 under churn | total time |
|---|---:|---:|---|
| **gamma** | **0.99+** ✅ | **0.1-0.2ms** ✅ | competitive |
| **HNSW (no rebuild)** | 0.59-0.94 ⚠️ | 4-25ms ⚠️ | fast (but wrong) |
| **HNSW + per-batch rebuild** | 0.99 ✅ | 1.5-3ms ⚠️ | 3-4× gamma |
| **IVF (no rebuild)** | high ✅ | 5-25ms ⚠️ | fast (but slow query) |
| **NSG** | n/a (Faiss NSG can't handle streaming inserts) | n/a | n/a |

**Recommendation for this regime:** gamma is the only solution that doesn't trade off recall vs query latency.

## Refined paper positioning

The strongest defensible claim:

> **GammaFresh wins WHEN offline rebuild is unavailable** (24/7 streaming SLA).
> When daily/weekly rebuild is allowed, IVF or HNSW+rebuild are simpler and faster.
> The contribution of GammaFresh is providing the **first ANN index that achieves
> SPFresh-class freshness without sacrificing query latency**.

## What this means for paper structure

1. **Don't claim gamma is universally best** — reviewers will catch this
2. **Frame the paper around the streaming/no-rebuild regime** — that's our turf
3. **Acknowledge IVF+rebuild and HNSW+rebuild as superior in offline-rebuild regime** — honest framing earns trust
4. **Show the regime boundary clearly** — Figure 1 candidate: 2D plot with x=churn intensity, y=algorithm choice region
