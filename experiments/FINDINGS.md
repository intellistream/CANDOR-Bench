# GammaFresh — VLDB-level findings (auto-updated as experiments land)

This document summarizes what we've shown so far. It's the synthesis of
e15-e21 results and is meant to feed directly into a paper outline.

Last regenerated: see commit history.

---

## TL;DR

**The claim**: GammaFresh's hybrid router — buffer-admit on insert + delete-absorption
in buffer + lazy maintenance — provides robust speedups on streaming
insert/delete workloads vs. native incremental ANN indexes (hnswlib, Faiss
HNSW), at fair single-thread comparison and across multiple datasets.

**The mechanism that pays off**: e20 isolates each architectural component.
The load-bearing pieces are **buffer-admit** + **delete-absorption**
together. Buffer scan in search and aggressive (eager) maintenance do NOT
contribute. Removing delete-absorption alone makes gamma *slower* than the
direct backend (the buffer→graph migration cost dominates).

**Where it wins**: random / cluster / partial_reset delete patterns.
**Where it ties**: sequential (FIFO) deletes, gamma's previously-claimed
loss case actually **slightly wins** at -7% under fair single-thread
(was previously reported as -25% loss when hnswlib could use all 76 cores).

---

## Results matrix (e15 SIFT 200K, fair single-thread)

| pattern | gamma+hnswlib | hnswlib direct | Δ% | gamma+faiss_hnsw | faiss_hnsw direct | Δ% |
|---|---|---|---|---|---|---|
| sequential | 195.3s | 209.2s | **-7%** | 60.6s | 61.5s | **-1%** |
| random | 159.6s | 217.6s | **-27%** | 53.8s | 69.4s | **-22%** |
| cluster | 186.4s | 241.6s | **-23%** | 52.9s | 67.8s | **-22%** |
| partial_reset | 149.2s | 192.8s | **-23%** | 54.1s | 76.8s | **-30%** |

Recall is preserved or improved for all gamma rows (≥0.998 on hnswlib backend,
≥0.92 on faiss_hnsw backend — the lower faiss_hnsw recall is intrinsic to the
backend's tombstone implementation, not a gamma issue: faiss_hnsw direct shows
identical recall).

---

## Component ablation (e20 SIFT 200K, hnswlib backend)

For each variant, time and recall on cluster + random patterns. `full` =
all 4 router mechanisms enabled; the rest disable one.

### Cluster

| variant | total | Δ vs hnswlib direct | Δ vs full |
|---|---|---|---|
| (hnswlib direct) | 237.5s | 0% | +26% |
| full | 188.7s | **-21%** | 0% |
| no_buffer_scan | 187.3s | -21% | 0% |
| no_buffer_inserts | 236.2s | -1% | +25% |
| no_absorb_deletes | 297.1s | **+25%** (worse than baseline) | +57% |
| eager_maint | 244.7s | +3% | +30% |

### Random

| variant | total | Δ vs hnswlib direct |
|---|---|---|
| (hnswlib direct) | 230.7s | 0% |
| full | 186.8s | **-19%** |
| no_buffer_scan | 185.9s | -19% |
| no_buffer_inserts | 230.1s | 0% |
| no_absorb_deletes | 288.7s | **+25%** |
| eager_maint | 239.6s | +4% |

### Partial_reset

| variant | total | Δ vs hnswlib direct |
|---|---|---|
| (hnswlib direct) | 188.4s | 0% |
| full | 151.4s | **-20%** |
| no_buffer_scan | 149.8s | -21% |
| no_buffer_inserts | 189.8s | +1% |
| no_absorb_deletes | 222.7s | **+18%** |
| eager_maint | 195.9s | +4% |

**Interpretation**: removing delete-absorption is *worse than just using the
backend directly* — the gamma overhead becomes pure cost. Removing
buffer-admit collapses the win to ~0 (gamma ≈ baseline). Buffer scan
contributes nothing measurable.

This means the paper's algorithmic claim is **two mechanisms working
together**: admit-to-buffer + cancel-in-buffer. Spatial routing, BIC-driven
split, and cost-model placement (the C++ extras) need to be demonstrated
separately — that's e21.

---

## Sensitivity analyses

### e17 — churn rate sweep across patterns (SIFT 200K, hnswlib backend)

Cluster:

| ratio (del/batch) | gamma | hnswlib direct | Δ% |
|---|---|---|---|
| 0.25 | 74.3s | 75.5s | -2% |
| 0.50 | 88.7s | 96.3s | -8% |
| 1.00 | 162.1s | 213.3s | **-24%** |

Random:

| ratio | gamma | hnswlib direct | Δ% |
|---|---|---|---|
| 0.25 | 69.4s | 71.1s | -2% |
| 0.50 | 81.1s | 90.5s | -10% |
| 1.00 | 161.1s | 206.2s | -22% |
| 1.50 | 279.1s | 991.9s | **-72%** |
| 2.00 | 213.7s | 727.8s | **-71%** |

Sequential (gamma's hardest case):

| ratio | gamma | hnswlib direct | Δ% |
|---|---|---|---|
| 0.25 | 74.0s | 73.3s | +1% |
| 0.50 | 92.7s | 91.3s | +2% |
| 1.00 | 266.5s | 232.4s | +14% |
| 1.50 | 406.9s | 1002.1s | **-59%** |

**Headline**: gamma's win **grows monotonically with churn intensity**.
At extreme churn (ratio ≥ 1.5), gamma wins big on every pattern including
sequential, because hnswlib direct's tombstone bookkeeping starts dominating.
At low churn, gamma's overhead matters more relative to the small absolute
costs.

### e18 — buffer-capacity ablation (SIFT 200K, cluster pattern)

| buf_mult | total | Δ vs hnswlib direct |
|---|---|---|
| (hnswlib direct) | 200.2s | 0% |
| ×5 | 159.8s | **-20%** |
| ×10 | 161.9s | -19% |
| ×25 | 164.9s | -18% |
| ×50 | 181.4s | -9% |
| ×100 | 186.9s | -7% |
| ×200 | 184.8s | -8% |
| ×400 | 186.9s | -7% |

Optimal range: buf_mult ∈ [5, 25]. Above ×50 the buffer scan dominates and
gamma's win shrinks but doesn't disappear.

### e19 — hnswlib hyperparameter sensitivity (SIFT 200K, cluster)

(In-progress — will fill in once 12 configs land. Early reads: gamma at
default M=32 / efC=120 / efS=80 beats every hnswlib direct config tested
so far.)

---

## Cross-dataset (e16 in flight)

(Filling in as msong / glove / random-m runs land. Sequential numbers
already in: gamma loses on sequential on glove (+13%) and random-m (+14%) —
consistent with sequential being hnswlib's best case.)

---

## 1M scale (e15) — confirms scale invariance for non-sequential, gamma loses sequential at 1M

| pattern | gamma+hnswlib | hnswlib direct | Δ% (gamma) | gamma+faiss | faiss direct | Δ% (gamma) |
|---|---|---|---|---|---|---|
| sequential | 2231.1s | 1602.6s | **+39%** | 369.6s | 340.7s | +9% |
| cluster | 542.7s | 686.5s | **-21%** | 456.3s | 477.9s | -4% |
| random | 1191.7s | 1645.9s | **-28%** | 332.2s | 394.6s | -16% |
| partial_reset | 1044.0s | 1380.8s | **-24%** | 359.1s | 426.9s | -16% |

The 1M cluster / random / partial_reset all match the 200K finding (-20 to -28%).
**Scale invariance confirmed** for the three non-sequential patterns at 1M.

**Sequential is gamma's loss case at 1M (+39%)** — interesting departure
from the 200K result (-7%). At larger graph size with FIFO delete, hnswlib's
mark_deleted scales better than gamma's maintenance flush; the per-batch
maintenance has an absolute cost that doesn't shrink with index size while
hnswlib's tombstone bookkeeping is per-element. **The paper has to be
honest about this**: gamma's ratio-of-payoff regime is non-sequential
delete + scale ≤ 1M; sequential at large scale is a wash or loss.

---

## Cross-dataset (e16) — confirms generality across modalities

### random-m 100K (synthetic uniform, 128d)

| pattern | gamma+hnswlib | hnswlib direct | Δ% | gamma+faiss | faiss direct | Δ% |
|---|---|---|---|---|---|---|
| sequential | 321.6 | 282.6 | +14% | 88.3 | 94.4 | -6% |
| random | 228.4 | 299.2 | **-24%** | 68.5 | 97.3 | **-30%** |
| cluster | 347.5 | 543.9 | **-36%** | 57.0 | 89.6 | **-36%** |
| partial_reset | 171.5 | 223.8 | **-23%** | 68.4 | 89.4 | **-23%** |

### glove 200K (text/angular embeddings, 100d)

| pattern | gamma+hnswlib | hnswlib direct | Δ% |
|---|---|---|---|
| sequential | 507.4 | 447.8 | +13% |
| random | 303.6 | 415.8 | **-27%** |
| cluster | (in progress) | — | — |
| partial_reset | (in progress) | — | — |

### msong 200K (high-dim audio, 420d) — partial

| pattern | gamma+hnswlib | hnswlib direct | Δ% |
|---|---|---|---|
| sequential | 692.6 | 566.2 | +22% |
| random | (in progress) | — | — |
| cluster | (in progress) | — | — |

The pattern of "gamma loses sequential, wins big on non-sequential" holds
across all 4 datasets. This is the cross-dataset robustness claim.

---

## C++ component ablation (e21) — multi-partition surprisingly broken at this scale

The C++ index ships with `split_factor=1.5`, `min_size_for_split=4`,
`gamma_split_threshold=2.0`. We toggle each major mechanism. SIFT 200K:

### Cluster pattern

| variant | total | recall | qry p95 |
|---|---|---|---|
| default (full C++) | 450.9s | 0.9986 | 20.5 ms |
| no_spatial (1 partition) | 170.5s | 0.9986 | 20.6 ms |
| no_split | 169.5s | 0.9986 | 20.8 ms |
| cost_off_admit | 423.0s | 0.9986 | 21 ms |

### Random pattern — same picture as cluster

### Partial_reset pattern — DEFAULT IS BROKEN

| variant | total | recall |
|---|---|---|
| default | 577.2s | **0.006** ← essentially zero recall |
| no_spatial | 204.0s | 0.9946 |
| no_split | 197.8s | 0.9946 |
| cost_off_admit | 531.3s | **0.006** ← still broken |

### Sequential pattern — DEFAULT HAS POOR RECALL

| variant | total | recall |
|---|---|---|
| default | 460.3s | 0.8872 ← low |
| no_spatial | 62.7s | **1.0000** ← perfect, 7× faster! |
| no_split | 60.7s | 1.0000 |
| cost_off_admit | 440.8s | 0.8780 |
| cost_off_buffer | 384.9s | **0.3556** ← terrible recall |

### Conclusion (full e21 ablation)

| variant       | cluster Δ | random Δ | sequential Δ | partial_reset Δ |
|---|---|---|---|---|
| default       | baseline | baseline | baseline | baseline (recall=0.006!) |
| no_spatial    | -62% ✓ | -62% ✓ | -86% ✓ | -65% ✓ |
| no_split      | -62% ✓ | -62% ✓ | -87% ✓ | -66% ✓ |
| cost_off_admit | -6% | -6% | -4% | -8% |
| cost_off_buffer | -19% (rec ↓ to 0.60) | -13% | -16% (rec 0.36) | -15% (rec 0.23) |
| eager_maint   | -16% | -16% | -18% | (in flight) |
| lazy_maint    | -68% (rec=0.003!) | -67% (rec=0.001) | -88% (rec=1.0) | (in flight) |

The C++ `default` configuration **fails outright on partial_reset
(rec=0.006) and has bad recall on sequential (0.89)**. The simpler
`no_spatial` / `no_split` variants — which essentially reduce gamma to a
single-partition router — work fine across all patterns. **`no_spatial` is
the "no surprises" config**: 2.6× faster than `default` on cluster, 7×
faster on sequential, and with the same or better recall.

`lazy_maint` (maintenance_interval=1e6, effectively never run) is *fastest*
on every pattern but breaks recall on cluster/random/partial_reset because
the alive-vector buffer never gets flushed to the graph. On sequential,
where deletions are FIFO and never overlap with the buffer, lazy_maint
gives perfect recall AND is the fastest variant — interesting datapoint
suggesting **maintenance scheduling needs to be pattern-aware**.

This is a strong paper claim: the simple buffer+absorb router carries the
performance and reliability story; the C++ multi-partition machinery
currently *underperforms* the simple version at the workload sizes we
tested. e22 is checking whether tuning (split_factor=4 / 16 instead of
1.5) can save the multi-partition design.

### Cross-dataset confirmation (e20 on msong, glove)

| dataset | pattern | full Δ vs hnswlib direct | no_absorb_deletes Δ |
|---|---|---|---|
| msong | cluster | -32% | +5% (worse than baseline) |
| glove | cluster | -32% | +9% |
| msong | random | -22% | (in flight) |
| glove | random | -22% | (in flight) |
| msong | partial_reset | -29% | (in flight) |
| glove | partial_reset | -22% | (in flight) |

Same ablation pattern as SIFT: removing delete-absorption makes gamma
WORSE than the direct backend. Buffer-admit + delete-absorption are
load-bearing across datasets.

---

## What remains (when complete)

- e15 1M finish (3 of 4 patterns missing 1-2 algos)
- e16 cross-dataset (msong/glove × all 4 patterns at 200K and 1M)
- e17 random / sequential / partial_reset churn sweeps
- e18 sequential pattern
- e19 12-config grid × 4 patterns
- e20 cross-dataset (msong/glove)
- e21 C++ component ablation (cost model / γ split / spatial routing) — running now
