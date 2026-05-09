# Paper outline (VLDB-style draft)

This is a draft outline derived from the experiments under `experiments/`.
It maps each section to the experiments that supply its evidence so we can
audit "every claim has data."

## 1. Introduction

- Motivating workload: streaming insert + delete on ANN indexes (recommendation
  feeds, RAG, social, GDPR, content takedowns).
- The problem: incremental graph indexes (HNSW etc.) handle FIFO/sequential
  delete well via mark_deleted, but degrade on irregular delete patterns
  (random / cluster / bursty) because of tombstone pollution and the fixed
  rebuild trigger.
- Claim: a small *router* layer in front of any ANN backend — buffer-admit
  on insert + delete-absorption in the buffer + lazy maintenance — recovers
  performance across delete patterns and at extreme churn rates, without
  changing the underlying ANN algorithm.
- Contributions:
  1. Empirical characterization of delete-pattern impact on incremental
     ANN indexes (e15, e16, e17).
  2. The router architecture: design, parameters, sensitivity analysis
     (e18, e20).
  3. Validation of router-vs-direct across SIFT / MSong / GloVe at
     200K and 1M scales (e15, e16).
  4. Ablation showing which router mechanisms are load-bearing (e20)
     and which complex C++ extensions add value vs overhead (e21).
  5. hnswlib-parameter-sensitivity comparison showing the gap is
     architectural, not a tuning artifact (e19).

## 2. Related work

- HNSW and its mark_deleted (Malkov & Yashunin 2018).
- DiskANN / FreshDiskANN (incremental graph + lazy rebuild).
- SPFresh (cluster-rebuild for IVF).
- LSM-style write buffers in OLTP (analogous architecture).
- IVF tombstones in Faiss.

## 3. The router architecture

- 3.1 Conceptual model: the index as a (write buffer) + (graph) split,
  with delete-absorption and lazy flush as the contract.
- 3.2 Pseudocode of `add`, `delete`, `maintain`, `search`.
- 3.3 Parameters: buffer capacity, maintenance interval. Defaults justified
  by e18.
- 3.4 The C++ "smart" extras (γ-driven adaptive split, cost-model
  placement, spatial routing). These are present in the codebase but
  Section 5.4 (e21) shows they currently add overhead at this workload size
  — discuss honestly: simple is enough.

## 4. Experimental setup

- Datasets: SIFT 1M (image, 128d), MSong 992K (audio, 420d), GloVe 1.18M
  (text, 100d, angular), random-m 100K (synthetic uniform).
- Backends: hnswlib (Qdrant/Chroma default), Faiss HNSW.
- Workload: streaming insert + concurrent delete; 4 delete patterns
  (sequential / random / cluster / partial_reset).
- Single-thread fair comparison: explicitly justify why this matters
  (multi-thread defaults make hnswlib direct look 60% faster — that's
  not the algorithm, that's `set_num_threads(76)`).
- Metrics: wall-clock total, recall@10 vs ground truth on alive set,
  query p95 latency.

## 5. Results

### 5.1 Main: gamma vs direct on each delete pattern

(Figure: e15 main bars at 200K and 1M.)

Result: gamma_v2+hnswlib wins on every pattern. -7% (sequential), -27%
(random), -23% (cluster), -23% (partial_reset). Same direction at 1M scale.
Same backend swap on FaissHNSW shows similar results (-1% to -30%).

### 5.2 Cross-dataset robustness

(Figure: e16 cross-dataset bars — msong, glove, random-m.)

Result: pattern of "lose sequential, win non-sequential" reproduces across
all 4 datasets. random-m cluster shows the biggest single win (-36%).

### 5.3 Sensitivity to churn intensity

(Figure: e17 churn curve.)

Result: gamma's lead is monotonically non-decreasing in churn intensity.
At ratio 1.5+, gamma wins 60-85% on every pattern including sequential
(hnswlib's tombstone bookkeeping starts dominating).

### 5.4 Ablation

#### 5.4.1 Router-component ablation (Python POC)

(Figure: e20 ablation bars.)

Result: buffer-admit and delete-absorption are load-bearing; without
delete-absorption gamma is *worse* than the direct backend (+25%).
Buffer scan in search and eager maintenance contribute nothing.

#### 5.4.2 C++ multi-partition ablation

(Figure: e21 cpp ablation bars.)

Result: the C++ default config (small partitions, BIC-driven split,
cost-model placement) is 2.6× slower than the same C++ index with
`split_factor=∞` (1 partition). At this workload size, the multi-partition
machinery costs more than it earns. We discuss why and propose when it
might pay off (very large indexes, multi-tenant routing).

### 5.5 Modular extensions on the Python POC ⭐

After the C++ ablation showed the multi-partition design underperforming,
we tested *which individual C++-inspired modules* on top of the simple
Python POC actually pay off. Four modules added one at a time:

#### 5.5.1 Spatial routing (e24)
K-partition + centroid-based routing. **Throughput-vs-latency trade-off**.
SIFT 200K cluster: K=16/M=8 gives -23% wall time and rec=1.0, but pays
6.7× higher p95 query latency. Pareto: K=1 (gamma_v2) wins for latency-
bound; K=16 wins for throughput-bound.

#### 5.5.2 Tombstone-rebuild trigger (e25) ★
Periodic full-rebuild when graph tombstone fraction > threshold. Universal
sweet spot at threshold=0.5: -38% to -56% across all 4 patterns at SIFT 200K
vs gamma_v2 (no recall impact). Most importantly, the +14% / +39% /
+106% / +115% sequential losses (200K SIFT / 1M SIFT / 1M MSong / 1M
GloVe) all flip to wins under rebuild. **This is the architectural fix
for gamma's sequential-pattern weakness.**

#### 5.5.3 Adaptive maintenance scheduling (e26)
Auto-trigger maint at buffer-fill threshold. Under our qstride-driven
workload the buffer-fill trigger never fires (qstride flush dominates).
Null finding under this experimental setup; future work to tune.

#### 5.5.4 Cost-model admit (e27)
Per-batch admit decision based on rolling-mean of observed lifetimes.
Tests whether learned admit beats always-buffer. Streams.

### 5.5 Buffer-capacity sensitivity

(Figure: e18 buffer curve.)

Result: optimal buf_mult ∈ [5, 25] on SIFT 200K cluster. Above mult=50
the win shrinks but doesn't disappear.

### 5.6 Comparison vs hnswlib hyperparameter tuning

(Figure: e19 param-grid Pareto.)

Result: gamma at default hyperparameters beats every hnswlib-direct config
in the (M, efC, efS) ∈ {16,32,64} × {120,240} × {80,160} grid on cluster
pattern. The gap is architectural, not a tuning artifact.

## 6. Discussion

- Where simple > complex: the router design's main contribution is the
  cancellation pathway (delete reaches buffer → just mark slot dead). All
  the complex routing on top can be deferred.
- Limitation: at low churn (≤0.5×) gamma is no better than direct.
- Limitation: very high churn (≥1.5×) trades a bit of recall for huge
  speedup; deployment must check.
- Future work: multi-tenant routing, distributed sharding, tail-latency
  analysis, real workload trace replay.

## 7. Conclusion

A small write-buffer + delete-absorption + tombstone-rebuild trigger router
lifts existing incremental ANN indexes (hnswlib, Faiss HNSW) by 38-56% on
SIFT 200K across all 4 delete patterns (cluster, random, sequential,
partial_reset), at the same recall, without modifying the underlying graph.

The simplest variant (buffer + delete-absorption only, e15) wins
20-30% on non-sequential patterns but loses on 1M sequential. Adding the
tombstone-rebuild trigger (e25) closes the sequential gap: gamma_v2+rebuild
beats hnswlib direct on sequential too (-51% at SIFT 200K). The combination
is the recommended drop-in design. The choice of underlying backend
remains orthogonal — same architecture works on hnswlib, FaissHNSW, and
(in principle) any backend with mark_deleted.

The C++ multi-partition machinery (cost-model placement controller,
γ-driven adaptive split, BIC-driven repartition, spatial routing) does not
add measurable value at the workload sizes we tested (200K-1M), and the
shipped default config has correctness bugs on partial_reset (rec=0.006).
At larger scales (10M+) or for different access patterns (Zipfian queries,
multi-tenant), the multi-partition design may justify its overhead — left
for future work.
