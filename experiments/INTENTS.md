# Experiment Intents — what we set out to prove and what we found

Each experiment in `experiments/eXX_*/` exists to answer ONE question.
This file is the index. Detailed methodology lives in each `eXX/README.md`,
raw data in `eXX/output*.json`, and (where present) per-experiment write-ups
in `eXX/RESULTS.md`.

---

## e01 — Write throughput

**Intent:** Does GammaFresh achieve higher write throughput than HNSW (and how does it compare to IVF)?

**Method:** Pure-insert stream on SIFT 200K and 1M; measure ops/sec, per-insert latency, final recall.

**Finding:** 
- gamma 6.6× FaissHNSW write throughput (27K vs 4K ops/s on SIFT 200K)
- BUT hnswlib (independent impl, not Faiss-based) is 2× faster than gamma (63K ops/s)
- The 6.6× claim is specifically vs FaissHNSW, NOT vs HNSW algorithm in general

**Status:** ✅ Completed. See `e01_write_throughput/output.json`.

---

## e02 — Recall and query latency under churn

**Intent:** As delete fraction grows, how do recall and query latency evolve for gamma vs HNSW vs IVF — across multiple datasets?

**Method:** Sweep `delete_frac ∈ {0, 0.25, 0.5, 0.75, 0.9}` on 3 datasets (SIFT/MSong/Random-m).

**Finding (vs Faiss HNSW):**
- HNSW recall drops 0.99 → 0.59-0.94 as churn grows (all 3 datasets)
- Gamma recall stays ≥ 0.99 (or rises as alive set shrinks)
- HNSW query p95 jumps to 2-5ms once any delete starts; gamma stays at 0.1ms

**Caveat:** This is vs Faiss HNSW only. e11 shows hnswlib doesn't have this problem.

**Status:** ✅ Completed across 3 datasets.

---

## e03 — HNSW + periodic rebuild ablation

**Intent:** Can Faiss HNSW recover from churn-induced recall drop by adding periodic graph rebuild? At what cost?

**Method:** Rebuild every {1, 4, 16} batches; measure total time and query latency p95.

**Finding:**
- HNSW + rebuild can recover recall to 0.99+ at any rebuild frequency
- BUT query p95 between rebuilds is 1.5-3.5ms (still 13-30× gamma)
- Total time is 1.5-3× gamma at moderate churn

**Caveat:** Same as e02 — vs Faiss HNSW only.

**Status:** ✅ Completed.

---

## e04 — Smart partition routing ablation

**Intent:** Does GammaFresh's `choose_owner_partition` (smart spatial routing) actually pay off vs cheap round-robin?

**Method:** Env var `GAMMA_DUMB_ROUTING=1` swaps to round-robin; A/B on buffer-resident query workload.

**Finding:** SMART beats DUMB by 28% total time + 1.6× query latency at same recall=1.0.

**Status:** ✅ Completed.

---

## e05 — Sliding-window streaming

**Intent:** On realistic streaming workloads (sliding window with insert + delete-oldest), does gamma win total time?

**Method:** `streaming_sliding(lag=N)` on SIFT 200K and 1M for `lag ∈ {2, 4, 8}`.

**Finding (vs Faiss HNSW):**
- lag=2/1M: gamma 416s vs faiss 556s (-25%, recall 0.994 vs 0.853)
- lag=4: similar pattern
- lag=8: gamma loses (vectors live long enough to migrate)

**Caveat:** Vs hnswlib direct (e11), gamma loses by ~25% on this workload.

**Status:** ✅ Completed.

---

## e06 — Bulk delete

**Intent:** Is gamma significantly better at bulk-delete (GDPR purge, TTL cleanup)?

**Method:** SIFT 1M, init 100K, stream 900K, then bulk-delete 30%.

**Finding (vs Faiss HNSW):**
- gamma 40.5s op + recall 1.000 vs faiss 290s + 0.973
- gamma -86% time + perfect recall

**Status:** ✅ Completed. ⚠️ vs hnswlib (e10), hnswlib wins on this scenario.

---

## e07 — Full scenario sweep

**Intent:** Across the 9 canonical ANN-stream scenarios, where does each algorithm win?

**Method:** Run all 9 scenarios on SIFT 200K and 1M for gamma/faiss/ivf.

**Finding:**
- gamma wins streaming_sliding + bulk_delete
- HNSW wins pure-write (insert_only / streaming / burst)
- IVF wins write-heavy / can-tolerate-slow-query
- Trade-offs across the matrix

**Status:** ✅ Completed.

---

## e08 — Where does gamma's recall come from?

**Intent:** Is gamma's recall stability under churn from (A) buffer scan acting as IVF or (B) the graph staying clean because deletes get absorbed by buffer?

**Method:** Force-drain gamma's buffer at every batch (so query goes graph-only); compare recall.

**Finding:** Recall is identical with or without buffer scan — so it's NOT from buffer scan.
It's from the graph staying cleaner due to fewer tombstones (gamma's maint absorbs deletes).

**Status:** ✅ Completed. Refined our thesis.

---

## e09 — Faiss-native baselines

**Intent:** Does gamma beat OTHER Faiss algorithms (NSG, IVF, IVFPQ) too, or just FaissHNSW?

**Method:** SIFT 1M bulk_delete with all 4 Faiss-native indexes; rebuild from surviving slice.

**Finding:**
- IVF wins op time (1.1s) — rebuild posting list super cheap
- NSG fastest query but slowest build (1164s)
- HNSW + rebuild competitive (159s)
- Gamma 226s — middle of pack
- IVFPQ recall destroyed by quantization (0.31)

**Two regimes emerge:** offline rebuild OK → IVF/HNSW+rebuild wins; streaming → see e02/e05.

**Status:** ✅ Completed.

---

## e10 — External libraries (hnswlib, scann, annoy)

**Intent:** How does gamma compare to industry-standard ANN libraries (Qdrant/ChromaDB use hnswlib)?

**Method:** SIFT 200K and 1M bulk_delete with hnswlib/scann/annoy + rebuild simulation.

**Finding (CRITICAL):**
- hnswlib at 1M bulk_delete: 53s vs gamma 224s — **hnswlib 4× faster than gamma**
- Same recall (0.984)
- hnswlib's mark_deleted is genuinely fast
- Faiss HNSW is 3× slower than hnswlib (lean impl)

**Status:** ✅ Completed. Forced us to revise the thesis.

---

## e11 — hnswlib in streaming

**Intent:** Does hnswlib also suffer tombstone-pollution under streaming, or is its mark_deleted fundamentally better?

**Method:** Streaming_sliding(lag=2) on SIFT 200K and 1M; hnswlib with mark_deleted only AND with periodic rebuild.

**Finding:** hnswlib's `mark_deleted` filters DURING search (not after), so recall under churn is preserved without rebuild.
- streaming_sliding(lag=2)/1M: hnswlib 125s/0.996 vs gamma 333s/0.994
- hnswlib with periodic rebuild: 49s/0.994 — **6.8× faster than gamma**

**Status:** ✅ Completed. Showed e02/e05's thesis was Faiss-specific.

---

## e12 — Gamma POC with hnswlib backend

**Intent:** If we replace gamma's FaissHNSW backend with hnswlib, does gamma's hybrid architecture still add value?

**Method:** Python POC of gamma's hybrid wrapping any backend; compare gamma+hnswlib vs hnswlib direct.

**Finding:**
- gamma+hnswlib: 14.6s on streaming_sliding(lag=2)/200K
- hnswlib raw: 11.7s
- Gamma adds 25% overhead with no recall improvement
- BUT this assumed sequential delete (lag-based)

**Status:** ✅ Completed.

---

## e13 — Fair backend-matched comparison

**Intent:** When comparing gamma vs algorithm X, swap gamma's backend to be X. Isolate architectural value from backend performance.

**Method:** gamma+X vs X direct for X ∈ {hnswlib, FaissHNSW} on bulk_delete, streaming_sliding(lag=2), streaming_sliding(lag=4).

**Finding:** Gamma's architecture provides clear value over FaissHNSW (because of bad tombstone),
but adds overhead with no benefit over hnswlib (because hnswlib's tombstone is already fast).

**Status:** ✅ Completed.

---

## e14 — Optimized buffer + diverse delete patterns ⭐

**Intent:** (a) Did Python overhead artificially inflate gamma's cost? (b) Does hnswlib's mark_deleted hold up under DIVERSE delete patterns (not just sequential)?

**Method:** Vectorized buffer (`gamma_py_v2`); test sequential / random / cluster delete patterns.

**Finding (BREAKTHROUGH):**
- gamma_v2+hnswlib **WINS hnswlib direct on random AND cluster delete** by 26-47% time
- Sequential is hnswlib's best case (gamma loses there)
- hnswlib's mark_deleted is fast for FIFO/LRU but degrades on irregular patterns

**Status:** ✅ Completed. Major finding — paper-defensible claim.

---

## e15 — Full delete-pattern × backend × architecture matrix ⭐

**Intent:** Confirm e14 finding across 4 patterns + 2 backends + 2 architectures + 2 scales.

**Method:** sequential / random / cluster / partial_reset × {hnswlib, faiss_hnsw} × {gamma_v2+, direct} × {200K, 1M}.

**Finding (200K, hnswlib backend):**
- sequential: hnswlib direct wins (-26% gamma)
- random: gamma -27% time + 0.06% recall
- cluster: gamma -29% time + 1.3× faster query
- partial_reset: gamma -28% time

**Status:** 200K complete; 1M complete (re-run on this machine for fair single-thread).

---

## e16 — Cross-dataset replication of e15 ⭐

**Intent:** Confirm e15 finding (gamma wins random/cluster/partial_reset) is
not SIFT-specific. MSong is high-dim audio (420d). Random-m is synthetic
uniform. GloVe is text embeddings (100d, angular).

**Method:** Same matrix as e15 — 4 patterns × 2 backends × 2 architectures
— but cycle through datasets + scales.

**Status:** msong 200K + 1M running; random-m 100K running; glove pending.

---

## e17 — Churn-rate sweep on each pattern

**Intent:** Does gamma's win grow, shrink, or invert as churn intensity
(deletes-per-batch / insert-batch) ranges from light (0.25×) to heavy (3×)?

**Method:** Each of the 4 delete patterns at SIFT 200K.
Sweep `del_per_batch / batch ∈ {0.25, 0.5, 1.0, 1.5, 2.0, 3.0}`.
gamma_v2+hnswlib vs hnswlib direct at each ratio.

**Status:** All 4 patterns running.

---

## e18 — Buffer-capacity ablation

**Intent:** How sensitive is gamma's win to the buffer capacity multiplier?
Maps the operating range and the failure modes (too-small → flush overhead;
too-large → buffer scan dominates query latency).

**Method:** Each pattern at SIFT 200K. Sweep
`buf_capacity = batch * mult` for `mult ∈ {5, 10, 25, 50, 100, 200, 400}`.

**Status:** All 4 patterns running.

---

## e21 — C++ component ablation ⭐

**Intent:** Same shape as e20 but for the *native* C++ index. The C++
ships with extras (BIC-driven adaptive split, cost-model placement
controller, spatial routing) — does each pay off?

**Method:** `candy.Index("GammaFresh")` configured with one mechanism
toggled to a degenerate value at a time: `no_spatial` (split_factor=1e9 →
single partition), `no_split` (γ=1e9), `cost_off_admit/buffer`,
`eager_maint`, `lazy_maint`. SIFT 200K, all 4 patterns.

**Finding (preliminary):** the *default* C++ config performs poorly
(2.6× slower than `no_spatial` on cluster/random; recall=0.006 on
partial_reset; recall=0.89 on sequential). The simpler `no_spatial`
variant works fine. Suggests the multi-partition machinery is mis-tuned
at this workload size. Ablation continuing.

---

## e22 — Tuned C++ config sweep ⭐

**Intent:** Can the C++ multi-partition design be saved by better tuning
than the shipped defaults? Tests `e04_tuned` (split_factor=4, γ=0.4) and
`large_split` (matches Python POC defaults: split_factor=16, γ=1.0)
against `default_cpp`, `no_spatial`, gamma_v2+FaissHNSW (Python POC),
and FaissHNSW direct.

**Status:** Just launched on all 4 patterns at SIFT 200K.

---

## e20 — Router-component ablation ⭐

**Intent:** Which mechanism of the simplified hybrid router actually pays off?
- buffer scan in search?
- buffer-admit on insert?
- delete absorption (vs always passing to graph)?
- lazy maintenance?

**Method:** `gamma_py_v3` exposes 4 toggles. Run each variant (one toggle off
at a time) on the 4 delete patterns (SIFT 200K), plus cross-dataset
(MSong/GloVe at cluster/random). Compare against hnswlib direct.

**Expected:** delete absorption is the load-bearing mechanism on cluster /
partial_reset; lazy maint matters for amortizing flush cost; buffer scan
adds recall on the most-recent inserts; buffer admit lets us be lazy.

**Status:** Running 12 configurations (4 SIFT patterns + 4 msong/glove × cluster/random).

---

## e19 — hnswlib parameter sensitivity

**Intent:** Can hnswlib close the gap to gamma on each pattern by tuning
M, efC, efS — or is the gap architectural?

**Method:** 12-config grid (M ∈ {16, 32, 64} × efC ∈ {120, 240} × efS ∈
{80, 160}) for hnswlib direct vs gamma_v2+hnswlib at default params.

**Status:** All 4 patterns running.

---

## Bottom line for paper

**Defensible claim:** GammaFresh's hybrid architecture provides **robustness across delete patterns**. While hnswlib's mark_deleted handles sequential delete fine, it degrades on random / cluster / bursty patterns — gamma wins these scenarios at fair backend matching.

**Realistic workloads (recommendation, social, e-commerce, RAG) have non-sequential delete patterns.** This is paper-defensible.

**What gamma does NOT win:** sequential / FIFO delete (hnswlib's best case), or pure-insert workloads.

---

## How to reproduce all

```bash
# Start any experiment from scratch
OMP_NUM_THREADS=1 uv run python experiments/eXX_*/run.py

# All experiments in sequence (~3-5 hours)
for d in experiments/e0* experiments/e1*; do
  OMP_NUM_THREADS=1 uv run python "$d/run.py"
done
```
