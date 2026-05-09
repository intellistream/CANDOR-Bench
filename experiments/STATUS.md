# Live experiment status

(Snapshot — re-run `experiments/_shared/aggregate.py` for fresh tables.)

## Done — high confidence

- ✅ **e15 200K** (4 patterns × 4 algos) — re-baselined under fair single-thread
- ✅ **e15 1M cluster + random + partial_reset** (3 of 4 patterns at 1M)
- ✅ **e16 random-m 100K** (4 patterns × 4 algos)
- ✅ **e18 buffer-capacity ablation** (all 4 patterns × 8 buf_mults)
- ✅ **e19 hnswlib param sensitivity** — most cells (still completing the 13-config grid for 3 patterns)
- ✅ **e20 SIFT** router-component ablation (all 4 patterns × 6 variants)
- ✅ **e21 SIFT C++ component ablation** — partial (4-5 of 7 variants per pattern)
- ✅ **e22 SIFT C++ tuned config** — just started

## In flight

- ⏳ **e15 1M sequential** — gamma+hnswlib done (2231s), waiting on hnswlib direct + faiss legs
- ⏳ **e16 cross-dataset 200K** — msong/glove still finishing patterns 3 and 4
- ⏳ **e16 cross-dataset 1M** — msong/glove just starting (will take 1-2 hr)
- ⏳ **e20 cross-dataset (msong/glove)** — most patterns 4-6 of 6 variants done
- ⏳ **e21 last variants** — eager_maint, lazy_maint, cost_off_buffer pending per pattern
- ⏳ **e22** — first variants running

## Headlines so far

1. **At fair single-thread, gamma_v2+hnswlib wins on EVERY pattern at 200K**
   (-7% sequential, -27% random, -23% cluster, -23% partial_reset).
2. **Same direction at 1M scale**: -21% to -28% on the 3 non-sequential patterns
   (sequential 1M still completing).
3. **Cross-dataset robust**: confirmed on glove (-27% random), random-m (-36% cluster),
   msong (in flight). Sequential consistently loses ~+13%, but recovers at extreme churn.
4. **Churn rate sensitivity**: gamma's lead grows monotonically with churn intensity.
   At ratio≥1.5×, gamma wins **60-85%** even on sequential.
5. **Component ablation**: buffer-admit + delete-absorption are load-bearing.
   Removing delete-absorption makes gamma *worse* than the direct backend (+25%).
   Buffer scan in search and eager maintenance contribute nothing measurable.
6. **C++ multi-partition default config is broken** at our workload sizes:
   2.6× slower than the same C++ with 1 partition; recall=0.006 on partial_reset.
   The simple Python POC is faster AND more correct. e22 testing whether tuning
   can save the C++ design.

## Documents

- `experiments/FINDINGS.md` — synthesis with tables for each finding
- `experiments/PAPER_OUTLINE.md` — VLDB-style outline mapping sections to experiments
- `experiments/INTENTS.md` — one-paragraph intent per experiment
- `experiments/ALL_RESULTS_TABLE.md` — auto-generated grand table of every result
- `experiments/ALL_DELTAS_TABLE.md` — gamma vs direct deltas, sorted by Δ%
- `experiments/figures/` — 7 PNG figures (e15-e21)
