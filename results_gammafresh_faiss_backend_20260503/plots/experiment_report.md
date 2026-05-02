# GammaFresh(FaissHNSW backend) vs FaissHNSW

Dataset: `sift-small`  
Runbook: `general_experiment_sift_small`  
Date: 2026-05-03

## Summary

| algorithm | mean_query_qps | mean_insert_qps | mean_latency_ms | p50_latency_ms | p95_latency_ms | p99_latency_ms | first10_query_latency_ms | last10_query_latency_ms | growth_last10_over_first10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GammaFresh + FaissHNSW backend | 2977.06 | 24801.42 | 33.59 | 35.32 | 41.14 | 49.61 | 20.28 | 38.22 | 1.88 |
| FaissHNSW | 19657.36 | 10022.29 | 5.09 | 6.70 | 7.61 | 8.15 | 3.11 | 6.14 | 1.97 |

FaissHNSW mean query QPS is `6.60x` GammaFresh(FaissHNSW backend). GammaFresh mean latency is `6.60x` FaissHNSW.

## Plots

- `query_qps_by_batch.png`
- `query_latency_by_batch.png`
- `query_timing_breakdown.png`
- `summary_bars.png`
