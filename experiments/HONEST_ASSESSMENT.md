# GammaFresh — 实验诚实评估

经过 11 个独立实验、3 个数据集、5 个外部 baseline 对比,这是诚实的最终评估。

## 之前的 thesis(已被推翻部分)

❌ **「Gamma 是唯一同时维持高 recall + 低 query latency 的 ANN」**

这个 claim 不成立 — 我们对比的 Faiss HNSW 用了 suboptimal tombstone 实现(filter-after-search),所以 recall 在 churn 下崩。但 hnswlib 用 filter-during-search 的 tombstone,在 churn 下 recall 不崩。

❌ **「Gamma 在 streaming_sliding 比 HNSW 快 24%」**

这只对 Faiss HNSW 成立。hnswlib 在 streaming_sliding(lag=2)/1M 上比 gamma **快 2.6×**。

❌ **「Gamma 写吞吐 4× HNSW」**

这只对 Faiss HNSW 成立。hnswlib **写吞吐比 gamma 快 2×**(63K vs 30K ops/s)。

## 仍然站得住的 claim

✅ **Gamma vs Faiss HNSW(我们 candy bench 集成的 backend):**
- 写吞吐 4-6× faster
- streaming_sliding total time -24% / -25%
- bulk_delete -62% + perfect recall
- 这些数据真实,但只对 Faiss HNSW 成立

✅ **可解释性 / 设计层面:**
- explicit maint API(operator-controlled scheduling)
- cost-model based placement decision
- 智能 partition routing(e04 验证有效)
- gamma 的 maint 顺序修复(e03 验证)

## 完整结果对比表

### Bulk delete (SIFT 1M, 30% delete)
| algo | op_total | qry | recall |
|---|---:|---:|---:|
| **hnswlib** | **52.7s** ⭐ | 0.03ms | 0.984 |
| faiss HNSW | 158.8s | 0.17ms | 0.983 |
| gamma | 224s | 0.22ms | 0.984 |
| faiss NSG | 1164s | 0.11ms | 0.980 |
| faiss IVF | 1.1s | 3.43ms | 0.999 |

### Streaming sliding(lag=2)/1M
| algo | total | recall | qry p95 |
|---|---:|---:|---:|
| **hnswlib + rebuild_4** | **48.8s** ⭐ | 0.994 | 0.020ms |
| **hnswlib (no rebuild)** | **125.2s** | 0.996 | 0.226ms |
| gamma | 332.6s | 0.994 | 0.134ms |
| faiss HNSW | 555s | 0.853 | 23.97ms |

### Pure insert SIFT 200K
| algo | ops/s | per-op |
|---|---:|---:|
| **hnswlib** | **63,423** | 15.8µs |
| gamma | 29,741 | 33.6µs |
| faiss HNSW | 5,308 | 188µs |

## 给老板/合作者的诚实结论

**论文 publish-ability 评估(基于现有数据):**

| 投稿目标 | 评估 |
|---|---|
| **VLDB / SIGMOD 主会** | ❌ **不行** — hnswlib 全场碾压,thesis 立不住 |
| **EDBT / CIKM** | ⚠️ **风险大** — 一个 reviewer 想到 hnswlib 就 reject |
| **VLDB demo / industrial** | 🟡 可投(focus 在 system design 而非性能)|
| **arxiv preprint** | ✅ 可发(诚实呈现 trade-off)|

## 可能的挽救路径

### 路径 1: 替换 backend 为 hnswlib(2-3 周工作)
- 把 gamma 的 graph 后端从 FaissHNSW 改为 hnswlib
- 期望:gamma + hnswlib backend 应该 inherit hnswlib 的写速度,加上 buffer-first 的额外好处
- 风险:可能 gamma + hnswlib 跟纯 hnswlib 性能相当,差 buffer 的额外开销
- **这是最有希望的路径**

### 路径 2: 找 hnswlib 真正失败的场景(1-2 周探索)
- 内存受限场景(hnswlib 必须 max_elements pre-alloc)
- 多线程并发写(hnswlib 锁竞争 vs gamma buffer-first)
- 极高 churn(>95% delete rate)
- 真实业务 trace(maybe Spotify / Twitter recall sensitive)

### 路径 3: 转向 system contribution(改写 paper 角度)
- "An extensible framework for hybrid ANN with operator-controlled maintenance"
- 重点放在 architecture / API / observability
- 性能不是主卖点
- 可能投 SIGMOD industrial track 或 SoCC / SOSP industrial

### 路径 4: 接受现状,转投较低级别会议
- EDBT industrial 或 CIKM short paper
- 如实呈现「在某些 scenario 下 vs 某些 backend 成立」
- 价值在于完整的 ablation 和 honest reporting

## 我的建议

**不要再投资 in 当前 thesis 了。** 数据给了我们一个明确信号:hnswlib 是一个我们无法在标准性能维度击败的 baseline。

**最务实的下一步:**
1. **试路径 1**(2-3 周)— 看 gamma + hnswlib backend 能不能赢纯 hnswlib
2. 如果路径 1 不行,**走路径 3**(reframe paper)
3. 平行做 arxiv preprint(零风险)

## 数据落盘

所有原始数据在 `experiments/e0X/output*.json`。详见:
- `experiments/ALL_RESULTS_TABLE.md` — 总览表
- `experiments/SCENARIO_BY_SCENARIO_ANALYSIS.md` — 13 场景独立分析  
- `experiments/e10_external_baselines/RESULTS.md` — vs hnswlib/scann/annoy(offline rebuild)
- `experiments/e11_hnswlib_streaming/output_*.json` — vs hnswlib(streaming)
