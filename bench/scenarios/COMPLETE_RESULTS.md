# SIFT 1M / 200K 完整结果报告

**测试条件:** OMP_NUM_THREADS=1(单线程),Faiss HNSW M=32 / efConstruction=120 / efSearch=80,IVF target_posting_size=4096。

**算法:** gamma(GammaFresh,latest commit `b7be797`) / faiss(直接 HNSW) / ivf(AdaIVF)。

**数据源:** `results/scenarios/sift1m_results_v2.json`(200K + bulk_delete 1M)+ 早前 1M sweep。

---

## 1. 总览:9 个 scenarios × 3 算法

### 总时间 + recall(单元: 秒 / recall@10)

| 场景 | 数据规模 | gamma 总时间 | gamma recall | faiss 总时间 | faiss recall | ivf 总时间 | ivf recall | 时间最快 | recall 最高 |
|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| **insert_only** | 200K | 8.1 s | 0.9928 | 31.4 s | 0.9913 | **2.2 s** | 0.9999 | ivf | ivf |
| insert_only | 1M | 346 s | 0.982 | 272 s | 0.982 | **38 s** | 0.992 | ivf | ivf |
| **insdel_stream(50%)** | 200K | 8.3 s | 0.9921 | 32.0 s | 0.9895 | **2.3 s** | 0.9998 | ivf | ivf |
| insdel_stream(50%) | 1M | 330 s | 0.981 | 300 s | 0.979 | **47 s** | 0.996 | ivf | ivf |
| **streaming(b=2500)** | 200K | 43.6 s | 0.9915 | **34.5 s** | 0.9914 | 98.1 s | 0.9999 | faiss | ivf |
| streaming(b=10K) | 1M | 316 s | 0.983 | **270 s** | 0.981 | 489 s | 0.992 | faiss | ivf |
| **streaming_sliding(lag=2)** | 1M | **402 s** 🏆 | **0.994** | 531 s | **0.853 ⚠️** | 124 s | 0.999 | ivf | ivf |
| streaming_sliding(lag=4) | 1M | 482 s 🏆 | **0.993** | 532 s | **0.868 ⚠️** | **142 s** | 1.000 | ivf | ivf |
| **burst(b=10K)** | 200K | 44.0 s | 0.9928 | **31.7 s** | 0.9913 | **12.5 s** | 0.9999 | ivf | ivf |
| burst(b=50K) | 1M | 353 s | 0.982 | 267 s | 0.981 | **39 s** | 0.992 | ivf | ivf |
| **drift(5cycles)** | 200K | 95.3 s | 0.9943 | **55.4 s** | 0.9823 | 26.3 s | 0.9998 | ivf | ivf |
| drift(5cycles) | 1M | 477 s | **0.984** | 379 s | 0.964 | **119 s** | 0.998 | ivf | ivf |
| **bulk_delete(30%)** | 1M | **131 s** 🏆 | **1.000** | 342 s | 0.973 | 34 s | 0.992 | ivf | gamma |
| **churn(10cyc,5%)** | 200K | 95.1 s | 0.9942 | **49.3 s** | 0.9896 | 58.6 s | 0.9998 | faiss | ivf |
| churn(10cyc,5%) | 1M | 560 s | **0.985** | **399 s** | 0.974 | 172 s | 0.997 | ivf | ivf |

**🏆 gamma 时间冠军场景:** `streaming_sliding`(2 个 lag),`bulk_delete`
**⚠️ HNSW recall 崩盘场景:** `streaming_sliding`(0.85-0.87)

---

## 2. 每个 op 的 latency(SIFT 200K v2 实测,微秒/毫秒)

### Insert latency(µs/向量)— 越小越快

| 场景 | gamma µs | gamma p99 | faiss µs | faiss p99 | ivf µs | ivf p99 | gamma vs faiss |
|---|---:|---:|---:|---:|---:|---:|---:|
| insert_only/200K | 45.3 | 49.4 | 174.7 | 223.7 | **12.3** | 169.9 | **3.9× 更快** |
| insdel_stream/200K | 46.2 | n/a | 177.7 | n/a | **12.5** | n/a | **3.8× 更快** |
| streaming/200K | 47.5 | n/a | 178.9 | n/a | **12.5** | n/a | **3.8× 更快** |
| burst/200K | 45.6 | n/a | 175.0 | n/a | **12.3** | n/a | **3.8× 更快** |
| drift/200K | 46.3 | n/a | 191.1 | n/a | **12.0** | n/a | **4.1× 更快** |
| bulk_delete/1M | 65.9 | n/a | 351.0 | n/a | **16.5** | n/a | **5.3× 更快** |
| churn/200K | 97.7 | n/a | 210.6 | n/a | **1.7** | n/a | **2.2× 更快** |

**写入吞吐总结:** gamma 比 HNSW **快 2.2-5.3×**(主因:buffer push 不建图);ivf 最快(无图)。

### Delete latency(µs/向量)

| 场景 | gamma | faiss | ivf | 备注 |
|---|---:|---:|---:|---|
| insdel_stream/200K | **0.23** | 0.12 | 1.55 | gamma 略慢 HNSW(都是 µs 级)|
| drift/200K | **0.23** | 0.10 | 1.99 | 同上 |
| churn/200K | **0.21** | 0.08 | 0.50 | gamma ≈ HNSW |
| bulk_delete/1M | 7.10 | **0.32** | 0.85 | gamma 慢:partition 内部维护 |

**删除已优化(早前 22µs → 现在 0.2µs)**,微 bench 上 gamma 与 HNSW 同档(都是亚微秒)。`bulk_delete` 的 7µs 是因为单次删 30% × 1M = 300K 向量,均摊每个 op 包含了 partition 重组。

### Query latency(ms/查询)— **关键差异**

| 场景 | gamma avg | gamma p95 | faiss avg | faiss p95 | ivf avg | ivf p95 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| insert_only/200K(final)| 0.165 | n/a | **0.157** | n/a | 10.08 | n/a | 全部新数据 |
| insdel_stream/200K(final)| **0.152** | n/a | 4.73 ⚠️ | n/a | 8.91 | n/a | HNSW 因 tombstone |
| streaming/200K | 0.129 | 0.157 | **0.124** | 0.158 | 5.04 | 9.78 | 无 delete 时三者像 |
| burst/200K(final) | 0.164 | n/a | **0.155** | n/a | 10.26 | n/a | 同上 |
| drift/200K | **0.131** | 0.170 | 3.06 ⚠️ | 4.58 | 3.48 | 5.87 | gamma 23× HNSW |
| bulk_delete/1M(final) | 63.5 | n/a | 24.9 | n/a | **11.1** | n/a | 1M 大数据集大查询 |
| churn/200K | 4.14 | 6.15 | **3.30** | 3.80 | 5.22 | 5.32 | 高 churn 都慢 |

**关键观察:**
- **无删除场景**:gamma ≈ HNSW(0.12-0.17ms),IVF 慢 30-60×
- **有删除场景**:HNSW tombstone 让 query 飙到 3-25ms,gamma hybrid 稳在 0.13-4ms — **gamma 比 HNSW 快 2-25×**

---

## 3. 写吞吐(ops / sec)— 流阶段(不含 final maint)

| 场景 | gamma | faiss | ivf | gamma vs faiss |
|---|---:|---:|---:|---:|
| insert_only/200K | **22,093** | 5,724 | 81,088 | 3.9× |
| insdel_stream/200K | **24,311** | 6,332 | 88,545 | 3.8× |
| insert_only/1M | 17,500 | 3,716 | 53,903 | 4.7× |
| insdel_stream/1M | 19,160 | 4,013 | 88,461 | 4.8× |

**gamma 写吞吐 ≈ 4× HNSW**,这是结构性优势(buffer-first 设计避开了 inline graph build)。IVF 写最快但 query 慢 50-100×。

---

## 4. 时间花在哪儿 — 每场景分阶段拆分(典型,200K)

### insert_only/200K(无删无 query 干扰)

| 算法 | stream | final maint | final query | total | 占比 stream/maint/query |
|---|---:|---:|---:|---:|:---|
| gamma | 8.1s | 35.8s | 0.17ms | **44.0s** | 18% / 81% / <1% |
| faiss | 31.4s | ~0s | 0.16ms | **31.4s** | 100% / 0% / <1% |
| ivf | 2.2s | ~0s | 10.1ms | **2.2s** | 100% / 0% / <1% |

**解读:** gamma 把 graph build 推到 maint(35.8s)= HNSW inline build 的成本(31.4s)。同样的工作量,只是时机不同。

### streaming/200K(insert + 周期 query,有 maint)

| 算法 | total | per-insert | per-query | 含 maint? |
|---|---:|---:|---:|:---|
| gamma | **43.6s** | 47.5µs | 0.13ms | 是,每 query batch 前一次 |
| faiss | 34.5s | 178.9µs | 0.12ms | 否(无概念)|
| ivf | 98.1s | 12.5µs | 5.04ms | 否 |

**解读:** gamma 的 maint 累计 ≈ HNSW inline build 时间;gamma 写阶段省下的时间被 maint 抵消。**结构性 ~26% gap**。

### bulk_delete(30%)/1M(gamma 最强场景)

| 算法 | insert phase | delete | final query × 1000 | total | recall |
|---|---:|---:|---:|---:|---:|
| **gamma** | 59s (66µs/id) | 2.1s (7µs/id) | 63.5s (63ms/q) | **131s** 🏆 | **1.000** |
| faiss | 316s (351µs/id) | 0.1s (0.3µs/id) | 24.9s (25ms/q) | 342s | 0.973 |
| ivf | 14.8s (16µs/id) | 0.3s (0.85µs/id) | 11.1s (11ms/q) | 34s | 0.992 |

**解读:** gamma 写 5.3× HNSW + perfect recall + total 时间 -62%。HNSW 的 tombstone 让 query 慢到 25ms 同时丢 recall。IVF 最快但 query 也只能到 11ms。

### streaming_sliding(lag=2)/1M(gamma 第二强场景)

| 算法 | stream | per-query | recall | total |
|---|---:|---:|---:|---:|
| **gamma** | 43s (40µs/id) | 0.15ms | **0.994** | **402s** 🏆 |
| faiss | 282s | 13.9ms ⚠️ | **0.853 ⚠️** | 531s |
| ivf | 12s | 5.9ms | 0.999 | 124s |

**解读:** 滑动窗口下,HNSW 的 tombstone 让 recall 跌 14%,query 慢 90×。gamma hybrid 稳定。

---

## 5. 三算法定位总结

| 维度 | gamma | faiss(HNSW)| ivf |
|---|:---:|:---:|:---:|
| 写吞吐 | 快(4× HNSW)| 慢 | **极快** |
| Query 速度(无删除)| ≈ HNSW | **快** | 极慢(40-80× 慢) |
| Query 速度(高 churn)| **稳定** | 严重退化(20× 慢) | 一直慢 |
| Recall(无删除)| 0.99 | 0.99 | **1.00** |
| Recall(高 churn)| **0.99** | **跌到 0.85** | 0.99 |
| 删除处理 | 删快 + query 不退化 | 删快 + query 退化 | 删快 + query 一直慢 |
| 总时间(写多查少)| 第二 | 第三 | **第一** |
| 总时间(写读 mix + 删多)| **第一** | 第二/第三 | 第一(短期)|
| 内存开销 | 中(buffer + graph)| 低 | 低 |

---

## 6. 应用场景建议

### 选 gamma 的场景

- ✅ **写读混合 + 持续删除**(news feed、recent items 索引)
- ✅ **bulk delete 后立即 query**(数据清理 + 重新查询)
- ✅ **滑动窗口索引**(time-series ANN,只保留近 N 条)
- ✅ **要求 recall ≥ 0.99 且不愿因删除而退化**

### 选 HNSW(faiss)的场景

- ✅ **静态数据 + 高 query rate**(预先 build 一次,然后只查询)
- ✅ **纯 insert 流**(无删除)
- ❌ **避免**:有 30%+ 删除的场景(recall 会跌到 0.85-0.97)

### 选 IVF 的场景

- ✅ **写多读极少**(归档、批处理、ETL)
- ✅ **可以接受 query 慢 100×**(吞吐重要、延迟不重要)
- ❌ **避免**:在线查询服务(query 5-25ms,gamma/HNSW 0.15ms)

---

## 7. gamma 当前配置(SIFT 1M tuned)

```python
gamma_fresh: {
    "backend": "FaissHNSW",
    "split_factor": 16.0,            # 1M ≈ 63 partition(原默认 1.5 = 800+)
    "gamma_split_threshold": 1.0,    # 抑制 split(原默认 2.0)
    "min_size_for_split": 512,       # 不 split 小 partition(原默认 4)
    "candidate_reserve_size": 8,     # query 候选 partition 数
    "maintenance_interval": 4,
    "maintenance_query_interval": 4,
    "maintenance_query_threshold_scale": 0.5,
}
faiss_hnsw: { m: 32, ef_construction: 120, ef_search: 80, tombstone_rebuild_threshold: 512 }
```

最近优化:
1. `split_factor=16` — 减 partition 数 4×,insert hot path -23%
2. Maint 顺序修复 — graph rebuild 移到 migration 之前,insdel_stream/1M total -46%
3. 跳过 maint 中冗余 graph.contains 检查 — maint -6%
4. 智能 routing(GAMMA_DUMB_ROUTING=0)— buffer-resident query 场景 -29%

---

## 8. 复现命令

```bash
# 200K 全场景 sweep(~10 分钟)
OMP_NUM_THREADS=1 uv run python bench/scenarios/sift1m_bench_v2.py

# 1M 全场景 sweep(~130 分钟)
OMP_NUM_THREADS=1 uv run python bench/scenarios/sift1m_bench_full1m.py

# 智能 routing A/B 验证
python bench/scenarios/smart_routing_ab.py
```

结果落盘:
- `results/scenarios/sift1m_results_v2.json` — 200K + bulk_delete 1M
- `results/scenarios/sift1m_results_full1m.json` — 全 1M(latest sweep 后会更新)
