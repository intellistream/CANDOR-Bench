# Related work positioning (from prior-art subagent, 2026-05-10)

Drop-in §2 Related Work for the paper. Three areas, then the positioning
paragraph.

---

## Area 1 — Streaming-friendly graph indexes

| Paper / system | Venue, year | What it does | Failure mode | GammaFresh differs by |
|---|---|---|---|---|
| **HNSW + hnswlib `mark_deleted`** | TPAMI 2018 (Malkov & Yashunin); library ongoing | Multi-layer proximity graph; `mark_deleted` flags a node, search filter-during-search keeps it as nav hub | Tombstones never decrement in-degree → under non-FIFO patterns dead hubs accumulate, search visits more dead nodes per query (latency drift). No built-in compaction trigger; users must do external rebuild. | Adds *router-level* delete-absorption: a delete that catches its target still in the buffer is just nulled, never reaches `mark_deleted`. The graph sees fewer total tombstones → filter-during-search overhead grows slower under random/cluster/bursty (e15 -22% to -36%). |
| **DiskANN / Vamana** | NeurIPS 2019 | Single-layer pruned proximity graph for SSD-resident search; static index | Not streaming-friendly; no deletes in original design | Doesn't compete; differs from FreshDiskANN below. |
| **FreshDiskANN** | arXiv 2105.09613, 2021 | In-RAM "FreshVamana" graph for new vectors; deletes go to a DeleteList; periodic *consolidate* pass rebuilds neighborhoods around deleted vertices in batches; periodic StreamingMerge folds in-RAM into SSD index | Batched consolidation interacts poorly with clustered/bursty runbooks; threshold-driven trigger is workload-blind | (i) Backend-agnostic router (we run on hnswlib AND Faiss HNSW); FreshDiskANN's buffer is tied to Vamana edge-update primitives. (ii) Our consolidation trigger is observed graph-side **tombstone fraction**, not deletion-count threshold. (iii) Our delete-absorption short-circuits inserts that died in-buffer; FreshDiskANN's in-RAM tier is itself a Vamana graph, so a delete still updates edges. |
| **IP-DiskANN** (In-Place Updates) | arXiv 2502.13826, Feb 2025 | Replaces FreshDiskANN's batched consolidation with in-place edge updates per-deletion via reverse-neighbor approximation | Per-delete CPU cost; requires custom Vamana modification; not yet ported to HNSW | IP-DiskANN attacks consolidation-cost axis; GammaFresh attacks tombstone-pollution axis with a buffer above an unmodified backend. **Complementary** — GammaFresh + IP-DiskANN-as-backend would compose. |
| **Greator** (Topology-Aware Localized Update) | PVLDB 19:495, 2025; arXiv 2503.00402 | Disk-resident graph index with page-level fine-grained updates; rewrites only SSD pages touching affected vertices. Up to 4.16× faster than FreshDiskANN-style consolidation. | Tightly coupled to disk layout; topology-copy memory cost grows with index | In-memory only and backend-agnostic; we don't optimize disk I/O — we eliminate work via buffer-side cancellation. Different bottleneck axis (I/O vs CPU under tombstone load). |

## Area 2 — Hybrid / two-tier ANN systems

| Paper / system | Venue, year | What it does | Differs from GammaFresh by |
|---|---|---|---|
| **SPANN** | NeurIPS 2021 | Memory-disk hybrid: hierarchical balanced clustering + IVF posting lists; centroids in RAM, posting lists on SSD | Solves *static* billion-scale memory budgets, not churn. No buffer; no delete-absorption. |
| **SPFresh** | SOSP 2023 (NOT SIGMOD) | Adds streaming to SPANN via LIRE: on update, only reassign vectors at the IVF *boundary* between partitions rather than rebuild the whole partition | LIRE assumes IVF backend with partition centroid + boundary; doesn't apply to graph indexes. SPFresh = "rebuild less of the IVF partition by recognizing only the boundary moved"; GammaFresh = "absorb the delete in a buffer above the graph." Architecturally orthogonal. |
| **Pinecone Serverless** (vendor blog 2024-2025) | — | Two-tier: *freshness layer* tails the WAL and builds a small in-memory index over un-flushed writes; query router fans out to (freshness + slab-resident static index); slabs are immutable in object storage. Memtable+slab+index-builder design borrowed from LSMs. | Pinecone's freshness tier is *throughput-and-latency* (decouple writes from immutable index builds); deletes handled via slab rewrites at object-storage level, not in hot search path. GammaFresh's contribution is in the *hot search-and-delete path* on a single node. |
| **Milvus growing/sealed segment** | open-source 2019- | Growing segment receives streaming inserts; once it reaches threshold it's sealed, flushed, and indexed (HNSW/IVF). Deletes are tombstones; *compaction* triggered when segment's deleted ratio passes ~20%. | Compaction at segment granularity (~123 MB); search-side cost paid until threshold trips. GammaFresh operates per-vector in the buffer and absorbs deletes *before* they touch the indexed tier. |
| **Qdrant Vacuum / Weaviate async cleanup** | open-source | External *async cleanup* worker rebuilds segments above tombstone threshold (Qdrant) or re-stitches HNSW graph (Weaviate `cleanupIntervalSeconds`) | These are external compaction loops on an unmodified graph — every insert and every delete still touches the graph. GammaFresh adds the *write-side buffer* and the *cancellation pathway*. **Complementary** — combining the two would help further. |
| **UBIS** | IEEE BigData 2025 | Critiques naive "buffer + global rebuild" under high update rates; proposes balanced index that schedules concurrent updates and prevents centroid imbalance. +77% accuracy, +45% update throughput vs SOTA. | UBIS targets update *concurrency conflicts and centroid drift* in IVF; GammaFresh targets *tombstone pollution* in graph indexes via buffer-cancellation contract. Both critique naive buffer+rebuild from different angles (theirs: balance/scheduling; ours: selective absorption + tombstone-rebuild trigger). |

## Area 3 — Cost-model / lifetime-aware placement (architectural analogy, not direct prior art for ANN)

| Paper | Venue, year | What it does | Analogy to GammaFresh |
|---|---|---|---|
| **Lehman & Yao, B-link trees** | TODS 1981 | Hand-off concurrency: split locally, leave a `link` pointer; defer split fix-up | Canonical "deferred split" pattern. GammaFresh's "delete reaches buffer first, may never reach graph" is the same defer-the-expensive-write idea. |
| **Luo & Carey, Adaptive Memory Management in LSM** | PVLDB 14:3, 2021 | Treats LSM memtable budget as tunable; reallocates memory between write buffer and buffer cache by monitoring write rates per LSM-tree | Same knob as our `buf_mult` (e18 sensitivity sweep). LSM-side justification for "buffer capacity is a meaningful per-workload tunable." |
| **ADOC / RocksDB write-stall mitigation** | various, 2020- | Adaptive tuning of compaction concurrency / batch size to keep memtable flushes from causing write-stall cascades | Plays the same role as our `maint_interval` and tombstone-rebuild trigger — control when expensive synchronous flush happens. |
| **QVCache** | arXiv 2602.02057 | Per-query caching of mini-indexes with LRU/LFU eviction tailored to ANN access patterns | Read-side caching, not write-side buffering. We mention only because "heat-aware vector caching" exists in literature, but it's not a write/delete-path mechanism. |

**There is no published vector-DB paper that specifically learns or models per-vector lifetime to drive tier admission.** GammaFresh's `cost-model admit` module (e27) and the rich version's heat-aware promotion (e30/e31, in flight) are in this gap.

---

## §2 Related-Work positioning paragraph (drop-in draft)

> The streaming-ANN literature splits cleanly into two camps. The first
> is *in-place graph maintenance*: hnswlib's `mark_deleted` (Malkov &
> Yashunin 2018), FreshDiskANN's batched DeleteList consolidation (Singh
> et al. 2021), IP-DiskANN's per-delete edge updates (Xu et al. 2025),
> and Greator's page-level localized updates (Yu et al. PVLDB 2025). The
> second is *partition-level rebuild*: SPFresh's LIRE boundary-rebalance
> over SPANN (Xu et al. SOSP 2023), Milvus's segment compaction, and
> Qdrant/Weaviate's async tombstone cleanup. Both camps modify the
> indexed tier directly. Our work occupies a third design point — a thin
> **router** above an unmodified incremental graph — and our
> contribution is not the buffer alone (Pinecone's freshness layer,
> Milvus's growing segment, and FreshDiskANN's in-RAM Vamana also buffer
> recent writes) but the **delete-absorption contract** between the
> buffer and the graph: a delete that catches its target still in the
> buffer never reaches the graph at all, and a separate
> **tombstone-fraction-driven** maintenance trigger replaces the
> buffer-fill or count-based triggers used in prior systems.
> Algorithmically this is more than an engineering improvement because
> (i) the absorption pathway changes the *workload* the underlying graph
> sees from "insert then delete" to "insert" — a structurally easier
> problem that explains why our gains hold across hnswlib *and* Faiss
> HNSW backends in §5; (ii) the tombstone-fraction trigger is
> *backend-observable* rather than workload-observable, so the same
> router is correct under sequential / random / cluster / bursty
> patterns where FreshDiskANN's count threshold and Milvus's
> per-segment threshold each have a known weakness; and (iii) the router
> composes with existing in-place techniques (IP-DiskANN, Greator)
> rather than competing with them — the buffer reduces the rate of
> consolidation work, while in-place techniques reduce the cost per
> consolidation. The simplicity of the design (≈50 LOC of router state
> above any incremental ANN library) is the point: it is the smallest
> change to the standard streaming-ANN stack that explains the headline
> win across non-FIFO delete patterns, and the empirical contribution of
> this paper is the demonstration (e15-e19, e25, e29) that this small
> change is sufficient and that the more elaborate multi-partition C++
> machinery in our v1 design (e21) is not.

---

## Citations to add to the paper

Required:
- Malkov & Yashunin TPAMI 2018 (HNSW)
- DiskANN NeurIPS 2019
- FreshDiskANN arXiv 2105.09613
- IP-DiskANN arXiv 2502.13826 (very recent, must cite)
- Greator PVLDB 2025
- SPANN NeurIPS 2021
- SPFresh SOSP 2023
- UBIS IEEE BigData 2025
- Lehman & Yao TODS 1981
- Luo & Carey PVLDB 2021

Useful to mention:
- Milvus blog (segment compaction)
- Qdrant Vacuum docs / Weaviate cleanup docs
- Pinecone serverless blog
- VDBBench 1.0 (real-world benchmark suite for vector DBs)
