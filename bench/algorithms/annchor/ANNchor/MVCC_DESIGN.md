# ANNchor: Batch-MVCC for Concurrent HNSW

## 1. 背景

### 1.1 HNSW 算法简介

HNSW (Hierarchical Navigable Small World) 是当前最主流的近似最近邻搜索算法，被 Milvus、Pinecone、Qdrant、Weaviate 等向量数据库广泛采用。

**核心结构 - 多层跳表式图**:

```
Layer 2:    [0] ─────────────────── [5]                 (稀疏，长距离跳跃)
             │                       │
Layer 1:    [0] ───── [3] ───── [5] ───── [8]           (中等密度)
             │         │         │         │
Layer 0:    [0]─[1]─[2]─[3]─[4]─[5]─[6]─[7]─[8]─[9]     (稠密，全量节点)
```

- 每个节点在 Layer 0 必定存在，以指数衰减概率出现在更高层
- 搜索从最高层开始，贪心逼近目标，逐层下降细化
- 每个节点最多 M 个邻居（M 通常为 16-64）

**插入流程**:

```
insert(new_node):
    1. 随机决定 new_node 的最高层 L
    2. 从顶层贪心搜索，找到每层的 entry point
    3. 在 Layer 0 用 beam search 找 efConstruction 个最近候选
    4. 从候选中选 ≤M 个作为邻居（启发式剪枝）
    5. 双向连接：new_node → neighbors, neighbors → new_node
    6. 若邻居的度数超过 M，触发邻居侧剪枝
```

**关键操作 - 剪枝 (Prune)**:

当节点 X 需要连接新邻居但度数已满时，需要从现有邻居中剔除一些：

```
prune(X, candidates):
    selected = []
    for candidate in sorted(candidates, by=distance_to_X):
        if len(selected) >= M:
            break
        if not is_dominated(candidate, selected):  # 启发式判断
            selected.append(candidate)
    X.neighbors = selected
```

剪枝的本质：新插入的节点可能"替代"某些旧邻居。

### 1.2 并发挑战

**问题场景**:

```
初始: Node X 的邻居 = [A, B, C, D]

Thread 1 (插入 P):                    Thread 2 (查询 Q):
─────────────────────                 ─────────────────────
1. P 搜索邻居，找到 X
2. P.neighbors.add(X)
3. X.neighbors.add(P)                 3'. Q 遍历 X.neighbors
4. X 度数超 M，剪枝                       → 读到 [A, B, C, P] 还是 [A, B, C, D]?
5. X.neighbors = [A, B, C, P]             → 不一致！
   (D 被删除)
```

**现有并发方案**:

| 方案 | 实现 | 问题 |
|------|------|------|
| 全局锁 | `mutex.lock(); insert(); mutex.unlock();` | 完全串行，吞吐 ≈ 0 |
| 节点级锁 | 每个节点一把锁 | 死锁风险（A→B, B→A），锁开销大 |
| 分层锁 | 先锁高层再锁低层 | 仍有竞争，实现复杂 |
| Lock-free | CAS 更新邻居列表 | ABA 问题，无法原子更新多个节点 |

**核心矛盾**:

- 读操作：遍历多个节点的邻居列表（路径搜索）
- 写操作：修改多个节点的邻居列表（双向连接 + 剪枝）
- 一次插入涉及 O(efConstruction × M) 次邻居列表修改
- 读写高度交织，传统锁方案难以兼顾正确性与性能

### 1.3 Batch 并发模型

我们采用 **Batch-Level 并发**，而非单点并发：

```
时间 ─────────────────────────────────────────────────────────────────►

Batch 0: ████████████████                        插入 nodes [0, 1000)
              Batch 1: ████████████████          插入 nodes [1000, 2000)
                            Batch 2: ████████████████    插入 nodes [2000, 3000)

Query Q1 (visible_batch=0): ──────●              只能看到 Batch 0
Query Q2 (visible_batch=1): ────────────●        能看到 Batch 0, 1
Query Q3 (visible_batch=2): ──────────────────●  能看到 Batch 0, 1, 2
```

**语义定义**:

- 每个 Batch 包含一组节点的插入
- 同一 Batch 内可并行（分区/细粒度同步）
- 查询携带 `visible_batch` 参数，只能看到 `batch_id ≤ visible_batch` 的节点
- **快照隔离 (Snapshot Isolation)**: 查询看到的图是某个时间点的一致快照

**为什么 Batch 模型**:

1. 实际场景：数据通常批量导入（ETL、streaming micro-batch）
2. 简化并发：Batch 边界是天然的同步点
3. 版本粒度：按 Batch 而非单点管理版本，开销可控

### 1.4 核心问题：边的版本一致性

**场景复现**:

```
初始状态:
  Node X.neighbors = [A, B, C, D]    (M=4, 已满)
  A, B, C, D 都属于 Batch 0

Batch 1 插入节点 P:
  Step 1: P 搜索邻居 → X 是候选
  Step 2: P.neighbors.add(X)         ✓ 正常
  Step 3: X.neighbors.add(P)         ✗ X 已满！
  Step 4: 剪枝判断 → D 可被 P 替代
          (P 离 X 更近，或 P "覆盖" D 的方向)
  Step 5: X.neighbors = [A, B, C, P]
          边 X→D 被删除！

问题出现:
  Query Q (visible_batch=0):
    - Q 不应该看到 P (P ∈ Batch 1)
    - 但 Q 遍历 X.neighbors 时只看到 [A, B, C, P]
    - D 不见了！
    - 从 Q 的视角，X→D 这条边"不应该被删除"
    - 搜索质量下降，recall 降低
```

**问题本质**:

```
边 X→D 被删除
  └── 原因：P 的插入触发了剪枝
        └── P ∈ Batch 1

对于 visible_batch < 1 的查询:
  - P 不可见 → 删除 D 的"原因"不存在
  - X→D 应该仍然存在
  - 需要"恢复"这条边
```

这就是 **边的版本一致性** 问题：同一条边，对不同 visible_batch 的查询，可见性不同。

### 1.5 MVCC 思路

**MVCC (Multi-Version Concurrency Control)**:

- 不直接覆盖/删除旧数据
- 保留多个版本，每个版本有时间戳/版本号
- 读操作根据自己的"可见性"选择对应版本
- 类似数据库的 Snapshot Isolation

**在 HNSW 中的应用**:

```
传统做法:
  X.neighbors = [A, B, C, P]        // D 直接消失

MVCC 做法:
  X.neighbors = [A, B, C, P]        // 当前版本
  X.deleted_edges = [
    { neighbor: D, caused_by: P, batch_id: 1 }
  ]

查询时:
  Q.visible_batch = 0
  for n in X.neighbors:
      if get_batch(n) <= 0: use(n)  // A, B, C ✓, P ✗
  for e in X.deleted_edges:
      if get_batch(e.caused_by) > 0:
          use(e.neighbor)           // D ✓ (恢复)
```

---

## 2. 要解决的问题

### 2.1 问题一：版本信息存储

**需要存储的信息**:

当边 `X→D` 因节点 P 的插入被剪枝时：

| 字段 | 含义 | 用途 |
|------|------|------|
| `node_id = X` | 哪个节点的邻居被删了 | 索引 |
| `deleted_neighbor = D` | 被删的邻居是谁 | 恢复时使用 |
| `caused_by = P` | 谁导致的删除 | 判断是否恢复 |
| `batch_id = 1` | 什么时候删的 | 版本过滤、GC |

**挑战**:

1. **空间效率**: 边的数量 = O(N × M)，版本信息不能太大
2. **访问效率**: 搜索路径上每个节点都要检查，必须快
3. **稀疏性**: 大部分边没有被删除，不应为所有边分配空间

**设计选项**:

```
Option A: 每条边存完整信息
  overhead = 12 bytes/edge (neighbor 4B + caused_by 4B + batch 4B)
  1M nodes × 32 edges = 384 MB overhead  ✗ 太大

Option B: Bitmap + 仅存被删边
  deleted_mask: 8 bytes/node (支持 M≤64)
  deleted_info: 仅被删边分配
  大部分节点无删除 → 开销极小  ✓

Option C: Delta 编码
  caused_by 通常与 node_id 接近（局部性）
  用 int16_t delta 代替 uint32_t id → 省 50%  ✓
```

### 2.2 问题二：可见性判断

**核心逻辑**:

```python
def should_restore(deleted_edge, visible_batch):
    """判断被删边是否应该恢复"""
    caused_by_batch = get_batch(deleted_edge.caused_by)

    if caused_by_batch > visible_batch:
        # caused_by 不可见 → 删除不应发生 → 恢复
        return True
    else:
        # caused_by 可见 → 删除合理 → 保持删除
        return False
```

**挑战**:

1. **get_batch() 效率**: 每次判断都要查 caused_by 的 batch，需要 O(1)
2. **批量判断**: 一个节点可能有多条被删边，逐条判断效率低
3. **分支预测**: if-else 分支可能导致 pipeline stall

**设计选项**:

```
Option A: Hash map
  batch_map[caused_by] → batch_id
  O(1) 但有 hash 开销

Option B: 数组直接索引
  batch_array[node_id] = batch_id
  O(1) 且 cache-friendly  ✓

Option C: SIMD 批量比较
  一次比较 8 个 caused_by_batch vs visible_batch
  3 条指令 vs 8 次分支  ✓
```

### 2.3 问题三：内存管理

**生命周期**:

```
1. 边被删除 → 分配 DeletedEdgeInfo
2. 查询使用 → 读取 DeletedEdgeInfo
3. 版本过期 → 回收 DeletedEdgeInfo (GC)
```

**何时可以 GC**:

```
当 min(all_active_queries.visible_batch) > edge.batch_id 时:
  - 所有查询都能看到 caused_by
  - 所有查询都认为删除合理
  - 不再需要恢复信息
  - 可以安全删除 DeletedEdgeInfo
```

**挑战**:

1. **安全回收**: 确保没有查询正在读取
2. **高效分配**: 版本信息频繁创建，需要快速分配
3. **内存复用**: 避免碎片化，回收的内存应复用

**设计选项**:

```
Option A: 引用计数
  每个 DeletedEdgeInfo 有 ref_count
  问题：原子操作开销，cache line bouncing  ✗

Option B: Epoch-Based Reclamation (EBR)
  按 epoch 批量回收，无需 per-object 追踪
  3 个 epoch 轮转，读者进入时记录 epoch
  当所有读者离开某 epoch，该 epoch 的对象可回收  ✓

Option C: Block Pool
  预分配固定大小块，free list 管理
  O(1) 分配/回收，无碎片  ✓
```

---

## 3. 初步设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         VersionManager                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │BatchRegistry │  │ VersionIndex │  │ VisibilityChecker    │  │
│  │              │  │              │  │                      │  │
│  │ node→batch   │  │ Two-Level    │  │ SIMD accelerated     │  │
│  │ O(1) lookup  │  │ sparse index │  │ batch comparison     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │  BlockPool   │  │     EBR      │                             │
│  │              │  │              │                             │
│  │ Memory reuse │  │ Safe GC      │                             │
│  └──────────────┘  └──────────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 数据结构设计

**BatchRegistry**: node ↔ batch 双向映射

```cpp
class BatchRegistry {
    vector<int16_t> node_to_batch_;      // node_id → batch_id, O(1)
    vector<pair<uint32_t, uint32_t>> batch_ranges_;  // batch → [start, end)
};
```

空间: 2 bytes/node

**NodeVersionInfo**: 每节点版本信息

```
┌─────────────────────────────────────────────────────────────┐
│                NodeVersionInfo (64 bytes)                    │
├─────────────────────────────────────────────────────────────┤
│ Tier 1: deleted_mask (8B)                                   │
│         bit[i] = 1 表示 neighbor[i] 被删了                   │
├─────────────────────────────────────────────────────────────┤
│ Tier 2: base_node_id (4B) + deleted_count (2B) + pad (2B)  │
├─────────────────────────────────────────────────────────────┤
│ Tier 3: details* (8B)                                       │
│         指向 DeletedEdgeCompact 数组                         │
├─────────────────────────────────────────────────────────────┤
│ Tier 4: caused_by_batches[8] (32B)                          │
│         预存前 8 个被删边的 caused_by batch，用于 SIMD        │
└─────────────────────────────────────────────────────────────┘
```

**DeletedEdgeCompact**: 被删边详情

```cpp
struct DeletedEdgeCompact {  // 6 bytes
    uint8_t slot_index;       // 邻居列表中的位置
    int16_t caused_by_delta;  // caused_by - base_node_id (delta 编码)
    int16_t batch_id;         // 删除时的 batch
};
```

**VersionIndex**: Two-level 稀疏索引

```
Level 0: block_masks[]     每个 uint64_t 管理 64 个节点
         bit = 1 表示该节点有版本信息

Level 1: blocks[]          NodeVersionInfo[64] 数组
         只有 Level 0 标记的 block 才分配
```

空间节省: 若 10% 节点有版本 → 节省 84%

### 3.3 可见性计算

**标量版本**:

```cpp
bool should_restore(int caused_by_batch, int visible_batch) {
    return caused_by_batch > visible_batch;
}
```

**SIMD 版本 (AVX2)**:

```cpp
uint8_t compute_restore_mask_simd(const int32_t* caused_by_batches,
                                   int visible_batch) {
    __m256i batches = _mm256_loadu_si256(caused_by_batches);
    __m256i threshold = _mm256_set1_epi32(visible_batch);
    __m256i cmp = _mm256_cmpgt_epi32(batches, threshold);
    return _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
}
// 返回 8-bit mask, bit[i]=1 表示第 i 条边需要恢复
```

**最终计算**:

```cpp
uint64_t get_effective_mask(uint32_t node_id, int visible_batch) {
    NodeVersionInfo* info = index.get(node_id);
    if (!info || !info->has_deletions()) {
        return ~0ULL;  // 无删除，全部可用
    }

    uint64_t restore_mask = compute_restore_mask(info, visible_batch);
    return ~info->deleted_mask | restore_mask;
    // 有效 = 未删除 OR 需要恢复
}
```

### 3.4 内存管理

**EBR (Epoch-Based Reclamation)**:

```cpp
class EBR {
    atomic<uint32_t> global_epoch;      // 0, 1, 2 轮转
    vector<ThreadState> thread_states;  // 每线程的 local_epoch

    void enter() {
        local_epoch = global_epoch | ACTIVE_FLAG;
    }

    void exit() {
        local_epoch = 0;
    }

    bool sync(unsigned& gc_epoch) {
        // 检查所有线程是否都在当前 epoch
        // 若是，推进 global_epoch，返回可 GC 的 epoch
    }
};
```

**BlockPool**:

```cpp
class BlockPool {
    size_t block_size;
    vector<char*> chunks;      // 预分配的大块内存
    vector<char*> free_list;   // 可用块列表

    void* allocate() {
        if (free_list.empty()) expand();
        return free_list.pop_back();
    }

    void deallocate(void* ptr) {
        free_list.push_back(ptr);  // 回收复用
    }
};
```

### 3.5 与 HNSW 集成

**插入时**:

```cpp
void addPoint(data, label) {
    // ... 正常插入流程 ...

    // 剪枝时记录被删边
    for (auto& pruned : pruned_neighbors) {
        version_mgr.record_deletion(
            node_id,
            pruned.slot,
            new_node_id,  // caused_by
            current_batch
        );
    }
}
```

**搜索时**:

```cpp
void searchLayer(query, entry, layer, visible_batch) {
    while (!candidates.empty()) {
        auto curr = candidates.pop();

        // 获取有效邻居 mask
        uint64_t mask = version_mgr.get_effective_mask(curr, visible_batch);

        for_each_visible_neighbor(neighbors, mask, [&](uint32_t neighbor) {
            // 只处理有效邻居
            float dist = distance(query, neighbor);
            candidates.push(neighbor, dist);
        });
    }
}
```

---

## 4. 当前实现

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/hnswlib/version_storage.h` | 框架完成 | 存储 + 查询主体逻辑 |
| `src/hnswlib/block_pool.h` | 完成 | 内存池 |
| `src/hnswlib/ebr.h` | 完成 | Epoch-Based Reclamation |
| `src/hnswlib/hnswalg.h` | 待集成 | 需对接 record_deletion / get_effective_mask |

---

## 5. TODO

1. **version_storage.h 细节补全**
   - `record_deletion()` 中 details 分配（使用 BlockPool）
   - `expand_simd_mask()` slot 索引映射
   - `compute_restore_scalar()` 完整实现

2. **hnswalg.h 集成**
   - 插入路径添加 `record_deletion()` 调用
   - 搜索路径使用 `get_effective_mask()` 过滤

3. **GC 流程**
   - EBR 与 BlockPool 对接
   - 按 epoch 批量回收版本信息

4. **测试与评估**
   - 正确性：多 batch 并发插入 + 不同 visible_batch 查询
   - 性能：版本管理开销、内存占用、recall 影响
