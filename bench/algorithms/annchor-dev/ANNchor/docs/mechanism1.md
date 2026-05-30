# Mechanism 1: MVCC Protocol + Undo Log + GC

## 概述

Mechanism 1 实现了 HNSW 图上的多版本并发控制（MVCC），让 insert 和 search 可以并发执行而不需要锁。对应 Wu et al. 的四个 MVCC 维度：

| 维度 | ANNchor 设计 |
|------|-------------|
| Concurrency Control Protocol | Batch Snapshot Isolation |
| Version Storage | Delta Storage, N2O (Newest-to-Oldest) |
| Garbage Collection | Inline GC (insert + search 顺手清) + Compact 兜底 |
| Index Management | 图即索引，无辅助结构 |

## 组件

### 1. Batch Snapshot Isolation

- **Watermark**: `global_watermark_` — 单调递增的逻辑时间戳，等于最新连续完成插入的 node ID
- **batch_ts**: 每个 batch 的时间戳 = `start_id + num_points - 1`（batch 最后一个元素的 ID）
- **view_limit**: search 的可见边界，`id < view_limit` 的节点才可见
- **advanceWatermark()**: 扫描 `insertion_complete_[]` 找到连续完成的最大 ID，CAS 推进

### 2. SnapshotTracker（`hnswalg.h`）

256 个 cache-line aligned slots，lock-free：
- search 开始 → `allocate_slot()` + `Guard(tracker, slot, view_limit)`
- search 结束 → Guard 析构自动释放
- `min_active_snapshot()` → 扫 256 slots 找最小值，用于 GC horizon

### 3. Undo Log（`undo_log.h`）

Delta Storage：只存被 prune 的边（不存完整 neighbor list）。

```
当前 neighbor list: [A, B, C, E]   ← 主存储，始终是最新版本

undo log (delta):
  batch=500: prune D, cause=E      ← "D 被删了，因为 E 插入"
  batch=300: prune F, cause=C      ← "F 被删了，因为 C 插入"
```

Search 要看 batch=400 的版本：
1. 读当前 list `[A, B, C, E]`
2. 从 undo log 恢复 `batch > 400` 的 prune → 把 D 加回来
3. 得到 `[A, B, C, D, E]`

每条 entry 存储：
- `edge`: 被删的邻居 ID
- `cause`: 导致删除的新插入节点 ID
- `batch_ts`: 发生的 batch 时间戳
- `dist_node_pruned`, `dist_cause_pruned`: prune 时计算的距离（为 Mechanism 2 预留）
- `rcs_sketch`: RCS 草图（为 Mechanism 2 预留，当前传 nullptr）

### 4. Node Modified Bitmap（`hnswalg.h`）

两级查询优化：
- **第一级 bitmap**: 1 bit/node，标记是否被 prune 过。1M 节点 = 128KB（L2 cache）
- **第二级 undo log**: 只有 bitmap=1 的节点才查 undo log

99% 的节点在第一级就返回（没被改过，直接读当前 neighbor list）。

### 5. GC（三路并行）

**gc_horizon_**: 每个 batch 开始时更新 = `min_active_snapshot()`，即当前最老的活跃 search 的 snapshot。

| 路径 | 触发时机 | 做什么 |
|------|---------|--------|
| Insert inline | recordPrune 后 | `advanceGcStart(node, gc_horizon_)` |
| Search inline | 遍历节点时 | `advanceGcStart(node, gc_horizon_)` |
| Compact | 手动调用 | 全量物理回收或按 safe_ts 部分回收 |

GC 分两步：
1. **逻辑截断**: `advanceGcStart()` — lock-free（atomic CAS），推进 `gc_start_`，任何线程可调
2. **物理回收**: `compact()` / `compactAll()` — `vector::erase`，只在单线程 compact 时调

### 6. Compact

```cpp
compact();        // 全删 undo log + bitmap 清零
compact(50000);   // 删 batch_ts < 50000 的 log，保留 >= 50000 的，bitmap 重建
```

## 文件清单

| 文件 | 改动 |
|------|------|
| `src/hnswlib/undo_log.h` | 新增。NodeUndoLog + UndoLogManager |
| `src/hnswlib/hnswalg.h` | 增加 SnapshotTracker、bitmap、gc_horizon、compact；修改 advanceWatermark、mutuallyConnectNewElement、searchBaseLayerST |

## 数据流

```
Insert:
  1. batch 开始 → advanceWatermark → gc_horizon_ = min_active_snapshot()
  2. 插入节点 → prune 旧边 → recordPrune() + markNodeModified() + inline GC
  3. markCompleted → advanceWatermark → watermark 推进

Search:
  1. 注册 snapshot (Guard)
  2. 遍历节点:
     a. bitmap 检查 → 未修改 → 直接读当前 neighbors
     b. bitmap=1 → 查 undo log → 恢复被 prune 的边加入候选集
     c. inline GC: advanceGcStart(node, gc_horizon_)
  3. Guard 析构 → 释放 snapshot slot
```
