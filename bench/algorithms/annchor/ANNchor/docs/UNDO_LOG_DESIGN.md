# Undo Log Design for Batch-MVCC HNSW

## 1. Problem Statement

### 1.1 Background

In HNSW, when a new node P is inserted, it may trigger **pruning** on existing nodes. If node X's neighbor list is full (size = M), inserting P as X's neighbor requires removing an existing neighbor D:

```
Before: X.neighbors = [A, B, C, D]   (M=4, full)
After:  X.neighbors = [A, B, C, P]   (D removed due to pruning)
```

### 1.2 The MVCC Problem

In Batch-MVCC, queries carry a `visible_batch` parameter and should only see nodes with `batch_id <= visible_batch`:

```
Timeline:
  Batch 0: nodes [0, 1000)    ← D belongs here
  Batch 1: nodes [1000, 2000) ← P belongs here

Query Q with visible_batch=0:
  - Cannot see P (P ∈ Batch 1)
  - But D is gone from X.neighbors!
  - Edge X→D should still exist for Q
  - Need to "undo" the deletion
```

### 1.3 Key Insight

We need an **Undo Log** to restore deleted edges for queries that cannot see the node causing the deletion.

```
Undo Log Entry:
  - deleted_neighbor: D (who was removed)
  - caused_by_batch: batch(P) (when P was inserted)
  - slot_index: original position in neighbor list

Restore Logic:
  if caused_by_batch > visible_batch:
      restore the edge (P not visible → deletion didn't happen)
  else:
      keep deleted (P visible → deletion is valid)
```

---

## 2. Design Goals

| Goal | Requirement |
|------|-------------|
| **Space Efficiency** | Most nodes have NO deleted edges; avoid allocating for all |
| **Query Speed** | O(1) to skip nodes without undo log |
| **SIMD Friendly** | Batch compare 16 entries at once |
| **GC Support** | Efficiently reclaim obsolete entries |

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       UndoLogManager                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐      ┌──────────────────────┐         │
│  │    SparseUndoLog     │      │    BatchRegistry     │         │
│  │                      │      │                      │         │
│  │  Two-Level Index:    │      │  node_to_batch_[]    │         │
│  │  - block_mask_[]     │      │  batch_ranges_[]     │         │
│  │  - blocks_[]         │      │                      │         │
│  └──────────────────────┘      └──────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Structures





### 4.1 Two-Level Sparse Index

```
Level 0: block_mask_[]
┌────────────────────────────────────────────────────────────────┐
│ block_mask_[0] = 0b...00100010  → nodes 1, 5 have undo logs    │
│ block_mask_[1] = 0b...00000001  → node 64 has undo log         │
│ block_mask_[2] = 0b...00000000  → nodes 128-191: none          │
└────────────────────────────────────────────────────────────────┘
  Each uint64_t covers 64 nodes (1 bit per node)

Level 1: blocks_[]
┌────────────────────────────────────────────────────────────────┐
│ blocks_[0] → UndoLogBlock[64]   (allocated, has entries)       │
│ blocks_[1] → UndoLogBlock[64]   (allocated)                    │
│ blocks_[2] → nullptr            (not allocated, saves memory)  │
└────────────────────────────────────────────────────────────────┘
```

**Space Saving**: If 10% of nodes have undo logs → ~90% memory saved vs dense allocation.

### 4.2 UndoEntry (8 bytes)

```cpp
struct UndoEntry {
    uint32_t deleted_neighbor;   // D: the removed neighbor
    uint16_t caused_by_batch;    // batch(P): when deletion happened
    uint8_t  slot_index;         // original position in neighbor list
    uint8_t  flags;              // reserved
};
```

### 4.3 NodeUndoLog (per node, 64-byte aligned)

```
┌─────────────────────────────────────────────────────────────────┐
│ Fast Path Metadata (8 bytes)                                    │
│   min_caused_by_batch: uint16_t   ← for fast skip               │
│   max_caused_by_batch: uint16_t   ← for fast skip               │
│   count: uint16_t                                               │
├─────────────────────────────────────────────────────────────────┤
│ SIMD-Friendly Batch Array (64 bytes)                            │
│   caused_by_batches[32]: uint16_t[]                             │
│   ← Separate array for SIMD batch comparison                    │
├─────────────────────────────────────────────────────────────────┤
│ Entry Details (256 bytes)                                       │
│   entries[32]: UndoEntry[]                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 BatchRegistry

```cpp
class BatchRegistry {
    vector<uint16_t> node_to_batch_;     // node_id → batch_id, O(1) lookup
    vector<pair<uint32_t, uint32_t>> batch_ranges_;  // batch → [start, end)
};
```

Space: 2 bytes per node.

---

## 5. Algorithms

### 5.1 Recording Deletion (Insert Time)

```cpp
void on_prune(uint32_t X, uint32_t D, uint32_t P, uint8_t slot) {
    // Called when edge X→D is pruned due to P's insertion

    uint16_t caused_by_batch = batch_registry.get_batch(P);

    undo_log.record(X, D, caused_by_batch, slot);
}
```

### 5.2 Restoring Edges (Query Time)

```cpp
vector<uint32_t> get_visible_neighbors(uint32_t X, uint16_t visible_batch) {
    vector<uint32_t> result;

    // Step 1: Add current neighbors that are visible
    for (auto n : X.neighbors) {
        if (batch_registry.get_batch(n) <= visible_batch) {
            result.push_back(n);
        }
    }

    // Step 2: Restore edges from undo log
    const NodeUndoLog* log = undo_log.get(X);
    if (log) {
        uint32_t restore_mask = log->get_restore_mask(visible_batch);

        while (restore_mask) {
            int idx = __builtin_ctz(restore_mask);
            result.push_back(log->entries[idx].deleted_neighbor);
            restore_mask &= (restore_mask - 1);
        }
    }

    return result;
}
```

### 5.3 Fast Path Optimization

```cpp
int NodeUndoLog::fast_path_check(uint16_t visible_batch) const {
    if (count == 0)
        return 0;  // No entries → restore nothing

    if (visible_batch >= max_caused_by_batch)
        return 0;  // All causes visible → restore nothing

    if (visible_batch < min_caused_by_batch)
        return 1;  // No causes visible → restore all

    return 2;      // Mixed → need per-entry check
}
```

**Benefit**: Most queries hit fast path (0 or 1), avoiding per-entry comparison.

### 5.4 SIMD Batch Comparison (AVX2)

```cpp
uint32_t get_restore_mask_simd(uint16_t visible_batch) const {
    __m256i threshold = _mm256_set1_epi16(visible_batch);

    // Process 16 entries at once (256 bits / 16 bits = 16)
    __m256i batches = _mm256_loadu_si256(&caused_by_batches[0]);
    __m256i cmp = _mm256_cmpgt_epi16(batches, threshold);
    uint32_t byte_mask = _mm256_movemask_epi8(cmp);
    uint16_t result = _pext_u32(byte_mask, 0xAAAAAAAA);

    return result;
}
```

**Benefit**: 3 SIMD instructions vs 16 scalar comparisons + branches.

### 5.5 Garbage Collection

```cpp
void gc(uint16_t min_visible_batch) {
    // min_visible_batch = min(all_active_queries.visible_batch)

    for (each node with undo log) {
        if (log.max_caused_by_batch <= min_visible_batch) {
            // All entries obsolete → clear entire log
            log.reset();
        } else {
            // Compact: remove entries with caused_by_batch <= min_visible_batch
            log.compact(min_visible_batch);
        }
    }

    // Free empty blocks
    for (each block) {
        if (block is empty) {
            delete block;
            clear bitmap;
        }
    }
}
```

**When to GC**: When the minimum `visible_batch` across all active queries increases (e.g., old queries complete).

---

## 6. API Reference

### 6.1 UndoLogManager

```cpp
class UndoLogManager {
public:
    // Construction
    explicit UndoLogManager(size_t max_nodes);

    // Batch Management
    void register_batch(uint16_t batch_id, uint32_t start, uint32_t end);
    uint16_t get_node_batch(uint32_t node_id) const;

    // Record Deletion (called during pruning)
    bool record_deletion(uint32_t node_id, uint32_t deleted_neighbor,
                         uint32_t caused_by, uint8_t slot_index);

    // Query (called during search)
    bool has_undo(uint32_t node_id) const;
    uint32_t get_restore_mask(uint32_t node_id, uint16_t visible_batch) const;
    size_t get_restore_edges(uint32_t node_id, uint16_t visible_batch,
                             uint32_t* out_neighbors, uint8_t* out_slots) const;

    // Garbage Collection
    void gc(uint16_t min_visible_batch);

    // Statistics
    Stats get_stats() const;
};
```

### 6.2 Usage Example

```cpp
// Setup
UndoLogManager manager(1000000);  // 1M nodes

// Register batches during insertion
manager.register_batch(0, 0, 10000);
manager.register_batch(1, 10000, 20000);

// During pruning in addPoint()
manager.record_deletion(
    node_X,           // whose neighbor was removed
    neighbor_D,       // the removed neighbor
    new_node_P,       // the node that caused removal
    slot_index        // original slot in neighbor list
);

// During search
uint16_t visible_batch = 0;  // query's snapshot

if (manager.has_undo(node_X)) {
    uint32_t neighbors[32];
    uint8_t slots[32];
    size_t n = manager.get_restore_edges(node_X, visible_batch, neighbors, slots);

    for (size_t i = 0; i < n; i++) {
        // Add neighbors[i] back to search candidates
    }
}

// Periodic GC
manager.gc(min_active_visible_batch);
```

---

## 7. Complexity Analysis

### 7.1 Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `has_undo()` | O(1) | Bitmap lookup |
| `record_deletion()` | O(1) amortized | May allocate block |
| `get_restore_mask()` | O(1) fast path | O(n/16) SIMD worst case |
| `gc()` | O(nodes with undo) | Periodic, not per-query |

### 7.2 Space Complexity

| Component | Size | Notes |
|-----------|------|-------|
| BatchRegistry | 2 bytes/node | Dense array |
| block_mask_ | 1 bit/node | ~128 KB for 1M nodes |
| blocks_ | 8 bytes/node | Pointers only |
| UndoLogBlock | ~21 KB | Only allocated when needed |
| NodeUndoLog | ~330 bytes | Per node with undo entries |

**Example**: 1M nodes, 10% have undo logs, avg 3 entries each:
- BatchRegistry: 2 MB
- Sparse Index overhead: ~1 MB
- Actual undo data: 100K × 330 bytes ≈ 33 MB
- **Total: ~36 MB** (vs ~330 MB if dense)

---

## 8. Integration with HNSW

### 8.1 Insert Path

```cpp
// In hnswalg.h: mutuallyConnectNewElement() or similar

void prune_neighbors(uint32_t node_id, vector<uint32_t>& candidates, int M) {
    // ... existing pruning logic ...

    for (auto pruned : pruned_neighbors) {
        undo_log_manager_.record_deletion(
            node_id,
            pruned.neighbor_id,
            new_node_id,      // caused_by
            pruned.slot_index
        );
    }
}
```

### 8.2 Search Path

```cpp
// In hnswalg.h: searchBaseLayerST() or similar

void search_layer(query, entry_point, visible_batch) {
    while (!candidates.empty()) {
        auto curr = candidates.top();

        // Get current neighbors
        auto* neighbors = get_linklist(curr);

        // Filter by batch visibility
        for (int i = 0; i < neighbor_count; i++) {
            uint32_t neighbor = neighbors[i];
            if (batch_registry_.get_batch(neighbor) <= visible_batch) {
                // Process neighbor...
            }
        }

        // Restore edges from undo log
        if (undo_log_manager_.has_undo(curr)) {
            uint32_t restored[32];
            uint8_t slots[32];
            size_t n = undo_log_manager_.get_restore_edges(
                curr, visible_batch, restored, slots);

            for (size_t i = 0; i < n; i++) {
                // Process restored[i]...
            }
        }
    }
}
```

---

## 9. Future Optimizations

### 9.1 Selective Version Storage (RNG-based)

Not all deleted edges need undo logging. If another visible neighbor "dominates" the deleted edge (RNG criterion), the edge is redundant:

```cpp
bool needs_undo_log(uint32_t X, uint32_t D, uint32_t P) {
    for (Z : X.neighbors) {
        if (Z == P) continue;
        if (batch(Z) > batch(D)) continue;  // Z must be visible when D is

        // RNG dominance check
        if (dist(X,Z) < dist(X,D) && dist(Z,D) < dist(X,D)) {
            return false;  // Z dominates D → no undo needed
        }
    }
    return true;  // Must store undo log
}
```

**Expected benefit**: 30-70% reduction in undo log entries.

### 9.2 Lock-Free Writes

Replace mutex with atomic operations for higher concurrency:

```cpp
// Atomic block allocation
UndoLogBlock* expected = nullptr;
UndoLogBlock* new_block = new UndoLogBlock();
if (!blocks_[idx].compare_exchange_strong(expected, new_block)) {
    delete new_block;  // Another thread allocated first
}

// Atomic entry append
uint16_t slot = log->count.fetch_add(1);
log->entries[slot] = entry;
```

### 9.3 Epoch-Based Reclamation (EBR)

For safe GC without stopping queries:

```cpp
class EBR {
    atomic<uint32_t> global_epoch;  // 0, 1, 2

    void enter_epoch() { /* reader enters */ }
    void exit_epoch() { /* reader exits */ }
    void gc_epoch(uint32_t epoch) { /* reclaim epoch's garbage */ }
};
```

---

## 10. References

1. HNSW: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using HNSW graphs", IEEE TPAMI 2018
2. NSG/MRNG: Fu et al., "Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph", VLDB 2019
3. MVCC: Bernstein & Goodman, "Concurrency Control in Distributed Database Systems", ACM Computing Surveys 1981
