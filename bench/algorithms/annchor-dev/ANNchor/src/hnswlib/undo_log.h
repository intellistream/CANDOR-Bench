#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

namespace annchor {

struct NodeUndoLog {
    static constexpr int RCS_K = 8;  // number of random coordinates (optimal from sweep) for directional pruning

    std::vector<uint32_t> batches;
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> edges;
    std::vector<uint32_t> causes;
    // Enriched Undo Log (EUL): store distances computed during pruning
    std::vector<float> dist_node_pruned;   // dist(owner_node, pruned_edge)
    std::vector<float> dist_cause_pruned;  // dist(caused_by, pruned_edge)
    // EUL-DP: RCS sketch of (pruned_neighbor - owner_node), k floats per edge
    std::vector<float> rcs_sketch;

    // Inline GC: logical start index into batches[]. Entries before this are garbage.
    // Can be advanced by any thread (insert or search) lock-free.
    // Physical compaction happens in compact (single-threaded).
    std::atomic<uint32_t> gc_start_{0};

    NodeUndoLog() = default;
    NodeUndoLog(NodeUndoLog&& o) noexcept
        : batches(std::move(o.batches)), offsets(std::move(o.offsets)),
          edges(std::move(o.edges)), causes(std::move(o.causes)),
          dist_node_pruned(std::move(o.dist_node_pruned)),
          dist_cause_pruned(std::move(o.dist_cause_pruned)),
          rcs_sketch(std::move(o.rcs_sketch)),
          gc_start_(o.gc_start_.load(std::memory_order_relaxed)) {}
    NodeUndoLog& operator=(NodeUndoLog&& o) noexcept {
        batches = std::move(o.batches); offsets = std::move(o.offsets);
        edges = std::move(o.edges); causes = std::move(o.causes);
        dist_node_pruned = std::move(o.dist_node_pruned);
        dist_cause_pruned = std::move(o.dist_cause_pruned);
        rcs_sketch = std::move(o.rcs_sketch);
        gc_start_.store(o.gc_start_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    inline void append(uint32_t batch, uint32_t neighbor, uint32_t caused_by,
                       float d_node_pruned = 0.0f, float d_cause_pruned = 0.0f,
                       const float* sketch = nullptr) noexcept {
        edges.push_back(neighbor);
        causes.push_back(caused_by);
        dist_node_pruned.push_back(d_node_pruned);
        dist_cause_pruned.push_back(d_cause_pruned);
        if (sketch) {
            for (int i = 0; i < RCS_K; i++)
                rcs_sketch.push_back(sketch[i]);
        } else {
            for (int i = 0; i < RCS_K; i++)
                rcs_sketch.push_back(0.0f);
        }
        if (__builtin_expect(batches.empty() || batches.back() != batch, 0)) {
            batches.push_back(batch);
            offsets.push_back(static_cast<uint32_t>(edges.size()));
        } else {
            offsets.back() = static_cast<uint32_t>(edges.size());
        }
    }

    inline size_t upperBound(uint32_t target) const noexcept {
        size_t lo = 0, hi = batches.size();
        while (lo < hi) {
            size_t mid = lo + ((hi - lo) >> 1);
            if (batches[mid] <= target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    inline std::pair<uint32_t, uint32_t> getRecoverRange(uint32_t visible_batch) const noexcept {
        if (__builtin_expect(batches.empty(), 0)) return {0, 0};
        uint32_t gs = gc_start_.load(std::memory_order_acquire);
        size_t n = batches.size();
        if (gs >= n) return {0, 0};
        // Search only in [gs, n) range
        size_t idx = upperBound(visible_batch);
        if (__builtin_expect(idx >= n, 0)) return {0, 0};
        // Clamp: don't return edges before gc_start_
        if (idx < gs) idx = gs;
        uint32_t start = (idx == 0) ? 0 : offsets[idx - 1];
        return {start, static_cast<uint32_t>(edges.size())};
    }

    inline size_t recover(uint32_t visible_batch, uint32_t* out, size_t max_out) const noexcept {
        auto [start, end] = getRecoverRange(visible_batch);
        size_t count = (end - start < max_out) ? (end - start) : max_out;
        if (__builtin_expect(count > 0, 1)) {
            std::memcpy(out, edges.data() + start, count * sizeof(uint32_t));
        }
        return count;
    }

    inline size_t getRecoverPointers(uint32_t visible_batch,
                                      const uint32_t** out_edges,
                                      const uint32_t** out_causes) const noexcept {
        auto [start, end] = getRecoverRange(visible_batch);
        if (end > start) {
            *out_edges = edges.data() + start;
            *out_causes = causes.data() + start;
            return end - start;
        }
        *out_edges = nullptr;
        *out_causes = nullptr;
        return 0;
    }

    // EUL: also return cached distances
    inline size_t getRecoverPointersEUL(uint32_t visible_batch,
                                         const uint32_t** out_edges,
                                         const uint32_t** out_causes,
                                         const float** out_dist_node,
                                         const float** out_dist_cause) const noexcept {
        auto [start, end] = getRecoverRange(visible_batch);
        if (end > start) {
            *out_edges = edges.data() + start;
            *out_causes = causes.data() + start;
            *out_dist_node = dist_node_pruned.data() + start;
            *out_dist_cause = dist_cause_pruned.data() + start;
            return end - start;
        }
        *out_edges = nullptr;
        *out_causes = nullptr;
        *out_dist_node = nullptr;
        *out_dist_cause = nullptr;
        return 0;
    }

    // EUL-DP: also return RCS sketch pointers
    inline size_t getRecoverPointersDP(uint32_t visible_batch,
                                        const uint32_t** out_edges,
                                        const uint32_t** out_causes,
                                        const float** out_dist_node,
                                        const float** out_dist_cause,
                                        const float** out_sketch) const noexcept {
        auto [start, end] = getRecoverRange(visible_batch);
        if (end > start) {
            *out_edges = edges.data() + start;
            *out_causes = causes.data() + start;
            *out_dist_node = dist_node_pruned.data() + start;
            *out_dist_cause = dist_cause_pruned.data() + start;
            *out_sketch = rcs_sketch.data() + start * RCS_K;
            return end - start;
        }
        *out_edges = nullptr; *out_causes = nullptr;
        *out_dist_node = nullptr; *out_dist_cause = nullptr;
        *out_sketch = nullptr;
        return 0;
    }

    inline bool needsRecover(uint32_t visible_batch) const noexcept {
        if (batches.empty()) return false;
        uint32_t gs = gc_start_.load(std::memory_order_acquire);
        return gs < batches.size() && batches.back() > visible_batch;
    }

    inline void clear() noexcept {
        batches.clear();
        offsets.clear();
        edges.clear();
        causes.clear();
        dist_node_pruned.clear();
        dist_cause_pruned.clear();
        rcs_sketch.clear();
        gc_start_.store(0, std::memory_order_relaxed);
    }

    inline bool empty() const noexcept {
        uint32_t gs = gc_start_.load(std::memory_order_relaxed);
        return edges.empty() || gs >= batches.size();
    }

    // Lock-free logical truncation: advance gc_start_ so entries before safe_batch
    // are skipped by readers. Safe to call from any thread (insert or search).
    inline void advanceGcStart(uint32_t safe_batch) noexcept {
        if (batches.empty()) return;
        uint32_t gs = gc_start_.load(std::memory_order_relaxed);
        // Find first batch >= safe_batch starting from current gc_start_
        uint32_t new_gs = gs;
        size_t n = batches.size();
        while (new_gs < n && batches[new_gs] < safe_batch) {
            new_gs++;
        }
        if (new_gs > gs) {
            // CAS loop to monotonically advance gc_start_
            uint32_t expected = gs;
            while (expected < new_gs) {
                if (gc_start_.compare_exchange_weak(expected, new_gs,
                        std::memory_order_release, std::memory_order_relaxed))
                    break;
                // If someone else advanced past us, done
                if (expected >= new_gs) break;
            }
        }
    }

    // Physical compaction: actually free memory. Call ONLY from single-threaded
    // context (compact). Moves data so gc_start_ resets to 0.
    inline void compact() noexcept {
        uint32_t gs = gc_start_.load(std::memory_order_relaxed);
        if (gs == 0 || batches.empty()) return;
        if (gs >= batches.size()) {
            clear();
            return;
        }
        uint32_t cut_at = (gs == 0) ? 0 : offsets[gs - 1];

        batches.erase(batches.begin(), batches.begin() + gs);
        offsets.erase(offsets.begin(), offsets.begin() + gs);
        for (auto &o : offsets) o -= cut_at;

        edges.erase(edges.begin(), edges.begin() + cut_at);
        causes.erase(causes.begin(), causes.begin() + cut_at);
        dist_node_pruned.erase(dist_node_pruned.begin(), dist_node_pruned.begin() + cut_at);
        dist_cause_pruned.erase(dist_cause_pruned.begin(), dist_cause_pruned.begin() + cut_at);
        if (rcs_sketch.size() >= cut_at * RCS_K) {
            rcs_sketch.erase(rcs_sketch.begin(), rcs_sketch.begin() + cut_at * RCS_K);
        }
        gc_start_.store(0, std::memory_order_relaxed);
    }

    // Legacy truncateBefore: now delegates to advanceGcStart (lock-free).
    inline void truncateBefore(uint32_t safe_batch) noexcept {
        advanceGcStart(safe_batch);
    }
};

class UndoLogManager {
public:
    explicit UndoLogManager(size_t max_nodes = 0) : max_nodes_(max_nodes) {
        if (max_nodes > 0) {
            logs_.resize(max_nodes);
            has_log_.resize(max_nodes, false);
        }
    }

    void resize(size_t new_max) {
        max_nodes_ = new_max;
        logs_.resize(new_max);
        has_log_.resize(new_max, false);
    }

    inline void recordPrune(uint32_t node_id, uint32_t deleted_neighbor,
                            uint32_t caused_by, uint32_t batch,
                            float dist_node_pruned = 0.0f,
                            float dist_cause_pruned = 0.0f,
                            const float* sketch = nullptr) {
        has_log_[node_id] = true;
        logs_[node_id].append(batch, deleted_neighbor, caused_by,
                              dist_node_pruned, dist_cause_pruned, sketch);
    }

    inline size_t recover(uint32_t node_id, uint32_t visible_batch,
                          uint32_t* out, size_t max_out) const noexcept {
        if (__builtin_expect(!has_log_[node_id], 1)) return 0;
        return logs_[node_id].recover(visible_batch, out, max_out);
    }

    inline size_t getRecoverPointers(uint32_t node_id, uint32_t visible_batch,
                                      const uint32_t** out_edges,
                                      const uint32_t** out_causes) const noexcept {
        if (__builtin_expect(!has_log_[node_id], 1)) {
            *out_edges = nullptr;
            *out_causes = nullptr;
            return 0;
        }
        return logs_[node_id].getRecoverPointers(visible_batch, out_edges, out_causes);
    }

    inline bool needsRecover(uint32_t node_id, uint32_t visible_batch) const noexcept {
        return has_log_[node_id] && logs_[node_id].needsRecover(visible_batch);
    }

    inline size_t getRecoverPointersEUL(uint32_t node_id, uint32_t visible_batch,
                                        const uint32_t** out_edges,
                                        const uint32_t** out_causes,
                                        const float** out_dist_node,
                                        const float** out_dist_cause) const noexcept {
        if (__builtin_expect(!has_log_[node_id], 1)) {
            *out_edges = nullptr; *out_causes = nullptr;
            *out_dist_node = nullptr; *out_dist_cause = nullptr;
            return 0;
        }
        return logs_[node_id].getRecoverPointersEUL(visible_batch, out_edges, out_causes,
                                                     out_dist_node, out_dist_cause);
    }

    inline size_t getRecoverPointersDP(uint32_t node_id, uint32_t visible_batch,
                                        const uint32_t** out_edges,
                                        const uint32_t** out_causes,
                                        const float** out_dist_node,
                                        const float** out_dist_cause,
                                        const float** out_sketch) const noexcept {
        if (__builtin_expect(!has_log_[node_id], 1)) {
            *out_edges = nullptr; *out_causes = nullptr;
            *out_dist_node = nullptr; *out_dist_cause = nullptr;
            *out_sketch = nullptr;
            return 0;
        }
        return logs_[node_id].getRecoverPointersDP(visible_batch, out_edges, out_causes,
                                                    out_dist_node, out_dist_cause, out_sketch);
    }

    inline bool hasLog(uint32_t node_id) const noexcept {
        return has_log_[node_id];
    }

    inline const NodeUndoLog* getLog(uint32_t node_id) const noexcept {
        return has_log_[node_id] ? &logs_[node_id] : nullptr;
    }

    // Lock-free: advance gc_start_ for a node. Safe from any thread.
    inline void advanceGcStart(uint32_t node_id, uint32_t safe_batch) noexcept {
        if (node_id < max_nodes_ && has_log_[node_id]) {
            logs_[node_id].advanceGcStart(safe_batch);
        }
    }

    inline void truncateBefore(uint32_t node_id, uint32_t safe_batch) noexcept {
        if (node_id < max_nodes_ && has_log_[node_id]) {
            logs_[node_id].truncateBefore(safe_batch);
            if (logs_[node_id].empty()) has_log_[node_id] = false;
        }
    }

    // Physical compaction for a single node. Thread-safe if different threads handle different nodes.
    inline void compactNode(uint32_t node_id) noexcept {
        if (node_id < max_nodes_ && has_log_[node_id]) {
            logs_[node_id].compact();
            if (logs_[node_id].empty()) has_log_[node_id] = false;
        }
    }

    // Physical compaction for all logs. Call from single-threaded compact only.
    inline void compactAll() noexcept {
        for (size_t i = 0; i < max_nodes_; ++i) {
            compactNode(i);
        }
    }

    inline void clearLog(uint32_t node_id) noexcept {
        if (node_id < max_nodes_ && has_log_[node_id]) {
            logs_[node_id].clear();
            has_log_[node_id] = false;
        }
    }

    inline void clear() noexcept {
        for (size_t i = 0; i < max_nodes_; ++i) {
            if (has_log_[i]) {
                logs_[i].clear();
                has_log_[i] = false;
            }
        }
    }

private:
    size_t max_nodes_ = 0;
    std::vector<NodeUndoLog> logs_;
    std::vector<uint8_t> has_log_;  // uint8_t is faster than vector<bool> for random access
};

}  // namespace annchor
