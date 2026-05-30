#pragma once

#include <assert.h>
#include <stdlib.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <future>
#include <limits>
#include <list>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <shared_mutex>
#include <thread>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hnswlib.h"
#include "visited_list_pool.h"
#include "undo_log.h"

namespace annchor {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

// Memory pool for versioned link lists (thread-safe)
class VersionMemoryPool {
   private:
    std::vector<char *> free_blocks_;
    size_t block_size_;
    size_t pool_batch_size_;
    mutable std::mutex pool_lock_;

   public:
    VersionMemoryPool(size_t block_size, size_t initial_capacity = 1024)
        : block_size_(block_size), pool_batch_size_(initial_capacity) {
        expand_unlocked();
    }

    ~VersionMemoryPool() {
        for (auto ptr : free_blocks_) {
            free(ptr);
        }
    }

    void expand_unlocked() {
        for (size_t i = 0; i < pool_batch_size_; ++i) {
            free_blocks_.push_back((char *)malloc(block_size_));
        }
    }

    char *allocate() {
        std::lock_guard<std::mutex> lock(pool_lock_);
        if (free_blocks_.empty()) {
            expand_unlocked();
        }
        char *ptr = free_blocks_.back();
        free_blocks_.pop_back();
        return ptr;
    }

    void recycle(char *ptr) {
        std::lock_guard<std::mutex> lock(pool_lock_);
        free_blocks_.push_back(ptr);
    }
};

template <typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
   public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{
        0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{0};

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};
    bool use_node_lock_in_search_{true};
    bool enable_mvcc_{true};  // toggle MVCC: false = pure HNSW (no watermark/undo/version)
    bool enable_undo_recovery_{true};  // toggle undo-log recovery on historical snapshots
    bool enable_trim_recovery_filter_{false};
    float trim_recovery_relax_factor_{1.5f};
    float trim_recovery_margin_ratio_{-1.0f};
    size_t trim_recovery_large_log_threshold_{4};
    size_t num_threads_{1};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    mutable std::vector<std::shared_mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element
    std::vector<uint16_t> trim_node_topk_indices_;
    std::vector<int8_t> trim_node_topk_values_;
    std::vector<float> trim_node_topk_scale_;
    std::vector<float> trim_node_rest_norm_;
    std::vector<float> trim_node_full_norm_sq_;

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};
    mutable std::atomic<size_t> metric_future_skip_queries{0};
    mutable std::atomic<size_t> metric_future_skip_hop_max{0};
    mutable std::atomic<size_t> metric_future_skip_hops_total{0};
    mutable std::atomic<size_t> metric_recovered_edges_total{0};
    mutable std::atomic<size_t> metric_recovered_edges_useful{0};
    mutable std::atomic<size_t> metric_recovered_edges_single_node_max{0};
    mutable std::atomic<size_t> metric_nodes_visited{0};
    mutable std::atomic<size_t> metric_visibility_checks{0};
    mutable std::atomic<size_t> metric_undo_log_lookups{0};
    mutable std::atomic<size_t> metric_trim_candidates_estimated{0};
    mutable std::atomic<size_t> metric_trim_candidates_filtered{0};
    mutable std::atomic<size_t> metric_trim_candidates_exact{0};
    mutable std::atomic<size_t> metric_trim_groups_total{0};
    mutable std::atomic<size_t> metric_trim_groups_pruned{0};
    mutable std::atomic<size_t> metric_trim_margin_filtered{0};

    // ---- Fine-grained profiling (rdtsc-based) ----
    // Separate counters for insert vs search operations
    // Using atomics for cross-thread accumulation
    struct OpProfile {
        std::atomic<uint64_t> count{0};            // number of operations
        std::atomic<uint64_t> dist_compute_ns{0};  // time in distance computation
        std::atomic<uint64_t> dist_count{0};       // number of distance computations
        std::atomic<uint64_t> lock_wait_ns{0};     // time waiting for locks
        std::atomic<uint64_t> lock_count{0};       // number of lock acquisitions
        std::atomic<uint64_t> upper_nav_ns{0};     // upper layer navigation time
        std::atomic<uint64_t> level0_beam_ns{0};   // level-0 beam search time
        std::atomic<uint64_t> pruning_ns{0};       // neighbor selection/pruning time
        std::atomic<uint64_t> total_ns{0};         // total operation time
        std::atomic<uint64_t> nodes_visited{0};    // nodes visited
    };
    mutable OpProfile prof_insert_;
    mutable OpProfile prof_search_;

    static inline uint64_t rdtsc_ns() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    }

    void initTrimRecoverySketchDims(size_t random_seed) {
        (void)random_seed;
        trim_node_topk_indices_.assign(max_elements_ * NodeUndoLog::TRIM_TOPK, 0);
        trim_node_topk_values_.assign(max_elements_ * NodeUndoLog::TRIM_TOPK, 0);
        trim_node_topk_scale_.assign(max_elements_, 0.0f);
        trim_node_rest_norm_.assign(max_elements_, 0.0f);
        trim_node_full_norm_sq_.assign(max_elements_, 0.0f);
    }

    void resizeTrimRecoveryCodes(size_t new_max_elements) {
        trim_node_topk_indices_.resize(new_max_elements * NodeUndoLog::TRIM_TOPK, 0);
        trim_node_topk_values_.resize(new_max_elements * NodeUndoLog::TRIM_TOPK, 0);
        trim_node_topk_scale_.resize(new_max_elements, 0.0f);
        trim_node_rest_norm_.resize(new_max_elements, 0.0f);
        trim_node_full_norm_sq_.resize(new_max_elements, 0.0f);
    }

    inline float computeTrimVectorNormSq(const float* vec) const {
        size_t dim = data_size_ / sizeof(float);
        float norm_sq = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            norm_sq += vec[i] * vec[i];
        }
        return norm_sq;
    }

    inline float trimRecoveryCosFactor() const {
        float cos_factor = 1.0f / trim_recovery_relax_factor_;
        if (cos_factor < 0.0f) cos_factor = 0.0f;
        if (cos_factor > 1.0f) cos_factor = 1.0f;
        return cos_factor;
    }

    inline void buildTrimRecoveryCode(const float* vec,
                                      uint16_t* out_indices,
                                      int8_t* out_values,
                                      float& out_scale,
                                      float& out_rest_norm,
                                      float& out_full_norm_sq) const {
        std::array<float, NodeUndoLog::TRIM_TOPK> top_abs{};
        std::array<float, NodeUndoLog::TRIM_TOPK> top_values{};
        std::array<uint16_t, NodeUndoLog::TRIM_TOPK> top_indices{};
        size_t dim = data_size_ / sizeof(float);
        size_t filled = 0;
        out_full_norm_sq = 0.0f;

        for (size_t d = 0; d < dim; ++d) {
            float value = vec[d];
            float abs_value = std::fabs(value);
            out_full_norm_sq += value * value;

            if (filled < NodeUndoLog::TRIM_TOPK) {
                top_abs[filled] = abs_value;
                top_values[filled] = value;
                top_indices[filled] = static_cast<uint16_t>(d);
                ++filled;
                size_t pos = filled - 1;
                while (pos > 0 && top_abs[pos] > top_abs[pos - 1]) {
                    std::swap(top_abs[pos], top_abs[pos - 1]);
                    std::swap(top_values[pos], top_values[pos - 1]);
                    std::swap(top_indices[pos], top_indices[pos - 1]);
                    --pos;
                }
            } else if (abs_value > top_abs[NodeUndoLog::TRIM_TOPK - 1]) {
                top_abs[NodeUndoLog::TRIM_TOPK - 1] = abs_value;
                top_values[NodeUndoLog::TRIM_TOPK - 1] = value;
                top_indices[NodeUndoLog::TRIM_TOPK - 1] = static_cast<uint16_t>(d);
                size_t pos = NodeUndoLog::TRIM_TOPK - 1;
                while (pos > 0 && top_abs[pos] > top_abs[pos - 1]) {
                    std::swap(top_abs[pos], top_abs[pos - 1]);
                    std::swap(top_values[pos], top_values[pos - 1]);
                    std::swap(top_indices[pos], top_indices[pos - 1]);
                    --pos;
                }
            }
        }

        float top_norm_sq = 0.0f;
        for (int i = 0; i < NodeUndoLog::TRIM_TOPK; ++i) {
            top_norm_sq += top_values[i] * top_values[i];
        }
        out_rest_norm = std::sqrt(std::max(0.0f, out_full_norm_sq - top_norm_sq));

        float max_abs = (filled > 0) ? top_abs[0] : 0.0f;
        out_scale = max_abs > 0.0f ? (max_abs / 127.0f) : 0.0f;
        for (int i = 0; i < NodeUndoLog::TRIM_TOPK; ++i) {
            out_indices[i] = top_indices[i];
            if (out_scale > 0.0f) {
                long quant = std::lround(top_values[i] / out_scale);
                if (quant < -127) quant = -127;
                if (quant > 127) quant = 127;
                out_values[i] = static_cast<int8_t>(quant);
            } else {
                out_values[i] = 0;
            }
        }
    }

    inline void refreshTrimRecoveryCode(tableint node_id) {
        if (node_id >= max_elements_ || trim_node_topk_scale_.empty()) return;
        size_t base = static_cast<size_t>(node_id) * NodeUndoLog::TRIM_TOPK;
        buildTrimRecoveryCode(
            reinterpret_cast<const float*>(getDataByInternalId(node_id)),
            trim_node_topk_indices_.data() + base,
            trim_node_topk_values_.data() + base,
            trim_node_topk_scale_[node_id], trim_node_rest_norm_[node_id],
            trim_node_full_norm_sq_[node_id]);
    }

    inline float computeTrimRecoveryUpperBound(const float* query_vec,
                                               float query_norm_sq,
                                               tableint node_id,
                                               float cos_factor) const {
        size_t base = static_cast<size_t>(node_id) * NodeUndoLog::TRIM_TOPK;
        const uint16_t* topk_indices = trim_node_topk_indices_.data() + base;
        const int8_t* topk_values = trim_node_topk_values_.data() + base;
        float top_inner = 0.0f;
        float q_top_norm_sq = 0.0f;
        for (int ti = 0; ti < NodeUndoLog::TRIM_TOPK; ++ti) {
            if (topk_values[ti] == 0) continue;
            uint16_t dim_idx = topk_indices[ti];
            float q_value = query_vec[dim_idx];
            q_top_norm_sq += q_value * q_value;
            top_inner += q_value *
                         (static_cast<float>(topk_values[ti]) *
                          trim_node_topk_scale_[node_id]);
        }
        float q_rest_norm =
            std::sqrt(std::max(0.0f, query_norm_sq - q_top_norm_sq));
        return top_inner + cos_factor * q_rest_norm * trim_node_rest_norm_[node_id];
    }

    std::string getProfileStats() const {
        auto fmt = [](const OpProfile& p, const char* name) -> std::string {
            std::stringstream ss;
            uint64_t cnt = p.count.load();
            if (cnt == 0) return "";
            double total_ms = p.total_ns.load() / 1e6;
            double dist_ms = p.dist_compute_ns.load() / 1e6;
            double lock_ms = p.lock_wait_ns.load() / 1e6;
            double upper_ms = p.upper_nav_ns.load() / 1e6;
            double l0_ms = p.level0_beam_ns.load() / 1e6;
            double prune_ms = p.pruning_ns.load() / 1e6;
            double other_ms = total_ms - dist_ms - lock_ms - upper_ms - prune_ms;
            if (other_ms < 0) other_ms = 0;

            ss << name << "_ops:" << cnt
               << ", " << name << "_total_ms:" << (uint64_t)total_ms
               << ", " << name << "_dist_ms:" << (uint64_t)dist_ms
               << " (" << (total_ms > 0 ? dist_ms/total_ms*100 : 0) << "%)"
               << ", " << name << "_lock_ms:" << (uint64_t)lock_ms
               << " (" << (total_ms > 0 ? lock_ms/total_ms*100 : 0) << "%)"
               << ", " << name << "_upper_ms:" << (uint64_t)upper_ms
               << ", " << name << "_l0beam_ms:" << (uint64_t)l0_ms
               << ", " << name << "_prune_ms:" << (uint64_t)prune_ms
               << ", " << name << "_dist_count:" << p.dist_count.load()
               << ", " << name << "_nodes:" << p.nodes_visited.load()
               << ", " << name << "_avg_dist_per_op:" << (cnt > 0 ? p.dist_count.load()/cnt : 0);
            return ss.str();
        };
        std::string result = fmt(prof_insert_, "ins") + ", " + fmt(prof_search_, "srch");
        // Also append global metrics for cross-check
        std::stringstream gs;
        gs << ", total_dist_computations:" << metric_distance_computations.load()
           << ", total_hops:" << metric_hops.load()
           << ", total_nodes_visited:" << metric_nodes_visited.load();
        
        // S3 metrics
        if (enable_s3_) {
            gs << ", s3_hits:" << s3_hits_.load()
               << ", s3_misses:" << s3_misses_.load()
               << ", s3_dist_saved:" << s3_dist_saved_.load()
               << ", s3_hit_rate:" << (s3_hits_.load() + s3_misses_.load() > 0 ? 
                   (double)s3_hits_.load() / (s3_hits_.load() + s3_misses_.load()) * 100 : 0);
        }
        
        // Piggyback metrics
        if (enable_piggyback_) {
            gs << ", piggyback_hits:" << piggyback_hits_.load()
               << ", piggyback_misses:" << piggyback_misses_.load()
               << ", piggyback_dist_saved:" << piggyback_dist_saved_.load()
               << ", piggyback_exact_computations:" << piggyback_exact_computations_.load()
               << ", piggyback_jl_computations:" << piggyback_jl_computations_.load()
               << ", piggyback_hit_rate:" << (piggyback_hits_.load() + piggyback_misses_.load() > 0 ? 
                   (double)piggyback_hits_.load() / (piggyback_hits_.load() + piggyback_misses_.load()) * 100 : 0);
        }
        
        return result + gs.str();
    }

    bool allow_replace_deleted_ = false;

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint>
        deleted_elements;  // contains internal ids of deleted elements

    std::unique_ptr<std::atomic<bool>[]> insertion_complete_;
    std::atomic<long> global_watermark_{-1};

    // MVCC support - version chain stored inline in linklist structure
    std::unique_ptr<VersionMemoryPool> version_pool_level0_;
    std::unique_ptr<VersionMemoryPool> version_pool_higher_;

    // Undo log for pruned edges
    std::unique_ptr<UndoLogManager> undo_log_;

    // S3 byproduct store for computation sharing
    struct Byproduct {
        tableint insert_id;          // which insert produced this
        std::vector<std::pair<dist_t, tableint>> candidates;  // top-ef_c candidates
        const void* insert_vector;   // pointer to inserted vector for distance checking
        uint64_t timestamp;          // when this byproduct was created
        mutable std::atomic<bool> consumed{false};
        
        Byproduct() : insert_id(0), insert_vector(nullptr), timestamp(0) {}
        Byproduct(tableint id, std::vector<std::pair<dist_t, tableint>>&& cands, 
                  const void* vec, uint64_t ts)
            : insert_id(id), candidates(std::move(cands)), insert_vector(vec), timestamp(ts) {}
            
        // Custom move constructor and assignment to handle atomic
        Byproduct(Byproduct&& other) 
            : insert_id(other.insert_id), 
              candidates(std::move(other.candidates)), 
              insert_vector(other.insert_vector),
              timestamp(other.timestamp),
              consumed(other.consumed.load()) {}
              
        Byproduct& operator=(Byproduct&& other) {
            if (this != &other) {
                insert_id = other.insert_id;
                candidates = std::move(other.candidates);
                insert_vector = other.insert_vector;
                timestamp = other.timestamp;
                consumed.store(other.consumed.load());
            }
            return *this;
        }

        Byproduct(const Byproduct& other)
            : insert_id(other.insert_id),
              candidates(other.candidates),
              insert_vector(other.insert_vector),
              timestamp(other.timestamp),
              consumed(other.consumed.load(std::memory_order_acquire)) {}

        Byproduct& operator=(const Byproduct& other) {
            if (this != &other) {
                insert_id = other.insert_id;
                candidates = other.candidates;
                insert_vector = other.insert_vector;
                timestamp = other.timestamp;
                consumed.store(other.consumed.load(std::memory_order_acquire),
                               std::memory_order_release);
            }
            return *this;
        }
    };

    // Thread-safe ring buffer for S3 byproducts
    struct S3ByproductStore {
        static const size_t DEFAULT_CAPACITY = 1024;
        std::unique_ptr<Byproduct[]> buffer;
        mutable std::mutex buffer_mutex;
        std::atomic<size_t> head{0};
        std::atomic<size_t> tail{0};
        size_t capacity;
        dist_t proximity_threshold;  // threshold for considering vectors "close"
        
        S3ByproductStore(size_t cap = DEFAULT_CAPACITY, dist_t threshold = 0.1) 
            : buffer(std::make_unique<Byproduct[]>(cap)), capacity(cap), proximity_threshold(threshold) {}
        
        // Add a byproduct (thread-safe)
        void add(tableint insert_id, std::vector<std::pair<dist_t, tableint>>&& candidates, 
                 const void* insert_vector, uint64_t timestamp) {
            std::lock_guard<std::mutex> guard(buffer_mutex);
            size_t current_tail = tail.load(std::memory_order_relaxed);
            size_t next_tail = (current_tail + 1) % capacity;
            
            // If buffer is full, overwrite oldest entry (ring buffer behavior)
            if (next_tail == head.load(std::memory_order_acquire)) {
                head.store((head.load(std::memory_order_relaxed) + 1) % capacity, std::memory_order_release);
            }
            
            buffer[current_tail] =
                Byproduct(insert_id, std::move(candidates), insert_vector, timestamp);
            tail.store(next_tail, std::memory_order_release);
        }

        // Claim and copy a suitable byproduct for query.
        // Returning a stable copy avoids exposing raw pointers into the ring buffer.
        bool claimByproduct(Byproduct* out, const void* query_vector,
                            dist_t (*dist_func)(const void*, const void*, const void*),
                            const void* dist_param,
                            uint64_t max_age_ns = 1000000000ULL) {  // 1 second max age
            std::lock_guard<std::mutex> guard(buffer_mutex);
            size_t current_head = head.load(std::memory_order_acquire);
            size_t current_tail = tail.load(std::memory_order_acquire);
            uint64_t now = rdtsc_ns();
            
            // Linear scan through ring buffer (newest to oldest)
            for (size_t i = current_tail; i != current_head; i = (i == 0 ? capacity - 1 : i - 1)) {
                size_t idx = (i == 0 ? capacity - 1 : i - 1);
                Byproduct& bp = buffer[idx];
                
                if (bp.insert_vector == nullptr) continue;
                if (now - bp.timestamp > max_age_ns) continue;  // too old
                if (bp.consumed.load(std::memory_order_acquire)) continue;  // already used
                
                // Check if query is close to insert vector
                dist_t dist = dist_func(query_vector, bp.insert_vector, dist_param);
                if (dist <= proximity_threshold) {
                    bp.consumed.store(true, std::memory_order_release);
                    if (out != nullptr) {
                        *out = bp;
                    }
                    return true;
                }
            }
            return false;
        }
    };
    
    std::unique_ptr<S3ByproductStore> s3_store_;

    static constexpr size_t SLIPSTREAM_CAPSULE_TABLE_SIZE = 128;
    static constexpr size_t SLIPSTREAM_MAX_SEEDS = 16;
    static constexpr uint32_t SLIPSTREAM_EPOCH_BAND = 4096;
    static constexpr size_t SLIPSTREAM_MIN_BETTER_SEEDS = 2;
    static constexpr double SLIPSTREAM_MIN_IMPROVEMENT_RATIO = 0.97;
    static constexpr uint32_t SLIPSTREAM_REGRET_WINDOW = 8;
    static constexpr uint64_t SLIPSTREAM_COOLDOWN_MULTIPLIER = 4;

    struct SlipstreamCapsule {
        std::atomic<uint64_t> gen{0};
        std::atomic<uint64_t> created_ts{0};
        std::atomic<uint64_t> ride_key{0};
        std::atomic<uint64_t> cooldown_until{0};
        std::atomic<uint32_t> num_seeds{0};
        std::atomic<uint32_t> attempts{0};
        std::atomic<uint32_t> regrets{0};
        std::array<tableint, SLIPSTREAM_MAX_SEEDS> seeds{};
    };

    struct SlipstreamCapsuleSnapshot {
        uint64_t ride_key{0};
        uint32_t num_seeds{0};
        std::array<tableint, SLIPSTREAM_MAX_SEEDS> seeds{};
    };

    struct SlipstreamCapsuleTable {
        std::array<SlipstreamCapsule, SLIPSTREAM_CAPSULE_TABLE_SIZE> slots;

        static uint64_t mixKey(uint64_t x) {
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccdULL;
            x ^= x >> 33;
            x *= 0xc4ceb9fe1a85ec53ULL;
            x ^= x >> 33;
            return x;
        }

        static size_t slotIndex(uint64_t ride_key) {
            return static_cast<size_t>(ride_key) &
                   (SLIPSTREAM_CAPSULE_TABLE_SIZE - 1);
        }

        static uint64_t makeRideKey(tableint ride_anchor, tableint basin_anchor,
                                    uint32_t epoch_band) {
            uint64_t x = mixKey(static_cast<uint32_t>(ride_anchor));
            x ^= mixKey((static_cast<uint64_t>(static_cast<uint32_t>(basin_anchor))
                         << 1) |
                        1ULL);
            x ^= mixKey(static_cast<uint64_t>(epoch_band) << 32);
            return mixKey(x);
        }

        void publish(tableint ride_anchor, tableint basin_anchor,
                     const std::array<tableint, SLIPSTREAM_MAX_SEEDS>& seeds,
                     uint32_t num_seeds, uint32_t epoch_band,
                     uint64_t created_ts) {
            uint64_t new_key = makeRideKey(ride_anchor, basin_anchor, epoch_band);
            SlipstreamCapsule& slot = slots[slotIndex(new_key)];
            slot.gen.fetch_add(1, std::memory_order_acq_rel);
            uint64_t old_key = slot.ride_key.load(std::memory_order_acquire);
            if (old_key != new_key) {
                slot.cooldown_until.store(0, std::memory_order_relaxed);
                slot.attempts.store(0, std::memory_order_relaxed);
                slot.regrets.store(0, std::memory_order_relaxed);
            }
            slot.ride_key.store(new_key, std::memory_order_release);
            slot.created_ts.store(created_ts, std::memory_order_release);
            slot.num_seeds.store(num_seeds, std::memory_order_release);
            for (size_t i = 0; i < SLIPSTREAM_MAX_SEEDS; ++i) {
                slot.seeds[i] = seeds[i];
            }
            slot.gen.fetch_add(1, std::memory_order_release);
        }

        bool consume(tableint ride_anchor, tableint basin_anchor,
                     uint32_t epoch_band, uint64_t ttl_ns,
                     SlipstreamCapsuleSnapshot* out,
                     bool* cooldown_blocked = nullptr) const {
            uint64_t expected_key = makeRideKey(ride_anchor, basin_anchor,
                                                epoch_band);
            const SlipstreamCapsule& slot = slots[slotIndex(expected_key)];
            uint64_t now = rdtsc_ns();
            if (cooldown_blocked != nullptr) {
                *cooldown_blocked = false;
            }

            for (int attempt = 0; attempt < 4; ++attempt) {
                uint64_t before = slot.gen.load(std::memory_order_acquire);
                if ((before & 1ULL) != 0) continue;

                uint64_t ride_key = slot.ride_key.load(std::memory_order_acquire);
                uint64_t created_ts =
                    slot.created_ts.load(std::memory_order_acquire);
                uint32_t num_seeds =
                    slot.num_seeds.load(std::memory_order_acquire);
                uint64_t cooldown_until =
                    slot.cooldown_until.load(std::memory_order_acquire);
                if (ride_key != expected_key || created_ts == 0 ||
                    now - created_ts > ttl_ns || num_seeds == 0 ||
                    num_seeds > SLIPSTREAM_MAX_SEEDS) {
                    return false;
                }
                if (cooldown_until > now) {
                    if (cooldown_blocked != nullptr) {
                        *cooldown_blocked = true;
                    }
                    return false;
                }

                SlipstreamCapsuleSnapshot snapshot;
                snapshot.ride_key = ride_key;
                snapshot.num_seeds = num_seeds;
                for (size_t i = 0; i < SLIPSTREAM_MAX_SEEDS; ++i) {
                    snapshot.seeds[i] = slot.seeds[i];
                }

                uint64_t after = slot.gen.load(std::memory_order_acquire);
                if (before == after) {
                    if (out != nullptr) {
                        *out = snapshot;
                    }
                    return true;
                }
            }
            return false;
        }

        void recordOutcome(uint64_t ride_key, bool regret, uint64_t ttl_ns) {
            SlipstreamCapsule& slot = slots[slotIndex(ride_key)];
            if (slot.ride_key.load(std::memory_order_acquire) != ride_key) {
                return;
            }

            uint32_t attempts =
                slot.attempts.fetch_add(1, std::memory_order_acq_rel) + 1;
            uint32_t regrets =
                slot.regrets.load(std::memory_order_acquire);
            if (regret) {
                regrets =
                    slot.regrets.fetch_add(1, std::memory_order_acq_rel) + 1;
            }

            if (attempts >= SLIPSTREAM_REGRET_WINDOW &&
                regrets * 2 >= attempts) {
                slot.cooldown_until.store(
                    rdtsc_ns() + ttl_ns * SLIPSTREAM_COOLDOWN_MULTIPLIER,
                    std::memory_order_release);
            }
        }
    };

    mutable std::unique_ptr<SlipstreamCapsuleTable> slipstream_capsules_;
    
    // S3 metrics
    mutable std::atomic<uint64_t> s3_hits_{0};
    mutable std::atomic<uint64_t> s3_misses_{0};
    mutable std::atomic<uint64_t> s3_dist_saved_{0};
    bool enable_s3_{false};
    bool enable_slipstream_capsule_{false};
    uint64_t slipstream_ttl_ns_{100000000ULL};
    int slipstream_mode_{0};
    bool slipstream_quality_gate_{true};
    dist_t slipstream_skip_ratio_{static_cast<dist_t>(0.98)};
    mutable std::atomic<uint64_t> slipstream_publish_{0};
    mutable std::atomic<uint64_t> slipstream_insert_publish_{0};
    mutable std::atomic<uint64_t> slipstream_search_publish_{0};
    mutable std::atomic<uint64_t> slipstream_seeds_injected_{0};
    mutable std::atomic<uint64_t> slipstream_quality_reject_{0};
    mutable std::atomic<uint64_t> slipstream_cooldown_reject_{0};
    mutable std::atomic<uint64_t> slipstream_regret_{0};
    mutable std::atomic<uint64_t> slipstream_helpful_{0};

    // ===== PIGGYBACK IMPLEMENTATION =====
    
    struct PiggybackRequest {
        const void* query;                               // Search query vector
        dist_t delta_sq;                                 // ||q - v||² precomputed
        std::vector<float> delta;                        // q - v vector
        std::atomic<bool> active{true};                  // Request is active
        
        // Results accumulation
        std::vector<std::pair<dist_t, tableint>> results;
        std::mutex results_mutex;
        
        // JL projection support
        bool use_jl_projection{false};
        int jl_dim{64};                                  // Johnson-Lindenstrauss dimension
        std::vector<float> delta_projected;              // δ projected to low-dim space
        std::unique_ptr<std::vector<std::vector<float>>> jl_matrix; // Random projection matrix
        
        PiggybackRequest(const void* q, const void* v, size_t dim, DISTFUNC<dist_t> dist_func, 
                        const void* dist_param, bool enable_jl = false) 
            : query(q), use_jl_projection(enable_jl) {
            // Compute δ = q - v
            delta.resize(dim);
            delta_sq = 0;
            const float* q_ptr = static_cast<const float*>(q);
            const float* v_ptr = static_cast<const float*>(v);
            
            for (size_t i = 0; i < dim; i++) {
                delta[i] = q_ptr[i] - v_ptr[i];
                delta_sq += delta[i] * delta[i];
            }
            
            // Initialize JL projection if enabled
            if (use_jl_projection) {
                initJLProjection(dim);
            }
        }
        
    private:
        void initJLProjection(size_t dim) {
            // Initialize random projection matrix: k × d Gaussian matrix
            jl_matrix = std::make_unique<std::vector<std::vector<float>>>(jl_dim, std::vector<float>(dim));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> normal(0.0f, 1.0f / sqrt(jl_dim));
            
            for (int i = 0; i < jl_dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    (*jl_matrix)[i][j] = normal(gen);
                }
            }
            
            // Project δ to low-dim space
            delta_projected.resize(jl_dim, 0.0f);
            for (int i = 0; i < jl_dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    delta_projected[i] += (*jl_matrix)[i][j] * delta[j];
                }
            }
        }
    };

    // Thread-safe store for active piggyback requests
    mutable std::vector<std::shared_ptr<PiggybackRequest>> active_piggyback_requests_;
    mutable std::shared_mutex piggyback_requests_mutex_;
    
    // Piggyback metrics
    mutable std::atomic<uint64_t> piggyback_hits_{0};
    mutable std::atomic<uint64_t> piggyback_misses_{0}; 
    mutable std::atomic<uint64_t> piggyback_dist_saved_{0};
    mutable std::atomic<uint64_t> piggyback_exact_computations_{0};
    mutable std::atomic<uint64_t> piggyback_jl_computations_{0};
    bool enable_piggyback_{false};

    HierarchicalNSW(SpaceInterface<dist_t> *s) {}

    HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location,
                    bool nmslib = false, size_t max_elements = 0,
                    bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }

    HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements,
                    size_t M = 16, size_t ef_construction = 200,
                    size_t random_seed = 100,
                    bool allow_replace_deleted = false,
                    bool use_node_lock_in_search = true, size_t num_threads = 1)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
          link_list_locks_(max_elements),
          element_levels_(max_elements),
          allow_replace_deleted_(allow_replace_deleted),
          use_node_lock_in_search_(use_node_lock_in_search),
          num_threads_(num_threads) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        // Initialize version memory pools (will be sized after M is determined)
        // Initialized later after size_links_level0_ and
        // size_links_per_element_ are computed
        if (M <= 10000) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to "
                       "adverse effects."
                    << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of "
                       "the processing."
                    << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) +
                             sizeof(linklistsizeint) + sizeof(uint32_t) +
                             sizeof(char *);  // +ts +version_ptr
        size_data_per_element_ =
            size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ =
            (char *)malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(
            new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error(
                "Not enough memory: HierarchicalNSW failed to allocate "
                "linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) +
                                  sizeof(linklistsizeint) + sizeof(uint32_t) +
                                  sizeof(char *);  // +ts +version_ptr
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
        initWatermarkState(max_elements_);

        // Initialize version memory pools
        version_pool_level0_ =
            std::make_unique<VersionMemoryPool>(size_links_level0_);
        version_pool_higher_ =
            std::make_unique<VersionMemoryPool>(size_links_per_element_);

        // Initialize undo log
        undo_log_ = std::make_unique<UndoLogManager>(max_elements_);
        initTrimRecoverySketchDims(random_seed + 7);
        
        // Initialize S3 byproduct store
        s3_store_ = std::make_unique<S3ByproductStore>();
    }

    ~HierarchicalNSW() { clear(); }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0) free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        trim_node_topk_indices_.clear();
        trim_node_topk_values_.clear();
        trim_node_topk_scale_.clear();
        trim_node_rest_norm_.clear();
        trim_node_full_norm_sq_.clear();
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
        insertion_complete_.reset();
        global_watermark_.store(-1, std::memory_order_relaxed);
    }

    void initWatermarkState(size_t max_elements) {
        insertion_complete_.reset(new std::atomic<bool>[max_elements]);
        for (size_t i = 0; i < max_elements; ++i)
            insertion_complete_[i].store(false, std::memory_order_relaxed);
        global_watermark_.store(-1, std::memory_order_relaxed);
    }

    void markReady(tableint internal_id) {
        insertion_complete_[internal_id].store(true, std::memory_order_release);
    }

    void advanceWatermark(long upper_bound = -1) {
        while (true) {
            long current = global_watermark_.load(std::memory_order_acquire);
            tableint probe = static_cast<tableint>(current + 1);
            while (probe < cur_element_count &&
                   insertion_complete_[probe].load(std::memory_order_acquire)) {
                ++probe;
            }
            long desired = static_cast<long>(probe) - 1;
            if (upper_bound >= 0 && desired > upper_bound) {
                desired = upper_bound;
            }
            if (desired <= current) {
                return;
            }
            if (global_watermark_.compare_exchange_weak(
                    current, desired, std::memory_order_release,
                    std::memory_order_acquire)) {
                return;
            }
        }
    }

    void markCompleted(tableint internal_id, long upper_bound = -1) {
        markReady(internal_id);
        advanceWatermark(upper_bound);
    }

    struct CompareByFirst {
        constexpr bool operator()(
            std::pair<dist_t, tableint> const &a,
            std::pair<dist_t, tableint> const &b) const noexcept {
            return a.first < b.first;
        }
    };

    void setEf(size_t ef) { ef_ = ef; }

    inline std::mutex &getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }

    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label,
               (data_level0_memory_ + internal_id * size_data_per_element_ +
                label_offset_),
               sizeof(labeltype));
        return return_label;
    }

    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ +
                label_offset_),
               &label, sizeof(labeltype));
    }

    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *)(data_level0_memory_ +
                             internal_id * size_data_per_element_ +
                             label_offset_);
    }

    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ +
                offsetData_);
    }

    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
    }

    size_t getMaxElements() { return max_elements_; }

    size_t getCurrentElementCount() { return cur_element_count; }

    size_t getDeletedCount() { return num_deleted_; }

    tableint getWatermark() const { return (tableint)global_watermark_; }

    // Inflight = IDs already allocated (addPoint entered and fetch_add'd)
    // but not yet committed to the query-visible watermark. Bounded by
    // physical thread-pool depth, not by the number of outer dispatchers.
    long getInflightPoints() const {
        long allocated = static_cast<long>(cur_element_count.load(std::memory_order_acquire));
        long visible_plus_one = global_watermark_.load(std::memory_order_acquire) + 1;
        long diff = allocated - visible_plus_one;
        return diff > 0 ? diff : 0;
    }

    void setEnableMvcc(bool enable) { enable_mvcc_ = enable; }
    bool isMvccEnabled() const { return enable_mvcc_; }
    void setEnableUndoRecovery(bool enable) { enable_undo_recovery_ = enable; }
    bool isUndoRecoveryEnabled() const { return enable_undo_recovery_; }
    void setEnableTrimRecoveryFilter(bool enable) {
        enable_trim_recovery_filter_ = enable;
    }
    bool isTrimRecoveryFilterEnabled() const {
        return enable_trim_recovery_filter_;
    }
    void setTrimRecoveryRelaxFactor(float factor) {
        trim_recovery_relax_factor_ = (factor < 1.0f) ? 1.0f : factor;
    }
    float getTrimRecoveryRelaxFactor() const { return trim_recovery_relax_factor_; }
    void setTrimRecoveryMarginRatio(float ratio) {
        trim_recovery_margin_ratio_ = ratio;
    }
    float getTrimRecoveryMarginRatio() const { return trim_recovery_margin_ratio_; }
    
    // S3 control methods
    void setEnableS3(bool enable) { enable_s3_ = enable; }
    bool isS3Enabled() const { return enable_s3_ && !enable_slipstream_capsule_; }
    void setS3ProximityThreshold(dist_t threshold) { 
        if (s3_store_) s3_store_->proximity_threshold = threshold; 
    }
    
    // S3 metrics
    uint64_t getS3Hits() const { return s3_hits_.load(); }
    uint64_t getS3Misses() const { return s3_misses_.load(); }
    uint64_t getS3DistSaved() const { return s3_dist_saved_.load(); }

    // Slipstream aliases over the S3 locality-reuse path.
    void setEnableSlipstream(bool enable) {
        enable_s3_ = enable;
        enable_slipstream_capsule_ = enable;
        if (enable && !slipstream_capsules_) {
            slipstream_capsules_ = std::make_unique<SlipstreamCapsuleTable>();
        }
    }
    bool isSlipstreamEnabled() const { return enable_slipstream_capsule_; }
    void setSlipstreamTTL(uint64_t ttl_ns) { slipstream_ttl_ns_ = ttl_ns; }
    void setSlipstreamMode(int mode) { slipstream_mode_ = mode; }
    void setSlipstreamQualityGate(bool enable) { slipstream_quality_gate_ = enable; }
    bool isSlipstreamQualityGateEnabled() const { return slipstream_quality_gate_; }
    void setSlipstreamSkipRatio(dist_t ratio) {
        if (ratio < static_cast<dist_t>(0)) ratio = static_cast<dist_t>(0);
        if (ratio > static_cast<dist_t>(1)) ratio = static_cast<dist_t>(1);
        slipstream_skip_ratio_ = ratio;
    }
    dist_t getSlipstreamSkipRatio() const { return slipstream_skip_ratio_; }
    uint64_t getSlipstreamPublishCount() const { return slipstream_publish_.load(); }
    uint64_t getSlipstreamInsertPublishCount() const {
        return slipstream_insert_publish_.load();
    }
    uint64_t getSlipstreamSearchPublishCount() const {
        return slipstream_search_publish_.load();
    }
    uint64_t getSlipstreamHitCount() const { return s3_hits_.load(); }
    uint64_t getSlipstreamMissCount() const { return s3_misses_.load(); }
    uint64_t getSlipstreamSeedsInjected() const { return slipstream_seeds_injected_.load(); }
    uint64_t getSlipstreamQualityRejectCount() const {
        return slipstream_quality_reject_.load();
    }
    uint64_t getSlipstreamCooldownRejectCount() const {
        return slipstream_cooldown_reject_.load();
    }
    uint64_t getSlipstreamRegretCount() const {
        return slipstream_regret_.load();
    }
    uint64_t getSlipstreamHelpfulCount() const {
        return slipstream_helpful_.load();
    }

    // ===== PIGGYBACK CONTROL METHODS =====
    
    void setEnablePiggyback(bool enable) { enable_piggyback_ = enable; }
    bool isPiggybackEnabled() const { return enable_piggyback_; }
    
    // Register a piggyback request (called by search thread)
    std::shared_ptr<PiggybackRequest> registerPiggybackRequest(const void* query, const void* insert_vector, bool enable_jl = false) const {
        if (!enable_piggyback_) return nullptr;
        
        auto request = std::make_shared<PiggybackRequest>(query, insert_vector, data_size_ / sizeof(float), 
                                                         fstdistfunc_, dist_func_param_, enable_jl);
        
        std::unique_lock<std::shared_mutex> lock(piggyback_requests_mutex_);
        active_piggyback_requests_.push_back(request);
        return request;
    }
    
    // Cleanup completed requests
    void cleanupPiggybackRequests() const {
        std::unique_lock<std::shared_mutex> lock(piggyback_requests_mutex_);
        active_piggyback_requests_.erase(
            std::remove_if(active_piggyback_requests_.begin(), active_piggyback_requests_.end(),
                          [](const std::weak_ptr<PiggybackRequest>& weak_req) {
                              auto req = weak_req.lock();
                              return !req || !req->active.load();
                          }), 
            active_piggyback_requests_.end());
    }
    
    // Piggyback metrics
    uint64_t getPiggybackHits() const { return piggyback_hits_.load(); }
    uint64_t getPiggybackMisses() const { return piggyback_misses_.load(); }
    uint64_t getPiggybackDistSaved() const { return piggyback_dist_saved_.load(); }
    uint64_t getPiggybackExactComputations() const { return piggyback_exact_computations_.load(); }
    uint64_t getPiggybackJLComputations() const { return piggyback_jl_computations_.load(); }
    
private:
    tableint computeSlipstreamScoutBasin(
        tableint ride_anchor, const void* query_data,
        BaseFilterFunctor* isIdAllowed,
        size_t view_limit = std::numeric_limits<size_t>::max()) const {
        (void)query_data;
        if (ride_anchor < 0 || ride_anchor >= cur_element_count) {
            return ride_anchor;
        }
        if (element_levels_[ride_anchor] < 1) {
            return ride_anchor;
        }

        const bool enforce_view_limit =
            view_limit != std::numeric_limits<size_t>::max();
        tableint basin_anchor = ride_anchor;

        std::shared_lock<std::shared_mutex> lock;
        if (use_node_lock_in_search_) {
            lock = std::shared_lock<std::shared_mutex>(
                link_list_locks_[ride_anchor]);
        }

        linklistsizeint* ll_data;
        if (enforce_view_limit) {
            ll_data = get_linklist_at_snapshot(
                ride_anchor, 1, static_cast<uint32_t>(view_limit - 1));
        } else {
            ll_data = get_linklist(ride_anchor, 1);
        }
        if (ll_data == nullptr) {
            return basin_anchor;
        }

        unsigned int* data = (unsigned int*)ll_data;
        int size = getListCount(data);
        tableint* datal = (tableint*)(data + 1);
        for (int i = 0; i < size; ++i) {
            tableint cand = datal[i];
            if (cand < 0 || cand >= cur_element_count) continue;
            if (enforce_view_limit &&
                static_cast<size_t>(cand) >= view_limit)
                continue;
            if (isMarkedDeleted(cand)) continue;
            if (isIdAllowed &&
                !(*isIdAllowed)(getExternalLabel(cand)))
                continue;
            if (basin_anchor == ride_anchor || cand < basin_anchor) {
                basin_anchor = cand;
            } 
        }

        return basin_anchor;
    }

    uint32_t getSlipstreamEpochBand(size_t view_limit) const {
        size_t watermark =
            (view_limit == std::numeric_limits<size_t>::max())
                ? static_cast<size_t>(
                      std::max<long>(global_watermark_.load(std::memory_order_acquire), 0))
                : view_limit;
        return static_cast<uint32_t>(watermark / SLIPSTREAM_EPOCH_BAND);
    }

    size_t countSlipstreamRetainedSeeds(
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates,
        const std::array<tableint, SLIPSTREAM_MAX_SEEDS>& injected_nodes,
        size_t injected_count, size_t keep_k) const {
        if (injected_count == 0) {
            return 0;
        }

        while (top_candidates.size() > keep_k) {
            top_candidates.pop();
        }

        size_t retained = 0;
        while (!top_candidates.empty()) {
            tableint node_id = top_candidates.top().second;
            top_candidates.pop();
            for (size_t i = 0; i < injected_count; ++i) {
                if (injected_nodes[i] == node_id) {
                    ++retained;
                    break;
                }
            }
        }
        return retained;
    }

    void publishSlipstreamCapsule(
        tableint ride_anchor, tableint basin_anchor,
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates,
        size_t view_limit, bool from_search) const {
        if (!enable_slipstream_capsule_ || !slipstream_capsules_ ||
            top_candidates.empty()) {
            return;
        }

        std::array<tableint, SLIPSTREAM_MAX_SEEDS> seeds{};
        uint32_t count = 0;
        while (!top_candidates.empty() && count < SLIPSTREAM_MAX_SEEDS) {
            const auto cand = top_candidates.top();
            top_candidates.pop();
            if (view_limit != std::numeric_limits<size_t>::max() &&
                static_cast<size_t>(cand.second) >= view_limit) {
                continue;
            }
            seeds[count++] = cand.second;
        }

        if (count == 0) return;

        std::reverse(seeds.begin(), seeds.begin() + count);
        slipstream_capsules_->publish(ride_anchor, basin_anchor, seeds, count,
                                      getSlipstreamEpochBand(view_limit),
                                      rdtsc_ns());
        slipstream_publish_.fetch_add(1, std::memory_order_relaxed);
        if (from_search) {
            slipstream_search_publish_.fetch_add(1, std::memory_order_relaxed);
        } else {
            slipstream_insert_publish_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Core Piggyback distance computation: dist(q,u)² = ||δ||² + dist(v,u)² + 2·δᵀ·(v-u)
    dist_t computePiggybackDistance(const PiggybackRequest& request, tableint node_u, 
                                   dist_t dist_v_u, const void* v_data) const {
        piggyback_exact_computations_.fetch_add(1, std::memory_order_relaxed);
        
        const float* u_data = reinterpret_cast<const float*>(getDataByInternalId(node_u));
        const float* v_ptr = static_cast<const float*>(v_data);
        
        // Compute inner product δᵀ·(v-u)
        float inner_product = 0.0f;
        size_t dim = data_size_ / sizeof(float);
        for (size_t i = 0; i < dim; i++) {
            inner_product += request.delta[i] * (v_ptr[i] - u_data[i]);
        }
        
        // Apply formula: dist(q,u)² = ||δ||² + dist(v,u)² + 2·δᵀ·(v-u)
        dist_t dist_qu_sq = request.delta_sq + dist_v_u * dist_v_u + 2 * inner_product;
        return sqrt(std::max(static_cast<dist_t>(0), dist_qu_sq));
    }
    
    // JL-projected approximate distance computation  
    dist_t computePiggybackDistanceJL(const PiggybackRequest& request, tableint node_u,
                                     dist_t dist_v_u, const void* v_data) const {
        if (!request.use_jl_projection || !request.jl_matrix) {
            // Fallback to exact computation
            return computePiggybackDistance(request, node_u, dist_v_u, v_data);
        }
        
        piggyback_jl_computations_.fetch_add(1, std::memory_order_relaxed);
        
        const float* u_data = reinterpret_cast<const float*>(getDataByInternalId(node_u));
        const float* v_ptr = static_cast<const float*>(v_data);
        size_t dim = data_size_ / sizeof(float);
        
        // Project (v-u) to low-dimensional space
        std::vector<float> vu_projected(request.jl_dim, 0.0f);
        for (int i = 0; i < request.jl_dim; i++) {
            for (size_t j = 0; j < dim; j++) {
                vu_projected[i] += (*request.jl_matrix)[i][j] * (v_ptr[j] - u_data[j]);
            }
        }
        
        // Approximate inner product in low-dim space
        float inner_product_approx = 0.0f;
        for (int i = 0; i < request.jl_dim; i++) {
            inner_product_approx += request.delta_projected[i] * vu_projected[i];
        }
        
        // Apply formula with approximation
        dist_t dist_qu_sq = request.delta_sq + dist_v_u * dist_v_u + 2 * inner_product_approx;
        return sqrt(std::max(static_cast<dist_t>(0), dist_qu_sq));
    }
    
    // Publish visited node to all active piggyback requests
    void publishToPiggyback(tableint node_u, dist_t dist_v_u, const void* v_data) const {
        if (!enable_piggyback_) return;
        
        std::shared_lock<std::shared_mutex> lock(piggyback_requests_mutex_);
        
        for (auto& request_ptr : active_piggyback_requests_) {
            if (!request_ptr || !request_ptr->active.load()) continue;
            
            // Compute piggyback distance
            dist_t piggyback_dist;
            if (request_ptr->use_jl_projection) {
                piggyback_dist = computePiggybackDistanceJL(*request_ptr, node_u, dist_v_u, v_data);
            } else {
                piggyback_dist = computePiggybackDistance(*request_ptr, node_u, dist_v_u, v_data);
            }
            
            // Add to results (thread-safe)
            std::lock_guard<std::mutex> results_lock(request_ptr->results_mutex);
            request_ptr->results.emplace_back(piggyback_dist, node_u);
        }
    }

public:

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id),
                                       dist_func_param_);
            
            // Piggyback: publish this distance computation
            publishToPiggyback(ep_id, dist, data_point);
            
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound &&
                top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock<std::shared_mutex> lock(
                link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum *
                        // size_links_per_element0_);
            if (layer == 0) {
                data = (int *)get_linklist0(curNodeNum);
            } else {
                data = (int *)get_linklist(curNodeNum, layer);
                //                    data = (int *) (linkLists_[curNodeNum] +
                //                    (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *(data + 1) + 64),
                         _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(datal + j + 1)),
                             _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)),
                             _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 =
                    fstdistfunc_(data_point, currObj1, dist_func_param_);
                
                // Piggyback: publish this distance computation  
                publishToPiggyback(candidate_id, dist1, data_point);
                
                if (top_candidates.size() < ef_construction_ ||
                    lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second),
                                 _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }

    // bare_bone_search means there is no check for deletions and stop condition
    // is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
    searchBaseLayerST(
        tableint ep_id, const void *data_point, size_t ef,
        BaseFilterFunctor *isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t> *stop_condition = nullptr,
        size_t view_limit = std::numeric_limits<size_t>::max(),
        bool is_insert_operation = false) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            candidate_set;

        const bool enforce_view_limit =
            view_limit != std::numeric_limits<size_t>::max();
        const bool effective_bare = bare_bone_search && !enforce_view_limit;

        auto is_visible = [&](tableint internal_id) -> bool {
            return !enforce_view_limit ||
                   static_cast<size_t>(internal_id) < view_limit;
        };

        const bool trim_filter_query_enabled =
            isTrimRecoveryFilterEnabled() && enforce_view_limit;
        const float* trim_query_vec = nullptr;
        float trim_query_norm_sq = 0.0f;
        if (trim_filter_query_enabled) {
            trim_query_vec = reinterpret_cast<const float*>(data_point);
            trim_query_norm_sq = computeTrimVectorNormSq(trim_query_vec);
        }

        dist_t lowerBound;
        bool ep_visible = is_visible(ep_id);
        bool ep_allowed =
            effective_bare ||
            (!isMarkedDeleted(ep_id) &&
             ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))));
        if (ep_allowed && ep_visible) {
            char *ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!effective_bare && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id),
                                                    ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else if (ep_allowed && !ep_visible) {
            // Entry point is future node: still use for navigation seed
            // but don't add to results
            char *ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (effective_bare) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(
                        candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound &&
                                       top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            std::shared_lock<std::shared_mutex> lock;
            if (use_node_lock_in_search_) {
                lock = std::shared_lock<std::shared_mutex>(
                    link_list_locks_[current_node_id]);
            }

            // Use MVCC snapshot if view_limit is enforced
            linklistsizeint *ll_data;
            if (enforce_view_limit) {
                ll_data = get_linklist_at_snapshot(
                    current_node_id, 0, static_cast<uint32_t>(view_limit - 1));
            } else {
                ll_data = get_linklist0(current_node_id);
            }
            int *data = (int *)ll_data;
            size_t size = getListCount((linklistsizeint *)data);

            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations += size;
            }

            // Always count nodes visited for baseline comparison
            metric_nodes_visited.fetch_add(1, std::memory_order_relaxed);

            // Fast path: if ts <= snapshot, no future neighbors exist, skip
            // filtering
            const bool need_neighbor_filter =
                enforce_view_limit &&
                get_linklist0_ts(current_node_id) > (view_limit - 1);

            if (need_neighbor_filter) {
                metric_visibility_checks.fetch_add(size, std::memory_order_relaxed);
            }

#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *(data + 1) + 64),
                         _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ +
                             (*(data + 1)) * size_data_per_element_ +
                             offsetData_,
                         _MM_HINT_T0);
            _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

            auto process_candidate = [&](int candidate_id) -> bool {
                if (visited_array[candidate_id] == visited_array_tag) return false;
                visited_array[candidate_id] = visited_array_tag;

                // No Ghost Routing: skip future nodes entirely
                // (don't compute distance, don't add to candidate_set)
                if (!is_visible(candidate_id)) return false;

                char *currObj1 = (getDataByInternalId(candidate_id));
                dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                
                // Piggyback: publish this distance computation (only during inserts)
                if (is_insert_operation) {
                    publishToPiggyback(candidate_id, dist, data_point);
                }

                bool flag_consider_candidate;
                if (!effective_bare && stop_condition) {
                    flag_consider_candidate =
                        stop_condition->should_consider_candidate(dist, lowerBound);
                } else {
                    flag_consider_candidate =
                        top_candidates.size() < ef || lowerBound > dist;
                }

                if (flag_consider_candidate) {
                    candidate_set.emplace(-dist, candidate_id);

                    bool candidate_allowed =
                        effective_bare ||
                        (!isMarkedDeleted(candidate_id) &&
                         ((!isIdAllowed) ||
                          (*isIdAllowed)(getExternalLabel(candidate_id))));
                    if (candidate_allowed) {
                        top_candidates.emplace(dist, candidate_id);
                        if (!effective_bare && stop_condition) {
                            stop_condition->add_point_to_result(
                                getExternalLabel(candidate_id), currObj1, dist);
                        }
                    }

                    bool flag_remove_extra = false;
                    if (!effective_bare && stop_condition) {
                        flag_remove_extra = stop_condition->should_remove_extra();
                    } else {
                        flag_remove_extra = top_candidates.size() > ef;
                    }
                    while (flag_remove_extra) {
                        tableint id = top_candidates.top().second;
                        top_candidates.pop();
                        if (!effective_bare && stop_condition) {
                            stop_condition->remove_point_from_result(
                                getExternalLabel(id), getDataByInternalId(id), dist);
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                    }

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                    return true;
                }
                return false;
            };

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + j + 1)),
                             _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ +
                                 (*(data + j + 1)) * size_data_per_element_ +
                                 offsetData_,
                             _MM_HINT_T0);
#endif
                if (need_neighbor_filter &&
                    static_cast<size_t>(candidate_id) >= view_limit)
                    continue;

                process_candidate(candidate_id);
            }

            // Only check undo log if this node has one (fast path: 89% of nodes do not)
            if (enable_undo_recovery_ && enforce_view_limit && undo_log_ &&
                undo_log_->hasLog(current_node_id)) {
                metric_undo_log_lookups.fetch_add(1, std::memory_order_relaxed);
                const uint32_t* recovered_edges = nullptr;
                const uint32_t* recovered_causes = nullptr;
                const float* recovered_dist_node = nullptr;
                const float* recovered_dist_cause = nullptr;
                size_t num_recovered = undo_log_->getRecoverPointersEUL(
                    current_node_id, static_cast<uint32_t>(view_limit - 1),
                    &recovered_edges, &recovered_causes, &recovered_dist_node,
                    &recovered_dist_cause);
                (void)recovered_dist_node;
                const bool large_recovery_log =
                    num_recovered >= trim_recovery_large_log_threshold_;
                bool use_trim_filter =
                    trim_filter_query_enabled &&
                    lowerBound < std::numeric_limits<dist_t>::max();
                const bool use_margin_filter =
                    trim_recovery_margin_ratio_ >= 0.0f && large_recovery_log &&
                    top_candidates.size() >= ef_ &&
                    lowerBound < std::numeric_limits<dist_t>::max() &&
                    recovered_dist_node != nullptr &&
                    recovered_dist_cause != nullptr;
                auto should_skip_by_margin = [&](size_t j) -> bool {
                    if (!use_margin_filter) return false;
                    const float d_node = recovered_dist_node[j];
                    if (d_node <= 1e-12f) return false;
                    const float d_cause = recovered_dist_cause[j];
                    const float margin_ratio = (d_node - d_cause) / d_node;
                    if (margin_ratio < trim_recovery_margin_ratio_) return false;
                    metric_trim_margin_filtered.fetch_add(
                        1, std::memory_order_relaxed);
                    return true;
                };
                if (num_recovered > 0) {
                    metric_recovered_edges_total.fetch_add(num_recovered, std::memory_order_relaxed);
                    size_t prev_max =
                        metric_recovered_edges_single_node_max.load(std::memory_order_relaxed);
                    while (prev_max < num_recovered &&
                           !metric_recovered_edges_single_node_max.compare_exchange_weak(
                               prev_max, num_recovered, std::memory_order_relaxed)) {
                    }
                    size_t useful = 0;
                    if (use_trim_filter) {
                        const float cos_factor = trimRecoveryCosFactor();
                        struct TrimRecoveryMember {
                            float upper_bound;
                            float cause_dist_sq;
                            tableint node;
                        };
                        struct TrimCauseGroup {
                            tableint cause;
                            float cause_upper_bound;
                            float cause_dist_lb_sq;
                            float min_cause_member_dist;
                            float best_member_upper_bound;
                            std::vector<TrimRecoveryMember> members;
                        };
                        std::vector<TrimCauseGroup> groups;
                        groups.reserve(num_recovered);
                        std::unordered_map<tableint, size_t> cause_to_group;
                        cause_to_group.reserve(num_recovered);

                        for (size_t j = 0; j < num_recovered; ++j) {
                            int recovered_id = static_cast<int>(recovered_edges[j]);
                            if (visited_array[recovered_id] == visited_array_tag) continue;
                            if (!is_visible(recovered_id)) continue;
                            if (should_skip_by_margin(j)) continue;

                            metric_trim_candidates_estimated.fetch_add(
                                1, std::memory_order_relaxed);

                            float upper_bound = computeTrimRecoveryUpperBound(
                                trim_query_vec, trim_query_norm_sq,
                                static_cast<tableint>(recovered_id), cos_factor);
                            tableint cause_id = static_cast<tableint>(recovered_causes[j]);
                            float cause_dist_sq = recovered_dist_cause
                                                      ? recovered_dist_cause[j]
                                                      : 0.0f;

                            auto group_it = cause_to_group.find(cause_id);
                            size_t group_idx = groups.size();
                            if (group_it == cause_to_group.end()) {
                                float cause_upper_bound = computeTrimRecoveryUpperBound(
                                    trim_query_vec, trim_query_norm_sq, cause_id,
                                    cos_factor);
                                float cause_dist_lb_sq = std::max(
                                    0.0f, trim_query_norm_sq +
                                              trim_node_full_norm_sq_[cause_id] -
                                              2.0f * cause_upper_bound);
                                groups.push_back(TrimCauseGroup{
                                    cause_id,
                                    cause_upper_bound,
                                    cause_dist_lb_sq,
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity(),
                                    {}});
                                groups.back().members.reserve(4);
                                cause_to_group.emplace(cause_id, group_idx);
                            } else {
                                group_idx = group_it->second;
                            }

                            TrimCauseGroup& group = groups[group_idx];
                            float cause_member_dist =
                                std::sqrt(std::max(0.0f, cause_dist_sq));
                            if (cause_member_dist < group.min_cause_member_dist) {
                                group.min_cause_member_dist = cause_member_dist;
                            }
                            if (upper_bound > group.best_member_upper_bound) {
                                group.best_member_upper_bound = upper_bound;
                            }
                            group.members.push_back(TrimRecoveryMember{
                                upper_bound, cause_dist_sq,
                                static_cast<tableint>(recovered_id)});
                        }

                        metric_trim_groups_total.fetch_add(
                            groups.size(), std::memory_order_relaxed);
                        for (auto& group : groups) {
                            if (group.members.empty()) continue;

                            const float cause_lb =
                                std::sqrt(std::max(0.0f, group.cause_dist_lb_sq));
                            const float best_member_lb =
                                std::max(0.0f, cause_lb - group.min_cause_member_dist);
                            const float best_member_lb_sq =
                                best_member_lb * best_member_lb;
                            const float relaxed_group_threshold =
                                static_cast<float>(lowerBound) /
                                trim_recovery_relax_factor_;
                            const float cause_theta = 0.5f *
                                (trim_query_norm_sq +
                                 trim_node_full_norm_sq_[group.cause] -
                                 relaxed_group_threshold);
                            const bool prune_group =
                                large_recovery_log && group.members.size() > 1 &&
                                group.cause_upper_bound <= cause_theta &&
                                best_member_lb_sq >= relaxed_group_threshold;
                            if (prune_group) {
                                metric_trim_groups_pruned.fetch_add(
                                    1, std::memory_order_relaxed);
                                metric_trim_candidates_filtered.fetch_add(
                                    group.members.size(), std::memory_order_relaxed);
                                continue;
                            }

                            std::sort(group.members.begin(), group.members.end(),
                                      [](const TrimRecoveryMember& a,
                                         const TrimRecoveryMember& b) {
                                          return a.upper_bound > b.upper_bound;
                                      });
                            size_t per_group_budget =
                                std::numeric_limits<size_t>::max();
                            if (large_recovery_log) {
                                if (group.members.size() >= 8 || num_recovered >= 16) {
                                    per_group_budget = 1;
                                } else if (group.members.size() >= 4 || num_recovered >= 8) {
                                    per_group_budget = 2;
                                } else {
                                    per_group_budget = 3;
                                }
                            }
                            const size_t exact_budget =
                                std::min(group.members.size(), per_group_budget);
                            for (size_t gi = 0; gi < group.members.size(); ++gi) {
                                float upper_bound = group.members[gi].upper_bound;
                                tableint recovered_id = group.members[gi].node;
                                float theta_tau = 0.5f *
                                    (trim_query_norm_sq +
                                     trim_node_full_norm_sq_[recovered_id] -
                                     static_cast<float>(lowerBound));
                                if (upper_bound <= theta_tau || gi >= exact_budget) {
                                    metric_trim_candidates_filtered.fetch_add(
                                        1, std::memory_order_relaxed);
                                    continue;
                                }

                                metric_trim_candidates_exact.fetch_add(
                                    1, std::memory_order_relaxed);
                                if (process_candidate(static_cast<int>(recovered_id))) {
                                    useful++;
                                }
                            }
                        }
                    } else {
                        for (size_t j = 0; j < num_recovered; ++j) {
                            int recovered_id = static_cast<int>(recovered_edges[j]);
                            if (visited_array[recovered_id] == visited_array_tag) continue;
                            if (!is_visible(recovered_id)) continue;
                            if (should_skip_by_margin(j)) continue;
                            if (process_candidate(recovered_id)) {
                                useful++;
                            }
                        }
                    }
                    metric_recovered_edges_useful.fetch_add(useful, std::memory_order_relaxed);
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    void getNeighborsByHeuristic(
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first,
                                  top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M) break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist = fstdistfunc_(
                    getDataByInternalId(second_pair.second),
                    getDataByInternalId(curent_pair.second), dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }

    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *)(data_level0_memory_ +
                                   internal_id * size_data_per_element_ +
                                   offsetLevel0_);
    }

    linklistsizeint *get_linklist0(tableint internal_id,
                                   char *data_level0_memory_) const {
        return (linklistsizeint *)(data_level0_memory_ +
                                   internal_id * size_data_per_element_ +
                                   offsetLevel0_);
    }

    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *)(linkLists_[internal_id] +
                                   (level - 1) * size_links_per_element_);
    }

    linklistsizeint *get_linklist_at_level(tableint internal_id,
                                           int level) const {
        return level == 0 ? get_linklist0(internal_id)
                          : get_linklist(internal_id, level);
    }

    // Get timestamp (batch id) for level 0 link list
    uint32_t get_linklist0_ts(tableint internal_id) const {
        char *ptr = data_level0_memory_ + internal_id * size_data_per_element_ +
                    offsetLevel0_ + sizeof(linklistsizeint) +
                    maxM0_ * sizeof(tableint);
        return *reinterpret_cast<uint32_t *>(ptr);
    }

    // Set timestamp (batch id) for level 0 link list
    void set_linklist0_ts(tableint internal_id, uint32_t ts) {
        char *ptr = data_level0_memory_ + internal_id * size_data_per_element_ +
                    offsetLevel0_ + sizeof(linklistsizeint) +
                    maxM0_ * sizeof(tableint);
        *reinterpret_cast<uint32_t *>(ptr) = ts;
    }

    // Get timestamp for higher level link list
    uint32_t get_linklist_ts(tableint internal_id, int level) const {
        char *ptr = linkLists_[internal_id] +
                    (level - 1) * size_links_per_element_ +
                    sizeof(linklistsizeint) + maxM_ * sizeof(tableint);
        return *reinterpret_cast<uint32_t *>(ptr);
    }

    // Set timestamp for higher level link list
    void set_linklist_ts(tableint internal_id, int level, uint32_t ts) {
        char *ptr = linkLists_[internal_id] +
                    (level - 1) * size_links_per_element_ +
                    sizeof(linklistsizeint) + maxM_ * sizeof(tableint);
        *reinterpret_cast<uint32_t *>(ptr) = ts;
    }

    // Get timestamp at any level
    uint32_t get_linklist_ts_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0_ts(internal_id)
                          : get_linklist_ts(internal_id, level);
    }

    // Set timestamp at any level
    void set_linklist_ts_at_level(tableint internal_id, int level,
                                  uint32_t ts) {
        if (level == 0) {
            set_linklist0_ts(internal_id, ts);
        } else {
            set_linklist_ts(internal_id, level, ts);
        }
    }

    // Get older version pointer for level 0
    char *get_linklist0_older(tableint internal_id) const {
        char *ptr = data_level0_memory_ + internal_id * size_data_per_element_ +
                    offsetLevel0_ + sizeof(linklistsizeint) +
                    maxM0_ * sizeof(tableint) + sizeof(uint32_t);
        return *reinterpret_cast<char **>(ptr);
    }

    // Set older version pointer for level 0
    void set_linklist0_older(tableint internal_id, char *older) {
        char *ptr = data_level0_memory_ + internal_id * size_data_per_element_ +
                    offsetLevel0_ + sizeof(linklistsizeint) +
                    maxM0_ * sizeof(tableint) + sizeof(uint32_t);
        *reinterpret_cast<char **>(ptr) = older;
    }

    // Get older version pointer for higher level
    char *get_linklist_older(tableint internal_id, int level) const {
        char *ptr = linkLists_[internal_id] +
                    (level - 1) * size_links_per_element_ +
                    sizeof(linklistsizeint) + maxM_ * sizeof(tableint) +
                    sizeof(uint32_t);
        return *reinterpret_cast<char **>(ptr);
    }

    // Set older version pointer for higher level
    void set_linklist_older(tableint internal_id, int level, char *older) {
        char *ptr = linkLists_[internal_id] +
                    (level - 1) * size_links_per_element_ +
                    sizeof(linklistsizeint) + maxM_ * sizeof(tableint) +
                    sizeof(uint32_t);
        *reinterpret_cast<char **>(ptr) = older;
    }

    // Get older version pointer at any level
    char *get_linklist_older_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0_older(internal_id)
                          : get_linklist_older(internal_id, level);
    }

    // Set older version pointer at any level
    void set_linklist_older_at_level(tableint internal_id, int level,
                                     char *older) {
        if (level == 0) {
            set_linklist0_older(internal_id, older);
        } else {
            set_linklist_older(internal_id, level, older);
        }
    }

    // Get ts from a version block (used for version chain traversal)
    uint32_t get_version_ts(char *version_data, int level) const {
        size_t offset = sizeof(linklistsizeint) +
                        (level == 0 ? maxM0_ : maxM_) * sizeof(tableint);
        return *reinterpret_cast<uint32_t *>(version_data + offset);
    }

    // Get older pointer from a version block
    char *get_version_older(char *version_data, int level) const {
        size_t offset = sizeof(linklistsizeint) +
                        (level == 0 ? maxM0_ : maxM_) * sizeof(tableint) +
                        sizeof(uint32_t);
        return *reinterpret_cast<char **>(version_data + offset);
    }

    // Set older pointer in a version block
    void set_version_older(char *version_data, int level, char *older) {
        size_t offset = sizeof(linklistsizeint) +
                        (level == 0 ? maxM0_ : maxM_) * sizeof(tableint) +
                        sizeof(uint32_t);
        *reinterpret_cast<char **>(version_data + offset) = older;
    }

    // Get linklist for a specific snapshot (MVCC support) - lock-free version
    // chain traversal
    linklistsizeint *get_linklist_at_snapshot(tableint internal_id, int level,
                                              uint32_t snapshot) const {
        // First check current version's ts
        uint32_t current_ts = get_linklist_ts_at_level(internal_id, level);
        if (current_ts <= snapshot) {
            // Current version is valid for this snapshot
            return get_linklist_at_level(internal_id, level);
        }

        // Traverse version chain to find appropriate version
        char *version = get_linklist_older_at_level(internal_id, level);
        while (version != nullptr) {
            uint32_t ver_ts = get_version_ts(version, level);
            if (ver_ts <= snapshot) {
                return (linklistsizeint *)version;
            }
            version = get_version_older(version, level);
        }

        // Fallback to current version (should not happen normally)
        return get_linklist_at_level(internal_id, level);
    }

    tableint mutuallyConnectNewElement(
        const void *data_point, tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> &top_candidates,
        int level, bool isUpdate, uint32_t batch_ts = 0) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error(
                "Should be not be more than M_ candidates returned by the "
                "heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already
            // acquired
            std::unique_lock<std::shared_mutex> lock(link_list_locks_[cur_c],
                                                     std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error(
                    "The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *)(ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error(
                        "Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
            // Update ts immediately for fast filtering optimization
            if (batch_ts > 0) {
                set_linklist_ts_at_level(cur_c, level, batch_ts);
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock<std::shared_mutex> lock(
                link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error(
                    "Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error(
                    "Trying to make a link on a non-existent level");

            tableint *data = (tableint *)(ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of
            // `selectedNeighbors[idx]` then no need to modify any connections
            // or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    // No pruning: just add neighbor
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                    // Update ts immediately for fast filtering
                    if (batch_ts > 0) {
                        set_linklist_ts_at_level(selectedNeighbors[idx], level,
                                                 batch_ts);
                    }
                } else {
                    // Pruning case: need to save old version and update ts
                    // finding the "weakest" element to replace it with the new
                    // one
                    dist_t d_max = fstdistfunc_(
                        getDataByInternalId(cur_c),
                        getDataByInternalId(selectedNeighbors[idx]),
                        dist_func_param_);
                    // Heuristic:
                    std::priority_queue<
                        std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
                        candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                            fstdistfunc_(
                                getDataByInternalId(data[j]),
                                getDataByInternalId(selectedNeighbors[idx]),
                                dist_func_param_),
                            data[j]);
                    }

                    getNeighborsByHeuristic(candidates, Mcurmax);

                    // Collect new neighbors from candidates
                    std::unordered_set<tableint> new_neighbors;
                    std::vector<tableint> new_neighbor_list;
                    {
                        auto tmp = candidates;
                        while (!tmp.empty()) {
                            new_neighbors.insert(tmp.top().second);
                            new_neighbor_list.push_back(tmp.top().second);
                            tmp.pop();
                        }
                    }

                    // MVCC: Record pruned edges to undo log (level 0 only)
                    if (enable_mvcc_ && level == 0 && batch_ts > 0) {
                        for (size_t j = 0; j < sz_link_list_other; j++) {
                            tableint old_neighbor = data[j];
                            if (new_neighbors.find(old_neighbor) == new_neighbors.end()) {
                                tableint caused_by = cur_c;
                                dist_t old_dist = fstdistfunc_(
                                    getDataByInternalId(old_neighbor),
                                    getDataByInternalId(selectedNeighbors[idx]),
                                    dist_func_param_);
                                dist_t caused_by_dist = old_dist;
                                for (tableint nn : new_neighbor_list) {
                                    dist_t nn_to_old = fstdistfunc_(
                                        getDataByInternalId(nn),
                                        getDataByInternalId(old_neighbor),
                                        dist_func_param_);
                                    if (nn_to_old < old_dist) {
                                        caused_by = nn;
                                        caused_by_dist = nn_to_old;
                                        break;
                                    }
                                }
                                undo_log_->recordPrune(
                                    selectedNeighbors[idx], old_neighbor, caused_by,
                                    batch_ts, static_cast<float>(old_dist),
                                    static_cast<float>(caused_by_dist));
                            }
                        }
                    }

                    // MVCC: Save old version before pruning (version chain)
                    if (enable_mvcc_) {
                        size_t block_size = level == 0 ? size_links_level0_
                                                       : size_links_per_element_;
                        char *old_data = (level == 0 ? version_pool_level0_
                                                     : version_pool_higher_)
                                             ->allocate();
                        memcpy(old_data, ll_other, block_size);
                        set_linklist_older_at_level(selectedNeighbors[idx], level,
                                                    old_data);
                    }

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);

                    // MVCC: Update ts
                    if (enable_mvcc_ && batch_ts > 0) {
                        set_linklist_ts_at_level(selectedNeighbors[idx], level,
                                                 batch_ts);
                    }
                }
            }
        }

        return next_closest_entry_point;
    }

    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error(
                "Cannot resize, max element is less than the current number of "
                "elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);
        resizeTrimRecoveryCodes(new_max_elements);

        std::vector<std::shared_mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char *data_level0_memory_new = (char *)realloc(
            data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error(
                "Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char **linkLists_new =
            (char **)realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error(
                "Not enough memory: resizeIndex failed to allocate other "
                "layers");
        linkLists_ = linkLists_new;

        if (undo_log_) undo_log_->resize(new_max_elements);
        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize =
                element_levels_[i] > 0
                    ? size_links_per_element_ * element_levels_[i]
                    : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_,
                     cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize =
                element_levels_[i] > 0
                    ? size_links_per_element_ * element_levels_[i]
                    : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize) output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }

    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s,
                   size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open()) throw std::runtime_error("Cannot open file");

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count) max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error(
                    "Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        if (input.tellg() != total_filesize)
            throw std::runtime_error(
                "Index seems to be corrupted or unsupported");

        input.clear();

        input.seekg(pos, input.beg);

        data_level0_memory_ =
            (char *)malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error(
                "Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_,
                   cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) +
                                  sizeof(linklistsizeint) + sizeof(uint32_t) +
                                  sizeof(char *);  // +ts +version_ptr

        size_links_level0_ = maxM0_ * sizeof(tableint) +
                             sizeof(linklistsizeint) + sizeof(uint32_t) +
                             sizeof(char *);  // +ts +version_ptr
        std::vector<std::shared_mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS)
            .swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **)malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error(
                "Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        initWatermarkState(max_elements_);
        undo_log_ = std::make_unique<UndoLogManager>(max_elements_);
        initTrimRecoverySketchDims(107);
        for (size_t i = 0; i < cur_element_count; ++i) {
            refreshTrimRecoveryCode(static_cast<tableint>(i));
        }
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *)malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error(
                        "Not enough memory: loadIndex failed to allocate "
                        "linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; ++i) {
            insertion_complete_[i].store(true, std::memory_order_relaxed);
        }
        global_watermark_.store(
            cur_element_count ? static_cast<long>(cur_element_count - 1) : -1,
            std::memory_order_relaxed);

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();

        return;
    }

    template <typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char *data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *)dist_func_param_);
        std::vector<data_t> data;
        data_t *data_ptr = (data_t *)data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }

    /*
     * Marks an element with the given label deleted, does NOT really change the
     * current graph.
     */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }

    /*
     * Uses the last 16 bits of the memory for the linked list size to store the
     * mark, whereas maxM0_ has to be limited to the lower 16 bits, however,
     * still large enough in almost all cases.
     */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur =
                ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock<std::mutex> lock_deleted_elements(
                    deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error(
                "The requested to delete element is already deleted");
        }
    }

    /*
     * Removes the deleted mark of the node, does NOT really change the current
     * graph.
     *
     * Note: the method is not safe to use when replacement of deleted elements
     * is enabled, because elements marked as deleted can be completely removed
     * by addPoint
     */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }

    /*
     * Remove the deleted mark of the node.
     */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur =
                ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock<std::mutex> lock_deleted_elements(
                    deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error(
                "The requested to undelete element is not deleted");
        }
    }

    /*
     * Checks the first 16 bits of the memory to see if the element is marked
     * deleted.
     */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur =
            ((unsigned char *)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }

    unsigned short int getListCount(linklistsizeint *ptr) const {
        return *((unsigned short int *)ptr);
    }

    void setListCount(linklistsizeint *ptr, unsigned short int size) const {
        *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
    }

    /*
     * Adds point. Updates the point if it is already in the index.
     * If replacement of deleted elements is enabled: replaces previously
     * deleted point if any, updating it with new point
     */
    void addPoint(const void *data_point, labeltype label,
                  bool replace_deleted = false) override {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error(
                "Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock<std::mutex> lock_deleted_elements(
            deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted
            // element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }

    void updatePoint(const void *dataPoint, tableint internalId,
                     float updateNeighborProbability) {
        // update the feature vector associated with existing point with new
        // vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);
        refreshTrimRecoveryCode(internalId);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single
        // element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1) return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop =
                getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0) continue;

            sCand.insert(internalId);

            for (auto &&elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) >
                    updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop =
                    getConnectionsWithLock(elOneHop, layer);
                for (auto &&elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto &&neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>,
                                    CompareByFirst>
                    candidates;
                size_t size =
                    sCand.find(neigh) == sCand.end()
                        ? sCand.size()
                        : sCand.size() -
                              1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto &&cand : sCand) {
                    if (cand == neigh) continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh),
                                                   getDataByInternalId(cand),
                                                   dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic(candidates,
                                        layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock<std::shared_mutex> lock(
                        link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *)(ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId,
                                   elemLevel, maxLevelCopy);
    }

    void repairConnectionsForUpdate(const void *dataPoint,
                                    tableint entryPointInternalId,
                                    tableint dataPointInternalId,
                                    int dataPointLevel, int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(
                dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock<std::shared_mutex> lock(
                        link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)),
                                     _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d =
                            fstdistfunc_(dataPoint, getDataByInternalId(cand),
                                         dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error(
                "Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
                topCandidates = searchBaseLayer(currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
                filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`,
            // there could be cases where `topCandidates` could just contains
            // entry point itself. To prevent self loops, the `topCandidates` is
            // filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(
                        fstdistfunc_(dataPoint,
                                     getDataByInternalId(entryPointInternalId),
                                     dist_func_param_),
                        entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(
                    dataPoint, dataPointInternalId, filteredTopCandidates,
                    level, true);
            }
        }
    }

    std::vector<tableint> getConnectionsWithLock(tableint internalId,
                                                 int level) {
        std::unique_lock<std::shared_mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *)(data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }

    tableint addPoint(
        const void *data_point, labeltype label, int level,
        bool defer_watermark = false,
        uint32_t batch_ts = std::numeric_limits<uint32_t>::max(),
        tableint alloc_id = std::numeric_limits<tableint>::max()) {
        uint64_t _prof_start = rdtsc_ns();
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error(
                            "Can't use addPoint to update deleted elements if "
                            "replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            // Use preallocated ID if provided, otherwise allocate dynamically
            if (alloc_id != std::numeric_limits<tableint>::max()) {
                cur_c = alloc_id;
            } else {
                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error(
                        "The number of elements exceeds the specified limit");
                }
                cur_c =
                    cur_element_count.fetch_add(1, std::memory_order_acq_rel);
            }
            label_lookup_[label] = cur_c;
        }

        insertion_complete_[cur_c].store(false, std::memory_order_relaxed);
        std::unique_lock<std::shared_mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0) curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock<std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy) templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ +
                   offsetLevel0_,
               0, size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);
        refreshTrimRecoveryCode(cur_c);

        if (curlevel) {
            linkLists_[cur_c] =
                (char *)malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error(
                    "Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0,
                   size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            tableint slipstream_ride_anchor = currObj;
            tableint slipstream_basin_anchor = currObj;
            uint64_t _prof_upper_start = rdtsc_ns();
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(
                    data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock<std::shared_mutex> lock(
                            link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *)(data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point,
                                                    getDataByInternalId(cand),
                                                    dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                    if (level == 2) {
                        slipstream_ride_anchor = currObj;
                    }
                }
            }
            if (enable_slipstream_capsule_ && slipstream_capsules_) {
                slipstream_basin_anchor = computeSlipstreamScoutBasin(
                    slipstream_ride_anchor, data_point, nullptr);
            }

            prof_insert_.upper_nav_ns.fetch_add(rdtsc_ns() - _prof_upper_start, std::memory_order_relaxed);

            uint64_t _prof_beam_acc = 0, _prof_prune_acc = 0;
            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0;
                 level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                uint64_t _t_beam = rdtsc_ns();
                std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>,
                                    CompareByFirst>
                    top_candidates =
                        searchBaseLayer(currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(
                        fstdistfunc_(data_point,
                                     getDataByInternalId(enterpoint_copy),
                                     dist_func_param_),
                        enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                _prof_beam_acc += rdtsc_ns() - _t_beam;

                // Publish locality state for reuse at level 0.
                if (level == 0) {
                    if (enable_slipstream_capsule_ && slipstream_capsules_) {
                        publishSlipstreamCapsule(slipstream_ride_anchor,
                                                 slipstream_basin_anchor,
                                                 top_candidates,
                                                 std::numeric_limits<size_t>::max(),
                                                 false);
                    } else if (enable_s3_ && s3_store_) {
                        std::vector<std::pair<dist_t, tableint>> byproduct_candidates;
                        auto temp_candidates = top_candidates;
                        while (!temp_candidates.empty()) {
                            byproduct_candidates.push_back(temp_candidates.top());
                            temp_candidates.pop();
                        }
                        s3_store_->add(cur_c, std::move(byproduct_candidates),
                                       data_point, rdtsc_ns());
                        slipstream_publish_.fetch_add(1, std::memory_order_relaxed);
                    }
                }

                // Use batch_ts if provided, otherwise use cur_c
                uint32_t effective_ts =
                    (batch_ts == std::numeric_limits<uint32_t>::max())
                        ? cur_c
                        : batch_ts;
                uint64_t _t_prune = rdtsc_ns();
                currObj =
                    mutuallyConnectNewElement(data_point, cur_c, top_candidates,
                                              level, false, effective_ts);
                _prof_prune_acc += rdtsc_ns() - _t_prune;
            }
            prof_insert_.level0_beam_ns.fetch_add(_prof_beam_acc, std::memory_order_relaxed);
            prof_insert_.pruning_ns.fetch_add(_prof_prune_acc, std::memory_order_relaxed);
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        if (!defer_watermark) {
            markCompleted(cur_c, -1);
        }
        prof_insert_.count.fetch_add(1, std::memory_order_relaxed);
        prof_insert_.total_ns.fetch_add(rdtsc_ns() - _prof_start, std::memory_order_relaxed);
        return cur_c;
    }

    tableint addPointBatch(const void *batch_data,
                           const labeltype *batch_labels, size_t num_points) {
        if (num_points == 0) return (tableint)getWatermark();

        const char *base_ptr = static_cast<const char *>(batch_data);
        size_t stride = data_size_;

        // Preallocate ID range for this batch
        tableint start_id = cur_element_count.fetch_add(
            static_cast<size_t>(num_points), std::memory_order_acq_rel);
        uint32_t batch_ts = start_id + num_points - 1;

        // Check capacity
        if (start_id + num_points > max_elements_) {
            throw std::runtime_error(
                "The number of elements exceeds the specified limit");
        }

        size_t threads = std::min(num_threads_, num_points);
        std::vector<tableint> ids(num_points);

        if (threads <= 1) {
            for (size_t i = 0; i < num_points; ++i) {
                const void *point =
                    static_cast<const void *>(base_ptr + i * stride);
                tableint alloc_id = start_id + i;
                ids[i] = addPoint(point, batch_labels[i], -1, true, batch_ts,
                                  alloc_id);
            }
        } else {
            // Use TBB to schedule batches and avoid tiny tasks in the custom
            // thread pool.
            size_t grain = std::max<size_t>(
                32, (num_points + threads * 4 - 1) / (threads * 4));
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, num_points, grain),
                [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        const void *point =
                            static_cast<const void *>(base_ptr + i * stride);
                        tableint alloc_id = start_id + i;
                        ids[i] = addPoint(point, batch_labels[i], -1, true,
                                          batch_ts, alloc_id);
                    }
                });
        }

        for (auto id : ids) {
            markReady(id);
        }

        advanceWatermark();
        return (tableint)getWatermark();
    }

    std::vector<std::priority_queue<std::pair<dist_t, labeltype>>>
    searchKnnBatch(const void *query_data, size_t num_queries, size_t k,
                   BaseFilterFunctor *isIdAllowed = nullptr) const {
        std::vector<std::priority_queue<std::pair<dist_t, labeltype>>> results(
            num_queries);
        if (num_queries == 0) return results;

        // Use current watermark as view_limit for consistent snapshot
        // If MVCC disabled, use max (no filtering)
        const size_t view_limit = enable_mvcc_
            ? static_cast<size_t>(getWatermark()) + 1
            : std::numeric_limits<size_t>::max();

        const char *base_ptr = static_cast<const char *>(query_data);
        size_t stride = data_size_;

        size_t threads = std::min(num_threads_, num_queries);
        if (threads <= 1) {
            for (size_t i = 0; i < num_queries; ++i) {
                const void *query =
                    static_cast<const void *>(base_ptr + i * stride);
                results[i] = searchKnn(query, k, isIdAllowed, view_limit);
            }
        } else {
            size_t grain = std::max<size_t>(
                32, (num_queries + threads * 4 - 1) / (threads * 4));
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, num_queries, grain),
                [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        const void *query =
                            static_cast<const void *>(base_ptr + i * stride);
                        results[i] =
                            searchKnn(query, k, isIdAllowed, view_limit);
                    }
                });
        }
        return results;
    }

    std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
        const void *query_data, size_t k,
        BaseFilterFunctor *isIdAllowed = nullptr,
        size_t view_limit = std::numeric_limits<size_t>::max()) const override {
        uint64_t _prof_start = rdtsc_ns();
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;

        const bool enforce_view_limit =
            view_limit != std::numeric_limits<size_t>::max();

        uint64_t _prof_upper_s = rdtsc_ns();
        tableint currObj = enterpoint_node_;
        dist_t curdist =
            fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_),
                         dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                std::shared_lock<std::shared_mutex> lock;
                if (use_node_lock_in_search_) {
                    lock = std::shared_lock<std::shared_mutex>(
                        link_list_locks_[currObj]);
                }

                // Use MVCC snapshot if view_limit is enforced
                linklistsizeint *ll_data;
                if (enforce_view_limit) {
                    ll_data = get_linklist_at_snapshot(
                        currObj, level, static_cast<uint32_t>(view_limit - 1));
                } else {
                    ll_data = get_linklist(currObj, level);
                }
                unsigned int *data = (unsigned int *)ll_data;
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;

                // Fast path: if ts <= snapshot, no future neighbors exist
                const bool need_neighbor_filter =
                    enforce_view_limit &&
                    get_linklist_ts(currObj, level) > (view_limit - 1);

                tableint *datal = (tableint *)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    // Skip nodes beyond view_limit (only when ts > snapshot)
                    if (need_neighbor_filter &&
                        static_cast<size_t>(cand) >= view_limit)
                        continue;
                    dist_t d =
                        fstdistfunc_(query_data, getDataByInternalId(cand),
                                     dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        prof_search_.upper_nav_ns.fetch_add(rdtsc_ns() - _prof_upper_s, std::memory_order_relaxed);
        uint64_t _prof_l0_s = rdtsc_ns();

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates =
                searchBaseLayerST<true>(currObj, query_data, std::max(ef_, k),
                                        isIdAllowed, nullptr, view_limit);
        } else {
            top_candidates =
                searchBaseLayerST<false>(currObj, query_data, std::max(ef_, k),
                                         isIdAllowed, nullptr, view_limit);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(
                rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        prof_search_.level0_beam_ns.fetch_add(rdtsc_ns() - _prof_l0_s, std::memory_order_relaxed);
        prof_search_.count.fetch_add(1, std::memory_order_relaxed);
        prof_search_.total_ns.fetch_add(rdtsc_ns() - _prof_start, std::memory_order_relaxed);
        return result;
    }

    // Piggyback-enabled search: tries to reuse concurrent Insert computations
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnPiggyback(
        const void *query_data, size_t k, const void *insert_vector = nullptr,
        BaseFilterFunctor *isIdAllowed = nullptr,
        size_t view_limit = std::numeric_limits<size_t>::max(),
        bool enable_jl = false) const {
        
        uint64_t _prof_start = rdtsc_ns();
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        
        if (!enable_piggyback_ || !insert_vector) {
            // Fallback to regular search
            return searchKnn(query_data, k, isIdAllowed, view_limit);
        }

        // Check if query is close enough to insert vector for Piggyback to be useful
        dist_t insert_query_dist = fstdistfunc_(query_data, insert_vector, dist_func_param_);
        size_t dim = data_size_ / sizeof(float);
        
        // Heuristic threshold: if ||q-v|| > 0.1 * typical distance, use regular search  
        // This can be made configurable later
        static const dist_t PIGGYBACK_THRESHOLD = 0.1;
        
        if (insert_query_dist > PIGGYBACK_THRESHOLD) {
            piggyback_misses_.fetch_add(1, std::memory_order_relaxed);
            return searchKnn(query_data, k, isIdAllowed, view_limit);
        }

        // Register piggyback request
        auto piggyback_request = registerPiggybackRequest(query_data, insert_vector, enable_jl);
        if (!piggyback_request) {
            return searchKnn(query_data, k, isIdAllowed, view_limit);
        }

        // Wait a brief moment for Insert to accumulate some results
        // In practice, this would be coordinated differently
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        
        // Deactivate request to stop accumulating more results
        piggyback_request->active.store(false);
        
        // Process accumulated results
        std::lock_guard<std::mutex> results_lock(piggyback_request->results_mutex);
        
        if (piggyback_request->results.size() < k) {
            // Not enough results from piggyback, fallback to regular search
            piggyback_misses_.fetch_add(1, std::memory_order_relaxed);
            return searchKnn(query_data, k, isIdAllowed, view_limit);
        }

        // Sort results by distance and take top k
        std::sort(piggyback_request->results.begin(), piggyback_request->results.end());
        
        size_t num_results = std::min(k, piggyback_request->results.size());
        for (size_t i = 0; i < num_results; ++i) {
            tableint internal_id = piggyback_request->results[i].second;
            dist_t distance = piggyback_request->results[i].first;
            
            if (!isMarkedDeleted(internal_id) && 
                (view_limit == std::numeric_limits<size_t>::max() || 
                 static_cast<size_t>(internal_id) < view_limit) &&
                (!isIdAllowed || (*isIdAllowed)(getExternalLabel(internal_id)))) {
                result.emplace(distance, getExternalLabel(internal_id));
            }
        }

        piggyback_hits_.fetch_add(1, std::memory_order_relaxed);
        piggyback_dist_saved_.fetch_add(piggyback_request->results.size(), std::memory_order_relaxed);
        
        prof_search_.count.fetch_add(1, std::memory_order_relaxed);
        prof_search_.total_ns.fetch_add(rdtsc_ns() - _prof_start, std::memory_order_relaxed);
        
        return result;
    }

    // S3-enabled search that tries to use byproducts from recent inserts
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnS3(
        const void *query_data, size_t k,
        BaseFilterFunctor *isIdAllowed = nullptr,
        size_t view_limit = std::numeric_limits<size_t>::max()) const {
        
        if (!enable_s3_) {
            return searchKnn(query_data, k, isIdAllowed, view_limit);
        }

        if (enable_slipstream_capsule_ && slipstream_capsules_) {
            uint64_t _prof_start = rdtsc_ns();
            std::priority_queue<std::pair<dist_t, labeltype>> result;
            if (cur_element_count == 0) return result;

            const bool enforce_view_limit =
                view_limit != std::numeric_limits<size_t>::max();

            uint64_t _prof_upper_s = rdtsc_ns();
            tableint ride_anchor = enterpoint_node_;
            dist_t ride_dist =
                fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_),
                             dist_func_param_);

            for (int level = maxlevel_; level > 1; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    std::shared_lock<std::shared_mutex> lock;
                    if (use_node_lock_in_search_) {
                        lock = std::shared_lock<std::shared_mutex>(
                            link_list_locks_[ride_anchor]);
                    }

                    linklistsizeint *ll_data;
                    if (enforce_view_limit) {
                        ll_data = get_linklist_at_snapshot(
                            ride_anchor, level,
                            static_cast<uint32_t>(view_limit - 1));
                    } else {
                        ll_data = get_linklist(ride_anchor, level);
                    }
                    unsigned int *data = (unsigned int *)ll_data;
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations += size;

                    const bool need_neighbor_filter =
                        enforce_view_limit &&
                        get_linklist_ts(ride_anchor, level) > (view_limit - 1);

                    tableint *datal = (tableint *)(data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        if (need_neighbor_filter &&
                            static_cast<size_t>(cand) >= view_limit)
                            continue;
                        dist_t d = fstdistfunc_(
                            query_data, getDataByInternalId(cand),
                            dist_func_param_);

                        if (d < ride_dist) {
                            ride_dist = d;
                            ride_anchor = cand;
                            changed = true;
                        }
                    }
                }
            }

            prof_search_.upper_nav_ns.fetch_add(
                rdtsc_ns() - _prof_upper_s, std::memory_order_relaxed);
            uint64_t _prof_l0_s = rdtsc_ns();

            bool bare_bone_search = !num_deleted_ && !isIdAllowed;
            size_t full_budget = std::max(ef_, k);
            size_t warm_budget = full_budget;
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst>
                top_candidates;
            tableint scout_basin_anchor = ride_anchor;
            if (enable_slipstream_capsule_ && slipstream_capsules_) {
                scout_basin_anchor = computeSlipstreamScoutBasin(
                    ride_anchor, query_data, isIdAllowed, view_limit);
            }

            SlipstreamCapsuleSnapshot capsule;
            bool used_capsule = false;
            bool cooldown_blocked = false;
            tableint warm_entry = ride_anchor;
            dist_t best_seed_dist = std::numeric_limits<dist_t>::max();
            size_t better_seed_count = 0;
            std::array<tableint, SLIPSTREAM_MAX_SEEDS> injected_nodes{};
            size_t injected_count = 0;

            auto inject_node = [&](tableint node_id) {
                if (node_id >= cur_element_count) return;
                if (enforce_view_limit &&
                    static_cast<size_t>(node_id) >= view_limit)
                    return;
                if (isMarkedDeleted(node_id)) return;
                if (isIdAllowed &&
                    !(*isIdAllowed)(getExternalLabel(node_id)))
                    return;
                dist_t dist = fstdistfunc_(
                    query_data, getDataByInternalId(node_id), dist_func_param_);
                top_candidates.emplace(dist, node_id);
                if (injected_count < SLIPSTREAM_MAX_SEEDS) {
                    injected_nodes[injected_count++] = node_id;
                }
                if (dist < ride_dist) {
                    ++better_seed_count;
                }
                if (dist < best_seed_dist) {
                    best_seed_dist = dist;
                    warm_entry = node_id;
                }
            };

            if (slipstream_capsules_->consume(
                    ride_anchor, scout_basin_anchor,
                    getSlipstreamEpochBand(view_limit), slipstream_ttl_ns_,
                    &capsule, &cooldown_blocked)) {
                if (slipstream_mode_ == 2) {
                    std::minstd_rand rng(static_cast<unsigned int>(
                        rdtsc_ns() ^ reinterpret_cast<uintptr_t>(query_data)));
                    std::uniform_int_distribution<tableint> node_dist(
                        0, static_cast<tableint>(cur_element_count - 1));
                    size_t target = std::max<size_t>(k, capsule.num_seeds);
                    if (target > SLIPSTREAM_MAX_SEEDS) {
                        target = SLIPSTREAM_MAX_SEEDS;
                    }
                    std::unordered_set<tableint> sampled;
                    while (sampled.size() < target) {
                        sampled.insert(node_dist(rng));
                    }
                    for (tableint node_id : sampled) {
                        inject_node(node_id);
                    }
                } else {
                    for (size_t i = 0; i < capsule.num_seeds; ++i) {
                        inject_node(capsule.seeds[i]);
                    }
                }

                slipstream_seeds_injected_.fetch_add(
                    top_candidates.size(), std::memory_order_relaxed);
                bool quality_gate_reject =
                    top_candidates.empty() ||
                    (slipstream_quality_gate_ &&
                     (better_seed_count < SLIPSTREAM_MIN_BETTER_SEEDS ||
                      best_seed_dist >=
                          static_cast<dist_t>(ride_dist *
                                              SLIPSTREAM_MIN_IMPROVEMENT_RATIO)));
                if (quality_gate_reject) {
                    slipstream_quality_reject_.fetch_add(
                        1, std::memory_order_relaxed);
                } else if (!top_candidates.empty()) {
                    used_capsule = true;
                    s3_hits_.fetch_add(1, std::memory_order_relaxed);
                    s3_dist_saved_.fetch_add(capsule.num_seeds,
                                             std::memory_order_relaxed);
                    warm_budget = static_cast<size_t>(
                        std::max<double>(k, full_budget * slipstream_skip_ratio_));
                    if (warm_budget < top_candidates.size()) {
                        warm_budget = top_candidates.size();
                    }
                }
            } else if (cooldown_blocked) {
                slipstream_cooldown_reject_.fetch_add(
                    1, std::memory_order_relaxed);
            }

            tableint currObj = ride_anchor;
            dist_t curdist = ride_dist;
            if (!used_capsule) {
                for (int level = std::min(maxlevel_, 1); level > 0; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        std::shared_lock<std::shared_mutex> lock;
                        if (use_node_lock_in_search_) {
                            lock = std::shared_lock<std::shared_mutex>(
                                link_list_locks_[currObj]);
                        }

                        linklistsizeint *ll_data;
                        if (enforce_view_limit) {
                            ll_data = get_linklist_at_snapshot(
                                currObj, level,
                                static_cast<uint32_t>(view_limit - 1));
                        } else {
                            ll_data = get_linklist(currObj, level);
                        }
                        unsigned int *data = (unsigned int *)ll_data;
                        int size = getListCount(data);
                        metric_hops++;
                        metric_distance_computations += size;

                        const bool need_neighbor_filter =
                            enforce_view_limit &&
                            get_linklist_ts(currObj, level) > (view_limit - 1);

                        tableint *datal = (tableint *)(data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            if (need_neighbor_filter &&
                                static_cast<size_t>(cand) >= view_limit)
                                continue;
                            dist_t d = fstdistfunc_(
                                query_data, getDataByInternalId(cand),
                                dist_func_param_);

                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
                s3_misses_.fetch_add(1, std::memory_order_relaxed);
                if (bare_bone_search) {
                    top_candidates = searchBaseLayerST<true>(
                        currObj, query_data, full_budget, isIdAllowed, nullptr,
                        view_limit);
                } else {
                    top_candidates = searchBaseLayerST<false>(
                        currObj, query_data, full_budget, isIdAllowed, nullptr,
                        view_limit);
                }
            } else {
                std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>,
                                    CompareByFirst>
                    additional;
                if (bare_bone_search) {
                    additional = searchBaseLayerST<true>(
                        warm_entry, query_data, warm_budget, isIdAllowed, nullptr,
                        view_limit);
                } else {
                    additional = searchBaseLayerST<false>(
                        warm_entry, query_data, warm_budget, isIdAllowed, nullptr,
                        view_limit);
                }
                while (!additional.empty()) {
                    top_candidates.push(additional.top());
                    additional.pop();
                    if (top_candidates.size() > warm_budget) {
                        top_candidates.pop();
                    }
                }
            }

            if (used_capsule) {
                size_t retained = countSlipstreamRetainedSeeds(
                    top_candidates, injected_nodes, injected_count, k);
                bool regret = retained == 0;
                slipstream_capsules_->recordOutcome(
                    capsule.ride_key, regret, slipstream_ttl_ns_);
                if (regret) {
                    slipstream_regret_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    slipstream_helpful_.fetch_add(1, std::memory_order_relaxed);
                }
            }

            publishSlipstreamCapsule(ride_anchor, scout_basin_anchor,
                                     top_candidates, view_limit, true);

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (!top_candidates.empty()) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(
                    rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }

            prof_search_.level0_beam_ns.fetch_add(
                rdtsc_ns() - _prof_l0_s, std::memory_order_relaxed);
            prof_search_.count.fetch_add(1, std::memory_order_relaxed);
            prof_search_.total_ns.fetch_add(rdtsc_ns() - _prof_start,
                                            std::memory_order_relaxed);
            return result;
        }

        if (!s3_store_) {
            return searchKnn(query_data, k, isIdAllowed, view_limit);
        }
        
        uint64_t _prof_start = rdtsc_ns();
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;

        // Try to find a suitable byproduct
        Byproduct byproduct;
        bool has_byproduct = s3_store_->claimByproduct(
            &byproduct, query_data, fstdistfunc_, dist_func_param_,
            slipstream_ttl_ns_);

        if (has_byproduct) {
            // S3 hit: use byproduct as starting set for hybrid search
            s3_hits_.fetch_add(1, std::memory_order_relaxed);
            
            const bool enforce_view_limit = 
                view_limit != std::numeric_limits<size_t>::max();
            
            // Start with byproduct candidates
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst> top_candidates;
            if (byproduct.candidates.size() > cur_element_count) {
                s3_misses_.fetch_add(1, std::memory_order_relaxed);
                return searchKnn(query_data, k, isIdAllowed, view_limit);
            }
            size_t dist_saved = byproduct.candidates.size();
            size_t seed_budget = std::max<size_t>(k, 16);
            if (seed_budget > byproduct.candidates.size()) {
                seed_budget = byproduct.candidates.size();
            }

            auto inject_seed = [&](const std::pair<dist_t, tableint>& cand) {
                if (enforce_view_limit && cand.second > view_limit) return;
                if (isMarkedDeleted(cand.second)) return;
                if (isIdAllowed && !(*isIdAllowed)(getExternalLabel(cand.second))) return;
                dist_t dist = fstdistfunc_(query_data, getDataByInternalId(cand.second),
                                           dist_func_param_);
                top_candidates.emplace(dist, cand.second);
            };

            if (slipstream_mode_ == 2) {
                std::vector<size_t> order(byproduct.candidates.size());
                for (size_t i = 0; i < order.size(); ++i) order[i] = i;
                std::minstd_rand rng(
                    static_cast<unsigned int>(rdtsc_ns() ^ reinterpret_cast<uintptr_t>(query_data)));
                std::shuffle(order.begin(), order.end(), rng);
                for (size_t i = 0; i < seed_budget; ++i) {
                    inject_seed(byproduct.candidates[order[i]]);
                }
            } else {
                for (size_t i = 0; i < seed_budget; ++i) {
                    inject_seed(byproduct.candidates[byproduct.candidates.size() - 1 - i]);
                }
            }

            s3_dist_saved_.fetch_add(dist_saved, std::memory_order_relaxed);
            slipstream_seeds_injected_.fetch_add(top_candidates.size(),
                                                 std::memory_order_relaxed);

            if (slipstream_quality_gate_ && top_candidates.empty()) {
                slipstream_quality_reject_.fetch_add(1, std::memory_order_relaxed);
                s3_misses_.fetch_add(1, std::memory_order_relaxed);
                return searchKnn(query_data, k, isIdAllowed, view_limit);
            }
            
            // If we don't have enough candidates, do a short beam search
            if (top_candidates.size() < k) {
                // Use the best candidate as entry point for additional search
                tableint entry_point = top_candidates.empty() ? 
                    enterpoint_node_ : top_candidates.top().second;
                
                auto additional = searchBaseLayerST<false>(
                    entry_point, query_data, std::max(ef_, k), 
                    isIdAllowed, nullptr, view_limit);
                
                // Merge results
                while (!additional.empty() && top_candidates.size() < std::max(ef_, k)) {
                    top_candidates.push(additional.top());
                    additional.pop();
                }
            }
            
            // Extract final results
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (!top_candidates.empty()) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(
                    rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            
        } else {
            // S3 miss: fall back to regular search
            s3_misses_.fetch_add(1, std::memory_order_relaxed);
            result = searchKnn(query_data, k, isIdAllowed, view_limit);
        }
        
        return result;
    }

    std::vector<std::pair<dist_t, labeltype>> searchStopConditionClosest(
        const void *query_data, BaseSearchStopCondition<dist_t> &stop_condition,
        BaseFilterFunctor *isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist =
            fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_),
                         dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                std::shared_lock<std::shared_mutex> lock;
                if (use_node_lock_in_search_) {
                    lock = std::shared_lock<std::shared_mutex>(
                        link_list_locks_[currObj]);
                }

                data = (unsigned int *)get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;

                tableint *datal = (tableint *)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d =
                        fstdistfunc_(query_data, getDataByInternalId(cand),
                                     dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0,
                                                  isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }

    void checkIntegrity() {
        int connections_checked = 0;
        std::vector<int> inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *)(ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0],
                max1 = inbound_connections_num[0];
            for (int i = 0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1
                      << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked
                  << " connections\n";
    }
};
}  // namespace annchor
