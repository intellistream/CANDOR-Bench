#ifndef ANNCHOR_MVCC_HNSWALG_H
#define ANNCHOR_MVCC_HNSWALG_H

#include <assert.h>
#include <stdlib.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <set>
#include <shared_mutex>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <time.h>

#include "hnswlib.h"
#include "util.h"
#include "visited_list_pool.h"
#include "undo_log.h"

namespace annchor {

// SnapshotTracker for lock-free snapshot management
class SnapshotTracker {
public:
    static constexpr size_t MAX_SLOTS = 256;
    static constexpr long INACTIVE = std::numeric_limits<long>::max();
    
private:
    struct alignas(64) Slot {  // Cache-line aligned
        std::atomic<long> timestamp{INACTIVE};
    };
    
    Slot slots_[MAX_SLOTS];
    std::atomic<size_t> next_slot_{0};
    
public:
    class Guard {
    private:
        SnapshotTracker* tracker_;
        size_t slot_;
        bool active_;
        
    public:
        Guard(SnapshotTracker* tracker, size_t slot, long timestamp) 
            : tracker_(tracker), slot_(slot), active_(false) {
            if (tracker_ && slot_ < MAX_SLOTS) {
                tracker_->enter(slot_, timestamp);
                active_ = true;
            }
        }
        
        ~Guard() {
            if (active_ && tracker_ && slot_ < MAX_SLOTS) {
                tracker_->exit(slot_);
            }
        }
        
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
        
        Guard(Guard&& other) noexcept 
            : tracker_(other.tracker_), slot_(other.slot_), active_(other.active_) {
            other.active_ = false;
        }
        
        Guard& operator=(Guard&& other) noexcept {
            if (this != &other) {
                if (active_ && tracker_ && slot_ < MAX_SLOTS) {
                    tracker_->exit(slot_);
                }
                tracker_ = other.tracker_;
                slot_ = other.slot_;
                active_ = other.active_;
                other.active_ = false;
            }
            return *this;
        }
    };
    
    size_t allocate_slot() {
        size_t attempts = 0;
        while (attempts < MAX_SLOTS * 2) {
            size_t slot = next_slot_.fetch_add(1, std::memory_order_relaxed) % MAX_SLOTS;
            long expected = INACTIVE;
            if (slots_[slot].timestamp.compare_exchange_weak(expected, INACTIVE, 
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                return slot;
            }
            attempts++;
        }
        return MAX_SLOTS;  // No available slot
    }
    
    void enter(size_t slot, long timestamp) {
        if (slot < MAX_SLOTS) {
            slots_[slot].timestamp.store(timestamp, std::memory_order_release);
        }
    }
    
    void exit(size_t slot) {
        if (slot < MAX_SLOTS) {
            slots_[slot].timestamp.store(INACTIVE, std::memory_order_release);
        }
    }
    
    long min_active_snapshot() const {
        long min_ts = INACTIVE;
        for (size_t i = 0; i < MAX_SLOTS; i++) {
            long ts = slots_[i].timestamp.load(std::memory_order_acquire);
            if (ts != INACTIVE && ts < min_ts) {
                min_ts = ts;
            }
        }
        return min_ts;
    }
};

// Thread-local snapshot slot
thread_local size_t tls_snapshot_slot = SnapshotTracker::MAX_SLOTS;

// Thread-local: true when this thread holds a node link lock (non-preemptible region)
thread_local bool tls_in_link_critical_section = false;
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;
static constexpr size_t SEARCH_PATH_CAPTURE_CAP = 128;

struct SearchWorkStats {
    bool capture_path = false;
    uint64_t entry_ns = 0;
    uint64_t upper_search_ns = 0;
    uint64_t base_search_ns = 0;
    uint64_t result_materialize_ns = 0;
    uint64_t snapshot_guard_ns = 0;
    uint64_t visited_list_get_ns = 0;
    uint64_t visited_list_release_ns = 0;
    uint64_t upper_lock_wait_ns = 0;
    uint64_t level0_lock_wait_ns = 0;
    uint64_t distance_computations = 0;
    uint64_t upper_distance_computations = 0;
    uint64_t level0_distance_computations = 0;
    uint64_t upper_hops = 0;
    uint64_t upper_edges_scanned = 0;
    uint64_t level0_expansions = 0;
    uint64_t level0_edges_scanned = 0;
    uint64_t candidate_pops = 0;
    uint64_t candidate_pushes = 0;
    uint64_t visited_nodes = 0;
    uint64_t result_pushes = 0;
    uint64_t invisible_expansions = 0;
    uint64_t invisible_edges = 0;
    uint64_t invisible_candidate_dist_comps = 0;
    uint64_t invisible_candidate_enqueues = 0;
    uint64_t future_skip_hops = 0;
    uint32_t path_count = 0;
    uint32_t path_labels[SEARCH_PATH_CAPTURE_CAP] = {};
    float path_dists[SEARCH_PATH_CAPTURE_CAP] = {};
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

    std::vector<tableint> m2_base_insert_neighbors_;
    std::vector<uint16_t> m2_base_insert_neighbor_counts_;
    size_t m2_base_insert_neighbor_stride_{0};

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
    mutable std::atomic<uint64_t> metric_graph_connect_calls{0};
    mutable std::atomic<uint64_t> metric_graph_connect_ns{0};
    mutable std::atomic<uint64_t> metric_graph_link_critical_ns{0};
    mutable std::atomic<uint64_t> metric_graph_unique_lock_wait_ns{0};
    mutable std::atomic<uint64_t> metric_graph_link_updates{0};
    mutable std::atomic<uint64_t> metric_graph_upper_search_ns{0};
    mutable std::atomic<uint64_t> metric_graph_upper_search_dist_comps{0};
    mutable std::atomic<uint64_t> metric_graph_upper_search_edges_scanned{0};
    mutable std::atomic<uint64_t> metric_graph_base_search_ns{0};
    mutable std::atomic<uint64_t> metric_graph_base_search_expansions{0};
    mutable std::atomic<uint64_t> metric_graph_base_search_edges_scanned{0};
    mutable std::atomic<uint64_t> metric_graph_base_search_dist_comps{0};
    mutable std::atomic<uint64_t> metric_graph_select_new_neighbors_ns{0};
    mutable std::atomic<uint64_t> metric_graph_select_new_neighbors_input{0};
    mutable std::atomic<uint64_t> metric_graph_select_new_neighbors_selected{0};
    mutable std::atomic<uint64_t> metric_graph_select_new_neighbors_heuristic_dist_comps{0};
    mutable std::atomic<uint64_t> metric_graph_inserted_node_link_ns{0};
    mutable std::atomic<uint64_t> metric_graph_inserted_node_edges_written{0};
    mutable std::atomic<uint64_t> metric_graph_existing_neighbor_update_loop_ns{0};
    mutable std::atomic<uint64_t> metric_graph_existing_neighbor_load_scan_ns{0};
    mutable std::atomic<uint64_t> metric_graph_existing_neighbor_loaded_edges{0};
    mutable std::atomic<uint64_t> metric_graph_existing_neighbor_append_ns{0};
    mutable std::atomic<uint64_t> metric_graph_existing_neighbor_prune_ns{0};
    mutable std::atomic<uint64_t> metric_graph_existing_neighbor_prune_candidates{0};
    mutable std::atomic<uint64_t> metric_graph_existing_neighbor_prune_dist_comps{0};
    mutable std::atomic<uint64_t> metric_graph_existing_neighbor_undo_record_ns{0};
    mutable std::atomic<uint64_t> metric_graph_existing_neighbor_rewrite_ns{0};
    mutable std::atomic<uint64_t> metric_graph_existing_neighbor_edges_written{0};
    mutable std::atomic<uint64_t> metric_graph_existing_neighbor_edges_pruned{0};
    mutable std::atomic<uint64_t> metric_graph_existing_neighbor_visits{0};
    mutable std::atomic<uint64_t> metric_graph_existing_neighbor_appends{0};
    mutable std::atomic<uint64_t> metric_graph_existing_neighbor_prunes{0};
    mutable std::atomic<uint64_t> metric_graph_existing_neighbor_pruned_edges_recorded{0};
    mutable std::atomic<uint64_t> metric_search_rewrite_active_expansions{0};
    mutable std::atomic<uint64_t> metric_search_rewrite_active_queries{0};
    mutable std::atomic<uint64_t> metric_search_rewrite_recent_expansions{0};
    mutable std::atomic<uint64_t> metric_search_rewrite_recent_queries{0};
    mutable std::atomic<uint64_t> metric_search_rewrite_period_expansions{0};
    mutable std::atomic<uint64_t> metric_search_rewrite_period_queries{0};
    mutable std::atomic<uint64_t> metric_search_rewrite_period_active_sum{0};
    mutable std::atomic<uint64_t> metric_search_rewrite_period_active_max{0};
    mutable std::atomic<uint64_t> metric_search_invisible_expansions{0};
    mutable std::atomic<uint64_t> metric_search_invisible_expansion_edges{0};
    mutable std::atomic<uint64_t> metric_search_invisible_candidate_dist_comps{0};
    mutable std::atomic<uint64_t> metric_search_invisible_candidate_enqueues{0};
    mutable std::once_flag phase_overlap_diag_init_once_;
    mutable bool phase_overlap_diag_enabled_{false};
    mutable std::atomic<uint64_t> active_phase_existing_update_{0};
    mutable std::atomic<uint64_t> active_phase_link_critical_{0};
    mutable std::atomic<uint64_t> active_phase_load_scan_{0};
    mutable std::atomic<uint64_t> active_phase_append_{0};
    mutable std::atomic<uint64_t> active_phase_prune_{0};
    mutable std::atomic<uint64_t> active_phase_undo_record_{0};
    mutable std::atomic<uint64_t> active_phase_rewrite_{0};
    mutable std::atomic<uint64_t> metric_search_phase_overlap_samples{0};
    mutable std::atomic<uint64_t> metric_search_phase_existing_update_samples{0};
    mutable std::atomic<uint64_t> metric_search_phase_existing_update_queries{0};
    mutable std::atomic<uint64_t> metric_search_phase_link_critical_samples{0};
    mutable std::atomic<uint64_t> metric_search_phase_link_critical_queries{0};
    mutable std::atomic<uint64_t> metric_search_phase_load_scan_samples{0};
    mutable std::atomic<uint64_t> metric_search_phase_load_scan_queries{0};
    mutable std::atomic<uint64_t> metric_search_phase_append_samples{0};
    mutable std::atomic<uint64_t> metric_search_phase_append_queries{0};
    mutable std::atomic<uint64_t> metric_search_phase_prune_samples{0};
    mutable std::atomic<uint64_t> metric_search_phase_prune_queries{0};
    mutable std::atomic<uint64_t> metric_search_phase_undo_record_samples{0};
    mutable std::atomic<uint64_t> metric_search_phase_undo_record_queries{0};
    mutable std::atomic<uint64_t> metric_search_phase_rewrite_samples{0};
    mutable std::atomic<uint64_t> metric_search_phase_rewrite_queries{0};
    mutable std::atomic<uint64_t> metric_hot_region_search_marks{0};
    mutable std::atomic<uint64_t> metric_hot_region_repair_checks{0};
    mutable std::atomic<uint64_t> metric_hot_region_repair_conflicts{0};
    mutable std::atomic<uint64_t> metric_hot_region_repair_hot_neighbors{0};
    mutable std::atomic<uint64_t> metric_hot_region_repair_reordered_calls{0};
    mutable std::atomic<uint64_t> metric_hot_region_candidate_filter_calls{0};
    mutable std::atomic<uint64_t> metric_hot_region_candidate_filter_cold_kept{0};
    mutable std::atomic<uint64_t> metric_hot_region_candidate_filter_hot_dropped{0};
    mutable std::atomic<uint64_t> global_rewrite_epoch_{1};
    mutable std::atomic<uint64_t> global_rewrite_active_{0};
    bool rewrite_active_diag_enabled_{true};
    mutable std::once_flag insert_phase_trace_init_once_;
    mutable bool insert_phase_trace_enabled_{false};
    mutable uint64_t insert_phase_trace_from_id_{0};
    mutable std::mutex insert_phase_trace_mu_;
    mutable std::ofstream insert_phase_trace_;

    static uint64_t monotonicRawNs() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
               static_cast<uint64_t>(ts.tv_nsec);
    }

    bool insertPhaseTraceEnabled() const {
        std::call_once(insert_phase_trace_init_once_, [&]() {
            const char *path = std::getenv("ANNCHOR_INSERT_PHASE_TRACE");
            if (path == nullptr || path[0] == '\0') {
                insert_phase_trace_enabled_ = false;
                return;
            }
            const char *from_id = std::getenv("ANNCHOR_INSERT_PHASE_TRACE_FROM_ID");
            if (from_id != nullptr && from_id[0] != '\0') {
                char *end = nullptr;
                unsigned long long value = std::strtoull(from_id, &end, 10);
                if (end != from_id) {
                    insert_phase_trace_from_id_ = static_cast<uint64_t>(value);
                }
            }
            insert_phase_trace_.open(path, std::ios::out | std::ios::trunc);
            insert_phase_trace_enabled_ = insert_phase_trace_.is_open();
            if (insert_phase_trace_enabled_) {
                insert_phase_trace_
                    << "start_ns,end_ns,tid_hash,phase,point_id,level,existing_neighbor,"
                       "existing_neighbor_degree_before,edges_loaded,prune_candidates,dist_comps,"
                       "edges_written,edges_pruned\n";
            }
        });
        return insert_phase_trace_enabled_;
    }

    void traceInsertPhase(const char *phase, uint64_t start_ns, uint64_t end_ns,
                          tableint point_id, int level,
                          tableint existing_neighbor = std::numeric_limits<tableint>::max(),
                          uint64_t existing_neighbor_degree = 0, uint64_t edges_loaded = 0,
                          uint64_t prune_candidates = 0, uint64_t dist_comps = 0,
                          uint64_t edges_written = 0, uint64_t edges_pruned = 0) const {
        if (start_ns == 0 || !insertPhaseTraceEnabled()) return;
        if (static_cast<uint64_t>(point_id) < insert_phase_trace_from_id_) return;
        const auto tid_hash =
            static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::lock_guard<std::mutex> guard(insert_phase_trace_mu_);
        insert_phase_trace_ << start_ns << ',' << end_ns << ',' << tid_hash << ','
                            << phase << ',' << static_cast<uint64_t>(point_id)
                            << ',' << level << ','
                            << static_cast<uint64_t>(existing_neighbor) << ','
                            << existing_neighbor_degree << ',' << edges_loaded << ','
                            << prune_candidates << ',' << dist_comps << ','
                            << edges_written << ',' << edges_pruned << '\n';
    }

    bool phaseOverlapDiagEnabled() const {
        std::call_once(phase_overlap_diag_init_once_, [&]() {
            const char *enabled = std::getenv("ANNCHOR_PHASE_OVERLAP_DIAG");
            phase_overlap_diag_enabled_ =
                enabled != nullptr && enabled[0] != '\0' && enabled[0] != '0';
        });
        return phase_overlap_diag_enabled_;
    }

    struct ActivePhaseGuard {
        std::atomic<uint64_t>& active;
        explicit ActivePhaseGuard(std::atomic<uint64_t>& active_ref)
            : active(active_ref) {
            active.fetch_add(1, std::memory_order_acq_rel);
        }
        ~ActivePhaseGuard() {
            active.fetch_sub(1, std::memory_order_acq_rel);
        }
    };

    void sampleSearchPhaseOverlap(bool& hit_existing_update,
                                  bool& hit_link_critical,
                                  bool& hit_load_scan,
                                  bool& hit_append,
                                  bool& hit_prune,
                                  bool& hit_undo_record,
                                  bool& hit_rewrite) const {
        if (!phaseOverlapDiagEnabled()) return;
        metric_search_phase_overlap_samples.fetch_add(
            1, std::memory_order_relaxed);
        if (active_phase_existing_update_.load(std::memory_order_acquire) > 0) {
            metric_search_phase_existing_update_samples.fetch_add(
                1, std::memory_order_relaxed);
            hit_existing_update = true;
        }
        if (active_phase_link_critical_.load(std::memory_order_acquire) > 0) {
            metric_search_phase_link_critical_samples.fetch_add(
                1, std::memory_order_relaxed);
            hit_link_critical = true;
        }
        if (active_phase_load_scan_.load(std::memory_order_acquire) > 0) {
            metric_search_phase_load_scan_samples.fetch_add(
                1, std::memory_order_relaxed);
            hit_load_scan = true;
        }
        if (active_phase_append_.load(std::memory_order_acquire) > 0) {
            metric_search_phase_append_samples.fetch_add(
                1, std::memory_order_relaxed);
            hit_append = true;
        }
        if (active_phase_prune_.load(std::memory_order_acquire) > 0) {
            metric_search_phase_prune_samples.fetch_add(
                1, std::memory_order_relaxed);
            hit_prune = true;
        }
        if (active_phase_undo_record_.load(std::memory_order_acquire) > 0) {
            metric_search_phase_undo_record_samples.fetch_add(
                1, std::memory_order_relaxed);
            hit_undo_record = true;
        }
        if (active_phase_rewrite_.load(std::memory_order_acquire) > 0) {
            metric_search_phase_rewrite_samples.fetch_add(
                1, std::memory_order_relaxed);
            hit_rewrite = true;
        }
    }

    bool allow_replace_deleted_ = false;

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint>
        deleted_elements;  // contains internal ids of deleted elements

    std::unique_ptr<std::atomic<bool>[]> insertion_complete_;
    std::unique_ptr<std::atomic<bool>[]> mid_insert_deferred_pending_;
    std::unique_ptr<std::atomic<tableint>[]> ready_commit_ids_;
    std::atomic<long> global_watermark_{-1};

    std::unique_ptr<ThreadPool> thread_pool_;
    mutable std::mutex ready_commit_publish_lock_;

    // MVCC Mechanism 1 components
    bool enable_mvcc_{true};
    bool enable_non_prefix_visibility_{false};  // M3: hide in-flight, see completed
    bool enable_undo_recovery_{true};
    // Experiment-only visibility controls for HNSW baselines.
    // 0: return-visible only, while future nodes may still guide traversal.
    // 1: search normally, then filter/refill from the efSearch candidate pool.
    // 2: drop future nodes before enqueueing/expanding them.
    int search_visibility_mode_{0};
    bool search_visibility_by_label_{false};
    bool existing_neighbor_update_read_scan_only_{false};
    bool disable_existing_neighbor_undo_record_{false};
    int existing_neighbor_update_read_scan_only_from_id_{-1};
    int disable_existing_neighbor_undo_record_from_id_{-1};
    int rewrite_active_hold_us_{0};
    uint64_t rewrite_recent_window_{0};
    std::unique_ptr<UndoLogManager> undo_log_;
    mutable SnapshotTracker snapshot_tracker_;
    std::atomic<long> gc_horizon_{-1};
    std::atomic<long> gc_safe_snapshot_{-1};  
    bool enable_online_gc_{true};
    std::atomic<uint8_t>* node_modified_bitmap_{nullptr};
    std::atomic<uint64_t>* node_link_version_{nullptr};
    std::unique_ptr<std::atomic<uint8_t>[]> node_rewrite_active_{nullptr};
    std::unique_ptr<std::atomic<uint64_t>[]> node_rewrite_epoch_{nullptr};
    bool hot_region_repair_enabled_{false};
    bool hot_region_candidate_filter_enabled_{true};
    bool hot_region_reorder_selected_enabled_{false};
    bool hot_region_mark_candidates_{false};
    int hot_region_filter_min_cold_{0};
    int hot_region_preserve_nearest_{0};
    int hot_region_shift_{16};
    uint32_t hot_region_ttl_epochs_{4096};
    size_t hot_region_table_size_{0};
    std::unique_ptr<std::atomic<uint64_t>[]> hot_region_keys_{nullptr};
    std::unique_ptr<std::atomic<uint32_t>[]> hot_region_epochs_{nullptr};
    mutable std::atomic<uint32_t> hot_region_epoch_{1};

    // ── M2 (Mechanism 2) config ──────────────────────────────────────
    // m2_mode: 0=off, 1=eager, 2=cnr, 3=degree_only, 4=stagnation_only
    int m2_mode_{0};
    float m2_degree_threshold_{0.5f};   // trigger when visible_degree < threshold * maxM0_
    int m2_max_recover_{4};             // max edges to recover per node per trigger
    int m2_stagnation_hops_{3};         // beam stagnation window

    // M2 stats (atomic, safe for concurrent search)
    mutable std::atomic<long> m2_triggered_{0};
    mutable std::atomic<long> m2_recovered_edges_{0};
    mutable std::atomic<long> m2_useful_edges_{0};
    mutable std::atomic<long> m2_degree_triggers_{0};
    mutable std::atomic<long> m2_stagnation_triggers_{0};
    mutable std::atomic<long> m2_triangle_filtered_{0};
    mutable std::atomic<long> m2_extra_dist_comps_{0};
    mutable std::atomic<long> m2_modified_expansions_{0};
    mutable std::atomic<long> m2_recovery_attempts_{0};
    mutable std::atomic<long> m2_recovery_candidate_edges_{0};
    mutable std::atomic<uint64_t> m2_recovery_get_ns_{0};
    mutable std::atomic<uint64_t> m2_recovery_loop_ns_{0};

    // Fine-grained yield callback: called at a searchBaseLayer safe point after
    // one node expansion. ANNchorPreempt uses the local frontier shape to decide
    // whether pausing the writer is worthwhile under live search pressure.
    struct FineGrainYieldContext {
        int expanded_neighbors{0};
        int candidate_queue_size{0};
        int top_candidate_size{0};
        dist_t lower_bound{};
        dist_t best_frontier_dist{};
        bool top_candidates_full{false};
    };
    inline static thread_local FineGrainYieldContext last_fine_grain_yield_ctx_{};
    // Expansion throttle: fraction of neighbors to evaluate (0.0-1.0).
    // Set by preempt callback based on search pressure + convergence.
    // 1.0 = complete expansion (normal), 0.5 = half neighbors, etc.
    // Atomic (not thread_local) to avoid cross-TU TLS issues.
    // Read once per candidate expansion — negligible overhead.
    std::atomic<float> expansion_throttle_{1.0f};
    std::function<void()> fine_grain_yield_cb_;
    std::atomic<bool> fine_grain_yield_active_{false};

    void setFineGrainYieldCallback(std::function<void()> cb) {
        fine_grain_yield_cb_ = std::move(cb);
    }
    void setFineGrainYieldActive(bool active) {
        fine_grain_yield_active_.store(active, std::memory_order_release);
    }
    FineGrainYieldContext getLastFineGrainYieldContext() const {
        return last_fine_grain_yield_ctx_;
    }

    struct MidInsertServicePressure {
        int search_backlog{0};
        int priority_searches{0};
    };
    std::function<MidInsertServicePressure()> mid_insert_service_pressure_cb_;

    void setMidInsertServicePressureCallback(
        std::function<MidInsertServicePressure()> cb) {
        mid_insert_service_pressure_cb_ = std::move(cb);
    }

    // Mid-insert preempt hook (L0 between searchBaseLayer and connect)
    bool mid_insert_preempt_enable_{false};
    int mid_insert_preempt_k_{0};
    bool mid_insert_preempt_revalidate_{false};
    bool mid_insert_preempt_shadow_replan_{false};
    bool mid_insert_preempt_harm_guard_{false};
    bool mid_insert_shadow_snapshot_teacher_{true};
    bool mid_insert_harm_micro_replan_{false};
    int mid_insert_harm_micro_probe_ef_{0};
    bool mid_insert_harm_online_policy_{false};
    int mid_insert_harm_busy_wait_commits_{32};
    int mid_insert_harm_search_backlog_threshold_{0};
    int mid_insert_harm_priority_search_threshold_{0};
    int mid_insert_harm_full_foreign_outrank_threshold_{2};
    int mid_insert_harm_full_selected_frontier_touched_threshold_{2};
    bool mid_insert_harm_defer_enabled_{false};
    int mid_insert_harm_defer_queue_cap_{4096};
    int mid_insert_harm_defer_drain_budget_{1};
    int mid_insert_harm_defer_high_watermark_pct_{75};
    int mid_insert_preempt_max_wait_us_{200000};
    int mid_insert_preempt_activate_after_commits_{0};
    int mid_insert_preempt_every_n_{1};
    int mid_insert_preempt_max_inflight_{1};
    int mid_insert_harm_micro_pool_cap_{64};
    mutable std::atomic<long> ready_commit_counter_{0};
    mutable std::atomic<long> mid_insert_preempt_hits_{0};
    mutable std::atomic<long> mid_insert_preempt_timeouts_{0};
    mutable std::atomic<long> mid_insert_preempt_wait_commits_sum_{0};
    mutable std::atomic<long> mid_insert_preempt_revalidations_{0};
    mutable std::atomic<long> mid_insert_shadow_replans_{0};
    mutable std::atomic<long> mid_insert_preempt_slot_denied_{0};
    mutable std::atomic<long> mid_insert_preempt_global_lock_skipped_{0};
    mutable std::atomic<long> mid_insert_preempt_quiesce_breaks_{0};
    mutable std::atomic<long> mid_insert_preempt_inflight_{0};
    mutable std::atomic<long> mid_insert_candidate_changed_{0};
    mutable std::atomic<long> mid_insert_candidate_before_total_{0};
    mutable std::atomic<long> mid_insert_candidate_after_total_{0};
    mutable std::atomic<long> mid_insert_candidate_intersection_total_{0};
    mutable std::atomic<long> mid_insert_candidate_union_total_{0};
    mutable std::atomic<long> mid_insert_selected_changed_{0};
    mutable std::atomic<long> mid_insert_selected_before_total_{0};
    mutable std::atomic<long> mid_insert_selected_after_total_{0};
    mutable std::atomic<long> mid_insert_selected_intersection_total_{0};
    mutable std::atomic<long> mid_insert_selected_union_total_{0};
    mutable std::atomic<long> mid_insert_pause_low_slack_hits_{0};
    mutable std::atomic<long> mid_insert_pause_mild_risk_hits_{0};
    mutable std::atomic<long> mid_insert_pause_service_busy_hits_{0};
    mutable std::atomic<long> mid_insert_pause_search_pressure_busy_hits_{0};
    mutable std::atomic<long> mid_insert_pause_severe_risk_hits_{0};
    mutable std::atomic<long> mid_insert_pause_selected_frontier_touched_hits_{0};
    mutable std::atomic<long> mid_insert_pause_selected_frontier_touched_nodes_{0};
    mutable std::atomic<long> mid_insert_pause_candidate_frontier_touched_hits_{0};
    mutable std::atomic<long> mid_insert_pause_candidate_frontier_touched_nodes_{0};
    mutable std::atomic<long> mid_insert_pause_foreign_commit_nodes_{0};
    mutable std::atomic<long> mid_insert_pause_foreign_outrank_hits_{0};
    mutable std::atomic<long> mid_insert_pause_foreign_outrank_nodes_{0};
    mutable std::atomic<long> mid_insert_pause_boundary_blocked_hits_{0};
    mutable std::atomic<long> mid_insert_pause_boundary_blocked_candidates_{0};
    mutable std::atomic<long> mid_insert_pause_boundary_blocker_touched_hits_{0};
    mutable std::atomic<long> mid_insert_pause_boundary_blocker_touched_nodes_{0};
    mutable std::atomic<long> mid_insert_pause_boundary_witness_invalidated_hits_{0};
    mutable std::atomic<long> mid_insert_pause_boundary_witness_invalidated_candidates_{0};
    mutable std::atomic<long> mid_insert_pause_boundary_harm_risk_hits_{0};
    mutable std::atomic<long> mid_insert_pause_boundary_harm_cert_true_positive_{0};
    mutable std::atomic<long> mid_insert_pause_boundary_harm_cert_false_positive_{0};
    mutable std::atomic<long> mid_insert_pause_boundary_harm_cert_false_negative_{0};
    mutable std::atomic<long> mid_insert_pause_harm_risk_hits_{0};
    mutable std::atomic<long> mid_insert_pause_harm_guard_replans_{0};
    mutable std::atomic<long> mid_insert_pause_action_continue_{0};
    mutable std::atomic<long> mid_insert_pause_action_continue_mild_{0};
    mutable std::atomic<long> mid_insert_pause_action_defer_{0};
    mutable std::atomic<long> mid_insert_pause_action_micro_{0};
    mutable std::atomic<long> mid_insert_pause_action_full_{0};
    mutable std::atomic<long> mid_insert_pause_deferred_enqueued_{0};
    mutable std::atomic<long> mid_insert_pause_deferred_deduped_{0};
    mutable std::atomic<long> mid_insert_pause_deferred_dropped_{0};
    mutable std::atomic<long> mid_insert_pause_deferred_drained_{0};
    mutable std::atomic<long> mid_insert_pause_deferred_drain_runs_{0};
    mutable std::atomic<long> mid_insert_pause_deferred_queue_max_{0};
    mutable std::atomic<long> mid_insert_pause_defer_backpressure_hits_{0};
    mutable std::atomic<long> mid_insert_pause_micro_replans_{0};
    mutable std::atomic<long> mid_insert_pause_micro_probe_hits_{0};
    mutable std::atomic<long> mid_insert_pause_micro_probe_candidates_total_{0};
    mutable std::atomic<long> mid_insert_pause_micro_probe_candidates_max_{0};
    mutable std::atomic<long> mid_insert_pause_micro_replan_pool_total_{0};
    mutable std::atomic<long> mid_insert_pause_micro_replan_pool_max_{0};
    mutable std::atomic<long> mid_insert_pause_harm_cert_true_positive_{0};
    mutable std::atomic<long> mid_insert_pause_harm_cert_false_positive_{0};
    mutable std::atomic<long> mid_insert_pause_harm_cert_false_negative_{0};
    mutable std::atomic<long> mid_insert_shadow_snapshot_teacher_hits_{0};
    mutable std::atomic<long> mid_insert_shadow_snapshot_teacher_future_entrypoint_hits_{0};
    mutable std::atomic<long> mid_insert_shadow_snapshot_teacher_postwait_commit_drift_hits_{0};
    mutable std::atomic<long> mid_insert_shadow_snapshot_teacher_postwait_commit_drift_sum_{0};
    mutable std::atomic<long> mid_insert_shadow_snapshot_teacher_postwait_watermark_drift_hits_{0};
    mutable std::atomic<long> mid_insert_shadow_snapshot_teacher_postwait_watermark_drift_sum_{0};
    mutable std::mutex mid_insert_deferred_repair_queue_lock_;
    mutable std::mutex mid_insert_deferred_repair_drain_lock_;
    mutable std::deque<tableint> mid_insert_deferred_repair_queue_;
    mutable std::unordered_set<tableint> mid_insert_deferred_repair_set_;

    void setM2Mode(int mode) { m2_mode_ = mode; }
    int getM2Mode() const { return m2_mode_; }
    void setM2DegreeThreshold(float t) { m2_degree_threshold_ = t; }
    void setM2MaxRecover(int m) { m2_max_recover_ = m; }
    void setM2StagnationHops(int h) { m2_stagnation_hops_ = h; }
    void resetM2Stats() {
        m2_triggered_.store(0); m2_recovered_edges_.store(0);
        m2_useful_edges_.store(0); m2_degree_triggers_.store(0);
        m2_stagnation_triggers_.store(0); m2_triangle_filtered_.store(0);
        m2_extra_dist_comps_.store(0);
        m2_modified_expansions_.store(0);
        m2_recovery_attempts_.store(0);
        m2_recovery_candidate_edges_.store(0);
        m2_recovery_get_ns_.store(0);
        m2_recovery_loop_ns_.store(0);
    }

    static int readEnvInt(const char* name, int fallback) {
        const char* v = std::getenv(name);
        if (!v || !*v) return fallback;
        char* end = nullptr;
        long x = std::strtol(v, &end, 10);
        if (end == v) return fallback;
        return static_cast<int>(x);
    }

    static bool readEnvBool(const char* name, bool fallback) {
        const char* v = std::getenv(name);
        if (!v || !*v) return fallback;
        if (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T') return true;
        if (v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F') return false;
        return fallback;
    }

    void initHotRegionRepairFromEnv() {
        hot_region_repair_enabled_ =
            readEnvBool("ANNCHOR_HOT_REGION_REPAIR", false);
        hot_region_candidate_filter_enabled_ =
            readEnvBool("ANNCHOR_HOT_REGION_FILTER_CANDIDATES", true);
        hot_region_reorder_selected_enabled_ =
            readEnvBool("ANNCHOR_HOT_REGION_REORDER_SELECTED", false);
        hot_region_mark_candidates_ =
            readEnvBool("ANNCHOR_HOT_REGION_MARK_CANDIDATES", false);
        hot_region_filter_min_cold_ =
            std::max(0, readEnvInt("ANNCHOR_HOT_REGION_FILTER_MIN_COLD", 0));
        hot_region_preserve_nearest_ =
            std::max(0, readEnvInt("ANNCHOR_HOT_REGION_PRESERVE_NEAREST", 0));
        hot_region_shift_ =
            std::min(30, std::max(6, readEnvInt("ANNCHOR_HOT_REGION_SHIFT", 16)));
        hot_region_ttl_epochs_ = static_cast<uint32_t>(
            std::max(1, readEnvInt("ANNCHOR_HOT_REGION_TTL_EPOCHS", 4096)));
        const int table_bits =
            std::min(24, std::max(10, readEnvInt("ANNCHOR_HOT_REGION_TABLE_BITS", 20)));
        hot_region_table_size_ = hot_region_repair_enabled_
                                     ? (static_cast<size_t>(1) << table_bits)
                                     : 0;
        if (!hot_region_repair_enabled_) return;

        hot_region_keys_ =
            std::make_unique<std::atomic<uint64_t>[]>(hot_region_table_size_);
        hot_region_epochs_ =
            std::make_unique<std::atomic<uint32_t>[]>(hot_region_table_size_);
        for (size_t i = 0; i < hot_region_table_size_; ++i) {
            hot_region_keys_[i].store(0, std::memory_order_relaxed);
            hot_region_epochs_[i].store(0, std::memory_order_relaxed);
        }
    }

    uint64_t hotRegionKeyForAddress(const void *ptr) const {
        const auto addr = reinterpret_cast<uintptr_t>(ptr);
        return (static_cast<uint64_t>(addr >> hot_region_shift_) + 1ull);
    }

    size_t hotRegionSlot(uint64_t key) const {
        uint64_t x = key;
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        return static_cast<size_t>(x) & (hot_region_table_size_ - 1);
    }

    uint32_t beginHotRegionSearchEpoch() const {
        if (!hot_region_repair_enabled_ || hot_region_table_size_ == 0) {
            return 0;
        }
        uint32_t epoch =
            hot_region_epoch_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (epoch == 0) {
            hot_region_epoch_.store(1, std::memory_order_relaxed);
            epoch = 1;
        }
        return epoch;
    }

    void markHotRegionAddress(const void *ptr, uint32_t epoch) const {
        if (epoch == 0 || ptr == nullptr || hot_region_table_size_ == 0) return;
        const uint64_t key = hotRegionKeyForAddress(ptr);
        const size_t slot = hotRegionSlot(key);
        hot_region_keys_[slot].store(key, std::memory_order_release);
        hot_region_epochs_[slot].store(epoch, std::memory_order_release);
        metric_hot_region_search_marks.fetch_add(1, std::memory_order_relaxed);
    }

    bool isHotRegionAddress(const void *ptr) const {
        if (!hot_region_repair_enabled_ || ptr == nullptr ||
            hot_region_table_size_ == 0) {
            return false;
        }
        const uint64_t key = hotRegionKeyForAddress(ptr);
        const size_t slot = hotRegionSlot(key);
        if (hot_region_keys_[slot].load(std::memory_order_acquire) != key) {
            return false;
        }
        const uint32_t seen =
            hot_region_epochs_[slot].load(std::memory_order_acquire);
        if (seen == 0) return false;
        const uint32_t now =
            hot_region_epoch_.load(std::memory_order_relaxed);
        return static_cast<uint32_t>(now - seen) <= hot_region_ttl_epochs_;
    }

    uint32_t hotRegionAgeForAddress(const void *ptr) const {
        if (!hot_region_repair_enabled_ || ptr == nullptr ||
            hot_region_table_size_ == 0) {
            return std::numeric_limits<uint32_t>::max();
        }
        const uint64_t key = hotRegionKeyForAddress(ptr);
        const size_t slot = hotRegionSlot(key);
        if (hot_region_keys_[slot].load(std::memory_order_acquire) != key) {
            return std::numeric_limits<uint32_t>::max();
        }
        const uint32_t seen =
            hot_region_epochs_[slot].load(std::memory_order_acquire);
        if (seen == 0) return std::numeric_limits<uint32_t>::max();
        const uint32_t now =
            hot_region_epoch_.load(std::memory_order_relaxed);
        const uint32_t age = static_cast<uint32_t>(now - seen);
        if (age > hot_region_ttl_epochs_) {
            return std::numeric_limits<uint32_t>::max();
        }
        return age;
    }

    uint32_t repairHotRegionAge(tableint old_node, int level) const {
        if (!hot_region_repair_enabled_ ||
            static_cast<size_t>(old_node) >= max_elements_) {
            return std::numeric_limits<uint32_t>::max();
        }
        uint32_t age = hotRegionAgeForAddress(getDataByInternalId(old_node));
        if (level == 0) {
            return std::min(age, hotRegionAgeForAddress(get_linklist0(old_node)));
        }
        if (level <= element_levels_[old_node] && linkLists_[old_node] != nullptr) {
            return std::min(age, hotRegionAgeForAddress(get_linklist(old_node, level)));
        }
        return age;
    }

    bool reorderLinkCandidatesByHotRegion(std::vector<tableint> &neighbors,
                                          int level) const {
        if (!hot_region_repair_enabled_ ||
            !hot_region_reorder_selected_enabled_ ||
            level != 0 ||
            neighbors.size() <= 1 ||
            metric_hot_region_search_marks.load(std::memory_order_relaxed) == 0) {
            return false;
        }
        struct RankedNeighbor {
            tableint node;
            uint32_t age;
            size_t original_pos;
        };
        std::vector<RankedNeighbor> ranked;
        ranked.reserve(neighbors.size());
        size_t hot_count = 0;
        for (size_t i = 0; i < neighbors.size(); ++i) {
            const uint32_t age = repairHotRegionAge(neighbors[i], level);
            ranked.push_back({neighbors[i], age, i});
            metric_hot_region_repair_checks.fetch_add(
                1, std::memory_order_relaxed);
            if (age != std::numeric_limits<uint32_t>::max()) {
                ++hot_count;
                metric_hot_region_repair_conflicts.fetch_add(
                    1, std::memory_order_relaxed);
            }
        }
        if (hot_count == 0) return false;

        std::stable_sort(ranked.begin(), ranked.end(),
                         [](const RankedNeighbor &a,
                            const RankedNeighbor &b) {
                             const bool a_hot =
                                 a.age != std::numeric_limits<uint32_t>::max();
                             const bool b_hot =
                                 b.age != std::numeric_limits<uint32_t>::max();
                             if (a_hot != b_hot) return !a_hot;
                             if (a_hot && b_hot && a.age != b.age) {
                                 return a.age > b.age;
                             }
                             return a.original_pos < b.original_pos;
                         });
        bool changed = false;
        for (size_t i = 0; i < ranked.size(); ++i) {
            if (neighbors[i] != ranked[i].node) changed = true;
            neighbors[i] = ranked[i].node;
        }
        if (changed) {
            metric_hot_region_repair_hot_neighbors.fetch_add(
                hot_count, std::memory_order_relaxed);
            metric_hot_region_repair_reordered_calls.fetch_add(
                1, std::memory_order_relaxed);
        }
        return changed;
    }

    struct CompareByFirst {
        constexpr bool operator()(
            std::pair<dist_t, tableint> const &a,
            std::pair<dist_t, tableint> const &b) const noexcept {
            return a.first < b.first;
        }
    };

    bool filterHotLinkCandidatePool(
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> &top_candidates,
        int level, size_t min_cold) const {
        if (!hot_region_repair_enabled_ ||
            !hot_region_candidate_filter_enabled_ ||
            level != 0 ||
            metric_hot_region_search_marks.load(std::memory_order_relaxed) == 0 ||
            top_candidates.size() <= min_cold) {
            return false;
        }

        struct RankedCandidate {
            dist_t dist;
            tableint node;
            uint32_t age;
            size_t original_pos;
            bool keep;
        };
        std::vector<RankedCandidate> candidates;
        candidates.reserve(top_candidates.size());
        size_t hot_count = 0;
        size_t original_pos = 0;
        while (!top_candidates.empty()) {
            auto item = top_candidates.top();
            top_candidates.pop();
            metric_hot_region_repair_checks.fetch_add(
                1, std::memory_order_relaxed);
            const uint32_t age = repairHotRegionAge(item.second, level);
            if (age != std::numeric_limits<uint32_t>::max()) {
                ++hot_count;
                metric_hot_region_repair_conflicts.fetch_add(
                    1, std::memory_order_relaxed);
            }
            candidates.push_back(
                {item.first, item.second, age, original_pos++, false});
        }

        if (hot_count == 0) {
            for (const auto &item : candidates) {
                top_candidates.emplace(item.dist, item.node);
            }
            return false;
        }

        std::vector<size_t> by_distance(candidates.size());
        for (size_t i = 0; i < by_distance.size(); ++i) by_distance[i] = i;
        std::stable_sort(
            by_distance.begin(), by_distance.end(),
            [&](size_t a, size_t b) {
                if (candidates[a].dist != candidates[b].dist) {
                    return candidates[a].dist < candidates[b].dist;
                }
                return candidates[a].original_pos < candidates[b].original_pos;
            });

        const size_t preserve_nearest =
            static_cast<size_t>(hot_region_preserve_nearest_ > 0
                                    ? hot_region_preserve_nearest_
                                    : static_cast<int>(M_ * 2));
        std::vector<size_t> kept;
        kept.reserve(candidates.size());
        auto keep_candidate = [&](size_t idx) {
            if (!candidates[idx].keep) {
                candidates[idx].keep = true;
                kept.push_back(idx);
            }
        };

        for (size_t i = 0; i < by_distance.size() && i < preserve_nearest; ++i) {
            keep_candidate(by_distance[i]);
        }
        for (size_t idx : by_distance) {
            if (candidates[idx].age == std::numeric_limits<uint32_t>::max()) {
                keep_candidate(idx);
            }
        }

        if (kept.size() < min_cold) {
            std::vector<size_t> hot_by_age;
            hot_by_age.reserve(hot_count);
            for (size_t i = 0; i < candidates.size(); ++i) {
                if (!candidates[i].keep &&
                    candidates[i].age != std::numeric_limits<uint32_t>::max()) {
                    hot_by_age.push_back(i);
                }
            }
            std::stable_sort(
                hot_by_age.begin(), hot_by_age.end(),
                [&](size_t a, size_t b) {
                    if (candidates[a].age != candidates[b].age) {
                        return candidates[a].age > candidates[b].age;
                    }
                    if (candidates[a].dist != candidates[b].dist) {
                        return candidates[a].dist < candidates[b].dist;
                    }
                    return candidates[a].original_pos < candidates[b].original_pos;
                });
            for (size_t idx : hot_by_age) {
                if (kept.size() >= min_cold) break;
                keep_candidate(idx);
            }
        }

        if (kept.size() == candidates.size() || kept.size() < M_) {
            for (const auto &item : candidates) {
                top_candidates.emplace(item.dist, item.node);
            }
            return false;
        }

        for (size_t idx : kept) {
            top_candidates.emplace(candidates[idx].dist, candidates[idx].node);
        }
        const size_t dropped = candidates.size() - kept.size();
        metric_hot_region_candidate_filter_calls.fetch_add(
            1, std::memory_order_relaxed);
        metric_hot_region_candidate_filter_cold_kept.fetch_add(
            kept.size(), std::memory_order_relaxed);
        metric_hot_region_candidate_filter_hot_dropped.fetch_add(
            dropped, std::memory_order_relaxed);
        return true;
    }

    void initMidInsertPreemptFromEnv() {
        mid_insert_preempt_enable_ = readEnvBool("ANN_MID_INSERT_PREEMPT_ENABLE", false);
        mid_insert_preempt_k_ = std::max(0, readEnvInt("ANN_MID_INSERT_PREEMPT_K", 0));
        mid_insert_preempt_revalidate_ = readEnvBool("ANN_MID_INSERT_PREEMPT_REVALIDATE", false);
        mid_insert_preempt_shadow_replan_ = readEnvBool("ANN_MID_INSERT_PREEMPT_SHADOW_REPLAN", false);
        mid_insert_preempt_harm_guard_ = readEnvBool("ANN_MID_INSERT_PREEMPT_HARM_GUARD", false);
        mid_insert_shadow_snapshot_teacher_ = readEnvBool("ANN_MID_INSERT_PREEMPT_SHADOW_SNAPSHOT", true);
        mid_insert_harm_micro_replan_ = readEnvBool("ANN_MID_INSERT_PREEMPT_HARM_MICRO_REPLAN", false);
        mid_insert_harm_micro_probe_ef_ = std::max(0, readEnvInt("ANN_MID_INSERT_PREEMPT_HARM_MICRO_PROBE_EF", 0));
        mid_insert_harm_online_policy_ = readEnvBool("ANN_MID_INSERT_PREEMPT_HARM_ONLINE_POLICY", false);
        mid_insert_harm_busy_wait_commits_ = std::max(0, readEnvInt("ANN_MID_INSERT_PREEMPT_HARM_BUSY_WAIT_COMMITS", 32));
        mid_insert_harm_search_backlog_threshold_ =
            std::max(0, readEnvInt("ANN_MID_INSERT_PREEMPT_HARM_SEARCH_BACKLOG_THRESHOLD", 0));
        mid_insert_harm_priority_search_threshold_ =
            std::max(0, readEnvInt("ANN_MID_INSERT_PREEMPT_HARM_PRIORITY_SEARCH_THRESHOLD", 0));
        mid_insert_harm_full_foreign_outrank_threshold_ =
            std::max(0, readEnvInt("ANN_MID_INSERT_PREEMPT_HARM_FULL_FOREIGN_OUTRANK", 2));
        mid_insert_harm_full_selected_frontier_touched_threshold_ =
            std::max(0, readEnvInt("ANN_MID_INSERT_PREEMPT_HARM_FULL_SELECTED_FRONTIER_TOUCHED", 2));
        mid_insert_harm_defer_enabled_ = readEnvBool("ANN_MID_INSERT_PREEMPT_HARM_DEFER_ENABLED", false);
        mid_insert_harm_defer_queue_cap_ =
            std::max(1, readEnvInt("ANN_MID_INSERT_PREEMPT_HARM_DEFER_QUEUE_CAP", 4096));
        mid_insert_harm_defer_drain_budget_ =
            std::max(1, readEnvInt("ANN_MID_INSERT_PREEMPT_HARM_DEFER_DRAIN_BUDGET", 1));
        mid_insert_harm_defer_high_watermark_pct_ =
            std::min(100, std::max(1, readEnvInt("ANN_MID_INSERT_PREEMPT_HARM_DEFER_HIGH_WATERMARK_PCT", 75)));
        mid_insert_preempt_max_wait_us_ = std::max(0, readEnvInt("ANN_MID_INSERT_PREEMPT_MAX_WAIT_US", 200000));
        mid_insert_preempt_activate_after_commits_ = std::max(0, readEnvInt("ANN_MID_INSERT_PREEMPT_ACTIVATE_AFTER_COMMITS", 0));
        mid_insert_preempt_every_n_ = std::max(1, readEnvInt("ANN_MID_INSERT_PREEMPT_EVERY_N", 1));
        mid_insert_preempt_max_inflight_ = std::max(1, readEnvInt("ANN_MID_INSERT_PREEMPT_MAX_INFLIGHT", 1));
        mid_insert_harm_micro_pool_cap_ = std::max(1, readEnvInt("ANN_MID_INSERT_PREEMPT_HARM_MICRO_POOL_CAP", 64));
    }

    // ── end M2 config ────────────────────────────────────────────────

    HierarchicalNSW(SpaceInterface<dist_t> *s) {}

    HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location,
                    bool nmslib = false, size_t max_elements = 0,
                    bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        initMidInsertPreemptFromEnv();
        loadIndex(location, s, max_elements);
        initHotRegionRepairFromEnv();
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
        thread_pool_ = std::make_unique<ThreadPool>(num_threads_);
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
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
        initM2BaseInsertNeighborStorage(max_elements);
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ =
            maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
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
        size_links_per_element_ =
            maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
        initWatermarkState(max_elements_);
        
        // Initialize MVCC components
        initModifiedBitmap(max_elements_);
        undo_log_ = std::make_unique<UndoLogManager>(max_elements_);
        existing_neighbor_update_read_scan_only_ =
            readEnvBool("ANNCHOR_EXISTING_NEIGHBOR_UPDATE_READ_SCAN_ONLY", false);
        disable_existing_neighbor_undo_record_ =
            readEnvBool("ANNCHOR_DISABLE_EXISTING_NEIGHBOR_UNDO_RECORD", false);
        existing_neighbor_update_read_scan_only_from_id_ =
            readEnvInt("ANNCHOR_EXISTING_NEIGHBOR_UPDATE_READ_SCAN_ONLY_FROM_ID", -1);
        disable_existing_neighbor_undo_record_from_id_ =
            readEnvInt("ANNCHOR_DISABLE_EXISTING_NEIGHBOR_UNDO_RECORD_FROM_ID", -1);
        rewrite_active_diag_enabled_ =
            readEnvBool("ANNCHOR_REWRITE_ACTIVE_DIAG", true);
        rewrite_active_hold_us_ =
            std::max(0, readEnvInt("ANNCHOR_REWRITE_ACTIVE_HOLD_US", 0));
        rewrite_recent_window_ = static_cast<uint64_t>(
            std::max(0, readEnvInt("ANNCHOR_REWRITE_RECENT_WINDOW", 100000)));
        node_rewrite_active_ =
            std::make_unique<std::atomic<uint8_t>[]>(max_elements_);
        node_rewrite_epoch_ =
            std::make_unique<std::atomic<uint64_t>[]>(max_elements_);
        for (size_t i = 0; i < max_elements_; ++i) {
            node_rewrite_active_[i].store(0, std::memory_order_relaxed);
            node_rewrite_epoch_[i].store(0, std::memory_order_relaxed);
        }
        initHotRegionRepairFromEnv();
        initMidInsertPreemptFromEnv();
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
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
        insertion_complete_.reset();
        ready_commit_ids_.reset();
        global_watermark_.store(-1, std::memory_order_relaxed);
        
        // Clean up MVCC components
        if (node_modified_bitmap_) {
            delete[] node_modified_bitmap_;
            node_modified_bitmap_ = nullptr;
        }
        if (node_link_version_) {
            delete[] node_link_version_;
            node_link_version_ = nullptr;
        }
        undo_log_.reset();
    }

    void initWatermarkState(size_t max_elements) {
        insertion_complete_.reset(new std::atomic<bool>[max_elements]);
        for (size_t i = 0; i < max_elements; ++i)
            insertion_complete_[i].store(false, std::memory_order_relaxed);
        mid_insert_deferred_pending_.reset(new std::atomic<bool>[max_elements]);
        for (size_t i = 0; i < max_elements; ++i)
            mid_insert_deferred_pending_[i].store(false, std::memory_order_relaxed);
        ready_commit_ids_.reset(new std::atomic<tableint>[max_elements]);
        for (size_t i = 0; i < max_elements; ++i)
            ready_commit_ids_[i].store(static_cast<tableint>(-1), std::memory_order_relaxed);
        global_watermark_.store(-1, std::memory_order_relaxed);
        ready_commit_counter_.store(0, std::memory_order_relaxed);
    }

    void initM2BaseInsertNeighborStorage(size_t max_elements) {
        m2_base_insert_neighbor_stride_ = std::max<size_t>(1, maxM0_);
        m2_base_insert_neighbor_counts_.assign(max_elements, 0);
        m2_base_insert_neighbors_.assign(
            max_elements * m2_base_insert_neighbor_stride_, tableint{0});
    }

    void resizeM2BaseInsertNeighborStorage(size_t new_max_elements) {
        if (m2_base_insert_neighbor_stride_ == 0) {
            initM2BaseInsertNeighborStorage(new_max_elements);
            return;
        }
        m2_base_insert_neighbor_counts_.resize(new_max_elements, 0);
        m2_base_insert_neighbors_.resize(
            new_max_elements * m2_base_insert_neighbor_stride_, tableint{0});
    }

    void markReady(tableint internal_id) {
        insertion_complete_[internal_id].store(true, std::memory_order_release);
        const bool need_commit_log =
            mid_insert_preempt_enable_ || mid_insert_preempt_revalidate_ ||
            mid_insert_preempt_shadow_replan_ || mid_insert_preempt_harm_guard_;
        if (!need_commit_log || !ready_commit_ids_) {
            ready_commit_counter_.fetch_add(1, std::memory_order_acq_rel);
            return;
        }
        std::lock_guard<std::mutex> guard(ready_commit_publish_lock_);
        const long seq = ready_commit_counter_.load(std::memory_order_relaxed);
        ready_commit_ids_[seq].store(internal_id, std::memory_order_relaxed);
        ready_commit_counter_.store(seq + 1, std::memory_order_release);
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
                // Update gc_horizon_ after successful watermark advance
                if (enable_mvcc_) {
                    long min_snap = snapshot_tracker_.min_active_snapshot();
                    if (min_snap == SnapshotTracker::INACTIVE) {
                        min_snap = desired;
                    }
                    gc_horizon_.store(min_snap, std::memory_order_release);
                }
                return;
            }
        }
    }

    void markCompleted(tableint internal_id, long upper_bound = -1) {
        markReady(internal_id);
        advanceWatermark(upper_bound);
    }

    void markMidInsertDeferredPending(tableint internal_id, bool pending) {
        if (!mid_insert_deferred_pending_) {
            return;
        }
        mid_insert_deferred_pending_[internal_id].store(
            pending, std::memory_order_relaxed);
    }

    bool consumeMidInsertDeferredPending(tableint internal_id) {
        if (!mid_insert_deferred_pending_) {
            return false;
        }
        return mid_insert_deferred_pending_[internal_id].exchange(
            false, std::memory_order_acq_rel);
    }

    bool tryEnqueueMidInsertDeferredRepair(tableint internal_id) const {
        if (!mid_insert_harm_defer_enabled_ || mid_insert_harm_defer_queue_cap_ <= 0) {
            return false;
        }
        std::lock_guard<std::mutex> guard(mid_insert_deferred_repair_queue_lock_);
        if (mid_insert_deferred_repair_set_.find(internal_id) !=
            mid_insert_deferred_repair_set_.end()) {
            mid_insert_pause_deferred_deduped_.fetch_add(
                1, std::memory_order_relaxed);
            return false;
        }
        if (static_cast<int>(mid_insert_deferred_repair_queue_.size()) >=
            mid_insert_harm_defer_queue_cap_) {
            mid_insert_pause_deferred_dropped_.fetch_add(
                1, std::memory_order_relaxed);
            return false;
        }
        mid_insert_deferred_repair_queue_.push_back(internal_id);
        mid_insert_deferred_repair_set_.insert(internal_id);
        mid_insert_pause_deferred_enqueued_.fetch_add(
            1, std::memory_order_relaxed);
        long queue_size =
            static_cast<long>(mid_insert_deferred_repair_queue_.size());
        long prev_queue_max =
            mid_insert_pause_deferred_queue_max_.load(std::memory_order_relaxed);
        while (queue_size > prev_queue_max &&
               !mid_insert_pause_deferred_queue_max_.compare_exchange_weak(
                   prev_queue_max, queue_size, std::memory_order_relaxed)) {
        }
        return true;
    }

    bool tryPopMidInsertDeferredRepair(tableint *internal_id) const {
        std::lock_guard<std::mutex> guard(mid_insert_deferred_repair_queue_lock_);
        if (mid_insert_deferred_repair_queue_.empty()) {
            return false;
        }
        *internal_id = mid_insert_deferred_repair_queue_.front();
        mid_insert_deferred_repair_queue_.pop_front();
        mid_insert_deferred_repair_set_.erase(*internal_id);
        return true;
    }

    size_t getMidInsertDeferredQueueSize() const {
        std::lock_guard<std::mutex> guard(mid_insert_deferred_repair_queue_lock_);
        return mid_insert_deferred_repair_queue_.size();
    }

    size_t drainMidInsertDeferredRepairs(size_t budget) {
        if (!mid_insert_harm_defer_enabled_ || budget == 0) {
            return 0;
        }
        if (mid_insert_preempt_inflight_.load(std::memory_order_acquire) > 0) {
            return 0;
        }
        std::unique_lock<std::mutex> drain_lock(
            mid_insert_deferred_repair_drain_lock_, std::try_to_lock);
        if (!drain_lock.owns_lock()) {
            return 0;
        }
        size_t drained = 0;
        while (drained < budget) {
            if (mid_insert_preempt_inflight_.load(std::memory_order_acquire) > 0) {
                break;
            }
            tableint deferred_id = static_cast<tableint>(-1);
            if (!tryPopMidInsertDeferredRepair(&deferred_id)) {
                break;
            }
            if (drained == 0) {
                mid_insert_pause_deferred_drain_runs_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (deferred_id < 0 || deferred_id >= cur_element_count ||
                isMarkedDeleted(deferred_id)) {
                continue;
            }
            updatePoint(getDataByInternalId(deferred_id), deferred_id, 1.0f);
            mid_insert_pause_deferred_drained_.fetch_add(
                1, std::memory_order_relaxed);
            ++drained;
        }
        return drained;
    }

    enum class MidInsertRepairAction {
        Continue = 0,
        Defer = 1,
        Micro = 2,
        Full = 3,
    };

    struct MidInsertRepairRiskState {
        bool harm_risky{false};
        bool boundary_harm_risky{false};
        bool mild_harm_risky{false};
        bool service_busy{false};
        bool search_pressure_busy{false};
        bool severe_risk{false};
        bool defer_backpressured{false};
        bool defer_eligible{false};
    };

    size_t getMidInsertDeferHighWatermark() const {
        if (!mid_insert_harm_defer_enabled_ || mid_insert_harm_defer_queue_cap_ <= 0) {
            return 0;
        }
        return std::max<size_t>(
            1,
            (static_cast<size_t>(mid_insert_harm_defer_queue_cap_) *
                 static_cast<size_t>(mid_insert_harm_defer_high_watermark_pct_) +
             99) /
                100);
    }

    MidInsertRepairRiskState classifyMidInsertRepairRisk(
        bool harm_risky, bool boundary_harm_risky, long waited_commits,
        long foreign_outrank_count, long selected_frontier_touched) const {
        MidInsertRepairRiskState state;
        state.harm_risky = harm_risky;
        state.boundary_harm_risky = boundary_harm_risky;
        state.mild_harm_risky = harm_risky && !boundary_harm_risky;
        MidInsertServicePressure pressure;
        if (mid_insert_service_pressure_cb_) {
            pressure = mid_insert_service_pressure_cb_();
        }
        const bool backlog_busy =
            mid_insert_harm_search_backlog_threshold_ > 0 &&
            pressure.search_backlog >= mid_insert_harm_search_backlog_threshold_;
        const bool priority_busy =
            mid_insert_harm_priority_search_threshold_ > 0 &&
            pressure.priority_searches >= mid_insert_harm_priority_search_threshold_;
        state.search_pressure_busy = harm_risky && (backlog_busy || priority_busy);
        state.service_busy =
            harm_risky &&
            (waited_commits >= static_cast<long>(mid_insert_harm_busy_wait_commits_) ||
             state.search_pressure_busy);

        const bool severe_by_outrank =
            mid_insert_harm_full_foreign_outrank_threshold_ > 0 &&
            foreign_outrank_count >=
                static_cast<long>(mid_insert_harm_full_foreign_outrank_threshold_);
        const bool severe_by_selected_touch =
            mid_insert_harm_full_selected_frontier_touched_threshold_ > 0 &&
            selected_frontier_touched >= static_cast<long>(
                mid_insert_harm_full_selected_frontier_touched_threshold_);
        state.severe_risk = harm_risky && (severe_by_outrank || severe_by_selected_touch);

        const size_t defer_high_watermark = getMidInsertDeferHighWatermark();
        if (defer_high_watermark > 0) {
            const size_t defer_queue_size = getMidInsertDeferredQueueSize();
            state.defer_backpressured = defer_queue_size >= defer_high_watermark;
        }

        state.defer_eligible =
            mid_insert_harm_online_policy_ && mid_insert_harm_defer_enabled_ &&
            state.service_busy && !state.defer_backpressured &&
            (state.severe_risk || boundary_harm_risky);
        return state;
    }

    void recordMidInsertRepairRiskStats(
        const MidInsertRepairRiskState &risk_state) const {
        if (risk_state.mild_harm_risky) {
            mid_insert_pause_mild_risk_hits_.fetch_add(
                1, std::memory_order_relaxed);
        }
        if (risk_state.service_busy) {
            mid_insert_pause_service_busy_hits_.fetch_add(
                1, std::memory_order_relaxed);
        }
        if (risk_state.search_pressure_busy) {
            mid_insert_pause_search_pressure_busy_hits_.fetch_add(
                1, std::memory_order_relaxed);
        }
        if (risk_state.severe_risk) {
            mid_insert_pause_severe_risk_hits_.fetch_add(
                1, std::memory_order_relaxed);
        }
        if (risk_state.defer_backpressured) {
            mid_insert_pause_defer_backpressure_hits_.fetch_add(
                1, std::memory_order_relaxed);
        }
    }

    MidInsertRepairAction chooseMidInsertRepairAction(
        const MidInsertRepairRiskState &risk_state) const {
        if (mid_insert_preempt_revalidate_) {
            return MidInsertRepairAction::Full;
        }
        if (!mid_insert_preempt_harm_guard_ || !risk_state.harm_risky) {
            return MidInsertRepairAction::Continue;
        }
        if (!mid_insert_harm_online_policy_) {
            return mid_insert_harm_micro_replan_ ? MidInsertRepairAction::Micro
                                                 : MidInsertRepairAction::Full;
        }
        if (risk_state.defer_eligible) {
            return MidInsertRepairAction::Defer;
        }
        if (risk_state.severe_risk && !risk_state.service_busy) {
            return MidInsertRepairAction::Full;
        }
        if (risk_state.boundary_harm_risky) {
            if (mid_insert_harm_micro_replan_) {
                return MidInsertRepairAction::Micro;
            }
            if (!risk_state.service_busy) {
                return MidInsertRepairAction::Full;
            }
        } else if (!risk_state.service_busy && mid_insert_harm_micro_replan_) {
            return MidInsertRepairAction::Micro;
        }
        return MidInsertRepairAction::Continue;
    }

    bool recordMidInsertRepairAction(
        MidInsertRepairAction repair_action,
        const MidInsertRepairRiskState &risk_state) const {
        if (repair_action == MidInsertRepairAction::Continue &&
            mid_insert_preempt_harm_guard_ && risk_state.harm_risky) {
            mid_insert_pause_action_continue_.fetch_add(
                1, std::memory_order_relaxed);
            if (risk_state.mild_harm_risky) {
                mid_insert_pause_action_continue_mild_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            return false;
        }
        if (repair_action == MidInsertRepairAction::Defer) {
            mid_insert_pause_action_defer_.fetch_add(
                1, std::memory_order_relaxed);
            return true;
        }
        if (repair_action == MidInsertRepairAction::Micro) {
            mid_insert_pause_action_micro_.fetch_add(
                1, std::memory_order_relaxed);
            return false;
        }
        if (repair_action == MidInsertRepairAction::Full) {
            mid_insert_pause_action_full_.fetch_add(
                1, std::memory_order_relaxed);
        }
        return false;
    }

    void setEf(size_t ef) { ef_ = ef; }
    size_t getEf() const { return ef_; }

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

    bool isMidInsertPreemptEnabled() const { return mid_insert_preempt_enable_ && mid_insert_preempt_k_ > 0; }
    int getMidInsertPreemptK() const { return mid_insert_preempt_k_; }
    bool isMidInsertPreemptRevalidateEnabled() const { return mid_insert_preempt_revalidate_; }
    bool isMidInsertShadowReplanEnabled() const { return mid_insert_preempt_revalidate_ || mid_insert_preempt_shadow_replan_; }
    bool isMidInsertHarmGuardEnabled() const { return mid_insert_preempt_harm_guard_; }
    bool isMidInsertShadowSnapshotTeacherEnabled() const { return mid_insert_shadow_snapshot_teacher_ && enable_mvcc_; }
    bool isMidInsertHarmMicroReplanEnabled() const { return mid_insert_harm_micro_replan_; }
    int getMidInsertHarmMicroProbeEf() const { return mid_insert_harm_micro_probe_ef_; }
    bool isMidInsertHarmOnlinePolicyEnabled() const { return mid_insert_harm_online_policy_; }
    int getMidInsertHarmBusyWaitCommits() const { return mid_insert_harm_busy_wait_commits_; }
    int getMidInsertHarmSearchBacklogThreshold() const { return mid_insert_harm_search_backlog_threshold_; }
    int getMidInsertHarmPrioritySearchThreshold() const { return mid_insert_harm_priority_search_threshold_; }
    int getMidInsertHarmFullForeignOutrankThreshold() const { return mid_insert_harm_full_foreign_outrank_threshold_; }
    int getMidInsertHarmFullSelectedFrontierTouchedThreshold() const { return mid_insert_harm_full_selected_frontier_touched_threshold_; }
    bool isMidInsertHarmDeferEnabled() const { return mid_insert_harm_defer_enabled_; }
    int getMidInsertHarmDeferQueueCap() const { return mid_insert_harm_defer_queue_cap_; }
    int getMidInsertHarmDeferDrainBudget() const { return mid_insert_harm_defer_drain_budget_; }
    int getMidInsertHarmDeferHighWatermarkPct() const { return mid_insert_harm_defer_high_watermark_pct_; }
    bool flushMidInsertDeferredReady(tableint internal_id) {
        if (!consumeMidInsertDeferredPending(internal_id)) {
            return false;
        }
        return tryEnqueueMidInsertDeferredRepair(internal_id);
    }
    size_t drainMidInsertDeferredRepairsBudgeted() {
        return drainMidInsertDeferredRepairs(
            static_cast<size_t>(mid_insert_harm_defer_drain_budget_));
    }
    long getMidInsertPreemptHits() const { return mid_insert_preempt_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPreemptTimeouts() const { return mid_insert_preempt_timeouts_.load(std::memory_order_relaxed); }
    long getMidInsertPreemptWaitCommitsSum() const { return mid_insert_preempt_wait_commits_sum_.load(std::memory_order_relaxed); }
    long getMidInsertPreemptRevalidations() const { return mid_insert_preempt_revalidations_.load(std::memory_order_relaxed); }
    long getMidInsertShadowReplans() const { return mid_insert_shadow_replans_.load(std::memory_order_relaxed); }
    int getMidInsertPreemptEveryN() const { return mid_insert_preempt_every_n_; }
    int getMidInsertPreemptMaxInflight() const { return mid_insert_preempt_max_inflight_; }
    long getMidInsertPreemptSlotDenied() const { return mid_insert_preempt_slot_denied_.load(std::memory_order_relaxed); }
    long getMidInsertPreemptGlobalLockSkipped() const { return mid_insert_preempt_global_lock_skipped_.load(std::memory_order_relaxed); }
    long getMidInsertPreemptQuiesceBreaks() const { return mid_insert_preempt_quiesce_breaks_.load(std::memory_order_relaxed); }
    long getMidInsertCandidateChanged() const { return mid_insert_candidate_changed_.load(std::memory_order_relaxed); }
    long getMidInsertCandidateBeforeTotal() const { return mid_insert_candidate_before_total_.load(std::memory_order_relaxed); }
    long getMidInsertCandidateAfterTotal() const { return mid_insert_candidate_after_total_.load(std::memory_order_relaxed); }
    long getMidInsertCandidateIntersectionTotal() const { return mid_insert_candidate_intersection_total_.load(std::memory_order_relaxed); }
    long getMidInsertCandidateUnionTotal() const { return mid_insert_candidate_union_total_.load(std::memory_order_relaxed); }
    long getMidInsertSelectedChanged() const { return mid_insert_selected_changed_.load(std::memory_order_relaxed); }
    long getMidInsertSelectedBeforeTotal() const { return mid_insert_selected_before_total_.load(std::memory_order_relaxed); }
    long getMidInsertSelectedAfterTotal() const { return mid_insert_selected_after_total_.load(std::memory_order_relaxed); }
    long getMidInsertSelectedIntersectionTotal() const { return mid_insert_selected_intersection_total_.load(std::memory_order_relaxed); }
    long getMidInsertSelectedUnionTotal() const { return mid_insert_selected_union_total_.load(std::memory_order_relaxed); }
    long getMidInsertPauseLowSlackHits() const { return mid_insert_pause_low_slack_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPauseMildRiskHits() const { return mid_insert_pause_mild_risk_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPauseServiceBusyHits() const { return mid_insert_pause_service_busy_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPauseSearchPressureBusyHits() const { return mid_insert_pause_search_pressure_busy_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPauseSevereRiskHits() const { return mid_insert_pause_severe_risk_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPauseSelectedFrontierTouchedHits() const { return mid_insert_pause_selected_frontier_touched_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPauseSelectedFrontierTouchedNodes() const { return mid_insert_pause_selected_frontier_touched_nodes_.load(std::memory_order_relaxed); }
    long getMidInsertPauseCandidateFrontierTouchedHits() const { return mid_insert_pause_candidate_frontier_touched_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPauseCandidateFrontierTouchedNodes() const { return mid_insert_pause_candidate_frontier_touched_nodes_.load(std::memory_order_relaxed); }
    long getMidInsertPauseForeignCommitNodes() const { return mid_insert_pause_foreign_commit_nodes_.load(std::memory_order_relaxed); }
    long getMidInsertPauseForeignOutrankHits() const { return mid_insert_pause_foreign_outrank_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPauseForeignOutrankNodes() const { return mid_insert_pause_foreign_outrank_nodes_.load(std::memory_order_relaxed); }
    long getMidInsertPauseBoundaryBlockedHits() const { return mid_insert_pause_boundary_blocked_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPauseBoundaryBlockedCandidates() const { return mid_insert_pause_boundary_blocked_candidates_.load(std::memory_order_relaxed); }
    long getMidInsertPauseBoundaryBlockerTouchedHits() const { return mid_insert_pause_boundary_blocker_touched_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPauseBoundaryBlockerTouchedNodes() const { return mid_insert_pause_boundary_blocker_touched_nodes_.load(std::memory_order_relaxed); }
    long getMidInsertPauseBoundaryWitnessInvalidatedHits() const { return mid_insert_pause_boundary_witness_invalidated_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPauseBoundaryWitnessInvalidatedCandidates() const { return mid_insert_pause_boundary_witness_invalidated_candidates_.load(std::memory_order_relaxed); }
    long getMidInsertPauseBoundaryHarmRiskHits() const { return mid_insert_pause_boundary_harm_risk_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPauseBoundaryHarmCertTruePositive() const { return mid_insert_pause_boundary_harm_cert_true_positive_.load(std::memory_order_relaxed); }
    long getMidInsertPauseBoundaryHarmCertFalsePositive() const { return mid_insert_pause_boundary_harm_cert_false_positive_.load(std::memory_order_relaxed); }
    long getMidInsertPauseBoundaryHarmCertFalseNegative() const { return mid_insert_pause_boundary_harm_cert_false_negative_.load(std::memory_order_relaxed); }
    long getMidInsertPauseHarmRiskHits() const { return mid_insert_pause_harm_risk_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPauseHarmGuardReplans() const { return mid_insert_pause_harm_guard_replans_.load(std::memory_order_relaxed); }
    long getMidInsertPauseActionContinue() const { return mid_insert_pause_action_continue_.load(std::memory_order_relaxed); }
    long getMidInsertPauseActionContinueMild() const { return mid_insert_pause_action_continue_mild_.load(std::memory_order_relaxed); }
    long getMidInsertPauseActionDefer() const { return mid_insert_pause_action_defer_.load(std::memory_order_relaxed); }
    long getMidInsertPauseActionMicro() const { return mid_insert_pause_action_micro_.load(std::memory_order_relaxed); }
    long getMidInsertPauseActionFull() const { return mid_insert_pause_action_full_.load(std::memory_order_relaxed); }
    long getMidInsertPauseDeferredEnqueued() const { return mid_insert_pause_deferred_enqueued_.load(std::memory_order_relaxed); }
    long getMidInsertPauseDeferredDeduped() const { return mid_insert_pause_deferred_deduped_.load(std::memory_order_relaxed); }
    long getMidInsertPauseDeferredDropped() const { return mid_insert_pause_deferred_dropped_.load(std::memory_order_relaxed); }
    long getMidInsertPauseDeferredDrained() const { return mid_insert_pause_deferred_drained_.load(std::memory_order_relaxed); }
    long getMidInsertPauseDeferredDrainRuns() const { return mid_insert_pause_deferred_drain_runs_.load(std::memory_order_relaxed); }
    long getMidInsertPauseDeferredQueueMax() const { return mid_insert_pause_deferred_queue_max_.load(std::memory_order_relaxed); }
    long getMidInsertPauseDeferBackpressureHits() const { return mid_insert_pause_defer_backpressure_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPauseMicroReplans() const { return mid_insert_pause_micro_replans_.load(std::memory_order_relaxed); }
    long getMidInsertPauseMicroProbeHits() const { return mid_insert_pause_micro_probe_hits_.load(std::memory_order_relaxed); }
    long getMidInsertPauseMicroProbeCandidatesTotal() const { return mid_insert_pause_micro_probe_candidates_total_.load(std::memory_order_relaxed); }
    long getMidInsertPauseMicroProbeCandidatesMax() const { return mid_insert_pause_micro_probe_candidates_max_.load(std::memory_order_relaxed); }
    long getMidInsertPauseMicroReplanPoolTotal() const { return mid_insert_pause_micro_replan_pool_total_.load(std::memory_order_relaxed); }
    long getMidInsertPauseMicroReplanPoolMax() const { return mid_insert_pause_micro_replan_pool_max_.load(std::memory_order_relaxed); }
    int getMidInsertHarmMicroPoolCap() const { return mid_insert_harm_micro_pool_cap_; }
    long getMidInsertPauseHarmCertTruePositive() const { return mid_insert_pause_harm_cert_true_positive_.load(std::memory_order_relaxed); }
    long getMidInsertPauseHarmCertFalsePositive() const { return mid_insert_pause_harm_cert_false_positive_.load(std::memory_order_relaxed); }
    long getMidInsertPauseHarmCertFalseNegative() const { return mid_insert_pause_harm_cert_false_negative_.load(std::memory_order_relaxed); }
    long getMidInsertShadowSnapshotTeacherHits() const { return mid_insert_shadow_snapshot_teacher_hits_.load(std::memory_order_relaxed); }
    long getMidInsertShadowSnapshotTeacherFutureEntrypointHits() const { return mid_insert_shadow_snapshot_teacher_future_entrypoint_hits_.load(std::memory_order_relaxed); }
    long getMidInsertShadowSnapshotTeacherPostwaitCommitDriftHits() const { return mid_insert_shadow_snapshot_teacher_postwait_commit_drift_hits_.load(std::memory_order_relaxed); }
    long getMidInsertShadowSnapshotTeacherPostwaitCommitDriftSum() const { return mid_insert_shadow_snapshot_teacher_postwait_commit_drift_sum_.load(std::memory_order_relaxed); }
    long getMidInsertShadowSnapshotTeacherPostwaitWatermarkDriftHits() const { return mid_insert_shadow_snapshot_teacher_postwait_watermark_drift_hits_.load(std::memory_order_relaxed); }
    long getMidInsertShadowSnapshotTeacherPostwaitWatermarkDriftSum() const { return mid_insert_shadow_snapshot_teacher_postwait_watermark_drift_sum_.load(std::memory_order_relaxed); }

    void setEnableMvcc(bool enabled) { enable_mvcc_ = enabled; }
    bool isMvccEnabled() const { return enable_mvcc_; }
    void setEnableNonPrefixVisibility(bool enabled) { enable_non_prefix_visibility_ = enabled; }
    bool isNonPrefixVisibilityEnabled() const { return enable_non_prefix_visibility_; }
    void setEnableUndoRecovery(bool enabled) { enable_undo_recovery_ = enabled; }
    bool isUndoRecoveryEnabled() const { return enable_undo_recovery_; }
    void setSearchVisibilityMode(int mode) {
        if (mode < 0 || mode > 2) mode = 0;
        search_visibility_mode_ = mode;
    }
    int getSearchVisibilityMode() const { return search_visibility_mode_; }
    void setSearchVisibilityByLabel(bool enabled) {
        search_visibility_by_label_ = enabled;
    }
    bool getSearchVisibilityByLabel() const {
        return search_visibility_by_label_;
    }

    void initModifiedBitmap(size_t max_elements) {
        if (node_modified_bitmap_) {
            delete[] node_modified_bitmap_;
        }
        node_modified_bitmap_ = new std::atomic<uint8_t>[max_elements];
        for (size_t i = 0; i < max_elements; i++) {
            node_modified_bitmap_[i].store(0, std::memory_order_relaxed);
        }
        if (node_link_version_) {
            delete[] node_link_version_;
        }
        node_link_version_ = new std::atomic<uint64_t>[max_elements];
        for (size_t i = 0; i < max_elements; i++) {
            node_link_version_[i].store(0, std::memory_order_relaxed);
        }
    }
    
    void markNodeModified(tableint node_id) {
        if (node_id < max_elements_ && node_modified_bitmap_) {
            node_modified_bitmap_[node_id].store(1, std::memory_order_release);
        }
    }

    void bumpNodeLinkVersion(tableint node_id) {
        if (node_id < max_elements_ && node_link_version_) {
            node_link_version_[node_id].fetch_add(1, std::memory_order_acq_rel);
        }
    }

    uint64_t getNodeLinkVersion(tableint node_id) const {
        if (node_id < max_elements_ && node_link_version_) {
            return node_link_version_[node_id].load(std::memory_order_acquire);
        }
        return 0;
    }
    
    bool isNodeModified(tableint node_id) const {
        if (node_id < max_elements_ && node_modified_bitmap_) {
            return node_modified_bitmap_[node_id].load(std::memory_order_acquire) != 0;
        }
        return false;
    }
    
    size_t compact(long safe_ts = -1) {
        if (!undo_log_) return 0;
        
        size_t compacted = 0;
        if (safe_ts < 0) {
            // Clear all undo logs
            undo_log_->clear();
            // Reset all modified bits
            if (node_modified_bitmap_) {
                for (size_t i = 0; i < max_elements_; i++) {
                    node_modified_bitmap_[i].store(0, std::memory_order_relaxed);
                }
            }
            compacted = max_elements_;
        } else {
            // Compact logs older than safe_ts
            for (size_t i = 0; i < max_elements_; i++) {
                if (undo_log_->hasLog(i)) {
                    undo_log_->advanceGcStart(i, safe_ts);
                }
            }
            undo_log_->compactAll();
            compacted = max_elements_;  // Approximate
        }
        return compacted;
    }

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer,
                    bool use_yield = false,
                    tableint trace_point_id = std::numeric_limits<tableint>::max()) {
        const uint64_t base_trace_start =
            insertPhaseTraceEnabled() ? monotonicRawNs() : 0;
        const auto base_search_start = std::chrono::steady_clock::now();
        uint64_t local_expansions = 0;
        uint64_t local_edges_scanned = 0;
        uint64_t local_dist_comps = 0;
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
            ++local_dist_comps;
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
            ++local_expansions;

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
            // Distance-computation-level scheduling: under search pressure,
            // reduce the number of neighbors evaluated per candidate.
            // expansion_throttle_ is set by the preempt callback (1.0 = complete, 0.5 = half).
            size_t effective_size = size;
            if (use_yield) {
                float throttle = expansion_throttle_.load(std::memory_order_relaxed);
                if (throttle < 1.0f) {
                    effective_size = std::max(size_t(1),
                        static_cast<size_t>(static_cast<float>(size) * throttle));
                }
            }
            local_edges_scanned += effective_size;
            tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *(data + 1) + 64),
                         _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < effective_size; j++) {
                tableint candidate_id = *(datal + j);
#ifdef USE_SSE
                if (j + 1 < effective_size) {
                    _mm_prefetch((char *)(visited_array + *(datal + j + 1)),
                                 _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)),
                                 _MM_HINT_T0);
                }
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 =
                    fstdistfunc_(data_point, currObj1, dist_func_param_);
                ++local_dist_comps;
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
            // Release node lock before yield — safe point
            lock.unlock();
            if (use_yield && fine_grain_yield_active_.load(std::memory_order_relaxed) && fine_grain_yield_cb_) {
                last_fine_grain_yield_ctx_.expanded_neighbors = static_cast<int>(size);
                last_fine_grain_yield_ctx_.candidate_queue_size = static_cast<int>(candidateSet.size());
                last_fine_grain_yield_ctx_.top_candidate_size = static_cast<int>(top_candidates.size());
                last_fine_grain_yield_ctx_.lower_bound = lowerBound;
                last_fine_grain_yield_ctx_.best_frontier_dist =
                    candidateSet.empty() ? lowerBound : -candidateSet.top().first;
                last_fine_grain_yield_ctx_.top_candidates_full =
                    top_candidates.size() == ef_construction_;
                fine_grain_yield_cb_();
            }
        }
        visited_list_pool_->releaseVisitedList(vl);
        const auto base_search_end = std::chrono::steady_clock::now();
        metric_graph_base_search_ns.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    base_search_end - base_search_start)
                    .count()),
            std::memory_order_relaxed);
        metric_graph_base_search_expansions.fetch_add(
            local_expansions, std::memory_order_relaxed);
        metric_graph_base_search_edges_scanned.fetch_add(
            local_edges_scanned, std::memory_order_relaxed);
        metric_graph_base_search_dist_comps.fetch_add(
            local_dist_comps, std::memory_order_relaxed);
        const uint64_t base_trace_end = base_trace_start ? monotonicRawNs() : 0;
        traceInsertPhase("insert_path_search", base_trace_start, base_trace_end,
                         trace_point_id, layer, std::numeric_limits<tableint>::max(),
                         0, local_edges_scanned, 0, local_dist_comps, 0, 0);

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
        SearchWorkStats *work_stats = nullptr) const {
        const auto snapshot_start = std::chrono::steady_clock::now();
        if (enable_mvcc_ && tls_snapshot_slot == SnapshotTracker::MAX_SLOTS) {
            tls_snapshot_slot = const_cast<SnapshotTracker&>(snapshot_tracker_).allocate_slot();
        }
        SnapshotTracker::Guard snap_guard(
            const_cast<SnapshotTracker*>(&snapshot_tracker_),
            tls_snapshot_slot, static_cast<long>(view_limit));
        if (work_stats) {
            work_stats->snapshot_guard_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - snapshot_start)
                    .count());
        }
        const auto visited_get_start = std::chrono::steady_clock::now();
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        if (work_stats) {
            work_stats->visited_list_get_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - visited_get_start)
                    .count());
        }
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
        const bool post_filter_refill =
            enforce_view_limit && search_visibility_mode_ == 1;
        const bool traversal_filter =
            enforce_view_limit && search_visibility_mode_ == 2;
        const bool effective_bare =
            bare_bone_search && (!enforce_view_limit || post_filter_refill);
        size_t future_skip_hops = 0;

        auto is_visible = [&](tableint internal_id) -> bool {
            if (!enforce_view_limit) return true;
            if (search_visibility_by_label_) {
                return static_cast<size_t>(getExternalLabel(internal_id)) <
                       view_limit;
            }
            if (enable_non_prefix_visibility_ && insertion_complete_) {
                // M3: non-prefix visibility — see any completed node, skip in-flight
                return static_cast<size_t>(internal_id) < view_limit ||
                       insertion_complete_[internal_id].load(std::memory_order_acquire);
            }
            return static_cast<size_t>(internal_id) < view_limit;
        };

        dist_t lowerBound;
        bool ep_allowed =
            effective_bare ||
            (!isMarkedDeleted(ep_id) &&
             ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))));
        if (ep_allowed) {
            char *ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            if (work_stats) {
                work_stats->distance_computations++;
                work_stats->level0_distance_computations++;
            }
            lowerBound = dist;
            const bool ep_visible = is_visible(ep_id);
            if (ep_visible || post_filter_refill) {
                top_candidates.emplace(dist, ep_id);
                if (work_stats) work_stats->result_pushes++;
            } else if (enforce_view_limit) {
                future_skip_hops++;
            }
            if (!effective_bare && stop_condition && ep_visible) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id),
                                                    ep_data, dist);
            }
            if (!traversal_filter || ep_visible) {
                candidate_set.emplace(-dist, ep_id);
                if (work_stats) work_stats->candidate_pushes++;
            }
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            if (!traversal_filter || is_visible(ep_id)) {
                candidate_set.emplace(-lowerBound, ep_id);
                if (work_stats) work_stats->candidate_pushes++;
            }
        }

        visited_array[ep_id] = visited_array_tag;
        if (work_stats) work_stats->visited_nodes++;

        // M2: beam stagnation tracking
        dist_t m2_prev_lb = lowerBound;
        int m2_stag_count = 0;
        size_t m2_expand_step = 0;
        bool rewrite_active_hit_query = false;
        bool rewrite_recent_hit_query = false;
        bool rewrite_period_hit_query = false;
        bool phase_existing_update_hit_query = false;
        bool phase_link_critical_hit_query = false;
        bool phase_load_scan_hit_query = false;
        bool phase_append_hit_query = false;
        bool phase_prune_hit_query = false;
        bool phase_undo_record_hit_query = false;
        bool phase_rewrite_hit_query = false;
        const uint32_t hot_region_search_epoch = beginHotRegionSearchEpoch();

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
            if (work_stats) {
                work_stats->candidate_pops++;
                work_stats->level0_expansions++;
            }
            m2_expand_step++;
            sampleSearchPhaseOverlap(phase_existing_update_hit_query,
                                     phase_link_critical_hit_query,
                                     phase_load_scan_hit_query,
                                     phase_append_hit_query,
                                     phase_prune_hit_query,
                                     phase_undo_record_hit_query,
                                     phase_rewrite_hit_query);

            tableint current_node_id = current_node_pair.second;
            if (work_stats && work_stats->capture_path &&
                work_stats->path_count < SEARCH_PATH_CAPTURE_CAP) {
                const uint32_t pos = work_stats->path_count++;
                work_stats->path_labels[pos] =
                    static_cast<uint32_t>(getExternalLabel(current_node_id));
                work_stats->path_dists[pos] =
                    static_cast<float>(candidate_dist);
            }
            const bool current_node_visible = is_visible(current_node_id);
            if (traversal_filter && !current_node_visible) {
                metric_search_invisible_expansions.fetch_add(
                    1, std::memory_order_relaxed);
                if (work_stats) work_stats->invisible_expansions++;
                future_skip_hops++;
                continue;
            }
            if (rewrite_active_diag_enabled_) {
                uint64_t active_rewriters =
                    global_rewrite_active_.load(std::memory_order_acquire);
                if (active_rewriters > 0) {
                    metric_search_rewrite_period_expansions.fetch_add(
                        1, std::memory_order_relaxed);
                    metric_search_rewrite_period_active_sum.fetch_add(
                        active_rewriters, std::memory_order_relaxed);
                    rewrite_period_hit_query = true;
                    uint64_t prev =
                        metric_search_rewrite_period_active_max.load(std::memory_order_relaxed);
                    while (active_rewriters > prev &&
                           !metric_search_rewrite_period_active_max.compare_exchange_weak(
                               prev, active_rewriters, std::memory_order_relaxed)) {
                    }
                }
                if (node_rewrite_active_ &&
                    static_cast<size_t>(current_node_id) < max_elements_ &&
                    node_rewrite_active_[current_node_id].load(std::memory_order_acquire) != 0) {
                    metric_search_rewrite_active_expansions.fetch_add(
                        1, std::memory_order_relaxed);
                    rewrite_active_hit_query = true;
                }
                if (rewrite_recent_window_ > 0 && node_rewrite_epoch_ &&
                    static_cast<size_t>(current_node_id) < max_elements_) {
                    uint64_t node_epoch =
                        node_rewrite_epoch_[current_node_id].load(std::memory_order_acquire);
                    uint64_t current_epoch =
                        global_rewrite_epoch_.load(std::memory_order_relaxed);
                    if (node_epoch != 0 && current_epoch >= node_epoch &&
                        current_epoch - node_epoch <= rewrite_recent_window_) {
                        metric_search_rewrite_recent_expansions.fetch_add(
                            1, std::memory_order_relaxed);
                        rewrite_recent_hit_query = true;
                    }
                }
            }
            std::shared_lock<std::shared_mutex> lock;
            if (use_node_lock_in_search_) {
                const auto lock_wait_start = std::chrono::steady_clock::now();
                lock = std::shared_lock<std::shared_mutex>(
                    link_list_locks_[current_node_id]);
                if (work_stats) {
                    work_stats->level0_lock_wait_ns += static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - lock_wait_start)
                            .count());
                }
	            }
	            int *data = (int *)get_linklist0(current_node_id);
            if (hot_region_search_epoch != 0) {
                markHotRegionAddress(data, hot_region_search_epoch);
                markHotRegionAddress(getDataByInternalId(current_node_id),
                                     hot_region_search_epoch);
            }
	            size_t size = getListCount((linklistsizeint *)data);
            if (work_stats) work_stats->level0_edges_scanned += size;
            if (enforce_view_limit && !current_node_visible) {
                metric_search_invisible_expansions.fetch_add(
                    1, std::memory_order_relaxed);
                metric_search_invisible_expansion_edges.fetch_add(
                    size, std::memory_order_relaxed);
                if (work_stats) {
                    work_stats->invisible_expansions++;
                    work_stats->invisible_edges += size;
                }
            }

	            if (collect_metrics) {
	                metric_hops++;
	                metric_distance_computations += size;
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

	            for (size_t j = 1; j <= size; j++) {
	                int candidate_id = *(data + j);
#ifdef USE_SSE
	                if (j < size) {
	                    _mm_prefetch((char *)(visited_array + *(data + j + 1)),
	                                 _MM_HINT_T0);
	                    _mm_prefetch(data_level0_memory_ +
	                                     (*(data + j + 1)) * size_data_per_element_ +
	                                     offsetData_,
	                                 _MM_HINT_T0);  ////////////
	                }
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;
                    if (work_stats) work_stats->visited_nodes++;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    if (hot_region_search_epoch != 0 &&
                        hot_region_mark_candidates_) {
                        markHotRegionAddress(currObj1, hot_region_search_epoch);
                    }
                    const bool candidate_visible = is_visible(candidate_id);
                    if (traversal_filter && !candidate_visible) {
                        future_skip_hops++;
                        continue;
                    }
                    if (enforce_view_limit && !candidate_visible) {
                        metric_search_invisible_candidate_dist_comps.fetch_add(
                            1, std::memory_order_relaxed);
                        if (work_stats) work_stats->invisible_candidate_dist_comps++;
                    }
                    dist_t dist =
                        fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (work_stats) {
                        work_stats->distance_computations++;
                        work_stats->level0_distance_computations++;
                    }

                    bool flag_consider_candidate;
                    if (!effective_bare && stop_condition) {
                        flag_consider_candidate =
                            stop_condition->should_consider_candidate(
                                dist, lowerBound);
                    } else {
                        flag_consider_candidate =
                            top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        if (work_stats && work_stats->capture_path &&
                            candidate_visible &&
                            work_stats->path_count < SEARCH_PATH_CAPTURE_CAP) {
                            const uint32_t pos = work_stats->path_count++;
                            work_stats->path_labels[pos] =
                                static_cast<uint32_t>(getExternalLabel(candidate_id));
                            work_stats->path_dists[pos] = static_cast<float>(dist);
                        }
                        candidate_set.emplace(-dist, candidate_id);
                        if (work_stats) work_stats->candidate_pushes++;
                        if (enforce_view_limit && !candidate_visible) {
                            metric_search_invisible_candidate_enqueues.fetch_add(
                                1, std::memory_order_relaxed);
                            if (work_stats) work_stats->invisible_candidate_enqueues++;
                        }
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ +
                                         candidate_set.top().second *
                                             size_data_per_element_ +
                                         offsetLevel0_,  ///////////
                                     _MM_HINT_T0);  ////////////////////////
#endif

                        bool candidate_allowed =
                            effective_bare ||
                            (!isMarkedDeleted(candidate_id) &&
                             ((!isIdAllowed) ||
                              (*isIdAllowed)(getExternalLabel(candidate_id))));
                        if (candidate_allowed) {
                            if (candidate_visible || post_filter_refill) {
                                top_candidates.emplace(dist, candidate_id);
                                if (work_stats) work_stats->result_pushes++;
                                if (!effective_bare && stop_condition &&
                                    candidate_visible) {
                                    stop_condition->add_point_to_result(
                                        getExternalLabel(candidate_id),
                                        currObj1, dist);
                                }
                            } else if (enforce_view_limit) {
                                future_skip_hops++;
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!effective_bare && stop_condition) {
                            flag_remove_extra =
                                stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!effective_bare && stop_condition) {
                                stop_condition->remove_point_from_result(
                                    getExternalLabel(id),
                                    getDataByInternalId(id), dist);
                                flag_remove_extra =
                                    stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }

            // ── M2: Mechanism 2 Recovery ─────────────────────────────────
	            if (enable_undo_recovery_ && enforce_view_limit && m2_mode_ > 0 &&
	                undo_log_ && isNodeModified(current_node_id)) {
                m2_modified_expansions_.fetch_add(1, std::memory_order_relaxed);

                // Count visible neighbors
                size_t visible_count = 0;
                for (size_t j = 1; j <= size; j++) {
                    if (is_visible(*(data + j))) visible_count++;
                }

                // Update stagnation tracker
                if (lowerBound < m2_prev_lb) {
                    m2_stag_count = 0;
                    m2_prev_lb = lowerBound;
                } else {
                    m2_stag_count++;
                }

                bool degree_deficit = visible_count < static_cast<size_t>(m2_degree_threshold_ * maxM0_);
                bool beam_stagnation = m2_stag_count >= m2_stagnation_hops_;

                // Decide whether to trigger based on mode
                bool do_recover = false;
                switch (m2_mode_) {
                    case 1: // eager: always recover for modified nodes
                        do_recover = true;
                        break;
                    case 2: // cnr: degree deficit OR beam stagnation
                        do_recover = degree_deficit || beam_stagnation;
                        break;
                    case 3: // degree_only
                        do_recover = degree_deficit;
                        break;
                    case 4: // stagnation_only
                        do_recover = beam_stagnation;
                        break;
                }

                if (do_recover) {
                    m2_recovery_attempts_.fetch_add(1, std::memory_order_relaxed);
                    const auto recovery_loop_start = std::chrono::steady_clock::now();
                    const uint32_t* rec_edges = nullptr;
                    const uint32_t* rec_causes = nullptr;
                    const float* rec_dist_node = nullptr;
                    const float* rec_dist_cause = nullptr;
                    const auto recovery_get_start = std::chrono::steady_clock::now();
                    size_t rec_count = undo_log_->getRecoverPointersEUL(
                        current_node_id, static_cast<uint32_t>(view_limit),
                        &rec_edges, &rec_causes, &rec_dist_node, &rec_dist_cause);
                    const auto recovery_get_end = std::chrono::steady_clock::now();
                    m2_recovery_get_ns_.fetch_add(
                        static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                recovery_get_end - recovery_get_start)
                                .count()),
                        std::memory_order_relaxed);
                    m2_recovery_candidate_edges_.fetch_add(
                        static_cast<long>(rec_count), std::memory_order_relaxed);

                    if (rec_count > 0) {
                        m2_triggered_.fetch_add(1, std::memory_order_relaxed);
                        if (degree_deficit) m2_degree_triggers_.fetch_add(1, std::memory_order_relaxed);
                        if (beam_stagnation) m2_stagnation_triggers_.fetch_add(1, std::memory_order_relaxed);

                        int max_rec = (m2_mode_ == 1) ? (int)rec_count : m2_max_recover_;
                        int recovered = 0;
                        for (size_t ri = 0; ri < rec_count && recovered < max_rec; ri++) {
                            tableint rec_id = rec_edges[ri];
                            if (!is_visible(rec_id)) continue;
                            if (visited_array[rec_id] == visited_array_tag) continue;

                            // Triangle inequality coarse filter (skip for eager mode)
                            if (m2_mode_ != 1) {
                                float d_owner_rec = rec_dist_node[ri];
                                dist_t lb_dist = std::abs(candidate_dist - d_owner_rec);
                                if (top_candidates.size() >= ef && lb_dist > lowerBound) {
                                    m2_triangle_filtered_.fetch_add(1, std::memory_order_relaxed);
                                    continue;
                                }
                            }

                            // Exact distance
                            visited_array[rec_id] = visited_array_tag;
                            char *rec_data_ptr = getDataByInternalId(rec_id);
                            dist_t rec_dist = fstdistfunc_(data_point, rec_data_ptr, dist_func_param_);
                            recovered++;
                            m2_recovered_edges_.fetch_add(1, std::memory_order_relaxed);
                            m2_extra_dist_comps_.fetch_add(1, std::memory_order_relaxed);

                            bool flag_consider = top_candidates.size() < ef || lowerBound > rec_dist;
                            if (flag_consider) {
                                candidate_set.emplace(-rec_dist, rec_id);
                                bool rec_allowed = effective_bare ||
                                    (!isMarkedDeleted(rec_id) &&
                                     ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(rec_id))));
                                if (rec_allowed && is_visible(rec_id)) {
                                    top_candidates.emplace(rec_dist, rec_id);
                                    m2_useful_edges_.fetch_add(1, std::memory_order_relaxed);
                                }
                                while (top_candidates.size() > ef) {
                                    top_candidates.pop();
                                }
                                if (!top_candidates.empty())
                                    lowerBound = top_candidates.top().first;
                            }
                        }
                        if (recovered > 0) m2_stag_count = 0;
                    }
                    const auto recovery_loop_end = std::chrono::steady_clock::now();
                    m2_recovery_loop_ns_.fetch_add(
                        static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                recovery_loop_end - recovery_loop_start)
                                .count()),
                        std::memory_order_relaxed);
                }

                // Inline GC during search
                long horizon = gc_horizon_.load(std::memory_order_relaxed);
                if (horizon > 0) {
                    undo_log_->advanceGcStart(current_node_id, horizon);
                }
            }
            // ── end M2 ──────────────────────────────────────────────────
        }

        if (rewrite_active_hit_query) {
            metric_search_rewrite_active_queries.fetch_add(
                1, std::memory_order_relaxed);
        }
        if (rewrite_recent_hit_query) {
            metric_search_rewrite_recent_queries.fetch_add(
                1, std::memory_order_relaxed);
        }
        if (rewrite_period_hit_query) {
            metric_search_rewrite_period_queries.fetch_add(
                1, std::memory_order_relaxed);
        }
        if (phaseOverlapDiagEnabled()) {
            if (phase_existing_update_hit_query) {
                metric_search_phase_existing_update_queries.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (phase_link_critical_hit_query) {
                metric_search_phase_link_critical_queries.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (phase_load_scan_hit_query) {
                metric_search_phase_load_scan_queries.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (phase_append_hit_query) {
                metric_search_phase_append_queries.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (phase_prune_hit_query) {
                metric_search_phase_prune_queries.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (phase_undo_record_hit_query) {
                metric_search_phase_undo_record_queries.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (phase_rewrite_hit_query) {
                metric_search_phase_rewrite_queries.fetch_add(
                    1, std::memory_order_relaxed);
            }
        }

        if (enforce_view_limit && future_skip_hops > 0) {
            metric_future_skip_queries.fetch_add(1, std::memory_order_relaxed);
            metric_future_skip_hops_total.fetch_add(future_skip_hops,
                                                    std::memory_order_relaxed);
            size_t prev =
                metric_future_skip_hop_max.load(std::memory_order_relaxed);
            while (future_skip_hops > prev &&
                   !metric_future_skip_hop_max.compare_exchange_weak(
                       prev, future_skip_hops, std::memory_order_relaxed)) {
            }
        }
        if (work_stats) {
            work_stats->future_skip_hops += future_skip_hops;
        }

        const auto visited_release_start = std::chrono::steady_clock::now();
        visited_list_pool_->releaseVisitedList(vl);
        if (work_stats) {
            work_stats->visited_list_release_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - visited_release_start)
                    .count());
        }
        return top_candidates;
    }

    void getNeighborsByHeuristic(
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> &top_candidates,
        const size_t M, uint64_t *heuristic_dist_comps = nullptr) {
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
                if (heuristic_dist_comps != nullptr) {
                    ++(*heuristic_dist_comps);
                }
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

    tableint mutuallyConnectNewElement(
        const void *data_point, tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> &top_candidates,
        int level, bool isUpdate) {
        const auto graph_connect_start = std::chrono::steady_clock::now();
        uint64_t local_link_updates = 0;
        size_t Mcurmax = level ? maxM_ : maxM0_;
        const auto select_neighbors_start = std::chrono::steady_clock::now();
        const uint64_t select_neighbors_input = top_candidates.size();
        uint64_t select_neighbors_dist_comps = 0;
        const size_t min_cold = hot_region_filter_min_cold_ > 0
                                    ? static_cast<size_t>(hot_region_filter_min_cold_)
                                    : M_ * 2;
        filterHotLinkCandidatePool(top_candidates, level, min_cold);
        getNeighborsByHeuristic(top_candidates, M_, &select_neighbors_dist_comps);
        const auto select_neighbors_end = std::chrono::steady_clock::now();
        metric_graph_select_new_neighbors_ns.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    select_neighbors_end - select_neighbors_start)
                    .count()),
            std::memory_order_relaxed);
        metric_graph_select_new_neighbors_input.fetch_add(
            select_neighbors_input, std::memory_order_relaxed);
        metric_graph_select_new_neighbors_heuristic_dist_comps.fetch_add(
            select_neighbors_dist_comps, std::memory_order_relaxed);
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
        metric_graph_select_new_neighbors_selected.fetch_add(
            selectedNeighbors.size(), std::memory_order_relaxed);

        tableint next_closest_entry_point = selectedNeighbors.back();
        reorderLinkCandidatesByHotRegion(selectedNeighbors, level);
        if (level == 0 &&
            static_cast<size_t>(cur_c) < m2_base_insert_neighbor_counts_.size() &&
            m2_base_insert_neighbor_stride_ > 0) {
            const size_t count =
                std::min(selectedNeighbors.size(),
                         std::min<size_t>(m2_base_insert_neighbor_stride_,
                                          std::numeric_limits<uint16_t>::max()));
            const size_t base =
                static_cast<size_t>(cur_c) * m2_base_insert_neighbor_stride_;
            for (size_t i = 0; i < count; ++i) {
                m2_base_insert_neighbors_[base + i] = selectedNeighbors[i];
            }
            m2_base_insert_neighbor_counts_[cur_c] =
                static_cast<uint16_t>(count);
        }

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already
            // acquired
            ActivePhaseGuard link_critical_phase(active_phase_link_critical_);
            tls_in_link_critical_section = true;
            const auto critical_start = std::chrono::steady_clock::now();
            std::unique_lock<std::shared_mutex> lock(link_list_locks_[cur_c],
                                                     std::defer_lock);
            if (isUpdate) {
                const auto wait_start = std::chrono::steady_clock::now();
                lock.lock();
                const auto wait_end = std::chrono::steady_clock::now();
                metric_graph_unique_lock_wait_ns.fetch_add(
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            wait_end - wait_start)
                            .count()),
                    std::memory_order_relaxed);
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
                ++local_link_updates;
            }
            metric_graph_inserted_node_edges_written.fetch_add(
                selectedNeighbors.size(), std::memory_order_relaxed);
            if (level == 0) {
                bumpNodeLinkVersion(cur_c);
            }
            const auto critical_end = std::chrono::steady_clock::now();
            metric_graph_link_critical_ns.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        critical_end - critical_start)
                        .count()),
                std::memory_order_relaxed);
            metric_graph_inserted_node_link_ns.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        critical_end - critical_start)
                        .count()),
                std::memory_order_relaxed);
            tls_in_link_critical_section = false;
        }

        const bool existing_update_read_scan_only_for_point =
            existing_neighbor_update_read_scan_only_ &&
            (existing_neighbor_update_read_scan_only_from_id_ < 0 ||
             cur_c >= static_cast<tableint>(existing_neighbor_update_read_scan_only_from_id_));
        const bool disable_existing_undo_record_for_point =
            disable_existing_neighbor_undo_record_ &&
            (disable_existing_neighbor_undo_record_from_id_ < 0 ||
             cur_c >= static_cast<tableint>(disable_existing_neighbor_undo_record_from_id_));

        const auto existing_neighbor_update_loop_start = std::chrono::steady_clock::now();
        ActivePhaseGuard existing_update_phase(active_phase_existing_update_);
        struct RewritePeriodGuard {
            std::atomic<uint64_t>& active;
            explicit RewritePeriodGuard(std::atomic<uint64_t>& active_ref)
                : active(active_ref) {
                active.fetch_add(1, std::memory_order_acq_rel);
            }
            ~RewritePeriodGuard() {
                active.fetch_sub(1, std::memory_order_acq_rel);
            }
        };
        std::unique_ptr<RewritePeriodGuard> rewrite_period_guard;
        if (rewrite_active_diag_enabled_) {
            rewrite_period_guard =
                std::make_unique<RewritePeriodGuard>(global_rewrite_active_);
        }
        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            const tableint neighbor = selectedNeighbors[idx];
            metric_graph_existing_neighbor_visits.fetch_add(1, std::memory_order_relaxed);
            tableint active_rewrite_node = neighbor;
            if (rewrite_active_diag_enabled_ && node_rewrite_active_ &&
                static_cast<size_t>(active_rewrite_node) < max_elements_) {
                uint64_t epoch = global_rewrite_epoch_.fetch_add(
                    1, std::memory_order_relaxed);
                if (node_rewrite_epoch_) {
                    node_rewrite_epoch_[active_rewrite_node].store(
                        epoch, std::memory_order_release);
                }
                node_rewrite_active_[active_rewrite_node].store(
                    1, std::memory_order_release);
            }
            tls_in_link_critical_section = true;
            const auto wait_start = std::chrono::steady_clock::now();
            std::unique_lock<std::shared_mutex> lock(
                link_list_locks_[neighbor], std::defer_lock);
            lock.lock();
            const auto critical_start = std::chrono::steady_clock::now();
            metric_graph_unique_lock_wait_ns.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        critical_start - wait_start)
                        .count()),
	                std::memory_order_relaxed);
            ActivePhaseGuard link_critical_phase(active_phase_link_critical_);

            const uint64_t load_trace_start =
                insertPhaseTraceEnabled() ? monotonicRawNs() : 0;
            const auto load_scan_start = std::chrono::steady_clock::now();
            linklistsizeint *ll_other;
            size_t sz_link_list_other = 0;
            tableint *data = nullptr;
            bool is_cur_c_present = false;
            {
                ActivePhaseGuard load_scan_phase(active_phase_load_scan_);
                if (level == 0)
                    ll_other = get_linklist0(neighbor);
                else
                    ll_other = get_linklist(neighbor, level);

                sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (neighbor == cur_c)
                    throw std::runtime_error(
                        "Trying to connect an element to itself");
                if (level > element_levels_[neighbor])
                    throw std::runtime_error(
                        "Trying to make a link on a non-existent level");

                data = (tableint *)(ll_other + 1);

                if (isUpdate) {
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if (data[j] == cur_c) {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }
            }
            const auto load_scan_end = std::chrono::steady_clock::now();
            metric_graph_existing_neighbor_load_scan_ns.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        load_scan_end - load_scan_start)
                        .count()),
                std::memory_order_relaxed);
            metric_graph_existing_neighbor_loaded_edges.fetch_add(
                sz_link_list_other, std::memory_order_relaxed);
            const uint64_t load_trace_end =
                load_trace_start ? monotonicRawNs() : 0;
            traceInsertPhase("existing_neighbor_load_scan", load_trace_start,
                             load_trace_end, cur_c, level,
                             neighbor, sz_link_list_other,
                             sz_link_list_other, 0, 0, 0, 0);

            // If cur_c is already present in the neighboring connections of
            // `neighbor` then no need to modify any connections
            // or run the heuristics.
		            if (!is_cur_c_present) {
		                if (sz_link_list_other < Mcurmax) {
                            const auto append_start = std::chrono::steady_clock::now();
                            ActivePhaseGuard append_phase(active_phase_append_);
		                    metric_graph_existing_neighbor_appends.fetch_add(1, std::memory_order_relaxed);
                            if (!existing_update_read_scan_only_for_point) {
			                        data[sz_link_list_other] = cur_c;
			                        setListCount(ll_other, sz_link_list_other + 1);
                            }
                            const auto append_end = std::chrono::steady_clock::now();
                            metric_graph_existing_neighbor_append_ns.fetch_add(
                                static_cast<uint64_t>(
                                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        append_end - append_start)
	                                .count()),
                                std::memory_order_relaxed);
	                            if (!existing_update_read_scan_only_for_point) {
			                        metric_graph_existing_neighbor_edges_written.fetch_add(
	                                    1, std::memory_order_relaxed);
			                        local_link_updates += 2;
                            }
		                    if (level == 0 && !existing_update_read_scan_only_for_point) {
		                        bumpNodeLinkVersion(neighbor);
		                    }
		                } else {
		                    metric_graph_existing_neighbor_prunes.fetch_add(1, std::memory_order_relaxed);
	                    metric_graph_existing_neighbor_prune_candidates.fetch_add(
	                        sz_link_list_other + 1, std::memory_order_relaxed);
                    const uint64_t prune_trace_start =
                        insertPhaseTraceEnabled() ? monotonicRawNs() : 0;
		                    const auto prune_heuristic_start = std::chrono::steady_clock::now();
                    ActivePhaseGuard prune_phase(active_phase_prune_);
	                    // finding the "weakest" element to replace it with the new
	                    // one
                    uint64_t prune_dist_comps = 0;
	                    dist_t d_max = fstdistfunc_(
	                        getDataByInternalId(cur_c),
	                        getDataByInternalId(neighbor),
	                        dist_func_param_);
                    ++prune_dist_comps;
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
	                                getDataByInternalId(neighbor),
	                                dist_func_param_),
	                            data[j]);
                        ++prune_dist_comps;
	                    }

                    uint64_t prune_heuristic_dist_comps = 0;
	                    getNeighborsByHeuristic(candidates, Mcurmax,
                                             &prune_heuristic_dist_comps);
                    prune_dist_comps += prune_heuristic_dist_comps;
	                    const auto prune_heuristic_end = std::chrono::steady_clock::now();
	                    metric_graph_existing_neighbor_prune_ns.fetch_add(
                        static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                prune_heuristic_end - prune_heuristic_start)
	                                .count()),
	                        std::memory_order_relaxed);
                    metric_graph_existing_neighbor_prune_dist_comps.fetch_add(
                        prune_dist_comps, std::memory_order_relaxed);
                    const uint64_t prune_trace_end =
                        prune_trace_start ? monotonicRawNs() : 0;
                    traceInsertPhase("existing_neighbor_prune", prune_trace_start,
                                     prune_trace_end, cur_c, level,
                                     neighbor, sz_link_list_other,
                                     0, sz_link_list_other + 1,
                                     prune_dist_comps, 0, 0);

                    std::vector<tableint> old_edges(data, data + sz_link_list_other);
                    std::vector<tableint> new_edges;
                    auto candidates_copy = candidates;
                    while (candidates_copy.size() > 0) {
                        new_edges.push_back(candidates_copy.top().second);
                        candidates_copy.pop();
                    }

                    std::vector<tableint> old_edges_sorted = old_edges;
                    std::vector<tableint> new_edges_sorted = new_edges;
                    std::sort(old_edges_sorted.begin(), old_edges_sorted.end());
                    std::sort(new_edges_sorted.begin(), new_edges_sorted.end());
                    const bool linkset_changed = old_edges_sorted != new_edges_sorted;

                    uint64_t edges_pruned = 0;
                    const bool record_undo =
                        !existing_update_read_scan_only_for_point &&
                        !disable_existing_undo_record_for_point &&
                        enable_mvcc_ && undo_log_;
                    const uint64_t undo_trace_start =
                        (record_undo && insertPhaseTraceEnabled()) ? monotonicRawNs() : 0;
                    const auto undo_record_start = std::chrono::steady_clock::now();
                    std::unique_ptr<ActivePhaseGuard> undo_phase;
                    if (record_undo) {
                        undo_phase = std::make_unique<ActivePhaseGuard>(
                            active_phase_undo_record_);
                    }
                    for (tableint old_edge : old_edges) {
                        bool found = false;
                        for (tableint new_edge : new_edges) {
                            if (old_edge == new_edge) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            ++edges_pruned;
                            if (record_undo) {
                                long batch_ts = global_watermark_.load(std::memory_order_acquire);
                                float dist_node_pruned = fstdistfunc_(
                                    getDataByInternalId(neighbor),
                                    getDataByInternalId(old_edge),
                                    dist_func_param_);
                                float dist_cause_pruned = fstdistfunc_(
                                    getDataByInternalId(cur_c),
                                    getDataByInternalId(old_edge),
                                    dist_func_param_);

                                undo_log_->recordPrune(neighbor, old_edge, cur_c,
                                    batch_ts, dist_node_pruned, dist_cause_pruned, nullptr);
                                metric_graph_existing_neighbor_pruned_edges_recorded.fetch_add(
                                    1, std::memory_order_relaxed);
                                markNodeModified(neighbor);

                                // Inline GC
                                long horizon = gc_horizon_.load(std::memory_order_relaxed);
                                if (horizon > 0) {
                                    undo_log_->advanceGcStart(neighbor, horizon);
                                }
                            }
                        }
                    }
                    metric_graph_existing_neighbor_edges_pruned.fetch_add(
                        edges_pruned, std::memory_order_relaxed);
                    if (record_undo) {
                        const auto undo_record_end = std::chrono::steady_clock::now();
                        metric_graph_existing_neighbor_undo_record_ns.fetch_add(
                            static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    undo_record_end - undo_record_start)
                                    .count()),
	                            std::memory_order_relaxed);
                        const uint64_t undo_trace_end =
                            undo_trace_start ? monotonicRawNs() : 0;
                        traceInsertPhase("existing_neighbor_undo_record", undo_trace_start,
                                         undo_trace_end, cur_c, level,
                                         neighbor,
                                         sz_link_list_other, 0, 0, 0, 0,
                                         edges_pruned);
                    }
                    undo_phase.reset();

                    if (!existing_update_read_scan_only_for_point) {
                        const uint64_t rewrite_trace_start =
                            insertPhaseTraceEnabled() ? monotonicRawNs() : 0;
                        const auto rewrite_start = std::chrono::steady_clock::now();
                        ActivePhaseGuard rewrite_phase(active_phase_rewrite_);
		                    int indx = 0;
		                    while (candidates.size() > 0) {
		                        data[indx] = candidates.top().second;
		                        candidates.pop();
	                        indx++;
	                        ++local_link_updates;
		                    }
		
		                    setListCount(ll_other, indx);
                        const auto rewrite_end = std::chrono::steady_clock::now();
                        metric_graph_existing_neighbor_rewrite_ns.fetch_add(
                            static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    rewrite_end - rewrite_start)
                                    .count()),
                            std::memory_order_relaxed);
                        metric_graph_existing_neighbor_edges_written.fetch_add(
                            static_cast<uint64_t>(indx), std::memory_order_relaxed);
                        const uint64_t rewrite_trace_end =
                            rewrite_trace_start ? monotonicRawNs() : 0;
                        traceInsertPhase("existing_neighbor_rewrite", rewrite_trace_start,
                                         rewrite_trace_end, cur_c, level,
                                         neighbor, sz_link_list_other,
                                         0, 0, 0, static_cast<uint64_t>(indx),
                                         edges_pruned);
		                    ++local_link_updates;
                    }
		                    if (level == 0 && linkset_changed &&
	                            !existing_update_read_scan_only_for_point) {
		                        bumpNodeLinkVersion(neighbor);
		                    }
	                }
            }
            const auto critical_end = std::chrono::steady_clock::now();
            metric_graph_link_critical_ns.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        critical_end - critical_start)
                        .count()),
                std::memory_order_relaxed);
            tls_in_link_critical_section = false;
            if (rewrite_active_hold_us_ > 0) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(rewrite_active_hold_us_));
            }
            if (rewrite_active_diag_enabled_ && node_rewrite_active_ &&
                static_cast<size_t>(active_rewrite_node) < max_elements_) {
                node_rewrite_active_[active_rewrite_node].store(
                    0, std::memory_order_release);
            }
        }
        const auto existing_neighbor_update_loop_end = std::chrono::steady_clock::now();
        metric_graph_existing_neighbor_update_loop_ns.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    existing_neighbor_update_loop_end - existing_neighbor_update_loop_start)
                    .count()),
            std::memory_order_relaxed);

        const auto graph_connect_end = std::chrono::steady_clock::now();
        metric_graph_connect_calls.fetch_add(1, std::memory_order_relaxed);
        metric_graph_connect_ns.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    graph_connect_end - graph_connect_start)
                    .count()),
            std::memory_order_relaxed);
        metric_graph_link_updates.fetch_add(local_link_updates,
                                            std::memory_order_relaxed);
        return next_closest_entry_point;
    }

    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error(
                "Cannot resize, max element is less than the current number of "
                "elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);
        resizeM2BaseInsertNeighborStorage(new_max_elements);

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

        size_links_per_element_ =
            maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ =
            maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::shared_mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS)
            .swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **)malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error(
                "Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        initM2BaseInsertNeighborStorage(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        initWatermarkState(max_elements_);
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
        ready_commit_counter_.store(static_cast<long>(cur_element_count), std::memory_order_relaxed);

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

    std::vector<tableint> getM2BaseInsertNeighbors(tableint internalId) const {
        if (static_cast<size_t>(internalId) >=
            m2_base_insert_neighbor_counts_.size()) {
            return {};
        }
        if (!insertion_complete_[internalId].load(std::memory_order_acquire)) {
            return {};
        }
        const size_t count = std::min<size_t>(
            m2_base_insert_neighbor_counts_[internalId],
            m2_base_insert_neighbor_stride_);
        std::vector<tableint> out;
        out.reserve(count);
        const size_t base =
            static_cast<size_t>(internalId) * m2_base_insert_neighbor_stride_;
        for (size_t i = 0; i < count; ++i) {
            out.push_back(m2_base_insert_neighbors_[base + i]);
        }
        return out;
    }

    tableint addPoint(const void *data_point, labeltype label, int level,
                      bool defer_watermark = false) {
        tableint cur_c = 0;
        bool enqueue_deferred_repair = false;
        bool allow_deferred_drain =
            mid_insert_harm_online_policy_ && mid_insert_harm_defer_enabled_ &&
            !defer_watermark;
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

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error(
                    "The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count.fetch_add(1, std::memory_order_acq_rel);
            label_lookup_[label] = cur_c;
        }

        insertion_complete_[cur_c].store(false, std::memory_order_relaxed);
        markMidInsertDeferredPending(cur_c, false);
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
            if (curlevel < maxlevelcopy) {
                const uint64_t upper_trace_start =
                    insertPhaseTraceEnabled() ? monotonicRawNs() : 0;
                const auto upper_search_start = std::chrono::steady_clock::now();
                uint64_t upper_search_dist_comps = 0;
                uint64_t upper_search_edges_scanned = 0;
                dist_t curdist = fstdistfunc_(
                    data_point, getDataByInternalId(currObj), dist_func_param_);
                ++upper_search_dist_comps;
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock<std::shared_mutex> lock(
                            link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);
                        upper_search_edges_scanned += static_cast<uint64_t>(size);

                        tableint *datal = (tableint *)(data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_ ||
                                !insertion_complete_[cand].load(std::memory_order_acquire))
                                continue;  // skip dirty/incomplete candidate (concurrent insert)
	                            dist_t d = fstdistfunc_(data_point,
	                                                    getDataByInternalId(cand),
	                                                    dist_func_param_);
                            ++upper_search_dist_comps;
	                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
                const auto upper_search_end = std::chrono::steady_clock::now();
                metric_graph_upper_search_ns.fetch_add(
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            upper_search_end - upper_search_start)
                            .count()),
                    std::memory_order_relaxed);
                metric_graph_upper_search_dist_comps.fetch_add(
                    upper_search_dist_comps, std::memory_order_relaxed);
                metric_graph_upper_search_edges_scanned.fetch_add(
                    upper_search_edges_scanned, std::memory_order_relaxed);
                const uint64_t upper_trace_end =
                    upper_trace_start ? monotonicRawNs() : 0;
                traceInsertPhase("insert_path_search", upper_trace_start, upper_trace_end,
                                 cur_c, curlevel,
                                 std::numeric_limits<tableint>::max(), 0,
                                 upper_search_edges_scanned, 0,
                                 upper_search_dist_comps, 0, 0);
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0;
                 level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                auto make_live_top_candidates = [&]() {
	                    std::priority_queue<std::pair<dist_t, tableint>,
	                                        std::vector<std::pair<dist_t, tableint>>,
	                                        CompareByFirst>
	                        tc = searchBaseLayer(currObj, data_point, level,
	                                             fine_grain_yield_cb_ ? true : false,
                                                 cur_c);
                    if (epDeleted) {
                        tc.emplace(
                            fstdistfunc_(data_point,
                                         getDataByInternalId(enterpoint_copy),
                                         dist_func_param_),
                            enterpoint_copy);
                        if (tc.size() > ef_construction_)
                            tc.pop();
                    }
                    return tc;
                };

                auto make_micro_probe_top_candidates = [&](size_t probe_ef) {
                    std::priority_queue<std::pair<dist_t, tableint>,
                                        std::vector<std::pair<dist_t, tableint>>,
                                        CompareByFirst>
                        tc;
                    if (probe_ef == 0) {
                        return tc;
                    }
                    if (level == 0) {
                        tc = searchBaseLayerST<false>(
                            currObj, data_point, probe_ef, nullptr, nullptr);
                    } else {
                        tc = make_live_top_candidates();
                    }
                    if (epDeleted) {
                        tc.emplace(
                            fstdistfunc_(data_point,
                                         getDataByInternalId(enterpoint_copy),
                                         dist_func_param_),
                            enterpoint_copy);
                        if (tc.size() > probe_ef) {
                            tc.pop();
                        }
                    }
                    return tc;
                };

                auto make_snapshot_top_candidates = [&](size_t view_limit) {
                    std::priority_queue<std::pair<dist_t, tableint>,
                                        std::vector<std::pair<dist_t, tableint>>,
                                        CompareByFirst>
                        tc;
                    if (level == 0) {
                        tc = searchBaseLayerST<false>(
                            currObj, data_point, ef_construction_, nullptr, nullptr,
                            view_limit);
                    } else {
                        tc = make_live_top_candidates();
                    }
                    if (epDeleted) {
                        const bool ep_visible =
                            view_limit == std::numeric_limits<size_t>::max() ||
                            static_cast<size_t>(enterpoint_copy) < view_limit;
                        if (ep_visible) {
                            tc.emplace(
                                fstdistfunc_(data_point,
                                             getDataByInternalId(enterpoint_copy),
                                             dist_func_param_),
                                enterpoint_copy);
                            if (tc.size() > ef_construction_)
                                tc.pop();
                        }
                    }
                    return tc;
                };

                auto top_candidates = make_live_top_candidates();

                auto snapshot_candidate_ids = [&](const auto& pq_like) {
                    auto pq = pq_like;
                    std::vector<tableint> ids;
                    ids.reserve(pq.size());
                    while (!pq.empty()) {
                        ids.push_back(pq.top().second);
                        pq.pop();
                    }
                    std::sort(ids.begin(), ids.end());
                    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
                    return ids;
                };

                auto snapshot_candidate_pairs = [&](const auto& pq_like) {
                    auto pq = pq_like;
                    std::vector<std::pair<dist_t, tableint>> pairs;
                    pairs.reserve(pq.size());
                    while (!pq.empty()) {
                        pairs.push_back(pq.top());
                        pq.pop();
                    }
                    std::sort(pairs.begin(), pairs.end(),
                              [](const std::pair<dist_t, tableint>& a,
                                 const std::pair<dist_t, tableint>& b) {
                                  if (a.first != b.first) {
                                      return a.first < b.first;
                                  }
                                  return a.second < b.second;
                              });
                    return pairs;
                };

                auto snapshot_selected_pairs = [&](const auto& pq_like) {
                    auto pq = pq_like;
                    getNeighborsByHeuristic(pq, M_);
                    std::vector<std::pair<dist_t, tableint>> pairs;
                    pairs.reserve(pq.size());
                    while (!pq.empty()) {
                        pairs.push_back(pq.top());
                        pq.pop();
                    }
                    std::sort(pairs.begin(), pairs.end(),
                              [](const std::pair<dist_t, tableint>& a,
                                 const std::pair<dist_t, tableint>& b) {
                                  if (a.first != b.first) {
                                      return a.first < b.first;
                                  }
                                  return a.second < b.second;
                              });
                    return pairs;
                };

                auto snapshot_selected_ids = [&](const auto& pq_like) {
                    auto pairs = snapshot_selected_pairs(pq_like);
                    std::vector<tableint> ids;
                    ids.reserve(pairs.size());
                    for (const auto& pair : pairs) {
                        ids.push_back(pair.second);
                    }
                    return ids;
                };

                auto snapshot_node_versions = [&](const std::vector<tableint>& ids) {
                    std::vector<uint64_t> versions;
                    versions.reserve(ids.size());
                    for (tableint id : ids) {
                        versions.push_back(getNodeLinkVersion(id));
                    }
                    return versions;
                };

                auto count_version_touches =
                    [&](const std::vector<tableint>& ids,
                        const std::vector<uint64_t>& versions) {
                        long touched = 0;
                        const size_t n = std::min(ids.size(), versions.size());
                        for (size_t i = 0; i < n; ++i) {
                            if (getNodeLinkVersion(ids[i]) != versions[i]) {
                                ++touched;
                            }
                        }
                        return touched;
                    };

                struct BoundaryWitnessInfo {
                    std::vector<tableint> blocked_candidate_ids;
                    std::vector<tableint> blocker_ids;
                    std::vector<uint64_t> blocker_versions;
                };

                auto snapshot_boundary_witnesses =
                    [&](const std::vector<std::pair<dist_t, tableint>>& candidate_pairs,
                        const std::vector<std::pair<dist_t, tableint>>& selected_pairs) {
                        BoundaryWitnessInfo info;
                        if (selected_pairs.empty()) {
                            return info;
                        }

                        const dist_t worst_selected_dist = selected_pairs.back().first;
                        std::unordered_set<tableint> selected_id_set;
                        selected_id_set.reserve(selected_pairs.size());
                        for (const auto& pair : selected_pairs) {
                            selected_id_set.insert(pair.second);
                        }

                        for (const auto& pair : candidate_pairs) {
                            if (selected_id_set.find(pair.second) != selected_id_set.end()) {
                                continue;
                            }
                            if (pair.first > worst_selected_dist) {
                                break;
                            }

                            tableint blocker = static_cast<tableint>(-1);
                            for (const auto& selected_pair : selected_pairs) {
                                dist_t blocker_dist = fstdistfunc_(
                                    getDataByInternalId(selected_pair.second),
                                    getDataByInternalId(pair.second), dist_func_param_);
                                if (blocker_dist < pair.first) {
                                    blocker = selected_pair.second;
                                    break;
                                }
                            }
                            if (blocker == static_cast<tableint>(-1)) {
                                continue;
                            }

                            info.blocked_candidate_ids.push_back(pair.second);
                            info.blocker_ids.push_back(blocker);
                            info.blocker_versions.push_back(getNodeLinkVersion(blocker));
                        }

                        return info;
                    };

                auto make_micro_replan_top_candidates =
                    [&](const std::vector<std::pair<dist_t, tableint>>& candidate_pairs,
                        const std::vector<std::pair<dist_t, tableint>>& selected_pairs,
                        const std::vector<std::pair<dist_t, tableint>>& foreign_outrank_pairs,
                        const std::vector<std::pair<dist_t, tableint>>& fresh_probe_pairs,
                        size_t* pool_size_out) {
                        std::priority_queue<std::pair<dist_t, tableint>,
                                            std::vector<std::pair<dist_t, tableint>>,
                                            CompareByFirst>
                            tc;
                        size_t pool_cap = static_cast<size_t>(
                            std::max(1, mid_insert_harm_micro_pool_cap_));
                        std::unordered_map<tableint, dist_t> best_dist;
                        best_dist.reserve(candidate_pairs.size() + fresh_probe_pairs.size() +
                                          foreign_outrank_pairs.size());

                        const auto absorb_pairs =
                            [&](const std::vector<std::pair<dist_t, tableint>>& pairs) {
                                for (const auto& pair : pairs) {
                                    tableint id = pair.second;
                                    if (id == static_cast<tableint>(-1) || id >= max_elements_) {
                                        continue;
                                    }
                                    auto it = best_dist.find(id);
                                    if (it == best_dist.end() || pair.first < it->second) {
                                        best_dist[id] = pair.first;
                                    }
                                }
                            };

                        absorb_pairs(selected_pairs);
                        absorb_pairs(fresh_probe_pairs);
                        absorb_pairs(foreign_outrank_pairs);
                        absorb_pairs(candidate_pairs);

                        std::vector<std::pair<dist_t, tableint>> merged_pairs;
                        merged_pairs.reserve(best_dist.size());
                        for (const auto& kv : best_dist) {
                            merged_pairs.emplace_back(kv.second, kv.first);
                        }
                        std::sort(merged_pairs.begin(), merged_pairs.end(),
                                  [](const std::pair<dist_t, tableint>& a,
                                     const std::pair<dist_t, tableint>& b) {
                                      if (a.first != b.first) {
                                          return a.first < b.first;
                                      }
                                      return a.second < b.second;
                                  });
                        if (merged_pairs.size() > pool_cap) {
                            merged_pairs.resize(pool_cap);
                        }
                        for (const auto& pair : merged_pairs) {
                            tc.emplace(pair.first, pair.second);
                        }

                        if (pool_size_out != nullptr) {
                            *pool_size_out = merged_pairs.size();
                        }
                        return tc;
                    };

                auto record_overlap_stats =
                    [&](const std::vector<tableint>& before_ids,
                        const std::vector<tableint>& after_ids,
                        std::atomic<long>& changed_counter,
                        std::atomic<long>& before_total_counter,
                        std::atomic<long>& after_total_counter,
                        std::atomic<long>& inter_counter,
                        std::atomic<long>& union_counter) {
                        long inter = 0;
                        size_t bi = 0;
                        size_t ai = 0;
                        while (bi < before_ids.size() && ai < after_ids.size()) {
                            if (before_ids[bi] == after_ids[ai]) {
                                ++inter;
                                ++bi;
                                ++ai;
                            } else if (before_ids[bi] < after_ids[ai]) {
                                ++bi;
                            } else {
                                ++ai;
                            }
                        }

                        const long before_sz = static_cast<long>(before_ids.size());
                        const long after_sz = static_cast<long>(after_ids.size());
                        const long uni = before_sz + after_sz - inter;
                        before_total_counter.fetch_add(before_sz, std::memory_order_relaxed);
                        after_total_counter.fetch_add(after_sz, std::memory_order_relaxed);
                        inter_counter.fetch_add(inter, std::memory_order_relaxed);
                        union_counter.fetch_add(uni, std::memory_order_relaxed);
                        if (uni != inter) {
                            changed_counter.fetch_add(1, std::memory_order_relaxed);
                        }
                    };

                const long ready_before_hook = ready_commit_counter_.load(std::memory_order_acquire);
                const bool cadence_ok = (mid_insert_preempt_every_n_ <= 1) ||
                                        ((static_cast<long>(cur_c) % static_cast<long>(mid_insert_preempt_every_n_)) == 0);
                const bool holds_global_lock = templock.owns_lock();
                const bool mid_insert_hook_eligible =
                    level == 0 && mid_insert_preempt_enable_ && mid_insert_preempt_k_ > 0 &&
                    ready_before_hook >= static_cast<long>(mid_insert_preempt_activate_after_commits_) &&
                    cadence_ok;
                if (mid_insert_hook_eligible && holds_global_lock) {
                    mid_insert_preempt_global_lock_skipped_.fetch_add(1, std::memory_order_relaxed);
                } else if (mid_insert_hook_eligible) {
                    bool acquired_slot = false;
                    long inflight = mid_insert_preempt_inflight_.load(std::memory_order_relaxed);
                    while (inflight < static_cast<long>(mid_insert_preempt_max_inflight_)) {
                        if (mid_insert_preempt_inflight_.compare_exchange_weak(
                                inflight, inflight + 1, std::memory_order_acq_rel,
                                std::memory_order_relaxed)) {
                            acquired_slot = true;
                            break;
                        }
                    }

                    if (!acquired_slot) {
                        mid_insert_preempt_slot_denied_.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        mid_insert_preempt_hits_.fetch_add(1, std::memory_order_relaxed);
                        const bool do_shadow_replan =
                            mid_insert_preempt_revalidate_ || mid_insert_preempt_shadow_replan_;
                        std::vector<tableint> before_ids;
                        std::vector<tableint> before_selected_ids;
                        std::vector<uint64_t> before_candidate_versions;
                        std::vector<uint64_t> before_selected_versions;
                        std::vector<std::pair<dist_t, tableint>> before_candidate_pairs =
                            snapshot_candidate_pairs(top_candidates);
                        std::vector<std::pair<dist_t, tableint>> before_selected_pairs =
                            snapshot_selected_pairs(top_candidates);
                        before_ids = snapshot_candidate_ids(top_candidates);
                        before_selected_ids = snapshot_selected_ids(top_candidates);
                        before_candidate_versions = snapshot_node_versions(before_ids);
                        before_selected_versions = snapshot_node_versions(before_selected_ids);
                        BoundaryWitnessInfo before_boundary_witnesses =
                            snapshot_boundary_witnesses(before_candidate_pairs, before_selected_pairs);

                        bool low_selection_slack = false;
                        bool have_selected_boundary = !before_selected_pairs.empty();
                        dist_t worst_selected_dist = dist_t();
                        if (have_selected_boundary) {
                            worst_selected_dist = before_selected_pairs.back().first;
                            std::unordered_set<tableint> selected_id_set;
                            selected_id_set.reserve(before_selected_ids.size());
                            for (tableint id : before_selected_ids) {
                                selected_id_set.insert(id);
                            }
                            for (const auto& pair : before_candidate_pairs) {
                                if (selected_id_set.find(pair.second) != selected_id_set.end()) {
                                    continue;
                                }
                                if (pair.first <= worst_selected_dist) {
                                    low_selection_slack = true;
                                }
                                break;
                            }
                        }
                        if (low_selection_slack) {
                            mid_insert_pause_low_slack_hits_.fetch_add(1, std::memory_order_relaxed);
                        }
                        const long start_commits = ready_before_hook;
                        const long target_commits = start_commits + static_cast<long>(mid_insert_preempt_k_);
                        const auto t0 = std::chrono::steady_clock::now();
                        const auto quiesce_grace = std::chrono::microseconds(50000);
                        size_t last_started = cur_element_count.load(std::memory_order_acquire);
                        auto impossible_since = std::chrono::steady_clock::time_point{};
                        while (ready_commit_counter_.load(std::memory_order_acquire) < target_commits) {
                            const auto now = std::chrono::steady_clock::now();
                            if (mid_insert_preempt_max_wait_us_ > 0) {
                                const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    now - t0).count();
                                if (elapsed_us >= mid_insert_preempt_max_wait_us_) {
                                    mid_insert_preempt_timeouts_.fetch_add(1, std::memory_order_relaxed);
                                    break;
                                }
                            } else {
                                const size_t started_now = cur_element_count.load(std::memory_order_acquire);
                                const long max_ready_without_self =
                                    started_now > 0 ? static_cast<long>(started_now) - 1 : 0;
                                if (max_ready_without_self < target_commits) {
                                    if (started_now != last_started) {
                                        last_started = started_now;
                                        impossible_since = std::chrono::steady_clock::time_point{};
                                    } else if (impossible_since == std::chrono::steady_clock::time_point{}) {
                                        impossible_since = now;
                                    } else if (now - impossible_since >= quiesce_grace) {
                                        mid_insert_preempt_quiesce_breaks_.fetch_add(1, std::memory_order_relaxed);
                                        break;
                                    }
                                } else {
                                    last_started = started_now;
                                    impossible_since = std::chrono::steady_clock::time_point{};
                                }
                            }
                            std::this_thread::yield();
                        }
                        const long now_commits = ready_commit_counter_.load(std::memory_order_acquire);
                        const long waited_commits = std::max<long>(0, now_commits - start_commits);
                        mid_insert_preempt_wait_commits_sum_.fetch_add(waited_commits, std::memory_order_relaxed);
                        mid_insert_pause_foreign_commit_nodes_.fetch_add(waited_commits, std::memory_order_relaxed);

                        const long selected_frontier_touched =
                            count_version_touches(before_selected_ids, before_selected_versions);
                        if (selected_frontier_touched > 0) {
                            mid_insert_pause_selected_frontier_touched_hits_.fetch_add(1, std::memory_order_relaxed);
                            mid_insert_pause_selected_frontier_touched_nodes_.fetch_add(
                                selected_frontier_touched, std::memory_order_relaxed);
                        }
                        const long candidate_frontier_touched =
                            count_version_touches(before_ids, before_candidate_versions);
                        if (candidate_frontier_touched > 0) {
                            mid_insert_pause_candidate_frontier_touched_hits_.fetch_add(1, std::memory_order_relaxed);
                            mid_insert_pause_candidate_frontier_touched_nodes_.fetch_add(
                                candidate_frontier_touched, std::memory_order_relaxed);
                        }

                        long foreign_outrank_count = 0;
                        std::vector<std::pair<dist_t, tableint>> foreign_outrank_pairs;
                        if (have_selected_boundary && ready_commit_ids_) {
                            for (long seq = start_commits; seq < now_commits; ++seq) {
                                tableint foreign_id =
                                    ready_commit_ids_[seq].load(std::memory_order_acquire);
                                if (foreign_id == static_cast<tableint>(-1) || foreign_id == cur_c ||
                                    foreign_id >= max_elements_) {
                                    continue;
                                }
                                if (!insertion_complete_[foreign_id].load(std::memory_order_acquire)) {
                                    continue;
                                }
                                dist_t foreign_dist =
                                    fstdistfunc_(data_point, getDataByInternalId(foreign_id),
                                                 dist_func_param_);
                                if (foreign_dist < worst_selected_dist) {
                                    ++foreign_outrank_count;
                                    foreign_outrank_pairs.emplace_back(foreign_dist, foreign_id);
                                }
                            }
                        }
                        if (foreign_outrank_count > 0) {
                            mid_insert_pause_foreign_outrank_hits_.fetch_add(1, std::memory_order_relaxed);
                            mid_insert_pause_foreign_outrank_nodes_.fetch_add(
                                foreign_outrank_count, std::memory_order_relaxed);
                        }

                        if (!before_boundary_witnesses.blocked_candidate_ids.empty()) {
                            mid_insert_pause_boundary_blocked_hits_.fetch_add(
                                1, std::memory_order_relaxed);
                            mid_insert_pause_boundary_blocked_candidates_.fetch_add(
                                static_cast<long>(before_boundary_witnesses.blocked_candidate_ids.size()),
                                std::memory_order_relaxed);
                        }

                        long boundary_blocker_touched = 0;
                        if (!before_boundary_witnesses.blocker_ids.empty()) {
                            std::unordered_set<tableint> seen_blockers;
                            seen_blockers.reserve(before_boundary_witnesses.blocker_ids.size());
                            for (size_t i = 0; i < before_boundary_witnesses.blocker_ids.size(); ++i) {
                                tableint blocker = before_boundary_witnesses.blocker_ids[i];
                                if (!seen_blockers.insert(blocker).second) {
                                    continue;
                                }
                                if (getNodeLinkVersion(blocker) !=
                                    before_boundary_witnesses.blocker_versions[i]) {
                                    ++boundary_blocker_touched;
                                }
                            }
                        }
                        if (boundary_blocker_touched > 0) {
                            mid_insert_pause_boundary_blocker_touched_hits_.fetch_add(
                                1, std::memory_order_relaxed);
                            mid_insert_pause_boundary_blocker_touched_nodes_.fetch_add(
                                boundary_blocker_touched, std::memory_order_relaxed);
                        }

                        long boundary_witness_invalidated = 0;
                        for (size_t i = 0; i < before_boundary_witnesses.blocker_ids.size(); ++i) {
                            if (getNodeLinkVersion(before_boundary_witnesses.blocker_ids[i]) !=
                                before_boundary_witnesses.blocker_versions[i]) {
                                ++boundary_witness_invalidated;
                            }
                        }
                        if (boundary_witness_invalidated > 0) {
                            mid_insert_pause_boundary_witness_invalidated_hits_.fetch_add(
                                1, std::memory_order_relaxed);
                            mid_insert_pause_boundary_witness_invalidated_candidates_.fetch_add(
                                boundary_witness_invalidated, std::memory_order_relaxed);
                        }

                        const bool boundary_harm_risky =
                            foreign_outrank_count > 0 || boundary_witness_invalidated > 0;
                        if (boundary_harm_risky) {
                            mid_insert_pause_boundary_harm_risk_hits_.fetch_add(
                                1, std::memory_order_relaxed);
                        }

                        const bool harm_risky =
                            foreign_outrank_count > 0 ||
                            (low_selection_slack && selected_frontier_touched > 0);
                        if (harm_risky) {
                            mid_insert_pause_harm_risk_hits_.fetch_add(1, std::memory_order_relaxed);
                        }
                        const MidInsertRepairRiskState repair_risk =
                            classifyMidInsertRepairRisk(
                                harm_risky, boundary_harm_risky, waited_commits,
                                foreign_outrank_count, selected_frontier_touched);
                        recordMidInsertRepairRiskStats(repair_risk);

                        const long teacher_watermark =
                            global_watermark_.load(std::memory_order_acquire);
                        const size_t teacher_view_limit =
                            teacher_watermark >= 0
                                ? static_cast<size_t>(teacher_watermark) + 1
                                : 0;
                        const bool use_snapshot_shadow_teacher =
                            do_shadow_replan && mid_insert_shadow_snapshot_teacher_ &&
                            enable_mvcc_ && level == 0 &&
                            teacher_watermark >= 0;

                        const MidInsertRepairAction repair_action =
                            chooseMidInsertRepairAction(repair_risk);
                        enqueue_deferred_repair =
                            recordMidInsertRepairAction(repair_action, repair_risk);

                        const bool did_inline_refresh =
                            repair_action == MidInsertRepairAction::Micro ||
                            repair_action == MidInsertRepairAction::Full;
                        const bool need_refresh =
                            do_shadow_replan ||
                            did_inline_refresh;
                        if (need_refresh) {
                            if (do_shadow_replan) {
                                mid_insert_shadow_replans_.fetch_add(1, std::memory_order_relaxed);
                            }

                            std::priority_queue<std::pair<dist_t, tableint>,
                                                std::vector<std::pair<dist_t, tableint>>,
                                                CompareByFirst>
                                shadow_top_candidates;
                            if (do_shadow_replan) {
                                if (use_snapshot_shadow_teacher) {
                                    mid_insert_shadow_snapshot_teacher_hits_.fetch_add(
                                        1, std::memory_order_relaxed);
                                    if (static_cast<size_t>(currObj) >= teacher_view_limit) {
                                        mid_insert_shadow_snapshot_teacher_future_entrypoint_hits_.fetch_add(
                                            1, std::memory_order_relaxed);
                                    }
                                    shadow_top_candidates =
                                        make_snapshot_top_candidates(teacher_view_limit);
                                    const long postwait_commit_drift = std::max<long>(
                                        0,
                                        ready_commit_counter_.load(std::memory_order_acquire) -
                                            now_commits);
                                    if (postwait_commit_drift > 0) {
                                        mid_insert_shadow_snapshot_teacher_postwait_commit_drift_hits_.fetch_add(
                                            1, std::memory_order_relaxed);
                                        mid_insert_shadow_snapshot_teacher_postwait_commit_drift_sum_.fetch_add(
                                            postwait_commit_drift, std::memory_order_relaxed);
                                    }
                                    const long postwait_watermark_drift = std::max<long>(
                                        0,
                                        global_watermark_.load(std::memory_order_acquire) -
                                            teacher_watermark);
                                    if (postwait_watermark_drift > 0) {
                                        mid_insert_shadow_snapshot_teacher_postwait_watermark_drift_hits_.fetch_add(
                                            1, std::memory_order_relaxed);
                                        mid_insert_shadow_snapshot_teacher_postwait_watermark_drift_sum_.fetch_add(
                                            postwait_watermark_drift, std::memory_order_relaxed);
                                    }
                                } else {
                                    shadow_top_candidates = make_live_top_candidates();
                                }
                            }

                            std::priority_queue<std::pair<dist_t, tableint>,
                                                std::vector<std::pair<dist_t, tableint>>,
                                                CompareByFirst>
                                apply_top_candidates;
                            bool have_apply_top_candidates = false;
                            if (repair_action == MidInsertRepairAction::Full) {
                                apply_top_candidates = make_live_top_candidates();
                                have_apply_top_candidates = true;
                            } else if (repair_action == MidInsertRepairAction::Micro) {
                                std::vector<std::pair<dist_t, tableint>> fresh_probe_pairs;
                                if (mid_insert_harm_micro_probe_ef_ > 0) {
                                    auto micro_probe_top_candidates =
                                        make_micro_probe_top_candidates(
                                            static_cast<size_t>(
                                                mid_insert_harm_micro_probe_ef_));
                                    fresh_probe_pairs = snapshot_candidate_pairs(
                                        micro_probe_top_candidates);
                                    mid_insert_pause_micro_probe_hits_.fetch_add(
                                        1, std::memory_order_relaxed);
                                    mid_insert_pause_micro_probe_candidates_total_.fetch_add(
                                        static_cast<long>(fresh_probe_pairs.size()),
                                        std::memory_order_relaxed);
                                    long prev_probe_max =
                                        mid_insert_pause_micro_probe_candidates_max_.load(
                                            std::memory_order_relaxed);
                                    while (static_cast<long>(fresh_probe_pairs.size()) >
                                               prev_probe_max &&
                                           !mid_insert_pause_micro_probe_candidates_max_
                                                .compare_exchange_weak(
                                                    prev_probe_max,
                                                    static_cast<long>(
                                                        fresh_probe_pairs.size()),
                                                    std::memory_order_relaxed)) {
                                    }
                                }
                                size_t micro_pool_size = 0;
                                apply_top_candidates = make_micro_replan_top_candidates(
                                    before_candidate_pairs, before_selected_pairs,
                                    foreign_outrank_pairs, fresh_probe_pairs,
                                    &micro_pool_size);
                                have_apply_top_candidates = true;
                                mid_insert_pause_micro_replans_.fetch_add(
                                    1, std::memory_order_relaxed);
                                mid_insert_pause_micro_replan_pool_total_.fetch_add(
                                    static_cast<long>(micro_pool_size),
                                    std::memory_order_relaxed);
                                long prev_pool_max = mid_insert_pause_micro_replan_pool_max_.load(
                                    std::memory_order_relaxed);
                                while (static_cast<long>(micro_pool_size) > prev_pool_max &&
                                       !mid_insert_pause_micro_replan_pool_max_.compare_exchange_weak(
                                           prev_pool_max,
                                           static_cast<long>(micro_pool_size),
                                           std::memory_order_relaxed)) {
                                }
                            }

                            if (do_shadow_replan) {
                                auto after_ids = snapshot_candidate_ids(shadow_top_candidates);
                                auto after_selected_ids = snapshot_selected_ids(shadow_top_candidates);
                                record_overlap_stats(
                                    before_ids, after_ids,
                                    mid_insert_candidate_changed_,
                                    mid_insert_candidate_before_total_,
                                    mid_insert_candidate_after_total_,
                                    mid_insert_candidate_intersection_total_,
                                    mid_insert_candidate_union_total_);
                                record_overlap_stats(
                                    before_selected_ids, after_selected_ids,
                                    mid_insert_selected_changed_,
                                    mid_insert_selected_before_total_,
                                    mid_insert_selected_after_total_,
                                    mid_insert_selected_intersection_total_,
                                    mid_insert_selected_union_total_);
                                const bool selected_changed_now =
                                    before_selected_ids != after_selected_ids;
                                if (harm_risky && selected_changed_now) {
                                    mid_insert_pause_harm_cert_true_positive_.fetch_add(
                                        1, std::memory_order_relaxed);
                                } else if (harm_risky && !selected_changed_now) {
                                    mid_insert_pause_harm_cert_false_positive_.fetch_add(
                                        1, std::memory_order_relaxed);
                                } else if (!harm_risky && selected_changed_now) {
                                    mid_insert_pause_harm_cert_false_negative_.fetch_add(
                                        1, std::memory_order_relaxed);
                                }
                                if (boundary_harm_risky && selected_changed_now) {
                                    mid_insert_pause_boundary_harm_cert_true_positive_.fetch_add(
                                        1, std::memory_order_relaxed);
                                } else if (boundary_harm_risky && !selected_changed_now) {
                                    mid_insert_pause_boundary_harm_cert_false_positive_.fetch_add(
                                        1, std::memory_order_relaxed);
                                } else if (!boundary_harm_risky && selected_changed_now) {
                                    mid_insert_pause_boundary_harm_cert_false_negative_.fetch_add(
                                        1, std::memory_order_relaxed);
                                }
                            }
                            if (did_inline_refresh) {
                                mid_insert_preempt_revalidations_.fetch_add(1, std::memory_order_relaxed);
                                if (!mid_insert_preempt_revalidate_ &&
                                    mid_insert_preempt_harm_guard_ && harm_risky) {
                                    mid_insert_pause_harm_guard_replans_.fetch_add(
                                        1, std::memory_order_relaxed);
                                }
                                if (have_apply_top_candidates) {
                                    top_candidates = std::move(apply_top_candidates);
                                }
                            }
                        }
                        allow_deferred_drain = !repair_risk.service_busy;
                        mid_insert_preempt_inflight_.fetch_sub(1, std::memory_order_acq_rel);
                    }
                }

                currObj = mutuallyConnectNewElement(
                    data_point, cur_c, top_candidates, level, false);
            }
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
            if (enqueue_deferred_repair) {
                tryEnqueueMidInsertDeferredRepair(cur_c);
            }
            if (allow_deferred_drain) {
                drainMidInsertDeferredRepairs(
                    static_cast<size_t>(mid_insert_harm_defer_drain_budget_));
            }
        } else if (enqueue_deferred_repair) {
            markMidInsertDeferredPending(cur_c, true);
        }
        return cur_c;
    }

    tableint addPointBatch(const void *batch_data,
                           const labeltype *batch_labels, size_t num_points) {
        if (num_points == 0) return (tableint)getWatermark();

        const char *base_ptr = static_cast<const char *>(batch_data);
        size_t stride = data_size_;

        size_t threads = std::min(num_threads_, num_points);
        std::vector<tableint> ids(num_points);

        if (threads <= 1) {
            for (size_t i = 0; i < num_points; ++i) {
                const void *point =
                    static_cast<const void *>(base_ptr + i * stride);
                ids[i] = addPoint(point, batch_labels[i], -1, true);
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
                        ids[i] = addPoint(point, batch_labels[i], -1, true);
                    }
                });
        }

        for (auto id : ids) {
            markReady(id);
            if (consumeMidInsertDeferredPending(id)) {
                tryEnqueueMidInsertDeferredRepair(id);
            }
        }
        advanceWatermark();
        if (mid_insert_harm_online_policy_ && mid_insert_harm_defer_enabled_) {
            drainMidInsertDeferredRepairs(
                static_cast<size_t>(mid_insert_harm_defer_drain_budget_));
        }
        return (tableint)getWatermark();
    }

    std::vector<std::priority_queue<std::pair<dist_t, labeltype>>>
    searchKnnBatch(const void *query_data, size_t num_queries, size_t k,
                   BaseFilterFunctor *isIdAllowed = nullptr) const {
        std::vector<std::priority_queue<std::pair<dist_t, labeltype>>> results(
            num_queries);
        if (num_queries == 0) return results;

        const char *base_ptr = static_cast<const char *>(query_data);
        size_t stride = data_size_;

        size_t threads = std::min(num_threads_, num_queries);
        if (threads <= 1) {
            for (size_t i = 0; i < num_queries; ++i) {
                const void *query =
                    static_cast<const void *>(base_ptr + i * stride);
                results[i] = searchKnn(query, k, isIdAllowed);
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
                        results[i] = searchKnn(query, k, isIdAllowed);
                    }
                });
        }
        return results;
    }

    std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
        const void *query_data, size_t k,
        BaseFilterFunctor *isIdAllowed = nullptr,
        size_t view_limit = std::numeric_limits<size_t>::max()) const override {
        return searchKnnWithStats(query_data, k, isIdAllowed, view_limit,
                                  nullptr);
    }

    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnWithStats(
        const void *query_data, size_t k,
        BaseFilterFunctor *isIdAllowed = nullptr,
        size_t view_limit = std::numeric_limits<size_t>::max(),
        SearchWorkStats *work_stats = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        const auto entry_start = std::chrono::steady_clock::now();
        dist_t curdist =
            fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_),
                         dist_func_param_);
        if (work_stats) {
            work_stats->entry_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - entry_start)
                    .count());
            work_stats->distance_computations++;
            work_stats->upper_distance_computations++;
        }

        const auto upper_start = std::chrono::steady_clock::now();
        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                std::shared_lock<std::shared_mutex> lock;
                if (use_node_lock_in_search_) {
                    const auto lock_wait_start = std::chrono::steady_clock::now();
                    lock = std::shared_lock<std::shared_mutex>(
                        link_list_locks_[currObj]);
                    if (work_stats) {
                        work_stats->upper_lock_wait_ns += static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - lock_wait_start)
                                .count());
                    }
                }
                unsigned int *data;

                data = (unsigned int *)get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;
                if (work_stats) {
                    work_stats->upper_hops++;
                    work_stats->upper_edges_scanned +=
                        static_cast<uint64_t>(size);
                }

                tableint *datal = (tableint *)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_ ||
                        !insertion_complete_[cand].load(std::memory_order_acquire))
                        continue;  // skip dirty/incomplete candidate (concurrent insert)
                    if (view_limit != std::numeric_limits<size_t>::max() &&
                        search_visibility_mode_ == 2) {
                        const bool visible_upper =
                            search_visibility_by_label_
                                ? static_cast<size_t>(getExternalLabel(cand)) <
                                      view_limit
                                : static_cast<size_t>(cand) < view_limit;
                        if (!visible_upper) {
                        if (work_stats) work_stats->future_skip_hops++;
                        continue;
                        }
                    }
                    dist_t d =
                        fstdistfunc_(query_data, getDataByInternalId(cand),
                                     dist_func_param_);
                    if (work_stats) {
                        work_stats->distance_computations++;
                        work_stats->upper_distance_computations++;
                    }

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
        if (work_stats) {
            work_stats->upper_search_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - upper_start)
                    .count());
        }

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        size_t base_view_limit = view_limit;
        if (base_view_limit != std::numeric_limits<size_t>::max() &&
            base_view_limit >=
                cur_element_count.load(std::memory_order_acquire)) {
            // The requested snapshot covers every currently allocated node.
            // Passing max preserves the same visible set while avoiding the
            // finite-watermark checks in the level-0 search loop.
            base_view_limit = std::numeric_limits<size_t>::max();
        }
        const auto base_start = std::chrono::steady_clock::now();
        if (bare_bone_search) {
            top_candidates =
                searchBaseLayerST<true>(currObj, query_data, std::max(ef_, k),
                                        isIdAllowed, nullptr, base_view_limit,
                                        work_stats);
        } else {
            top_candidates =
                searchBaseLayerST<false>(currObj, query_data, std::max(ef_, k),
                                         isIdAllowed, nullptr, base_view_limit,
                                         work_stats);
        }
        if (work_stats) {
            work_stats->base_search_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - base_start)
                    .count());
        }

        const auto materialize_start = std::chrono::steady_clock::now();
        const bool post_filter_refill =
            view_limit != std::numeric_limits<size_t>::max() &&
            search_visibility_mode_ == 1;
        auto is_materialize_visible = [&](tableint internal_id) -> bool {
            if (!post_filter_refill) return true;
            if (search_visibility_by_label_) {
                return static_cast<size_t>(getExternalLabel(internal_id)) <
                       view_limit;
            }
            return static_cast<size_t>(internal_id) < view_limit;
        };
        while (top_candidates.size() > k) {
            if (post_filter_refill) break;
            top_candidates.pop();
        }
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            visible_candidates;
        while (!top_candidates.empty()) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            top_candidates.pop();
            if (is_materialize_visible(rez.second)) {
                visible_candidates.emplace(rez.first, rez.second);
            }
        }
        while (visible_candidates.size() > k) {
            visible_candidates.pop();
        }
        while (visible_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = visible_candidates.top();
            result.push(std::pair<dist_t, labeltype>(
                rez.first, getExternalLabel(rez.second)));
            visible_candidates.pop();
        }
        if (work_stats) {
            work_stats->result_materialize_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - materialize_start)
                    .count());
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
                    if (cand < 0 || cand > max_elements_ ||
                        !insertion_complete_[cand].load(std::memory_order_acquire))
                        continue;  // skip dirty/incomplete candidate (concurrent insert)
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

#endif  // ANNCHOR_MVCC_HNSWALG_H
