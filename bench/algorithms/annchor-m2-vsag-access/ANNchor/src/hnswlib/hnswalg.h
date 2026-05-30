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

#ifndef ANNCHOR_VSAG_PREFETCH_STRIDE
#define ANNCHOR_VSAG_PREFETCH_STRIDE 1
#endif

#ifndef ANNCHOR_VSAG_PREFETCH_DEPTH_LINES
#define ANNCHOR_VSAG_PREFETCH_DEPTH_LINES 1
#endif

#ifndef ANNCHOR_VSAG_PREFETCH_MAX_CODE_LINES
#define ANNCHOR_VSAG_PREFETCH_MAX_CODE_LINES 8
#endif

namespace annchor {

enum class TailIssueRole : uint8_t {
    None = 0,
    Reader = 1,
    Writer = 2,
};

struct TailIssueThreadState {
    TailIssueRole role{TailIssueRole::None};
    uint64_t op_start_ns{0};
    bool urgent{false};
    uint32_t reader_budget_left{0};
    uint32_t issue_checks{0};
    uint32_t writer_tail_countdown{0};
    bool writer_tail_cached{false};
    uint32_t reader_deferred_lookahead_left{0};
    uint32_t reader_deferred_lookahead_id{0};
    bool reader_deferred_lookahead_valid{false};
    bool writer_repair_deferred_pending{false};
    bool writer_repair_drain_active{false};
    bool reader_clamped{false};
    bool reader_force_full_ef{false};
    bool reader_confidence_resolved{false};
    size_t reader_k{0};
};

static thread_local TailIssueThreadState tail_issue_tls;

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
    bool phase_existing_update_hit = false;
    bool phase_link_critical_hit = false;
    bool phase_load_scan_hit = false;
    bool phase_append_hit = false;
    bool phase_prune_hit = false;
    bool phase_undo_record_hit = false;
    bool phase_rewrite_hit = false;
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
    bool tail_issue_enabled_{false};
    uint64_t tail_issue_reader_guard_ns_{0};
    uint64_t tail_issue_reader_guard2_ns_{0};
    uint64_t tail_issue_lease_ns_{0};
    uint32_t tail_issue_reader_budget_lines_{0};
    uint32_t tail_issue_writer_tail_lines_{1};
    uint32_t tail_issue_check_period_{256};
    bool tail_issue_reader_urgent_prefetch_enabled_{true};
    uint32_t tail_issue_reader_limit_mode_{0};
    uint32_t tail_issue_reader_expansion_budget_{0};
    uint32_t tail_issue_reader_stop_alpha_permille_{1000};
    uint32_t tail_issue_reader_batch_size_{0};
    uint32_t tail_issue_reader_confidence_margin_permille_{0};
    uint32_t tail_issue_reader_lookahead_mode_{0};
    uint32_t tail_issue_reader_lookahead_budget_{0};
    uint32_t tail_issue_reader_stag_clamp_hops_{0};
    uint32_t tail_issue_reader_stag_clamp_min_expansions_{0};
    uint32_t tail_issue_reader_frontier_ef_{0};
    uint32_t tail_issue_reader_frontier_stag_hops_{0};
    uint32_t tail_issue_reader_frontier_min_expansions_{0};
    uint32_t tail_issue_reader_frontier_continue_budget_{0};
    bool tail_issue_reader_retry_enabled_{false};
    uint32_t tail_issue_reader_retry_margin_permille_{1100};
    bool tail_issue_reader_continue_enabled_{false};
    uint32_t tail_issue_reader_continue_margin_permille_{1050};
    bool tail_issue_reader_bounded_distance_enabled_{false};
    bool tail_issue_reader_prefix_guard_enabled_{false};
    uint32_t tail_issue_reader_prefix_guard_lines_{1};
    bool tail_issue_reader_prefix_estimate_enabled_{false};
    uint32_t tail_issue_reader_prefix_estimate_lines_{1};
    uint32_t tail_issue_reader_prefix_estimate_margin_permille_{1200};
    uint32_t tail_issue_reader_current_lines_{0};
    uint32_t tail_issue_reader_refine_seeds_{0};
    uint32_t tail_issue_reader_refine_edge_limit_{0};
    uint32_t tail_issue_writer_ef_budget_{0};
    uint32_t tail_issue_writer_ef_budget_cap_{0};
    uint32_t tail_issue_writer_ef_limit_{0};
    uint32_t tail_issue_writer_repair_defer_budget_{0};
    uint32_t tail_issue_writer_repair_defer_cap_{0};
    uint32_t tail_issue_writer_repair_drain_budget_{0};
    uint32_t tail_issue_writer_repair_queue_cap_{0};
    bool tail_issue_writer_repair_no_drain_{false};
    uint32_t tail_issue_writer_lowvalue_skip_budget_{0};
    uint32_t tail_issue_writer_lowvalue_skip_cap_{0};
    uint32_t tail_issue_writer_lowvalue_margin_permille_{1000};
    uint32_t tail_issue_writer_level0_budget_{0};
    uint32_t tail_issue_writer_level0_cap_{0};
    uint32_t tail_issue_writer_budget_lines_{0};
    uint32_t tail_issue_writer_action_{0};
    bool tail_issue_reader_policy_enabled_{false};
    bool tail_issue_writer_policy_enabled_{false};
    bool tail_issue_prefetch_policy_enabled_{false};
    uint32_t tail_issue_reader_window_extra_{0};
    uint32_t tail_issue_reader_ef_limit_{0};
    uint32_t tail_issue_reader_ef_limit2_{0};
    uint32_t tail_issue_writer_dist_mode_{0};
    uint32_t tail_issue_writer_dist_budget_{0};
    uint32_t tail_issue_writer_dist_budget_cap_{0};
    uint32_t tail_issue_writer_dist_prefix_guard_lines_{1};
    uint32_t tail_issue_writer_dist_prefix_estimate_lines_{1};
    uint32_t tail_issue_writer_dist_prefix_estimate_margin_permille_{1200};
    uint32_t tail_issue_writer_lead_skip_budget_{0};
    uint32_t tail_issue_writer_lead_skip_cap_{0};
    uint32_t tail_issue_writer_lead_hint_budget_{0};
    uint32_t tail_issue_writer_lead_hint_cap_{0};
    uint32_t tail_issue_writer_lead_hint_action_{0};
    uint32_t tail_issue_writer_frontier_budget_{0};
    uint32_t tail_issue_writer_frontier_cap_{0};
    uint32_t tail_issue_writer_frontier_margin_permille_{990};
    uint32_t tail_issue_writer_shared_scan_budget_{0};
    uint32_t tail_issue_writer_shared_scan_cap_{0};
    bool tail_issue_writer_shared_scan_always_{false};
    uint32_t tail_issue_writer_demote_budget_{0};
    uint32_t tail_issue_writer_demote_cap_{0};
    uint32_t tail_issue_writer_demote_line_count_{1};
    uint32_t tail_issue_writer_offlock_prune_budget_{0};
    uint32_t tail_issue_writer_offlock_prune_cap_{0};
    mutable std::atomic<uint64_t> tail_issue_until_ns_{0};
    mutable std::atomic<uint32_t> tail_issue_writer_budget_global_{0};
    mutable std::atomic<uint32_t> tail_issue_writer_dist_budget_global_{0};
    mutable std::atomic<uint32_t> tail_issue_writer_lead_skip_global_{0};
    mutable std::atomic<uint32_t> tail_issue_writer_lead_hint_global_{0};
    mutable std::atomic<uint32_t> tail_issue_writer_frontier_global_{0};
    mutable std::atomic<uint32_t> tail_issue_writer_shared_scan_global_{0};
    mutable std::atomic<uint32_t> tail_issue_writer_demote_global_{0};
    mutable std::atomic<uint32_t> tail_issue_writer_offlock_prune_global_{0};
    mutable std::atomic<uint32_t> tail_issue_writer_ef_budget_global_{0};
    mutable std::atomic<uint32_t> tail_issue_writer_repair_defer_global_{0};
    mutable std::atomic<uint32_t> tail_issue_writer_lowvalue_skip_global_{0};
    mutable std::atomic<uint32_t> tail_issue_writer_level0_global_{0};
    mutable std::atomic<uint64_t> tail_issue_last_dist_grant_ns_{0};
    mutable std::atomic<uint64_t> tail_issue_last_lead_skip_grant_ns_{0};
    mutable std::atomic<uint64_t> tail_issue_last_lead_hint_grant_ns_{0};
    mutable std::atomic<uint64_t> tail_issue_last_frontier_grant_ns_{0};
    mutable std::atomic<uint64_t> tail_issue_last_shared_scan_grant_ns_{0};
    mutable std::atomic<uint64_t> tail_issue_last_demote_grant_ns_{0};
    mutable std::atomic<uint64_t> tail_issue_last_offlock_prune_grant_ns_{0};
    mutable std::atomic<uint64_t> tail_issue_last_ef_grant_ns_{0};
    mutable std::atomic<uint64_t> tail_issue_last_repair_defer_grant_ns_{0};
    mutable std::atomic<uint64_t> tail_issue_last_lowvalue_skip_grant_ns_{0};
    mutable std::atomic<uint64_t> tail_issue_last_level0_grant_ns_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_urgent_ops_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_urgent_prefetch_lines_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_window_prefetches_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_beam_clamps_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_beam_trimmed_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_beam_clamps2_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_beam_trimmed2_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_expansion_stops_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_margin_stops_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_confidence_checks_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_confidence_grants_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_confidence_rejects_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_batch_prefetches_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_lookahead_deferred_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_lookahead_skipped_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_frontier_checks_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_frontier_eligible_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_frontier_stops_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_frontier_continue_checks_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_frontier_continue_grants_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_retry_checks_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_retry_runs_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_retry_confident_skips_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_continue_checks_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_continue_runs_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_continue_confident_stops_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_bounded_distance_ops_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_bounded_distance_early_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_prefix_guard_ops_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_prefix_guard_early_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_prefix_guard_full_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_prefix_estimate_ops_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_prefix_estimate_skips_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_prefix_estimate_full_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_current_prefetch_lines_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_refine_runs_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_refine_edges_scanned_{0};
    mutable std::atomic<uint64_t> tail_issue_reader_refine_inserted_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_ef_budget_granted_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_ef_limited_searches_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_ef_trimmed_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_repair_defer_granted_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_repair_deferred_points_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_repair_deferred_edges_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_repair_defer_dropped_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_repair_drained_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_repair_drain_runs_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_repair_queue_max_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_lowvalue_skip_granted_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_lowvalue_skip_checks_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_lowvalue_skipped_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_level0_granted_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_level0_forced_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_budget_granted_lines_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_dist_budget_granted_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_dist_shaped_ops_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_dist_bounded_early_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_dist_prefix_full_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_dist_prefix_estimate_skips_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_dist_prefix_estimate_full_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_dist_admission_skips_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_lead_skip_granted_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_lead_prefetch_skipped_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_lead_hint_granted_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_lead_prefetch_hinted_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_frontier_granted_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_frontier_checks_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_frontier_skipped_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_shared_scan_granted_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_shared_scan_used_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_demote_granted_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_demote_used_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_demote_lines_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_offlock_prune_granted_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_offlock_prune_attempts_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_offlock_prune_success_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_offlock_prune_validate_fail_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_offlock_prune_skipped_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_tail_hits_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_hint_lines_{0};
    mutable std::atomic<uint64_t> tail_issue_writer_suppressed_lines_{0};

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
    mutable std::mutex tail_issue_deferred_repair_queue_lock_;
    mutable std::mutex tail_issue_deferred_repair_drain_lock_;
    mutable std::deque<tableint> tail_issue_deferred_repair_queue_;
    mutable std::unordered_set<tableint> tail_issue_deferred_repair_set_;

    void setM2Mode(int mode) { m2_mode_ = mode; }
    int getM2Mode() const { return m2_mode_; }
    void setM2DegreeThreshold(float t) { m2_degree_threshold_ = t; }
    void setM2MaxRecover(int m) { m2_max_recover_ = m; }
    void setM2StagnationHops(int h) { m2_stagnation_hops_ = h; }
    uint64_t getTailIssueReaderUrgentOps() const {
        return tail_issue_reader_urgent_ops_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderUrgentPrefetchLines() const {
        return tail_issue_reader_urgent_prefetch_lines_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderWindowPrefetches() const {
        return tail_issue_reader_window_prefetches_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderBeamClamps() const {
        return tail_issue_reader_beam_clamps_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderBeamTrimmed() const {
        return tail_issue_reader_beam_trimmed_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderBeamClamps2() const {
        return tail_issue_reader_beam_clamps2_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderBeamTrimmed2() const {
        return tail_issue_reader_beam_trimmed2_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderExpansionStops() const {
        return tail_issue_reader_expansion_stops_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderMarginStops() const {
        return tail_issue_reader_margin_stops_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderConfidenceChecks() const {
        return tail_issue_reader_confidence_checks_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderConfidenceGrants() const {
        return tail_issue_reader_confidence_grants_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderConfidenceRejects() const {
        return tail_issue_reader_confidence_rejects_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderBatchPrefetches() const {
        return tail_issue_reader_batch_prefetches_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderLookaheadDeferred() const {
        return tail_issue_reader_lookahead_deferred_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderLookaheadSkipped() const {
        return tail_issue_reader_lookahead_skipped_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderFrontierChecks() const {
        return tail_issue_reader_frontier_checks_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderFrontierEligible() const {
        return tail_issue_reader_frontier_eligible_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderFrontierStops() const {
        return tail_issue_reader_frontier_stops_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderFrontierContinueChecks() const {
        return tail_issue_reader_frontier_continue_checks_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderFrontierContinueGrants() const {
        return tail_issue_reader_frontier_continue_grants_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderRetryChecks() const {
        return tail_issue_reader_retry_checks_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderRetryRuns() const {
        return tail_issue_reader_retry_runs_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderRetryConfidentSkips() const {
        return tail_issue_reader_retry_confident_skips_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderContinueChecks() const {
        return tail_issue_reader_continue_checks_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderContinueRuns() const {
        return tail_issue_reader_continue_runs_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderContinueConfidentStops() const {
        return tail_issue_reader_continue_confident_stops_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderBoundedDistanceOps() const {
        return tail_issue_reader_bounded_distance_ops_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderBoundedDistanceEarly() const {
        return tail_issue_reader_bounded_distance_early_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderPrefixGuardOps() const {
        return tail_issue_reader_prefix_guard_ops_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderPrefixGuardEarly() const {
        return tail_issue_reader_prefix_guard_early_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderPrefixGuardFull() const {
        return tail_issue_reader_prefix_guard_full_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderPrefixEstimateOps() const {
        return tail_issue_reader_prefix_estimate_ops_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderPrefixEstimateSkips() const {
        return tail_issue_reader_prefix_estimate_skips_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderPrefixEstimateFull() const {
        return tail_issue_reader_prefix_estimate_full_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderCurrentPrefetchLines() const {
        return tail_issue_reader_current_prefetch_lines_.load(
            std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderRefineRuns() const {
        return tail_issue_reader_refine_runs_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderRefineEdges() const {
        return tail_issue_reader_refine_edges_scanned_.load(
            std::memory_order_relaxed);
    }
    uint64_t getTailIssueReaderRefineInserted() const {
        return tail_issue_reader_refine_inserted_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterEfBudgetGranted() const {
        return tail_issue_writer_ef_budget_granted_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterEfLimitedSearches() const {
        return tail_issue_writer_ef_limited_searches_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterEfTrimmed() const {
        return tail_issue_writer_ef_trimmed_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterRepairDeferGranted() const {
        return tail_issue_writer_repair_defer_granted_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterRepairDeferredPoints() const {
        return tail_issue_writer_repair_deferred_points_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterRepairDeferredEdges() const {
        return tail_issue_writer_repair_deferred_edges_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterRepairDeferDropped() const {
        return tail_issue_writer_repair_defer_dropped_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterRepairDrained() const {
        return tail_issue_writer_repair_drained_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterRepairDrainRuns() const {
        return tail_issue_writer_repair_drain_runs_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterRepairQueueMax() const {
        return tail_issue_writer_repair_queue_max_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterLowvalueSkipGranted() const {
        return tail_issue_writer_lowvalue_skip_granted_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterLowvalueSkipChecks() const {
        return tail_issue_writer_lowvalue_skip_checks_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterLowvalueSkipped() const {
        return tail_issue_writer_lowvalue_skipped_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterLevel0Granted() const {
        return tail_issue_writer_level0_granted_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterLevel0Forced() const {
        return tail_issue_writer_level0_forced_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterBudgetGrantedLines() const {
        return tail_issue_writer_budget_granted_lines_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterDistBudgetGranted() const {
        return tail_issue_writer_dist_budget_granted_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterDistShapedOps() const {
        return tail_issue_writer_dist_shaped_ops_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterDistBoundedEarly() const {
        return tail_issue_writer_dist_bounded_early_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterDistPrefixFull() const {
        return tail_issue_writer_dist_prefix_full_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterDistPrefixEstimateSkips() const {
        return tail_issue_writer_dist_prefix_estimate_skips_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterDistPrefixEstimateFull() const {
        return tail_issue_writer_dist_prefix_estimate_full_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterDistAdmissionSkips() const {
        return tail_issue_writer_dist_admission_skips_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterLeadSkipGranted() const {
        return tail_issue_writer_lead_skip_granted_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterLeadPrefetchSkipped() const {
        return tail_issue_writer_lead_prefetch_skipped_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterLeadHintGranted() const {
        return tail_issue_writer_lead_hint_granted_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterLeadPrefetchHinted() const {
        return tail_issue_writer_lead_prefetch_hinted_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterFrontierGranted() const {
        return tail_issue_writer_frontier_granted_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterFrontierChecks() const {
        return tail_issue_writer_frontier_checks_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterFrontierSkipped() const {
        return tail_issue_writer_frontier_skipped_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterSharedScanGranted() const {
        return tail_issue_writer_shared_scan_granted_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterSharedScanUsed() const {
        return tail_issue_writer_shared_scan_used_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterDemoteGranted() const {
        return tail_issue_writer_demote_granted_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterDemoteUsed() const {
        return tail_issue_writer_demote_used_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterDemoteLines() const {
        return tail_issue_writer_demote_lines_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterOfflockPruneGranted() const {
        return tail_issue_writer_offlock_prune_granted_.load(
            std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterOfflockPruneAttempts() const {
        return tail_issue_writer_offlock_prune_attempts_.load(
            std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterOfflockPruneSuccess() const {
        return tail_issue_writer_offlock_prune_success_.load(
            std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterOfflockPruneValidateFail() const {
        return tail_issue_writer_offlock_prune_validate_fail_.load(
            std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterOfflockPruneSkipped() const {
        return tail_issue_writer_offlock_prune_skipped_.load(
            std::memory_order_relaxed);
    }
    // reverse-link repair op breakdown: append (room) vs prune+rewrite (full)
    uint64_t getExistingNeighborAppends() const {
        return metric_graph_existing_neighbor_appends.load(std::memory_order_relaxed);
    }
    uint64_t getExistingNeighborPrunes() const {
        return metric_graph_existing_neighbor_prunes.load(std::memory_order_relaxed);
    }
    uint64_t getExistingNeighborEdgesWritten() const {
        return metric_graph_existing_neighbor_edges_written.load(std::memory_order_relaxed);
    }
    uint64_t getExistingNeighborEdgesPruned() const {
        return metric_graph_existing_neighbor_edges_pruned.load(std::memory_order_relaxed);
    }
    uint64_t getSearchPhaseOverlapSamples() const {
        return metric_search_phase_overlap_samples.load(std::memory_order_relaxed);
    }
    uint64_t getSearchPhaseExistingUpdateSamples() const {
        return metric_search_phase_existing_update_samples.load(std::memory_order_relaxed);
    }
    uint64_t getSearchPhaseExistingUpdateQueries() const {
        return metric_search_phase_existing_update_queries.load(std::memory_order_relaxed);
    }
    uint64_t getSearchPhaseLinkCriticalSamples() const {
        return metric_search_phase_link_critical_samples.load(std::memory_order_relaxed);
    }
    uint64_t getSearchPhaseLinkCriticalQueries() const {
        return metric_search_phase_link_critical_queries.load(std::memory_order_relaxed);
    }
    uint64_t getSearchPhaseLoadScanSamples() const {
        return metric_search_phase_load_scan_samples.load(std::memory_order_relaxed);
    }
    uint64_t getSearchPhaseLoadScanQueries() const {
        return metric_search_phase_load_scan_queries.load(std::memory_order_relaxed);
    }
    uint64_t getSearchPhaseAppendSamples() const {
        return metric_search_phase_append_samples.load(std::memory_order_relaxed);
    }
    uint64_t getSearchPhaseAppendQueries() const {
        return metric_search_phase_append_queries.load(std::memory_order_relaxed);
    }
    uint64_t getSearchPhasePruneSamples() const {
        return metric_search_phase_prune_samples.load(std::memory_order_relaxed);
    }
    uint64_t getSearchPhasePruneQueries() const {
        return metric_search_phase_prune_queries.load(std::memory_order_relaxed);
    }
    uint64_t getSearchPhaseUndoRecordSamples() const {
        return metric_search_phase_undo_record_samples.load(std::memory_order_relaxed);
    }
    uint64_t getSearchPhaseUndoRecordQueries() const {
        return metric_search_phase_undo_record_queries.load(std::memory_order_relaxed);
    }
    uint64_t getSearchPhaseRewriteSamples() const {
        return metric_search_phase_rewrite_samples.load(std::memory_order_relaxed);
    }
    uint64_t getSearchPhaseRewriteQueries() const {
        return metric_search_phase_rewrite_queries.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterTailHits() const {
        return tail_issue_writer_tail_hits_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterHintLines() const {
        return tail_issue_writer_hint_lines_.load(std::memory_order_relaxed);
    }
    uint64_t getTailIssueWriterSuppressedLines() const {
        return tail_issue_writer_suppressed_lines_.load(std::memory_order_relaxed);
    }
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

    void initTailIssueFromEnv() {
        tail_issue_enabled_ = readEnvBool("ANNCHOR_TAIL_ISSUE_ENABLE", false);
        tail_issue_reader_guard_ns_ =
            static_cast<uint64_t>(std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_GUARD_NS", 50000)));
        tail_issue_reader_guard2_ns_ =
            static_cast<uint64_t>(std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_GUARD2_NS", 0)));
        tail_issue_lease_ns_ =
            static_cast<uint64_t>(std::max(1, readEnvInt("ANNCHOR_TAIL_ISSUE_LEASE_NS", 50000)));
        tail_issue_reader_budget_lines_ =
            static_cast<uint32_t>(std::max(1, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_BUDGET_LINES", 64)));
        tail_issue_writer_tail_lines_ =
            static_cast<uint32_t>(std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_LINES", 0)));
        tail_issue_check_period_ =
            static_cast<uint32_t>(std::max(1, readEnvInt("ANNCHOR_TAIL_ISSUE_CHECK_PERIOD", 256)));
        tail_issue_reader_urgent_prefetch_enabled_ =
            readEnvBool("ANNCHOR_TAIL_ISSUE_READER_PREFETCH", true);
        tail_issue_reader_limit_mode_ =
            static_cast<uint32_t>(std::min(1, std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_LIMIT_MODE", 0))));
        tail_issue_reader_expansion_budget_ =
            static_cast<uint32_t>(std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_EXPANSION_BUDGET", 0)));
        tail_issue_reader_stop_alpha_permille_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_STOP_ALPHA_PERMILLE", 1000)));
        tail_issue_reader_confidence_margin_permille_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_CONF_MARGIN_PERMILLE", 0)));
        tail_issue_reader_batch_size_ =
            static_cast<uint32_t>(std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_BATCH_SIZE", 0)));
        tail_issue_reader_lookahead_mode_ =
            static_cast<uint32_t>(std::min(
                2, std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_LOOKAHEAD_MODE", 0))));
        tail_issue_reader_lookahead_budget_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_LOOKAHEAD_BUDGET", 0)));
        tail_issue_reader_stag_clamp_hops_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_STAG_CLAMP_HOPS", 0)));
        tail_issue_reader_stag_clamp_min_expansions_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_STAG_CLAMP_MIN_EXPANSIONS", 0)));
        tail_issue_reader_frontier_ef_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_FRONTIER_EF", 0)));
        tail_issue_reader_frontier_stag_hops_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_FRONTIER_STAG_HOPS", 0)));
        tail_issue_reader_frontier_min_expansions_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_FRONTIER_MIN_EXPANSIONS", 0)));
        tail_issue_reader_frontier_continue_budget_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_FRONTIER_CONTINUE_BUDGET", 0)));
        tail_issue_reader_retry_enabled_ =
            readEnvBool("ANNCHOR_TAIL_ISSUE_READER_RETRY_ENABLE", false);
        tail_issue_reader_retry_margin_permille_ =
            static_cast<uint32_t>(std::max(
                1000, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_RETRY_MARGIN_PERMILLE", 1100)));
        tail_issue_reader_continue_enabled_ =
            readEnvBool("ANNCHOR_TAIL_ISSUE_READER_CONTINUE_ENABLE", false);
        tail_issue_reader_continue_margin_permille_ =
            static_cast<uint32_t>(std::max(
                1000, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_CONTINUE_MARGIN_PERMILLE", 1050)));
        tail_issue_reader_bounded_distance_enabled_ =
            readEnvBool("ANNCHOR_TAIL_ISSUE_READER_BOUNDED_DISTANCE", false);
        tail_issue_reader_prefix_guard_enabled_ =
            readEnvBool("ANNCHOR_TAIL_ISSUE_READER_PREFIX_GUARD", false);
        tail_issue_reader_prefix_guard_lines_ =
            static_cast<uint32_t>(std::min(
                8, std::max(1, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_PREFIX_GUARD_LINES", 1))));
        tail_issue_reader_prefix_estimate_enabled_ =
            readEnvBool("ANNCHOR_TAIL_ISSUE_READER_PREFIX_ESTIMATE", false);
        tail_issue_reader_prefix_estimate_lines_ =
            static_cast<uint32_t>(std::min(
                8, std::max(1, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_PREFIX_ESTIMATE_LINES", 1))));
        tail_issue_reader_prefix_estimate_margin_permille_ =
            static_cast<uint32_t>(std::max(
                1000, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_PREFIX_ESTIMATE_MARGIN_PERMILLE", 1200)));
        tail_issue_reader_current_lines_ =
            static_cast<uint32_t>(std::min(
                8, std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_CURRENT_LINES", 0))));
        tail_issue_reader_refine_seeds_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_REFINE_SEEDS", 0)));
        tail_issue_reader_refine_edge_limit_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_REFINE_EDGES", 0)));
        tail_issue_writer_ef_budget_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_EF_BUDGET", 0)));
        tail_issue_writer_ef_budget_cap_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_EF_BUDGET_CAP",
                              static_cast<int>(tail_issue_writer_ef_budget_))));
        tail_issue_writer_ef_limit_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_EF_LIMIT", 0)));
        tail_issue_writer_repair_defer_budget_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_REPAIR_DEFER_BUDGET", 0)));
        tail_issue_writer_repair_defer_cap_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_REPAIR_DEFER_CAP",
                              static_cast<int>(tail_issue_writer_repair_defer_budget_))));
        tail_issue_writer_repair_drain_budget_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_REPAIR_DRAIN_BUDGET", 0)));
        tail_issue_writer_repair_no_drain_ =
            readEnvBool("ANNCHOR_TAIL_ISSUE_WRITER_REPAIR_NO_DRAIN", false);
        tail_issue_writer_repair_queue_cap_ =
            static_cast<uint32_t>(std::max(
                1, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_REPAIR_QUEUE_CAP", 4096)));
        tail_issue_writer_lowvalue_skip_budget_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_LOWVALUE_SKIP_BUDGET", 0)));
        tail_issue_writer_lowvalue_skip_cap_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_LOWVALUE_SKIP_CAP",
                              static_cast<int>(tail_issue_writer_lowvalue_skip_budget_))));
        tail_issue_writer_lowvalue_margin_permille_ =
            static_cast<uint32_t>(std::max(
                1000, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_LOWVALUE_MARGIN_PERMILLE", 1100)));
        tail_issue_writer_level0_budget_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_LEVEL0_BUDGET", 0)));
        tail_issue_writer_level0_cap_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_LEVEL0_CAP",
                              static_cast<int>(tail_issue_writer_level0_budget_))));
        tail_issue_writer_budget_lines_ =
            static_cast<uint32_t>(std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_BUDGET_LINES", 0)));
        tail_issue_writer_action_ =
            static_cast<uint32_t>(std::min(3, std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_ACTION", 0))));
        tail_issue_reader_window_extra_ =
            static_cast<uint32_t>(std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_WINDOW_EXTRA", 0)));
        tail_issue_reader_ef_limit_ =
            static_cast<uint32_t>(std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_EF_LIMIT", 0)));
        tail_issue_reader_ef_limit2_ =
            static_cast<uint32_t>(std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_READER_EF_LIMIT2", 0)));
        tail_issue_writer_dist_mode_ =
            static_cast<uint32_t>(std::min(7, std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_DIST_MODE", 0))));
        tail_issue_writer_dist_budget_ =
            static_cast<uint32_t>(std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_DIST_BUDGET", 0)));
        tail_issue_writer_dist_budget_cap_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_DIST_BUDGET_CAP",
                              static_cast<int>(tail_issue_writer_dist_budget_))));
        tail_issue_writer_dist_prefix_guard_lines_ =
            static_cast<uint32_t>(std::min(
                8, std::max(1, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_DIST_PREFIX_GUARD_LINES", 1))));
        tail_issue_writer_dist_prefix_estimate_lines_ =
            static_cast<uint32_t>(std::min(
                8, std::max(1, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_DIST_PREFIX_ESTIMATE_LINES", 1))));
        tail_issue_writer_dist_prefix_estimate_margin_permille_ =
            static_cast<uint32_t>(std::max(
                1000, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_DIST_PREFIX_ESTIMATE_MARGIN_PERMILLE", 1200)));
        tail_issue_writer_lead_skip_budget_ =
            static_cast<uint32_t>(std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_LEAD_SKIP_BUDGET", 0)));
        tail_issue_writer_lead_skip_cap_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_LEAD_SKIP_CAP",
                              static_cast<int>(tail_issue_writer_lead_skip_budget_))));
        tail_issue_writer_lead_hint_budget_ =
            static_cast<uint32_t>(std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_LEAD_HINT_BUDGET", 0)));
        tail_issue_writer_lead_hint_cap_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_LEAD_HINT_CAP",
                              static_cast<int>(tail_issue_writer_lead_hint_budget_))));
        tail_issue_writer_lead_hint_action_ =
            static_cast<uint32_t>(std::min(
                3, std::max(0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_LEAD_HINT_ACTION", 0))));
        tail_issue_writer_frontier_budget_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_FRONTIER_BUDGET", 0)));
        tail_issue_writer_frontier_cap_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_FRONTIER_CAP",
                              static_cast<int>(tail_issue_writer_frontier_budget_))));
        tail_issue_writer_frontier_margin_permille_ =
            static_cast<uint32_t>(std::min(
                1000, std::max(
                    0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_FRONTIER_MARGIN_PERMILLE", 990))));
        tail_issue_writer_shared_scan_budget_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_SHARED_SCAN_BUDGET", 0)));
        tail_issue_writer_shared_scan_cap_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_SHARED_SCAN_CAP",
                              static_cast<int>(tail_issue_writer_shared_scan_budget_))));
        tail_issue_writer_shared_scan_always_ =
            readEnvBool("ANNCHOR_TAIL_ISSUE_WRITER_SHARED_SCAN_ALWAYS", false);
        tail_issue_writer_demote_budget_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_DEMOTE_BUDGET", 0)));
        tail_issue_writer_demote_cap_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_DEMOTE_CAP",
                              static_cast<int>(tail_issue_writer_demote_budget_))));
        tail_issue_writer_demote_line_count_ =
            static_cast<uint32_t>(std::min(
                8, std::max(
                    1, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_DEMOTE_LINES", 1))));
        tail_issue_writer_offlock_prune_budget_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_OFFLOCK_PRUNE_BUDGET", 0)));
        tail_issue_writer_offlock_prune_cap_ =
            static_cast<uint32_t>(std::max(
                0, readEnvInt("ANNCHOR_TAIL_ISSUE_WRITER_OFFLOCK_PRUNE_CAP",
                              static_cast<int>(tail_issue_writer_offlock_prune_budget_))));
        tail_issue_reader_policy_enabled_ =
            tail_issue_enabled_ && tail_issue_reader_guard_ns_ > 0 &&
            ((tail_issue_reader_urgent_prefetch_enabled_ &&
              tail_issue_reader_budget_lines_ > 0) ||
             tail_issue_reader_window_extra_ > 0 ||
             tail_issue_reader_ef_limit_ > 0 ||
             tail_issue_reader_expansion_budget_ > 0 ||
             tail_issue_reader_stop_alpha_permille_ < 1000 ||
             tail_issue_reader_batch_size_ > 0 ||
             (tail_issue_reader_lookahead_mode_ > 0 &&
              tail_issue_reader_lookahead_budget_ > 0) ||
             (tail_issue_reader_stag_clamp_hops_ > 0 &&
              tail_issue_reader_ef_limit_ > 0) ||
             (tail_issue_reader_frontier_ef_ > 0 &&
              tail_issue_reader_frontier_stag_hops_ > 0) ||
             tail_issue_reader_frontier_continue_budget_ > 0 ||
             tail_issue_reader_retry_enabled_ ||
             tail_issue_reader_continue_enabled_ ||
             tail_issue_reader_bounded_distance_enabled_ ||
             tail_issue_reader_prefix_guard_enabled_ ||
             tail_issue_reader_prefix_estimate_enabled_ ||
             tail_issue_reader_current_lines_ > 0 ||
             (tail_issue_reader_refine_seeds_ > 0 &&
              tail_issue_reader_refine_edge_limit_ > 0) ||
             (tail_issue_writer_ef_budget_ > 0 &&
              tail_issue_writer_ef_limit_ > 0) ||
             (tail_issue_writer_repair_defer_budget_ > 0 &&
              (tail_issue_writer_repair_drain_budget_ > 0 ||
               tail_issue_writer_repair_no_drain_)) ||
             tail_issue_writer_lowvalue_skip_budget_ > 0 ||
             tail_issue_writer_level0_budget_ > 0 ||
             tail_issue_writer_budget_lines_ > 0 ||
             tail_issue_writer_dist_budget_ > 0 ||
             tail_issue_writer_lead_skip_budget_ > 0 ||
             (tail_issue_writer_lead_hint_budget_ > 0 &&
              tail_issue_writer_lead_hint_action_ > 0) ||
             tail_issue_writer_frontier_budget_ > 0 ||
             tail_issue_writer_shared_scan_budget_ > 0 ||
             tail_issue_writer_shared_scan_always_ ||
             tail_issue_writer_demote_budget_ > 0 ||
             tail_issue_writer_offlock_prune_budget_ > 0);
        tail_issue_writer_policy_enabled_ =
            tail_issue_enabled_ &&
            (tail_issue_writer_budget_lines_ > 0 || tail_issue_writer_dist_budget_ > 0 ||
             tail_issue_writer_lead_skip_budget_ > 0 ||
             (tail_issue_writer_lead_hint_budget_ > 0 &&
              tail_issue_writer_lead_hint_action_ > 0) ||
             tail_issue_writer_frontier_budget_ > 0 ||
             tail_issue_writer_shared_scan_budget_ > 0 ||
             tail_issue_writer_shared_scan_always_ ||
             tail_issue_writer_demote_budget_ > 0 ||
             tail_issue_writer_offlock_prune_budget_ > 0 ||
             (tail_issue_writer_ef_budget_ > 0 &&
              tail_issue_writer_ef_limit_ > 0) ||
             (tail_issue_writer_repair_defer_budget_ > 0 &&
              (tail_issue_writer_repair_drain_budget_ > 0 ||
               tail_issue_writer_repair_no_drain_)) ||
             tail_issue_writer_lowvalue_skip_budget_ > 0 ||
             tail_issue_writer_level0_budget_ > 0);
        tail_issue_prefetch_policy_enabled_ =
            tail_issue_reader_policy_enabled_ || tail_issue_writer_policy_enabled_;
    }

    struct TailIssueScope {
        TailIssueRole prev_role;
        uint64_t prev_start_ns;
        bool prev_urgent;
        uint32_t prev_reader_budget_left;
        uint32_t prev_issue_checks;
        uint32_t prev_reader_deferred_lookahead_left;
        uint32_t prev_reader_deferred_lookahead_id;
        bool prev_reader_deferred_lookahead_valid;
        bool prev_writer_repair_deferred_pending;
        bool prev_writer_repair_drain_active;
        bool prev_reader_clamped;
        bool prev_reader_force_full_ef;
        bool prev_reader_confidence_resolved;
        size_t prev_reader_k;

        TailIssueScope(TailIssueRole role, bool enabled)
            : prev_role(tail_issue_tls.role),
              prev_start_ns(tail_issue_tls.op_start_ns),
              prev_urgent(tail_issue_tls.urgent),
              prev_reader_budget_left(tail_issue_tls.reader_budget_left),
              prev_issue_checks(tail_issue_tls.issue_checks),
              prev_reader_deferred_lookahead_left(
                  tail_issue_tls.reader_deferred_lookahead_left),
              prev_reader_deferred_lookahead_id(
                  tail_issue_tls.reader_deferred_lookahead_id),
              prev_reader_deferred_lookahead_valid(
                  tail_issue_tls.reader_deferred_lookahead_valid),
              prev_writer_repair_deferred_pending(
                  tail_issue_tls.writer_repair_deferred_pending),
              prev_writer_repair_drain_active(
                  tail_issue_tls.writer_repair_drain_active),
              prev_reader_clamped(tail_issue_tls.reader_clamped),
              prev_reader_force_full_ef(tail_issue_tls.reader_force_full_ef),
              prev_reader_confidence_resolved(
                  tail_issue_tls.reader_confidence_resolved),
              prev_reader_k(tail_issue_tls.reader_k) {
            if (enabled) {
                tail_issue_tls.role = role;
                tail_issue_tls.urgent = false;
                tail_issue_tls.reader_budget_left = 0;
                tail_issue_tls.issue_checks = 0;
                tail_issue_tls.reader_deferred_lookahead_left = 0;
                tail_issue_tls.reader_deferred_lookahead_id = 0;
                tail_issue_tls.reader_deferred_lookahead_valid = false;
                tail_issue_tls.writer_repair_deferred_pending = false;
                tail_issue_tls.reader_clamped = false;
                tail_issue_tls.reader_force_full_ef = false;
                tail_issue_tls.reader_confidence_resolved = false;
                tail_issue_tls.reader_k = 0;
                if (role == TailIssueRole::Reader) {
                    tail_issue_tls.op_start_ns = monotonicRawNs();
                }
            }
        }

        ~TailIssueScope() {
            tail_issue_tls.role = prev_role;
            tail_issue_tls.op_start_ns = prev_start_ns;
            tail_issue_tls.urgent = prev_urgent;
            tail_issue_tls.reader_budget_left = prev_reader_budget_left;
            tail_issue_tls.issue_checks = prev_issue_checks;
            tail_issue_tls.reader_deferred_lookahead_left =
                prev_reader_deferred_lookahead_left;
            tail_issue_tls.reader_deferred_lookahead_id =
                prev_reader_deferred_lookahead_id;
            tail_issue_tls.reader_deferred_lookahead_valid =
                prev_reader_deferred_lookahead_valid;
            tail_issue_tls.writer_repair_deferred_pending =
                prev_writer_repair_deferred_pending;
            tail_issue_tls.writer_repair_drain_active =
                prev_writer_repair_drain_active;
            tail_issue_tls.reader_clamped = prev_reader_clamped;
            tail_issue_tls.reader_force_full_ef = prev_reader_force_full_ef;
            tail_issue_tls.reader_confidence_resolved =
                prev_reader_confidence_resolved;
            tail_issue_tls.reader_k = prev_reader_k;
        }
    };

    inline bool tailIssueActive(uint64_t now_ns) const {
        return tail_issue_enabled_ &&
               now_ns < tail_issue_until_ns_.load(std::memory_order_acquire);
    }

    inline void grantTailIssueWriterDistBudget(uint64_t now_ns) const {
        if (tail_issue_writer_dist_budget_ == 0) {
            return;
        }
        if (tail_issue_writer_dist_budget_cap_ > 0) {
            uint64_t last = tail_issue_last_dist_grant_ns_.load(std::memory_order_acquire);
            while (now_ns > last && now_ns - last >= tail_issue_lease_ns_) {
                if (tail_issue_last_dist_grant_ns_.compare_exchange_weak(
                        last, now_ns, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    break;
                }
            }
            if (last != 0 && now_ns - last < tail_issue_lease_ns_) {
                return;
            }
            uint32_t old =
                tail_issue_writer_dist_budget_global_.load(std::memory_order_acquire);
            while (old < tail_issue_writer_dist_budget_cap_) {
                uint32_t grant = tail_issue_writer_dist_budget_;
                if (grant > tail_issue_writer_dist_budget_cap_ - old) {
                    grant = tail_issue_writer_dist_budget_cap_ - old;
                }
                if (grant == 0) {
                    return;
                }
                if (tail_issue_writer_dist_budget_global_.compare_exchange_weak(
                        old, old + grant, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    tail_issue_writer_dist_budget_granted_.fetch_add(
                        grant, std::memory_order_relaxed);
                    return;
                }
            }
            return;
        }
        tail_issue_writer_dist_budget_global_.fetch_add(
            tail_issue_writer_dist_budget_, std::memory_order_release);
        tail_issue_writer_dist_budget_granted_.fetch_add(
            tail_issue_writer_dist_budget_, std::memory_order_relaxed);
    }

    inline void grantTailIssueWriterLeadSkipBudget(uint64_t now_ns) const {
        if (tail_issue_writer_lead_skip_budget_ == 0) {
            return;
        }
        if (tail_issue_writer_lead_skip_cap_ > 0) {
            uint64_t last =
                tail_issue_last_lead_skip_grant_ns_.load(std::memory_order_acquire);
            while (now_ns > last && now_ns - last >= tail_issue_lease_ns_) {
                if (tail_issue_last_lead_skip_grant_ns_.compare_exchange_weak(
                        last, now_ns, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    break;
                }
            }
            if (last != 0 && now_ns - last < tail_issue_lease_ns_) {
                return;
            }
            uint32_t old =
                tail_issue_writer_lead_skip_global_.load(std::memory_order_acquire);
            while (old < tail_issue_writer_lead_skip_cap_) {
                uint32_t grant = tail_issue_writer_lead_skip_budget_;
                if (grant > tail_issue_writer_lead_skip_cap_ - old) {
                    grant = tail_issue_writer_lead_skip_cap_ - old;
                }
                if (grant == 0) {
                    return;
                }
                if (tail_issue_writer_lead_skip_global_.compare_exchange_weak(
                        old, old + grant, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    tail_issue_writer_lead_skip_granted_.fetch_add(
                        grant, std::memory_order_relaxed);
                    return;
                }
            }
            return;
        }
        tail_issue_writer_lead_skip_global_.fetch_add(
            tail_issue_writer_lead_skip_budget_, std::memory_order_release);
        tail_issue_writer_lead_skip_granted_.fetch_add(
            tail_issue_writer_lead_skip_budget_, std::memory_order_relaxed);
    }

    inline void grantTailIssueWriterLeadHintBudget(uint64_t now_ns) const {
        if (tail_issue_writer_lead_hint_budget_ == 0 ||
            tail_issue_writer_lead_hint_action_ == 0) {
            return;
        }
        if (tail_issue_writer_lead_hint_cap_ > 0) {
            uint64_t last =
                tail_issue_last_lead_hint_grant_ns_.load(std::memory_order_acquire);
            while (now_ns > last && now_ns - last >= tail_issue_lease_ns_) {
                if (tail_issue_last_lead_hint_grant_ns_.compare_exchange_weak(
                        last, now_ns, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    break;
                }
            }
            if (last != 0 && now_ns - last < tail_issue_lease_ns_) {
                return;
            }
            uint32_t old =
                tail_issue_writer_lead_hint_global_.load(std::memory_order_acquire);
            while (old < tail_issue_writer_lead_hint_cap_) {
                uint32_t grant = tail_issue_writer_lead_hint_budget_;
                if (grant > tail_issue_writer_lead_hint_cap_ - old) {
                    grant = tail_issue_writer_lead_hint_cap_ - old;
                }
                if (grant == 0) {
                    return;
                }
                if (tail_issue_writer_lead_hint_global_.compare_exchange_weak(
                        old, old + grant, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    tail_issue_writer_lead_hint_granted_.fetch_add(
                        grant, std::memory_order_relaxed);
                    return;
                }
            }
            return;
        }
        tail_issue_writer_lead_hint_global_.fetch_add(
            tail_issue_writer_lead_hint_budget_, std::memory_order_release);
        tail_issue_writer_lead_hint_granted_.fetch_add(
            tail_issue_writer_lead_hint_budget_, std::memory_order_relaxed);
    }

    inline void grantTailIssueWriterFrontierBudget(uint64_t now_ns) const {
        if (tail_issue_writer_frontier_budget_ == 0) {
            return;
        }
        if (tail_issue_writer_frontier_cap_ > 0) {
            uint64_t last =
                tail_issue_last_frontier_grant_ns_.load(std::memory_order_acquire);
            while (now_ns > last && now_ns - last >= tail_issue_lease_ns_) {
                if (tail_issue_last_frontier_grant_ns_.compare_exchange_weak(
                        last, now_ns, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    break;
                }
            }
            if (last != 0 && now_ns - last < tail_issue_lease_ns_) {
                return;
            }
            uint32_t old =
                tail_issue_writer_frontier_global_.load(std::memory_order_acquire);
            while (old < tail_issue_writer_frontier_cap_) {
                uint32_t grant = tail_issue_writer_frontier_budget_;
                if (grant > tail_issue_writer_frontier_cap_ - old) {
                    grant = tail_issue_writer_frontier_cap_ - old;
                }
                if (grant == 0) {
                    return;
                }
                if (tail_issue_writer_frontier_global_.compare_exchange_weak(
                        old, old + grant, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    tail_issue_writer_frontier_granted_.fetch_add(
                        grant, std::memory_order_relaxed);
                    return;
                }
            }
            return;
        }
        tail_issue_writer_frontier_global_.fetch_add(
            tail_issue_writer_frontier_budget_, std::memory_order_release);
        tail_issue_writer_frontier_granted_.fetch_add(
            tail_issue_writer_frontier_budget_, std::memory_order_relaxed);
    }

    inline void grantTailIssueWriterSharedScanBudget(uint64_t now_ns) const {
        if (tail_issue_writer_shared_scan_budget_ == 0) {
            return;
        }
        if (tail_issue_writer_shared_scan_cap_ > 0) {
            uint64_t last =
                tail_issue_last_shared_scan_grant_ns_.load(std::memory_order_acquire);
            while (now_ns > last && now_ns - last >= tail_issue_lease_ns_) {
                if (tail_issue_last_shared_scan_grant_ns_.compare_exchange_weak(
                        last, now_ns, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    break;
                }
            }
            if (last != 0 && now_ns - last < tail_issue_lease_ns_) {
                return;
            }
            uint32_t old =
                tail_issue_writer_shared_scan_global_.load(std::memory_order_acquire);
            while (old < tail_issue_writer_shared_scan_cap_) {
                uint32_t grant = tail_issue_writer_shared_scan_budget_;
                if (grant > tail_issue_writer_shared_scan_cap_ - old) {
                    grant = tail_issue_writer_shared_scan_cap_ - old;
                }
                if (grant == 0) {
                    return;
                }
                if (tail_issue_writer_shared_scan_global_.compare_exchange_weak(
                        old, old + grant, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    tail_issue_writer_shared_scan_granted_.fetch_add(
                        grant, std::memory_order_relaxed);
                    return;
                }
            }
            return;
        }
        tail_issue_writer_shared_scan_global_.fetch_add(
            tail_issue_writer_shared_scan_budget_, std::memory_order_release);
        tail_issue_writer_shared_scan_granted_.fetch_add(
            tail_issue_writer_shared_scan_budget_, std::memory_order_relaxed);
    }

    inline void grantTailIssueWriterDemoteBudget(uint64_t now_ns) const {
        if (tail_issue_writer_demote_budget_ == 0) {
            return;
        }
        if (tail_issue_writer_demote_cap_ > 0) {
            uint64_t last =
                tail_issue_last_demote_grant_ns_.load(std::memory_order_acquire);
            while (now_ns > last && now_ns - last >= tail_issue_lease_ns_) {
                if (tail_issue_last_demote_grant_ns_.compare_exchange_weak(
                        last, now_ns, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    break;
                }
            }
            if (last != 0 && now_ns - last < tail_issue_lease_ns_) {
                return;
            }
            uint32_t old =
                tail_issue_writer_demote_global_.load(std::memory_order_acquire);
            while (old < tail_issue_writer_demote_cap_) {
                uint32_t grant = tail_issue_writer_demote_budget_;
                if (grant > tail_issue_writer_demote_cap_ - old) {
                    grant = tail_issue_writer_demote_cap_ - old;
                }
                if (grant == 0) {
                    return;
                }
                if (tail_issue_writer_demote_global_.compare_exchange_weak(
                        old, old + grant, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    tail_issue_writer_demote_granted_.fetch_add(
                        grant, std::memory_order_relaxed);
                    return;
                }
            }
            return;
        }
        tail_issue_writer_demote_global_.fetch_add(
            tail_issue_writer_demote_budget_, std::memory_order_release);
        tail_issue_writer_demote_granted_.fetch_add(
            tail_issue_writer_demote_budget_, std::memory_order_relaxed);
    }

    inline void grantTailIssueWriterOfflockPruneBudget(uint64_t now_ns) const {
        if (tail_issue_writer_offlock_prune_budget_ == 0) {
            return;
        }
        if (tail_issue_writer_offlock_prune_cap_ > 0) {
            uint64_t last =
                tail_issue_last_offlock_prune_grant_ns_.load(
                    std::memory_order_acquire);
            while (now_ns > last && now_ns - last >= tail_issue_lease_ns_) {
                if (tail_issue_last_offlock_prune_grant_ns_.compare_exchange_weak(
                        last, now_ns, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    break;
                }
            }
            if (last != 0 && now_ns - last < tail_issue_lease_ns_) {
                return;
            }
            uint32_t old =
                tail_issue_writer_offlock_prune_global_.load(
                    std::memory_order_acquire);
            while (old < tail_issue_writer_offlock_prune_cap_) {
                uint32_t grant = tail_issue_writer_offlock_prune_budget_;
                if (grant > tail_issue_writer_offlock_prune_cap_ - old) {
                    grant = tail_issue_writer_offlock_prune_cap_ - old;
                }
                if (grant == 0) {
                    return;
                }
                if (tail_issue_writer_offlock_prune_global_.compare_exchange_weak(
                        old, old + grant, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    tail_issue_writer_offlock_prune_granted_.fetch_add(
                        grant, std::memory_order_relaxed);
                    return;
                }
            }
            return;
        }
        tail_issue_writer_offlock_prune_global_.fetch_add(
            tail_issue_writer_offlock_prune_budget_, std::memory_order_release);
        tail_issue_writer_offlock_prune_granted_.fetch_add(
            tail_issue_writer_offlock_prune_budget_, std::memory_order_relaxed);
    }

    inline void grantTailIssueWriterEfBudget(uint64_t now_ns) const {
        if (tail_issue_writer_ef_budget_ == 0 || tail_issue_writer_ef_limit_ == 0) {
            return;
        }
        if (tail_issue_writer_ef_budget_cap_ > 0) {
            uint64_t last =
                tail_issue_last_ef_grant_ns_.load(std::memory_order_acquire);
            while (now_ns > last && now_ns - last >= tail_issue_lease_ns_) {
                if (tail_issue_last_ef_grant_ns_.compare_exchange_weak(
                        last, now_ns, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    break;
                }
            }
            if (last != 0 && now_ns - last < tail_issue_lease_ns_) {
                return;
            }
            uint32_t old =
                tail_issue_writer_ef_budget_global_.load(std::memory_order_acquire);
            while (old < tail_issue_writer_ef_budget_cap_) {
                uint32_t grant = tail_issue_writer_ef_budget_;
                if (grant > tail_issue_writer_ef_budget_cap_ - old) {
                    grant = tail_issue_writer_ef_budget_cap_ - old;
                }
                if (grant == 0) {
                    return;
                }
                if (tail_issue_writer_ef_budget_global_.compare_exchange_weak(
                        old, old + grant, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    tail_issue_writer_ef_budget_granted_.fetch_add(
                        grant, std::memory_order_relaxed);
                    return;
                }
            }
            return;
        }
        tail_issue_writer_ef_budget_global_.fetch_add(
            tail_issue_writer_ef_budget_, std::memory_order_release);
        tail_issue_writer_ef_budget_granted_.fetch_add(
            tail_issue_writer_ef_budget_, std::memory_order_relaxed);
    }

    inline void grantTailIssueWriterRepairDeferBudget(uint64_t now_ns) const {
        if (tail_issue_writer_repair_defer_budget_ == 0 ||
            (tail_issue_writer_repair_drain_budget_ == 0 &&
             !tail_issue_writer_repair_no_drain_)) {
            return;
        }
        if (tail_issue_writer_repair_defer_cap_ > 0) {
            uint64_t last =
                tail_issue_last_repair_defer_grant_ns_.load(std::memory_order_acquire);
            while (now_ns > last && now_ns - last >= tail_issue_lease_ns_) {
                if (tail_issue_last_repair_defer_grant_ns_.compare_exchange_weak(
                        last, now_ns, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    break;
                }
            }
            if (last != 0 && now_ns - last < tail_issue_lease_ns_) {
                return;
            }
            uint32_t old =
                tail_issue_writer_repair_defer_global_.load(std::memory_order_acquire);
            while (old < tail_issue_writer_repair_defer_cap_) {
                uint32_t grant = tail_issue_writer_repair_defer_budget_;
                if (grant > tail_issue_writer_repair_defer_cap_ - old) {
                    grant = tail_issue_writer_repair_defer_cap_ - old;
                }
                if (grant == 0) {
                    return;
                }
                if (tail_issue_writer_repair_defer_global_.compare_exchange_weak(
                        old, old + grant, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    tail_issue_writer_repair_defer_granted_.fetch_add(
                        grant, std::memory_order_relaxed);
                    return;
                }
            }
            return;
        }
        tail_issue_writer_repair_defer_global_.fetch_add(
            tail_issue_writer_repair_defer_budget_, std::memory_order_release);
        tail_issue_writer_repair_defer_granted_.fetch_add(
            tail_issue_writer_repair_defer_budget_, std::memory_order_relaxed);
    }

    inline void grantTailIssueWriterLowvalueSkipBudget(uint64_t now_ns) const {
        if (tail_issue_writer_lowvalue_skip_budget_ == 0) {
            return;
        }
        if (tail_issue_writer_lowvalue_skip_cap_ > 0) {
            uint64_t last =
                tail_issue_last_lowvalue_skip_grant_ns_.load(std::memory_order_acquire);
            while (now_ns > last && now_ns - last >= tail_issue_lease_ns_) {
                if (tail_issue_last_lowvalue_skip_grant_ns_.compare_exchange_weak(
                        last, now_ns, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    break;
                }
            }
            if (last != 0 && now_ns - last < tail_issue_lease_ns_) {
                return;
            }
            uint32_t old =
                tail_issue_writer_lowvalue_skip_global_.load(std::memory_order_acquire);
            while (old < tail_issue_writer_lowvalue_skip_cap_) {
                uint32_t grant = tail_issue_writer_lowvalue_skip_budget_;
                if (grant > tail_issue_writer_lowvalue_skip_cap_ - old) {
                    grant = tail_issue_writer_lowvalue_skip_cap_ - old;
                }
                if (grant == 0) {
                    return;
                }
                if (tail_issue_writer_lowvalue_skip_global_.compare_exchange_weak(
                        old, old + grant, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    tail_issue_writer_lowvalue_skip_granted_.fetch_add(
                        grant, std::memory_order_relaxed);
                    return;
                }
            }
            return;
        }
        tail_issue_writer_lowvalue_skip_global_.fetch_add(
            tail_issue_writer_lowvalue_skip_budget_, std::memory_order_release);
        tail_issue_writer_lowvalue_skip_granted_.fetch_add(
            tail_issue_writer_lowvalue_skip_budget_, std::memory_order_relaxed);
    }

    inline void grantTailIssueWriterLevel0Budget(uint64_t now_ns) const {
        if (tail_issue_writer_level0_budget_ == 0) {
            return;
        }
        if (tail_issue_writer_level0_cap_ > 0) {
            uint64_t last =
                tail_issue_last_level0_grant_ns_.load(std::memory_order_acquire);
            while (now_ns > last && now_ns - last >= tail_issue_lease_ns_) {
                if (tail_issue_last_level0_grant_ns_.compare_exchange_weak(
                        last, now_ns, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    break;
                }
            }
            if (last != 0 && now_ns - last < tail_issue_lease_ns_) {
                return;
            }
            uint32_t old =
                tail_issue_writer_level0_global_.load(std::memory_order_acquire);
            while (old < tail_issue_writer_level0_cap_) {
                uint32_t grant = tail_issue_writer_level0_budget_;
                if (grant > tail_issue_writer_level0_cap_ - old) {
                    grant = tail_issue_writer_level0_cap_ - old;
                }
                if (grant == 0) {
                    return;
                }
                if (tail_issue_writer_level0_global_.compare_exchange_weak(
                        old, old + grant, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    tail_issue_writer_level0_granted_.fetch_add(
                        grant, std::memory_order_relaxed);
                    return;
                }
            }
            return;
        }
        tail_issue_writer_level0_global_.fetch_add(
            tail_issue_writer_level0_budget_, std::memory_order_release);
        tail_issue_writer_level0_granted_.fetch_add(
            tail_issue_writer_level0_budget_, std::memory_order_relaxed);
    }

    inline bool maybeActivateReaderTailIssue(uint64_t now_ns) const {
        if (!tail_issue_reader_policy_enabled_ || tail_issue_tls.role != TailIssueRole::Reader ||
            tail_issue_tls.urgent || tail_issue_reader_guard_ns_ == 0) {
            return tail_issue_tls.urgent;
        }
        if (now_ns - tail_issue_tls.op_start_ns >= tail_issue_reader_guard_ns_) {
            tail_issue_tls.urgent = true;
            tail_issue_tls.reader_budget_left = tail_issue_reader_budget_lines_;
            tail_issue_tls.reader_deferred_lookahead_left =
                tail_issue_reader_lookahead_budget_;
            tail_issue_tls.reader_deferred_lookahead_valid = false;
            tail_issue_until_ns_.store(now_ns + tail_issue_lease_ns_, std::memory_order_release);
            if (tail_issue_writer_budget_lines_ > 0) {
                tail_issue_writer_budget_global_.fetch_add(
                    tail_issue_writer_budget_lines_, std::memory_order_release);
                tail_issue_writer_budget_granted_lines_.fetch_add(
                    tail_issue_writer_budget_lines_, std::memory_order_relaxed);
            }
            grantTailIssueWriterDistBudget(now_ns);
            grantTailIssueWriterLeadSkipBudget(now_ns);
            grantTailIssueWriterLeadHintBudget(now_ns);
            grantTailIssueWriterFrontierBudget(now_ns);
            grantTailIssueWriterSharedScanBudget(now_ns);
            grantTailIssueWriterDemoteBudget(now_ns);
            grantTailIssueWriterOfflockPruneBudget(now_ns);
            grantTailIssueWriterEfBudget(now_ns);
            grantTailIssueWriterRepairDeferBudget(now_ns);
            grantTailIssueWriterLowvalueSkipBudget(now_ns);
            grantTailIssueWriterLevel0Budget(now_ns);
            tail_issue_reader_urgent_ops_.fetch_add(1, std::memory_order_relaxed);
        }
        return tail_issue_tls.urgent;
    }

    inline void tailIssueReaderExpansionCheckpoint() const {
        if (!tail_issue_reader_policy_enabled_ || tail_issue_tls.role != TailIssueRole::Reader ||
            tail_issue_tls.urgent || tail_issue_reader_guard_ns_ == 0) {
            return;
        }
        ++tail_issue_tls.issue_checks;
        if (tail_issue_tls.issue_checks % tail_issue_check_period_ == 0) {
            const uint64_t now_ns = monotonicRawNs();
            if (tail_issue_tls.op_start_ns == 0) {
                tail_issue_tls.op_start_ns = now_ns;
                return;
            }
            maybeActivateReaderTailIssue(now_ns);
        }
    }

    inline bool tailIssueWriterTailCached() const {
        if (!tail_issue_writer_policy_enabled_ || tail_issue_tls.role != TailIssueRole::Writer ||
            tail_issue_writer_tail_lines_ >= 1 || tail_issue_writer_budget_lines_ == 0) {
            return false;
        }
        if (tail_issue_tls.writer_tail_countdown == 0) {
            const uint64_t now_ns = monotonicRawNs();
            tail_issue_tls.writer_tail_cached =
                tailIssueActive(now_ns) &&
                tail_issue_writer_budget_global_.load(std::memory_order_acquire) > 0;
            tail_issue_tls.writer_tail_countdown = tail_issue_check_period_;
        } else {
            --tail_issue_tls.writer_tail_countdown;
        }
        return tail_issue_tls.writer_tail_cached;
    }

    inline bool consumeTailIssueWriterBudget(size_t lines) const {
        if (lines == 0) {
            return false;
        }
        const uint32_t need = static_cast<uint32_t>(
            std::min<size_t>(lines, std::numeric_limits<uint32_t>::max()));
        uint32_t old = tail_issue_writer_budget_global_.load(std::memory_order_acquire);
        while (old >= need && need > 0) {
            if (tail_issue_writer_budget_global_.compare_exchange_weak(
                    old, old - need, std::memory_order_acq_rel, std::memory_order_acquire)) {
                return true;
            }
        }
        return false;
    }

    inline void __attribute__((always_inline))
    prefetchLineWithTailHint(const char* ptr, uint32_t action) const {
        if (action == 1) {
            _mm_prefetch(ptr, _MM_HINT_T1);
        } else if (action == 2) {
            _mm_prefetch(ptr, _MM_HINT_T2);
        } else if (action == 3) {
            _mm_prefetch(ptr, _MM_HINT_NTA);
        } else {
            _mm_prefetch(ptr, _MM_HINT_T0);
        }
    }

    inline bool consumeTailIssueWriterDistBudget() const {
        uint32_t old = tail_issue_writer_dist_budget_global_.load(std::memory_order_acquire);
        while (old > 0) {
            if (tail_issue_writer_dist_budget_global_.compare_exchange_weak(
                    old, old - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                return true;
            }
        }
        return false;
    }

    inline bool shouldSkipWriterLeadPrefetch() const {
        if (!tail_issue_writer_policy_enabled_ ||
            tail_issue_tls.role != TailIssueRole::Writer ||
            tail_issue_writer_lead_skip_budget_ == 0) {
            return false;
        }
        if (tail_issue_writer_lead_skip_global_.load(std::memory_order_acquire) == 0) {
            return false;
        }
        const uint64_t now_ns = monotonicRawNs();
        if (!tailIssueActive(now_ns)) {
            tail_issue_writer_lead_skip_global_.store(0, std::memory_order_release);
            return false;
        }
        uint32_t old = tail_issue_writer_lead_skip_global_.load(std::memory_order_acquire);
        while (old > 0) {
            if (tail_issue_writer_lead_skip_global_.compare_exchange_weak(
                    old, old - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                tail_issue_writer_lead_prefetch_skipped_.fetch_add(
                    1, std::memory_order_relaxed);
                return true;
            }
        }
        return false;
    }

    inline uint32_t writerLeadPrefetchHintAction() const {
        if (!tail_issue_writer_policy_enabled_ ||
            tail_issue_tls.role != TailIssueRole::Writer ||
            tail_issue_writer_lead_hint_budget_ == 0 ||
            tail_issue_writer_lead_hint_action_ == 0) {
            return 0;
        }
        if (tail_issue_writer_lead_hint_global_.load(std::memory_order_acquire) == 0) {
            return 0;
        }
        const uint64_t now_ns = monotonicRawNs();
        if (!tailIssueActive(now_ns)) {
            tail_issue_writer_lead_hint_global_.store(0, std::memory_order_release);
            return 0;
        }
        uint32_t old =
            tail_issue_writer_lead_hint_global_.load(std::memory_order_acquire);
        while (old > 0) {
            if (tail_issue_writer_lead_hint_global_.compare_exchange_weak(
                    old, old - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                tail_issue_writer_lead_prefetch_hinted_.fetch_add(
                    1, std::memory_order_relaxed);
                return tail_issue_writer_lead_hint_action_;
            }
        }
        return 0;
    }

    inline bool shouldSkipWriterFrontierExpansion(dist_t dist,
                                                  dist_t lower_bound,
                                                  bool exact_required) const {
        if (!tail_issue_writer_policy_enabled_ ||
            tail_issue_tls.role != TailIssueRole::Writer ||
            tail_issue_writer_frontier_budget_ == 0 ||
            tail_issue_writer_frontier_margin_permille_ >= 1000 ||
            exact_required ||
            lower_bound == std::numeric_limits<dist_t>::max()) {
            return false;
        }
        if (dist >= lower_bound) {
            return false;
        }
        if (tail_issue_writer_frontier_global_.load(std::memory_order_acquire) == 0) {
            return false;
        }
        const uint64_t now_ns = monotonicRawNs();
        if (!tailIssueActive(now_ns)) {
            tail_issue_writer_frontier_global_.store(0, std::memory_order_release);
            return false;
        }
        tail_issue_writer_frontier_checks_.fetch_add(1, std::memory_order_relaxed);
        const double lhs = static_cast<double>(dist) * 1000.0;
        const double rhs = static_cast<double>(lower_bound) *
                           static_cast<double>(tail_issue_writer_frontier_margin_permille_);
        if (lhs < rhs) {
            return false;
        }
        uint32_t old =
            tail_issue_writer_frontier_global_.load(std::memory_order_acquire);
        while (old > 0) {
            if (tail_issue_writer_frontier_global_.compare_exchange_weak(
                    old, old - 1, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                tail_issue_writer_frontier_skipped_.fetch_add(
                    1, std::memory_order_relaxed);
                return true;
            }
        }
        return false;
    }

    inline bool shouldUseWriterSharedScanLock() const {
        if (!tail_issue_writer_policy_enabled_ ||
            tail_issue_tls.role != TailIssueRole::Writer ||
            (tail_issue_writer_shared_scan_budget_ == 0 &&
             !tail_issue_writer_shared_scan_always_)) {
            return false;
        }
        if (tail_issue_writer_shared_scan_always_) {
            tail_issue_writer_shared_scan_used_.fetch_add(
                1, std::memory_order_relaxed);
            return true;
        }
        if (tail_issue_writer_shared_scan_global_.load(std::memory_order_acquire) == 0) {
            return false;
        }
        const uint64_t now_ns = monotonicRawNs();
        if (!tailIssueActive(now_ns)) {
            tail_issue_writer_shared_scan_global_.store(0, std::memory_order_release);
            return false;
        }
        uint32_t old =
            tail_issue_writer_shared_scan_global_.load(std::memory_order_acquire);
        while (old > 0) {
            if (tail_issue_writer_shared_scan_global_.compare_exchange_weak(
                    old, old - 1, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                tail_issue_writer_shared_scan_used_.fetch_add(
                    1, std::memory_order_relaxed);
                return true;
            }
        }
        return false;
    }

    inline bool consumeTailIssueWriterDemoteToken() const {
        if (!tail_issue_writer_policy_enabled_ ||
            tail_issue_tls.role != TailIssueRole::Writer ||
            tail_issue_writer_demote_budget_ == 0) {
            return false;
        }
        if (tail_issue_writer_demote_global_.load(std::memory_order_acquire) == 0) {
            return false;
        }
        const uint64_t now_ns = monotonicRawNs();
        if (!tailIssueActive(now_ns)) {
            tail_issue_writer_demote_global_.store(0, std::memory_order_release);
            return false;
        }
        uint32_t old =
            tail_issue_writer_demote_global_.load(std::memory_order_acquire);
        while (old > 0) {
            if (tail_issue_writer_demote_global_.compare_exchange_weak(
                    old, old - 1, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
        }
        return false;
    }

    inline void tailIssueWriterDemoteLines(void* ptr, size_t lines) const {
        if (ptr == nullptr || lines == 0 || !consumeTailIssueWriterDemoteToken()) {
            return;
        }
#if defined(__CLDEMOTE__)
        if (lines > tail_issue_writer_demote_line_count_) {
            lines = tail_issue_writer_demote_line_count_;
        }
        if (lines > 8) {
            lines = 8;
        }
        char* base = static_cast<char*>(ptr);
        if (lines > 0) _cldemote(base);
        if (lines > 1) _cldemote(base + 64);
        if (lines > 2) _cldemote(base + 128);
        if (lines > 3) _cldemote(base + 192);
        if (lines > 4) _cldemote(base + 256);
        if (lines > 5) _cldemote(base + 320);
        if (lines > 6) _cldemote(base + 384);
        if (lines > 7) _cldemote(base + 448);
        tail_issue_writer_demote_used_.fetch_add(1, std::memory_order_relaxed);
        tail_issue_writer_demote_lines_.fetch_add(lines, std::memory_order_relaxed);
#else
        (void)ptr;
        (void)lines;
#endif
    }

    inline bool consumeTailIssueWriterOfflockPruneToken() const {
        if (!tail_issue_writer_policy_enabled_ ||
            tail_issue_tls.role != TailIssueRole::Writer ||
            tail_issue_writer_offlock_prune_budget_ == 0) {
            return false;
        }
        if (tail_issue_writer_offlock_prune_global_.load(
                std::memory_order_acquire) == 0) {
            return false;
        }
        const uint64_t now_ns = monotonicRawNs();
        if (!tailIssueActive(now_ns)) {
            tail_issue_writer_offlock_prune_global_.store(
                0, std::memory_order_release);
            return false;
        }
        uint32_t old =
            tail_issue_writer_offlock_prune_global_.load(
                std::memory_order_acquire);
        while (old > 0) {
            if (tail_issue_writer_offlock_prune_global_.compare_exchange_weak(
                    old, old - 1, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
        }
        return false;
    }

    inline bool shouldShapeWriterDistance() const {
        if (!tail_issue_writer_policy_enabled_ ||
            tail_issue_tls.role != TailIssueRole::Writer ||
            tail_issue_writer_dist_mode_ == 0 || tail_issue_writer_dist_budget_ == 0) {
            return false;
        }
        if (tail_issue_writer_dist_budget_global_.load(std::memory_order_acquire) == 0) {
            return false;
        }
        const uint64_t now_ns = monotonicRawNs();
        if (!tailIssueActive(now_ns)) {
            tail_issue_writer_dist_budget_global_.store(0, std::memory_order_release);
            return false;
        }
        return consumeTailIssueWriterDistBudget();
    }

    inline size_t tailIssueWriterEffectiveEfConstruction() const {
        const size_t requested = ef_construction_;
        if (!tail_issue_writer_policy_enabled_ ||
            tail_issue_tls.role != TailIssueRole::Writer ||
            tail_issue_writer_ef_limit_ == 0 ||
            tail_issue_writer_ef_budget_ == 0 ||
            requested <= M_) {
            return requested;
        }
        if (tail_issue_writer_ef_budget_global_.load(std::memory_order_acquire) == 0) {
            return requested;
        }
        const uint64_t now_ns = monotonicRawNs();
        if (!tailIssueActive(now_ns)) {
            tail_issue_writer_ef_budget_global_.store(0, std::memory_order_release);
            return requested;
        }
        uint32_t old =
            tail_issue_writer_ef_budget_global_.load(std::memory_order_acquire);
        while (old > 0) {
            if (tail_issue_writer_ef_budget_global_.compare_exchange_weak(
                    old, old - 1, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                const size_t limited = std::max<size_t>(
                    M_, static_cast<size_t>(tail_issue_writer_ef_limit_));
                if (limited < requested) {
                    tail_issue_writer_ef_limited_searches_.fetch_add(
                        1, std::memory_order_relaxed);
                    tail_issue_writer_ef_trimmed_.fetch_add(
                        requested - limited, std::memory_order_relaxed);
                    return limited;
                }
                return requested;
            }
        }
        return requested;
    }

    inline bool consumeTailIssueWriterRepairDeferToken() const {
        if (!tail_issue_writer_policy_enabled_ ||
            tail_issue_tls.role != TailIssueRole::Writer ||
            tail_issue_tls.writer_repair_drain_active ||
            tail_issue_writer_repair_defer_budget_ == 0 ||
            (tail_issue_writer_repair_drain_budget_ == 0 &&
             !tail_issue_writer_repair_no_drain_)) {
            return false;
        }
        if (tail_issue_writer_repair_defer_global_.load(std::memory_order_acquire) == 0) {
            return false;
        }
        const uint64_t now_ns = monotonicRawNs();
        if (!tailIssueActive(now_ns)) {
            tail_issue_writer_repair_defer_global_.store(0, std::memory_order_release);
            return false;
        }
        uint32_t old =
            tail_issue_writer_repair_defer_global_.load(std::memory_order_acquire);
        while (old > 0) {
            if (tail_issue_writer_repair_defer_global_.compare_exchange_weak(
                    old, old - 1, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
        }
        return false;
    }

    inline bool consumeTailIssueWriterLowvalueSkipToken() const {
        if (!tail_issue_writer_policy_enabled_ ||
            tail_issue_tls.role != TailIssueRole::Writer ||
            tail_issue_writer_lowvalue_skip_budget_ == 0) {
            return false;
        }
        if (tail_issue_writer_lowvalue_skip_global_.load(std::memory_order_acquire) == 0) {
            return false;
        }
        const uint64_t now_ns = monotonicRawNs();
        if (!tailIssueActive(now_ns)) {
            tail_issue_writer_lowvalue_skip_global_.store(0, std::memory_order_release);
            return false;
        }
        uint32_t old =
            tail_issue_writer_lowvalue_skip_global_.load(std::memory_order_acquire);
        while (old > 0) {
            if (tail_issue_writer_lowvalue_skip_global_.compare_exchange_weak(
                    old, old - 1, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
        }
        return false;
    }

    inline bool shouldSkipTailIssueLowvalueReverseLink(
        dist_t candidate_dist, dist_t worst_old_dist) const {
        if (tail_issue_writer_lowvalue_skip_budget_ == 0 ||
            tail_issue_writer_lowvalue_margin_permille_ <= 1000 ||
            !(worst_old_dist > static_cast<dist_t>(0))) {
            return false;
        }
        const double lhs = static_cast<double>(candidate_dist) * 1000.0;
        const double rhs = static_cast<double>(worst_old_dist) *
                           static_cast<double>(tail_issue_writer_lowvalue_margin_permille_);
        if (lhs < rhs) {
            return false;
        }
        tail_issue_writer_lowvalue_skip_checks_.fetch_add(
            1, std::memory_order_relaxed);
        if (!consumeTailIssueWriterLowvalueSkipToken()) {
            return false;
        }
        tail_issue_writer_lowvalue_skipped_.fetch_add(
            1, std::memory_order_relaxed);
        return true;
    }

    inline bool consumeTailIssueWriterLevel0Token() const {
        if (!tail_issue_writer_policy_enabled_ ||
            tail_issue_tls.role != TailIssueRole::Writer ||
            tail_issue_writer_level0_budget_ == 0) {
            return false;
        }
        if (tail_issue_writer_level0_global_.load(std::memory_order_acquire) == 0) {
            return false;
        }
        const uint64_t now_ns = monotonicRawNs();
        if (!tailIssueActive(now_ns)) {
            tail_issue_writer_level0_global_.store(0, std::memory_order_release);
            return false;
        }
        uint32_t old =
            tail_issue_writer_level0_global_.load(std::memory_order_acquire);
        while (old > 0) {
            if (tail_issue_writer_level0_global_.compare_exchange_weak(
                    old, old - 1, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
        }
        return false;
    }

    inline dist_t writerDistanceScalar(const void* a, const void* b) const {
        const float* pa = static_cast<const float*>(a);
        const float* pb = static_cast<const float*>(b);
        const size_t qty = *static_cast<const size_t*>(dist_func_param_);
        float sum = 0.0f;
        for (size_t i = 0; i < qty; ++i) {
            const float diff = pa[i] - pb[i];
            sum += diff * diff;
        }
        return static_cast<dist_t>(sum);
    }

    inline dist_t writerDistanceSse4(const void* a, const void* b) const {
#ifdef USE_SSE
        const float* pa = static_cast<const float*>(a);
        const float* pb = static_cast<const float*>(b);
        const size_t qty = *static_cast<const size_t*>(dist_func_param_);
        const size_t qty4 = qty & ~static_cast<size_t>(3);
        __m128 sum = _mm_setzero_ps();
        for (size_t i = 0; i < qty4; i += 4) {
            __m128 va = _mm_loadu_ps(pa + i);
            __m128 vb = _mm_loadu_ps(pb + i);
            __m128 diff = _mm_sub_ps(va, vb);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        float tmp[4];
        _mm_storeu_ps(tmp, sum);
        float res = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        for (size_t i = qty4; i < qty; ++i) {
            const float diff = pa[i] - pb[i];
            res += diff * diff;
        }
        return static_cast<dist_t>(res);
#else
        return writerDistanceScalar(a, b);
#endif
    }

    inline dist_t writerDistanceBounded(const void* a, const void* b,
                                        dist_t bound,
                                        bool* early_exit) const {
        const float* pa = static_cast<const float*>(a);
        const float* pb = static_cast<const float*>(b);
        const size_t qty = *static_cast<const size_t*>(dist_func_param_);
        float sum = 0.0f;
        const float fbound = static_cast<float>(bound);
        for (size_t i = 0; i < qty; ++i) {
            const float diff = pa[i] - pb[i];
            sum += diff * diff;
            if (sum > fbound) {
                *early_exit = true;
                return static_cast<dist_t>(sum);
            }
        }
        *early_exit = false;
        return static_cast<dist_t>(sum);
    }

    inline dist_t writerDistanceBoundedVector(const void* a, const void* b,
                                              dist_t bound,
                                              bool* early_exit) const {
#ifdef USE_AVX
        const float* pa = static_cast<const float*>(a);
        const float* pb = static_cast<const float*>(b);
        const size_t qty = *static_cast<const size_t*>(dist_func_param_);
        const size_t qty16 = qty & ~static_cast<size_t>(15);
        const float fbound = static_cast<float>(bound);
        float sum_scalar = 0.0f;
        for (size_t i = 0; i < qty16; i += 16) {
            __m256 va0 = _mm256_loadu_ps(pa + i);
            __m256 vb0 = _mm256_loadu_ps(pb + i);
            __m256 diff0 = _mm256_sub_ps(va0, vb0);
            __m256 sum0 = _mm256_mul_ps(diff0, diff0);

            __m256 va1 = _mm256_loadu_ps(pa + i + 8);
            __m256 vb1 = _mm256_loadu_ps(pb + i + 8);
            __m256 diff1 = _mm256_sub_ps(va1, vb1);
            __m256 sum1 = _mm256_mul_ps(diff1, diff1);

            __m256 sum = _mm256_add_ps(sum0, sum1);
            float tmp[8];
            _mm256_storeu_ps(tmp, sum);
            sum_scalar += tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                          tmp[4] + tmp[5] + tmp[6] + tmp[7];
            if (sum_scalar > fbound) {
                *early_exit = true;
                return static_cast<dist_t>(sum_scalar);
            }
        }
        for (size_t i = qty16; i < qty; ++i) {
            const float diff = pa[i] - pb[i];
            sum_scalar += diff * diff;
            if (sum_scalar > fbound) {
                *early_exit = true;
                return static_cast<dist_t>(sum_scalar);
            }
        }
        *early_exit = false;
        return static_cast<dist_t>(sum_scalar);
#else
        return writerDistanceBounded(a, b, bound, early_exit);
#endif
    }

    inline dist_t writerDistancePrefixGuard(const void* a, const void* b,
                                            dist_t bound,
                                            bool* early_exit) const {
        const float* pa = static_cast<const float*>(a);
        const float* pb = static_cast<const float*>(b);
        const size_t qty = *static_cast<const size_t*>(dist_func_param_);
        const size_t prefix_qty = std::min<size_t>(
            qty, static_cast<size_t>(tail_issue_writer_dist_prefix_guard_lines_) * 16);
        const float fbound = static_cast<float>(bound);
        float sum = 0.0f;
        size_t i = 0;
#ifdef USE_AVX
        for (; i + 15 < prefix_qty; i += 16) {
            __m256 va0 = _mm256_loadu_ps(pa + i);
            __m256 vb0 = _mm256_loadu_ps(pb + i);
            __m256 diff0 = _mm256_sub_ps(va0, vb0);
            __m256 prod0 = _mm256_mul_ps(diff0, diff0);
            __m256 va1 = _mm256_loadu_ps(pa + i + 8);
            __m256 vb1 = _mm256_loadu_ps(pb + i + 8);
            __m256 diff1 = _mm256_sub_ps(va1, vb1);
            __m256 prod1 = _mm256_mul_ps(diff1, diff1);
            __m256 v_sum = _mm256_add_ps(prod0, prod1);
            float tmp[8];
            _mm256_storeu_ps(tmp, v_sum);
            sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                   tmp[4] + tmp[5] + tmp[6] + tmp[7];
        }
#elif defined(USE_SSE)
        for (; i + 15 < prefix_qty; i += 16) {
            __m128 va0 = _mm_loadu_ps(pa + i);
            __m128 vb0 = _mm_loadu_ps(pb + i);
            __m128 diff0 = _mm_sub_ps(va0, vb0);
            __m128 sum0 = _mm_mul_ps(diff0, diff0);
            __m128 va1 = _mm_loadu_ps(pa + i + 4);
            __m128 vb1 = _mm_loadu_ps(pb + i + 4);
            __m128 diff1 = _mm_sub_ps(va1, vb1);
            __m128 sum1 = _mm_mul_ps(diff1, diff1);
            __m128 va2 = _mm_loadu_ps(pa + i + 8);
            __m128 vb2 = _mm_loadu_ps(pb + i + 8);
            __m128 diff2 = _mm_sub_ps(va2, vb2);
            __m128 sum2 = _mm_mul_ps(diff2, diff2);
            __m128 va3 = _mm_loadu_ps(pa + i + 12);
            __m128 vb3 = _mm_loadu_ps(pb + i + 12);
            __m128 diff3 = _mm_sub_ps(va3, vb3);
            __m128 sum3 = _mm_mul_ps(diff3, diff3);
            sum0 = _mm_add_ps(_mm_add_ps(sum0, sum1), _mm_add_ps(sum2, sum3));
            float tmp[4];
            _mm_storeu_ps(tmp, sum0);
            sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
        }
#endif
        for (; i < prefix_qty; ++i) {
            const float diff = pa[i] - pb[i];
            sum += diff * diff;
        }
        if (sum > fbound) {
            *early_exit = true;
            return static_cast<dist_t>(sum);
        }
        *early_exit = false;
        return fstdistfunc_(a, b, dist_func_param_);
    }

    inline dist_t writerDistancePrefixEstimate(const void* a, const void* b,
                                               dist_t bound,
                                               bool* skipped_full) const {
        const float* pa = static_cast<const float*>(a);
        const float* pb = static_cast<const float*>(b);
        const size_t qty = *static_cast<const size_t*>(dist_func_param_);
        const size_t prefix_qty = std::min<size_t>(
            qty, static_cast<size_t>(tail_issue_writer_dist_prefix_estimate_lines_) * 16);
        float sum = 0.0f;
        size_t i = 0;
#ifdef USE_AVX
        for (; i + 15 < prefix_qty; i += 16) {
            __m256 va0 = _mm256_loadu_ps(pa + i);
            __m256 vb0 = _mm256_loadu_ps(pb + i);
            __m256 diff0 = _mm256_sub_ps(va0, vb0);
            __m256 prod0 = _mm256_mul_ps(diff0, diff0);
            __m256 va1 = _mm256_loadu_ps(pa + i + 8);
            __m256 vb1 = _mm256_loadu_ps(pb + i + 8);
            __m256 diff1 = _mm256_sub_ps(va1, vb1);
            __m256 prod1 = _mm256_mul_ps(diff1, diff1);
            __m256 v_sum = _mm256_add_ps(prod0, prod1);
            float tmp[8];
            _mm256_storeu_ps(tmp, v_sum);
            sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                   tmp[4] + tmp[5] + tmp[6] + tmp[7];
        }
#elif defined(USE_SSE)
        for (; i + 15 < prefix_qty; i += 16) {
            __m128 va0 = _mm_loadu_ps(pa + i);
            __m128 vb0 = _mm_loadu_ps(pb + i);
            __m128 diff0 = _mm_sub_ps(va0, vb0);
            __m128 sum0 = _mm_mul_ps(diff0, diff0);
            __m128 va1 = _mm_loadu_ps(pa + i + 4);
            __m128 vb1 = _mm_loadu_ps(pb + i + 4);
            __m128 diff1 = _mm_sub_ps(va1, vb1);
            __m128 sum1 = _mm_mul_ps(diff1, diff1);
            __m128 va2 = _mm_loadu_ps(pa + i + 8);
            __m128 vb2 = _mm_loadu_ps(pb + i + 8);
            __m128 diff2 = _mm_sub_ps(va2, vb2);
            __m128 sum2 = _mm_mul_ps(diff2, diff2);
            __m128 va3 = _mm_loadu_ps(pa + i + 12);
            __m128 vb3 = _mm_loadu_ps(pb + i + 12);
            __m128 diff3 = _mm_sub_ps(va3, vb3);
            __m128 sum3 = _mm_mul_ps(diff3, diff3);
            sum0 = _mm_add_ps(_mm_add_ps(sum0, sum1), _mm_add_ps(sum2, sum3));
            float tmp[4];
            _mm_storeu_ps(tmp, sum0);
            sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
        }
#endif
        for (; i < prefix_qty; ++i) {
            const float diff = pa[i] - pb[i];
            sum += diff * diff;
        }
        if (prefix_qty == 0 || prefix_qty >= qty) {
            *skipped_full = false;
            return static_cast<dist_t>(sum);
        }

        const double estimated =
            static_cast<double>(sum) *
            static_cast<double>(qty) / static_cast<double>(prefix_qty);
        const double threshold =
            static_cast<double>(bound) *
            static_cast<double>(tail_issue_writer_dist_prefix_estimate_margin_permille_) /
            1000.0;
        if (estimated > threshold) {
            *skipped_full = true;
            return static_cast<dist_t>(estimated);
        }

        *skipped_full = false;
        return fstdistfunc_(a, b, dist_func_param_);
    }

    inline dist_t readerDistanceBoundedVector(const void* a, const void* b,
                                              dist_t bound,
                                              bool* early_exit) const {
        return writerDistanceBoundedVector(a, b, bound, early_exit);
    }

    inline dist_t readerDistancePrefixGuard(const void* a, const void* b,
                                            dist_t bound,
                                            bool* early_exit) const {
        const float* pa = static_cast<const float*>(a);
        const float* pb = static_cast<const float*>(b);
        const size_t qty = *static_cast<const size_t*>(dist_func_param_);
        const size_t prefix_qty =
            std::min<size_t>(qty, static_cast<size_t>(tail_issue_reader_prefix_guard_lines_) * 16);
        const float fbound = static_cast<float>(bound);
        float sum = 0.0f;
        size_t i = 0;
#ifdef USE_AVX
        for (; i + 15 < prefix_qty; i += 16) {
            __m256 va0 = _mm256_loadu_ps(pa + i);
            __m256 vb0 = _mm256_loadu_ps(pb + i);
            __m256 diff0 = _mm256_sub_ps(va0, vb0);
            __m256 prod0 = _mm256_mul_ps(diff0, diff0);
            __m256 va1 = _mm256_loadu_ps(pa + i + 8);
            __m256 vb1 = _mm256_loadu_ps(pb + i + 8);
            __m256 diff1 = _mm256_sub_ps(va1, vb1);
            __m256 prod1 = _mm256_mul_ps(diff1, diff1);
            __m256 v_sum = _mm256_add_ps(prod0, prod1);
            float tmp[8];
            _mm256_storeu_ps(tmp, v_sum);
            sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                   tmp[4] + tmp[5] + tmp[6] + tmp[7];
        }
#elif defined(USE_SSE)
        for (; i + 15 < prefix_qty; i += 16) {
            __m128 va0 = _mm_loadu_ps(pa + i);
            __m128 vb0 = _mm_loadu_ps(pb + i);
            __m128 diff0 = _mm_sub_ps(va0, vb0);
            __m128 sum0 = _mm_mul_ps(diff0, diff0);
            __m128 va1 = _mm_loadu_ps(pa + i + 4);
            __m128 vb1 = _mm_loadu_ps(pb + i + 4);
            __m128 diff1 = _mm_sub_ps(va1, vb1);
            __m128 sum1 = _mm_mul_ps(diff1, diff1);
            __m128 va2 = _mm_loadu_ps(pa + i + 8);
            __m128 vb2 = _mm_loadu_ps(pb + i + 8);
            __m128 diff2 = _mm_sub_ps(va2, vb2);
            __m128 sum2 = _mm_mul_ps(diff2, diff2);
            __m128 va3 = _mm_loadu_ps(pa + i + 12);
            __m128 vb3 = _mm_loadu_ps(pb + i + 12);
            __m128 diff3 = _mm_sub_ps(va3, vb3);
            __m128 sum3 = _mm_mul_ps(diff3, diff3);
            sum0 = _mm_add_ps(_mm_add_ps(sum0, sum1), _mm_add_ps(sum2, sum3));
            float tmp[4];
            _mm_storeu_ps(tmp, sum0);
            sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
        }
#endif
        for (; i < prefix_qty; ++i) {
            const float diff = pa[i] - pb[i];
            sum += diff * diff;
        }

        if (sum > fbound) {
            *early_exit = true;
            return static_cast<dist_t>(sum);
        }

        if (prefix_qty < qty) {
            *early_exit = false;
            return fstdistfunc_(a, b, dist_func_param_);
        }
        *early_exit = false;
        return static_cast<dist_t>(sum);
    }

    inline dist_t readerDistancePrefixEstimate(const void* a, const void* b,
                                               dist_t bound,
                                               bool* skipped_full) const {
        const float* pa = static_cast<const float*>(a);
        const float* pb = static_cast<const float*>(b);
        const size_t qty = *static_cast<const size_t*>(dist_func_param_);
        const size_t prefix_qty =
            std::min<size_t>(qty, static_cast<size_t>(tail_issue_reader_prefix_estimate_lines_) * 16);
        float prefix_sum = 0.0f;
        size_t i = 0;
#ifdef USE_AVX
        for (; i + 15 < prefix_qty; i += 16) {
            __m256 va0 = _mm256_loadu_ps(pa + i);
            __m256 vb0 = _mm256_loadu_ps(pb + i);
            __m256 diff0 = _mm256_sub_ps(va0, vb0);
            __m256 prod0 = _mm256_mul_ps(diff0, diff0);
            __m256 va1 = _mm256_loadu_ps(pa + i + 8);
            __m256 vb1 = _mm256_loadu_ps(pb + i + 8);
            __m256 diff1 = _mm256_sub_ps(va1, vb1);
            __m256 prod1 = _mm256_mul_ps(diff1, diff1);
            __m256 v_sum = _mm256_add_ps(prod0, prod1);
            float tmp[8];
            _mm256_storeu_ps(tmp, v_sum);
            prefix_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                          tmp[4] + tmp[5] + tmp[6] + tmp[7];
        }
#elif defined(USE_SSE)
        for (; i + 15 < prefix_qty; i += 16) {
            __m128 va0 = _mm_loadu_ps(pa + i);
            __m128 vb0 = _mm_loadu_ps(pb + i);
            __m128 diff0 = _mm_sub_ps(va0, vb0);
            __m128 sum0 = _mm_mul_ps(diff0, diff0);
            __m128 va1 = _mm_loadu_ps(pa + i + 4);
            __m128 vb1 = _mm_loadu_ps(pb + i + 4);
            __m128 diff1 = _mm_sub_ps(va1, vb1);
            __m128 sum1 = _mm_mul_ps(diff1, diff1);
            __m128 va2 = _mm_loadu_ps(pa + i + 8);
            __m128 vb2 = _mm_loadu_ps(pb + i + 8);
            __m128 diff2 = _mm_sub_ps(va2, vb2);
            __m128 sum2 = _mm_mul_ps(diff2, diff2);
            __m128 va3 = _mm_loadu_ps(pa + i + 12);
            __m128 vb3 = _mm_loadu_ps(pb + i + 12);
            __m128 diff3 = _mm_sub_ps(va3, vb3);
            __m128 sum3 = _mm_mul_ps(diff3, diff3);
            sum0 = _mm_add_ps(_mm_add_ps(sum0, sum1), _mm_add_ps(sum2, sum3));
            float tmp[4];
            _mm_storeu_ps(tmp, sum0);
            prefix_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
        }
#endif
        for (; i < prefix_qty; ++i) {
            const float diff = pa[i] - pb[i];
            prefix_sum += diff * diff;
        }
        if (prefix_qty == 0 || prefix_qty >= qty) {
            *skipped_full = false;
            return static_cast<dist_t>(prefix_sum);
        }
        const double estimated =
            static_cast<double>(prefix_sum) *
            static_cast<double>(qty) / static_cast<double>(prefix_qty);
        const double threshold =
            static_cast<double>(bound) *
            static_cast<double>(tail_issue_reader_prefix_estimate_margin_permille_) /
            1000.0;
        if (estimated > threshold) {
            *skipped_full = true;
            return static_cast<dist_t>(estimated);
        }
        for (; i < qty; ++i) {
            const float diff = pa[i] - pb[i];
            prefix_sum += diff * diff;
        }
        *skipped_full = false;
        return static_cast<dist_t>(prefix_sum);
    }

    inline dist_t readerDistanceForTail(const void* a, const void* b,
                                        dist_t bound,
                                        bool exact_required) const {
        if (tail_issue_reader_prefix_estimate_enabled_ &&
            tail_issue_tls.role == TailIssueRole::Reader &&
            tail_issue_tls.urgent &&
            !exact_required &&
            bound < std::numeric_limits<dist_t>::max()) {
            bool skipped_full = false;
            dist_t d = readerDistancePrefixEstimate(a, b, bound, &skipped_full);
            tail_issue_reader_prefix_estimate_ops_.fetch_add(
                1, std::memory_order_relaxed);
            if (skipped_full) {
                tail_issue_reader_prefix_estimate_skips_.fetch_add(
                    1, std::memory_order_relaxed);
            } else {
                tail_issue_reader_prefix_estimate_full_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            return d;
        }
        if (tail_issue_reader_prefix_guard_enabled_ &&
            tail_issue_tls.role == TailIssueRole::Reader &&
            tail_issue_tls.urgent &&
            !exact_required &&
            bound < std::numeric_limits<dist_t>::max()) {
            bool early_exit = false;
            dist_t d = readerDistancePrefixGuard(a, b, bound, &early_exit);
            tail_issue_reader_prefix_guard_ops_.fetch_add(
                1, std::memory_order_relaxed);
            if (early_exit) {
                tail_issue_reader_prefix_guard_early_.fetch_add(
                    1, std::memory_order_relaxed);
            } else {
                tail_issue_reader_prefix_guard_full_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            return d;
        }
        if (tail_issue_reader_bounded_distance_enabled_ &&
            tail_issue_tls.role == TailIssueRole::Reader &&
            tail_issue_tls.urgent &&
            !exact_required &&
            bound < std::numeric_limits<dist_t>::max()) {
            bool early_exit = false;
            dist_t d = readerDistanceBoundedVector(a, b, bound, &early_exit);
            tail_issue_reader_bounded_distance_ops_.fetch_add(
                1, std::memory_order_relaxed);
            if (early_exit) {
                tail_issue_reader_bounded_distance_early_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            return d;
        }
        return fstdistfunc_(a, b, dist_func_param_);
    }

    inline dist_t distanceForCurrentRole(const void* a, const void* b) const {
        if (shouldShapeWriterDistance()) {
            tail_issue_writer_dist_shaped_ops_.fetch_add(1, std::memory_order_relaxed);
            if (tail_issue_writer_dist_mode_ == 1) {
                return writerDistanceScalar(a, b);
            }
            if (tail_issue_writer_dist_mode_ == 2) {
                return writerDistanceSse4(a, b);
            }
        }
        return fstdistfunc_(a, b, dist_func_param_);
    }

    inline dist_t distanceForCurrentRoleBounded(const void* a, const void* b,
                                                dist_t bound,
                                                bool exact_required) const {
        if (!exact_required && bound < std::numeric_limits<dist_t>::max() &&
            (tail_issue_writer_dist_mode_ == 3 || tail_issue_writer_dist_mode_ == 4) &&
            shouldShapeWriterDistance()) {
            bool early_exit = false;
            dist_t d = tail_issue_writer_dist_mode_ == 4
                           ? writerDistanceBoundedVector(a, b, bound, &early_exit)
                           : writerDistanceBounded(a, b, bound, &early_exit);
            tail_issue_writer_dist_shaped_ops_.fetch_add(1, std::memory_order_relaxed);
            if (early_exit) {
                tail_issue_writer_dist_bounded_early_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            return d;
        }
        if (!exact_required && bound < std::numeric_limits<dist_t>::max() &&
            tail_issue_writer_dist_mode_ == 5 && shouldShapeWriterDistance()) {
            bool early_exit = false;
            dist_t d = writerDistancePrefixGuard(a, b, bound, &early_exit);
            tail_issue_writer_dist_shaped_ops_.fetch_add(1, std::memory_order_relaxed);
            if (early_exit) {
                tail_issue_writer_dist_bounded_early_.fetch_add(
                    1, std::memory_order_relaxed);
            } else {
                tail_issue_writer_dist_prefix_full_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            return d;
        }
        if (!exact_required && bound < std::numeric_limits<dist_t>::max() &&
            tail_issue_writer_dist_mode_ == 6 && shouldShapeWriterDistance()) {
            bool skipped_full = false;
            dist_t d = writerDistancePrefixEstimate(a, b, bound, &skipped_full);
            tail_issue_writer_dist_shaped_ops_.fetch_add(1, std::memory_order_relaxed);
            if (skipped_full) {
                tail_issue_writer_dist_prefix_estimate_skips_.fetch_add(
                    1, std::memory_order_relaxed);
            } else {
                tail_issue_writer_dist_prefix_estimate_full_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            return d;
        }
        if (!exact_required && bound < std::numeric_limits<dist_t>::max() &&
            tail_issue_writer_dist_mode_ == 7 && shouldShapeWriterDistance()) {
            tail_issue_writer_dist_shaped_ops_.fetch_add(1, std::memory_order_relaxed);
            tail_issue_writer_dist_admission_skips_.fetch_add(
                1, std::memory_order_relaxed);
            return bound;
        }
        return distanceForCurrentRole(a, b);
    }

    inline void __attribute__((always_inline))
    prefetchVectorPayloadPtr(const char* ptr) const {
#ifdef USE_SSE
        const size_t payload_lines = (data_size_ + 63) / 64;
        bool reader_urgent = false;
        bool writer_tail = false;
        if (tail_issue_prefetch_policy_enabled_) {
            if (tail_issue_tls.role == TailIssueRole::Reader) {
                reader_urgent = tail_issue_tls.urgent;
            } else if (tail_issue_tls.role == TailIssueRole::Writer) {
                writer_tail = tailIssueWriterTailCached();
            }
        }
        size_t lines =
#if ANNCHOR_VSAG_PREFETCH_DEPTH_LINES > 0
            ANNCHOR_VSAG_PREFETCH_DEPTH_LINES;
#else
            payload_lines;
#if ANNCHOR_VSAG_PREFETCH_MAX_CODE_LINES > 0
        if (lines > ANNCHOR_VSAG_PREFETCH_MAX_CODE_LINES) {
            lines = ANNCHOR_VSAG_PREFETCH_MAX_CODE_LINES;
        }
#endif
#endif
        if (lines > payload_lines) {
            lines = payload_lines;
        }
        if (lines > 8) {
            lines = 8;
        }
        if (reader_urgent && tail_issue_reader_urgent_prefetch_enabled_) {
            size_t urgent_lines = payload_lines;
            if (tail_issue_tls.reader_budget_left == 0) {
                urgent_lines = lines;
            } else if (urgent_lines > tail_issue_tls.reader_budget_left) {
                urgent_lines = tail_issue_tls.reader_budget_left;
            }
            if (urgent_lines > 8) {
                urgent_lines = 8;
            }
            if (urgent_lines > lines) {
                const size_t extra_lines = urgent_lines - lines;
                tail_issue_reader_urgent_prefetch_lines_.fetch_add(
                    extra_lines, std::memory_order_relaxed);
                tail_issue_tls.reader_budget_left -= static_cast<uint32_t>(extra_lines);
                lines = urgent_lines;
            }
        }
        uint32_t writer_action = 0;
        if (writer_tail && tail_issue_writer_action_ > 0 &&
            consumeTailIssueWriterBudget(lines)) {
            writer_action = tail_issue_writer_action_;
            tail_issue_writer_tail_hits_.fetch_add(1, std::memory_order_relaxed);
            tail_issue_writer_hint_lines_.fetch_add(lines, std::memory_order_relaxed);
        } else if (writer_tail && tail_issue_writer_action_ == 0 &&
                   lines > tail_issue_writer_tail_lines_ &&
                   consumeTailIssueWriterBudget(lines - tail_issue_writer_tail_lines_)) {
            tail_issue_writer_tail_hits_.fetch_add(1, std::memory_order_relaxed);
            tail_issue_writer_suppressed_lines_.fetch_add(
                lines - tail_issue_writer_tail_lines_, std::memory_order_relaxed);
            lines = tail_issue_writer_tail_lines_;
        }
        if (lines > 0) prefetchLineWithTailHint(ptr, writer_action);
        if (lines > 1) prefetchLineWithTailHint(ptr + 64, writer_action);
        if (lines > 2) prefetchLineWithTailHint(ptr + 128, writer_action);
        if (lines > 3) prefetchLineWithTailHint(ptr + 192, writer_action);
        if (lines > 4) prefetchLineWithTailHint(ptr + 256, writer_action);
        if (lines > 5) prefetchLineWithTailHint(ptr + 320, writer_action);
        if (lines > 6) prefetchLineWithTailHint(ptr + 384, writer_action);
        if (lines > 7) prefetchLineWithTailHint(ptr + 448, writer_action);
#else
        (void)ptr;
#endif
    }

    inline size_t __attribute__((always_inline))
    prefetchVectorStride() const {
#if ANNCHOR_VSAG_PREFETCH_STRIDE > 0
        return ANNCHOR_VSAG_PREFETCH_STRIDE;
#else
        const size_t half_line_groups = data_size_ / (64 * 2);
        return half_line_groups > 1 ? half_line_groups - 1 : 1;
#endif
    }

    inline void __attribute__((always_inline))
    prefetchVectorPayload(tableint id) const {
        prefetchVectorPayloadPtr(getDataByInternalId(id));
    }

    inline void __attribute__((always_inline))
    prefetchVectorPayloadPtrWithAction(const char* ptr, uint32_t action) const {
#ifdef USE_SSE
        const size_t payload_lines = (data_size_ + 63) / 64;
        size_t lines =
#if ANNCHOR_VSAG_PREFETCH_DEPTH_LINES > 0
            ANNCHOR_VSAG_PREFETCH_DEPTH_LINES;
#else
            payload_lines;
#if ANNCHOR_VSAG_PREFETCH_MAX_CODE_LINES > 0
        if (lines > ANNCHOR_VSAG_PREFETCH_MAX_CODE_LINES) {
            lines = ANNCHOR_VSAG_PREFETCH_MAX_CODE_LINES;
        }
#endif
#endif
        if (lines > payload_lines) {
            lines = payload_lines;
        }
        if (lines > 8) {
            lines = 8;
        }
        if (lines > 0) prefetchLineWithTailHint(ptr, action);
        if (lines > 1) prefetchLineWithTailHint(ptr + 64, action);
        if (lines > 2) prefetchLineWithTailHint(ptr + 128, action);
        if (lines > 3) prefetchLineWithTailHint(ptr + 192, action);
        if (lines > 4) prefetchLineWithTailHint(ptr + 256, action);
        if (lines > 5) prefetchLineWithTailHint(ptr + 320, action);
        if (lines > 6) prefetchLineWithTailHint(ptr + 384, action);
        if (lines > 7) prefetchLineWithTailHint(ptr + 448, action);
#else
        (void)ptr;
        (void)action;
#endif
    }

    inline void __attribute__((always_inline))
    prefetchVectorPayloadPtrLines(const char* ptr, size_t lines,
                                  uint32_t action) const {
#ifdef USE_SSE
        const size_t payload_lines = (data_size_ + 63) / 64;
        if (lines > payload_lines) {
            lines = payload_lines;
        }
        if (lines > 8) {
            lines = 8;
        }
        if (lines > 0) prefetchLineWithTailHint(ptr, action);
        if (lines > 1) prefetchLineWithTailHint(ptr + 64, action);
        if (lines > 2) prefetchLineWithTailHint(ptr + 128, action);
        if (lines > 3) prefetchLineWithTailHint(ptr + 192, action);
        if (lines > 4) prefetchLineWithTailHint(ptr + 256, action);
        if (lines > 5) prefetchLineWithTailHint(ptr + 320, action);
        if (lines > 6) prefetchLineWithTailHint(ptr + 384, action);
        if (lines > 7) prefetchLineWithTailHint(ptr + 448, action);
#else
        (void)ptr;
        (void)lines;
        (void)action;
#endif
    }

    inline void __attribute__((always_inline))
    maybePrefetchReaderCurrent(const char* ptr) const {
        if (tail_issue_reader_current_lines_ == 0 ||
            tail_issue_tls.role != TailIssueRole::Reader ||
            !tail_issue_tls.urgent ||
            tail_issue_tls.reader_budget_left == 0) {
            return;
        }
#if ANNCHOR_VSAG_PREFETCH_DEPTH_LINES > 0
        const size_t base_lines = ANNCHOR_VSAG_PREFETCH_DEPTH_LINES;
#else
        const size_t base_lines = 0;
#endif
        if (tail_issue_reader_current_lines_ <= base_lines) {
            return;
        }
        const uint32_t extra_need =
            tail_issue_reader_current_lines_ - static_cast<uint32_t>(base_lines);
        const uint32_t extra =
            std::min<uint32_t>(extra_need, tail_issue_tls.reader_budget_left);
        if (extra == 0) {
            return;
        }
        const size_t lines = static_cast<size_t>(base_lines) + extra;
        prefetchVectorPayloadPtrLines(ptr, lines, 0);
        tail_issue_tls.reader_budget_left -= extra;
        tail_issue_reader_current_prefetch_lines_.fetch_add(
            extra, std::memory_order_relaxed);
    }

    inline void __attribute__((always_inline))
    prefetchVectorPayloadLookahead(tableint id) const {
        if (shouldSkipWriterLeadPrefetch()) {
            return;
        }
        const uint32_t hint_action = writerLeadPrefetchHintAction();
        if (hint_action > 0) {
            prefetchVectorPayloadPtrWithAction(getDataByInternalId(id), hint_action);
            return;
        }
        prefetchVectorPayload(id);
    }

    inline void __attribute__((always_inline))
    prefetchVectorPayloadFromLevel0Base(tableint id) const {
        prefetchVectorPayloadPtr(
            data_level0_memory_ + id * size_data_per_element_ + offsetData_);
    }

    inline void __attribute__((always_inline))
    prefetchVectorPayloadFromLevel0BaseLookahead(tableint id) const {
        if (shouldSkipWriterLeadPrefetch()) {
            return;
        }
        const uint32_t hint_action = writerLeadPrefetchHintAction();
        if (hint_action > 0) {
            prefetchVectorPayloadPtrWithAction(
                data_level0_memory_ + id * size_data_per_element_ + offsetData_,
                hint_action);
            return;
        }
        prefetchVectorPayloadFromLevel0Base(id);
    }

    inline bool __attribute__((always_inline))
    maybeStageReaderLookahead(tableint id) const {
        if (tail_issue_reader_lookahead_mode_ == 0 ||
            tail_issue_tls.role != TailIssueRole::Reader) {
            return false;
        }
        if (!tail_issue_tls.urgent ||
            tail_issue_tls.reader_deferred_lookahead_left == 0) {
            return false;
        }
        --tail_issue_tls.reader_deferred_lookahead_left;
        if (tail_issue_reader_lookahead_mode_ == 1) {
            tail_issue_reader_lookahead_skipped_.fetch_add(
                1, std::memory_order_relaxed);
            return true;
        }
        tail_issue_tls.reader_deferred_lookahead_id =
            static_cast<uint32_t>(id);
        tail_issue_tls.reader_deferred_lookahead_valid = true;
        tail_issue_reader_lookahead_deferred_.fetch_add(
            1, std::memory_order_relaxed);
        return true;
    }

    inline void __attribute__((always_inline))
    issueDeferredReaderLookaheadFromLevel0Base() const {
        if (tail_issue_tls.reader_deferred_lookahead_valid) {
            const tableint id =
                static_cast<tableint>(tail_issue_tls.reader_deferred_lookahead_id);
            tail_issue_tls.reader_deferred_lookahead_valid = false;
            prefetchVectorPayloadFromLevel0Base(id);
        }
    }

    inline void __attribute__((always_inline))
    issueDeferredReaderLookahead() const {
        if (tail_issue_tls.reader_deferred_lookahead_valid) {
            const tableint id =
                static_cast<tableint>(tail_issue_tls.reader_deferred_lookahead_id);
            tail_issue_tls.reader_deferred_lookahead_valid = false;
            prefetchVectorPayload(id);
        }
    }

    inline void __attribute__((always_inline))
    maybePrefetchReaderWindow(tableint id) const {
        if (tail_issue_reader_window_extra_ == 0 ||
            tail_issue_tls.role != TailIssueRole::Reader ||
            !tail_issue_tls.urgent || tail_issue_tls.reader_budget_left == 0) {
            return;
        }
        prefetchVectorPayloadFromLevel0Base(id);
        --tail_issue_tls.reader_budget_left;
        tail_issue_reader_window_prefetches_.fetch_add(1, std::memory_order_relaxed);
    }

    inline size_t tailIssueReaderEffectiveEf(size_t requested_ef,
                                             size_t k) const {
        if (tail_issue_reader_ef_limit_ == 0 ||
            tail_issue_reader_stag_clamp_hops_ > 0 ||
            tail_issue_tls.reader_force_full_ef ||
            tail_issue_tls.role != TailIssueRole::Reader ||
            tail_issue_reader_guard_ns_ == 0) {
            return requested_ef;
        }
        uint64_t now_ns = monotonicRawNs();
        if (tail_issue_tls.op_start_ns == 0) {
            tail_issue_tls.op_start_ns = now_ns;
        }
        if (!maybeActivateReaderTailIssue(now_ns) || !tail_issue_tls.urgent) {
            return requested_ef;
        }
        const size_t floor_ef = std::max<size_t>(k, 1);
        size_t limited_ef = std::max<size_t>(floor_ef, tail_issue_reader_ef_limit_);
        bool stage2 = false;
        if (tail_issue_reader_ef_limit2_ > 0 &&
            tail_issue_reader_guard2_ns_ > tail_issue_reader_guard_ns_ &&
            now_ns - tail_issue_tls.op_start_ns >= tail_issue_reader_guard2_ns_) {
            const size_t limited2 =
                std::max<size_t>(floor_ef, tail_issue_reader_ef_limit2_);
            if (limited2 < limited_ef) {
                limited_ef = limited2;
                stage2 = true;
            }
        }
        if (limited_ef < requested_ef) {
            tail_issue_tls.reader_clamped = true;
            tail_issue_reader_beam_clamps_.fetch_add(1, std::memory_order_relaxed);
            tail_issue_reader_beam_trimmed_.fetch_add(
                requested_ef - limited_ef, std::memory_order_relaxed);
            if (stage2) {
                tail_issue_reader_beam_clamps2_.fetch_add(
                    1, std::memory_order_relaxed);
                tail_issue_reader_beam_trimmed2_.fetch_add(
                    requested_ef - limited_ef, std::memory_order_relaxed);
            }
            return limited_ef;
        }
        return requested_ef;
    }

    inline size_t tailIssueReaderEfLimit(size_t current_ef) const {
        if (tail_issue_reader_ef_limit_ == 0 ||
            tail_issue_tls.reader_force_full_ef ||
            tail_issue_tls.role != TailIssueRole::Reader ||
            tail_issue_tls.op_start_ns == 0) {
            return current_ef;
        }
        const uint64_t now_ns = monotonicRawNs();
        if (!maybeActivateReaderTailIssue(now_ns) || !tail_issue_tls.urgent) {
            return current_ef;
        }
        size_t limited_ef = std::max<size_t>(1, tail_issue_reader_ef_limit_);
        bool stage2 = false;
        if (tail_issue_reader_ef_limit2_ > 0 &&
            tail_issue_reader_guard2_ns_ > tail_issue_reader_guard_ns_ &&
            now_ns - tail_issue_tls.op_start_ns >= tail_issue_reader_guard2_ns_) {
            const size_t limited2 =
                std::max<size_t>(1, tail_issue_reader_ef_limit2_);
            if (limited2 < limited_ef) {
                limited_ef = limited2;
                stage2 = true;
            }
        }
        if (limited_ef < current_ef) {
            tail_issue_tls.reader_clamped = true;
            if (stage2) {
                tail_issue_reader_beam_clamps2_.fetch_add(
                    1, std::memory_order_relaxed);
                tail_issue_reader_beam_trimmed2_.fetch_add(
                    current_ef - limited_ef, std::memory_order_relaxed);
            }
            return limited_ef;
        }
        return current_ef;
    }

    template <typename TopCandidates>
    inline bool tailIssueReaderClampConfidence(const TopCandidates& top_candidates,
                                               size_t target_ef) const {
        if (tail_issue_reader_confidence_margin_permille_ == 0) {
            return true;
        }
        if (tail_issue_tls.reader_confidence_resolved) {
            return false;
        }
        tail_issue_reader_confidence_checks_.fetch_add(
            1, std::memory_order_relaxed);
        const size_t k = tail_issue_tls.reader_k;
        if (k == 0 || target_ef <= k || top_candidates.size() <= target_ef) {
            tail_issue_tls.reader_confidence_resolved = true;
            tail_issue_reader_confidence_rejects_.fetch_add(
                1, std::memory_order_relaxed);
            return false;
        }
        auto copy = top_candidates;
        while (copy.size() > target_ef + 1) {
            copy.pop();
        }
        if (copy.size() <= target_ef) {
            tail_issue_tls.reader_confidence_resolved = true;
            tail_issue_reader_confidence_rejects_.fetch_add(
                1, std::memory_order_relaxed);
            return false;
        }
        const dist_t boundary = copy.top().first;
        while (copy.size() > k) {
            copy.pop();
        }
        if (copy.empty()) {
            tail_issue_tls.reader_confidence_resolved = true;
            tail_issue_reader_confidence_rejects_.fetch_add(
                1, std::memory_order_relaxed);
            return false;
        }
        const dist_t kth = copy.top().first;
        const double lhs = static_cast<double>(boundary) * 1000.0;
        const double rhs = static_cast<double>(kth) *
                           static_cast<double>(tail_issue_reader_confidence_margin_permille_);
        if (lhs >= rhs) {
            tail_issue_tls.reader_confidence_resolved = true;
            tail_issue_reader_confidence_grants_.fetch_add(
                1, std::memory_order_relaxed);
            return true;
        }
        tail_issue_tls.reader_confidence_resolved = true;
        tail_issue_reader_confidence_rejects_.fetch_add(
            1, std::memory_order_relaxed);
        return false;
    }

    inline size_t tailIssueReaderStagnationEfLimit(
        size_t current_ef, uint32_t stagnant_hops,
        uint32_t expansions_after_urgent) const {
        if (tail_issue_reader_stag_clamp_hops_ == 0 ||
            tail_issue_reader_ef_limit_ == 0 ||
            tail_issue_tls.reader_force_full_ef ||
            tail_issue_tls.role != TailIssueRole::Reader ||
            !tail_issue_tls.urgent ||
            stagnant_hops < tail_issue_reader_stag_clamp_hops_ ||
            expansions_after_urgent < tail_issue_reader_stag_clamp_min_expansions_) {
            return current_ef;
        }
        const size_t limited_ef =
            std::max<size_t>(1, tail_issue_reader_ef_limit_);
        if (limited_ef < current_ef) {
            tail_issue_tls.reader_clamped = true;
            return limited_ef;
        }
        return current_ef;
    }

    template <typename TopCandidates>
    inline bool tailIssueReaderFrontierStop(
        const TopCandidates& top_candidates,
        dist_t candidate_dist, uint32_t stagnant_hops,
        uint32_t expansions_after_urgent) const {
        if (tail_issue_reader_frontier_ef_ > 0) {
            tail_issue_reader_frontier_checks_.fetch_add(
                1, std::memory_order_relaxed);
        }
        if (tail_issue_reader_frontier_ef_ == 0 ||
            tail_issue_reader_frontier_stag_hops_ == 0 ||
            tail_issue_tls.role != TailIssueRole::Reader ||
            !tail_issue_tls.urgent ||
            stagnant_hops < tail_issue_reader_frontier_stag_hops_ ||
            expansions_after_urgent < tail_issue_reader_frontier_min_expansions_ ||
            top_candidates.size() <= tail_issue_reader_frontier_ef_ ||
            top_candidates.empty()) {
            return false;
        }
        tail_issue_reader_frontier_eligible_.fetch_add(
            1, std::memory_order_relaxed);

        auto copy = top_candidates;
        while (copy.size() > tail_issue_reader_frontier_ef_) {
            copy.pop();
        }
        if (copy.empty()) {
            return false;
        }
        const dist_t frontier_bound = copy.top().first;
        if (candidate_dist > frontier_bound) {
            tail_issue_reader_frontier_stops_.fetch_add(
                1, std::memory_order_relaxed);
            return true;
        }
        return false;
    }

    template <typename TopCandidates>
    inline bool tailIssueReaderRetryNeeded(const TopCandidates& top_candidates,
                                           size_t k) const {
        if (!tail_issue_reader_retry_enabled_ ||
            !tail_issue_tls.reader_clamped ||
            tail_issue_tls.reader_force_full_ef ||
            k == 0 ||
            top_candidates.size() <= k ||
            tail_issue_reader_retry_margin_permille_ <= 1000) {
            return false;
        }
        tail_issue_reader_retry_checks_.fetch_add(1, std::memory_order_relaxed);
        auto copy = top_candidates;
        while (copy.size() > k + 1) {
            copy.pop();
        }
        if (copy.size() <= k) {
            return true;
        }
        const dist_t next_dist = copy.top().first;
        copy.pop();
        if (copy.empty()) {
            return true;
        }
        const dist_t kth_dist = copy.top().first;
        const double lhs = static_cast<double>(next_dist) * 1000.0;
        const double rhs = static_cast<double>(kth_dist) *
                           static_cast<double>(tail_issue_reader_retry_margin_permille_);
        if (lhs >= rhs) {
            tail_issue_reader_retry_confident_skips_.fetch_add(
                1, std::memory_order_relaxed);
            return false;
        }
        return true;
    }

    template <typename TopCandidates>
    inline bool tailIssueReaderContinueNeeded(const TopCandidates& top_candidates,
                                              dist_t candidate_dist,
                                              size_t full_ef) const {
        if (!tail_issue_reader_continue_enabled_ ||
            !tail_issue_tls.reader_clamped ||
            tail_issue_tls.reader_force_full_ef ||
            top_candidates.empty() ||
            top_candidates.size() >= full_ef ||
            tail_issue_reader_continue_margin_permille_ <= 1000) {
            return false;
        }
        tail_issue_reader_continue_checks_.fetch_add(
            1, std::memory_order_relaxed);
        const dist_t boundary = top_candidates.top().first;
        const double lhs = static_cast<double>(candidate_dist) * 1000.0;
        const double rhs = static_cast<double>(boundary) *
                           static_cast<double>(tail_issue_reader_continue_margin_permille_);
        if (lhs > rhs) {
            tail_issue_reader_continue_confident_stops_.fetch_add(
                1, std::memory_order_relaxed);
            return false;
        }
        tail_issue_tls.reader_force_full_ef = true;
        tail_issue_reader_continue_runs_.fetch_add(
            1, std::memory_order_relaxed);
        return true;
    }

    template <typename TopCandidates, typename CandidateSet>
    inline void tailIssueReaderLocalRefine(TopCandidates& top_candidates,
                                           const CandidateSet& candidate_set,
                                           const void* data_point,
                                           size_t effective_ef,
                                           size_t full_ef,
                                           vl_type* visited_array,
                                           vl_type visited_array_tag,
                                           bool effective_bare,
                                           BaseFilterFunctor* isIdAllowed,
                                           size_t view_limit,
                                           dist_t& lowerBound) const {
        if (tail_issue_reader_refine_seeds_ == 0 ||
            tail_issue_reader_refine_edge_limit_ == 0 ||
            !tail_issue_tls.reader_clamped ||
            tail_issue_tls.reader_force_full_ef ||
            tail_issue_tls.role != TailIssueRole::Reader ||
            !tail_issue_tls.urgent ||
            candidate_set.empty()) {
            return;
        }
        std::vector<tableint> seeds;
        seeds.reserve(tail_issue_reader_refine_seeds_);
        auto copy = candidate_set;
        while (!copy.empty() && seeds.size() < tail_issue_reader_refine_seeds_) {
            seeds.push_back(copy.top().second);
            copy.pop();
        }
        if (seeds.empty()) {
            return;
        }
        tail_issue_reader_refine_runs_.fetch_add(1, std::memory_order_relaxed);
        const bool enforce_view_limit =
            view_limit != std::numeric_limits<size_t>::max();
        const size_t target_ef = std::max(effective_ef, full_ef);
        for (tableint seed : seeds) {
            int* data = reinterpret_cast<int*>(get_linklist0(seed));
            const size_t size = getListCount(reinterpret_cast<linklistsizeint*>(data));
            const size_t edge_limit =
                std::min<size_t>(size, tail_issue_reader_refine_edge_limit_);
            for (size_t j = 1; j <= edge_limit; ++j) {
                const tableint candidate_id = static_cast<tableint>(*(data + j));
                tail_issue_reader_refine_edges_scanned_.fetch_add(
                    1, std::memory_order_relaxed);
                if (visited_array[candidate_id] == visited_array_tag) {
                    continue;
                }
                visited_array[candidate_id] = visited_array_tag;
                const bool candidate_visible =
                    !enforce_view_limit ||
                    (search_visibility_by_label_
                         ? static_cast<size_t>(getExternalLabel(candidate_id)) <
                               view_limit
                         : ((enable_non_prefix_visibility_ &&
                             insertion_complete_ &&
                             insertion_complete_[candidate_id].load(
                                 std::memory_order_acquire)) ||
                            static_cast<size_t>(candidate_id) < view_limit));
                if (!candidate_visible) {
                    continue;
                }
                char* currObj1 = getDataByInternalId(candidate_id);
                const dist_t dist =
                    fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() >= target_ef && dist >= lowerBound) {
                    continue;
                }
                const bool candidate_allowed =
                    effective_bare ||
                    (!isMarkedDeleted(candidate_id) &&
                     ((!isIdAllowed) ||
                      (*isIdAllowed)(getExternalLabel(candidate_id))));
                if (!candidate_allowed) {
                    continue;
                }
                top_candidates.emplace(dist, candidate_id);
                tail_issue_reader_refine_inserted_.fetch_add(
                    1, std::memory_order_relaxed);
                while (top_candidates.size() > target_ef) {
                    top_candidates.pop();
                }
                if (!top_candidates.empty()) {
                    lowerBound = top_candidates.top().first;
                }
            }
        }
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
        initTailIssueFromEnv();
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
        initTailIssueFromEnv();
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

    bool tryEnqueueTailIssueDeferredRepair(tableint internal_id,
                                           size_t edge_count) const {
        if (tail_issue_writer_repair_defer_budget_ == 0 ||
            (tail_issue_writer_repair_drain_budget_ == 0 &&
             !tail_issue_writer_repair_no_drain_)) {
            return false;
        }
        std::lock_guard<std::mutex> guard(tail_issue_deferred_repair_queue_lock_);
        if (tail_issue_deferred_repair_set_.find(internal_id) !=
            tail_issue_deferred_repair_set_.end()) {
            return false;
        }
        if (tail_issue_deferred_repair_queue_.size() >=
            static_cast<size_t>(tail_issue_writer_repair_queue_cap_)) {
            tail_issue_writer_repair_defer_dropped_.fetch_add(
                1, std::memory_order_relaxed);
            return false;
        }
        tail_issue_deferred_repair_queue_.push_back(internal_id);
        tail_issue_deferred_repair_set_.insert(internal_id);
        tail_issue_writer_repair_deferred_points_.fetch_add(
            1, std::memory_order_relaxed);
        tail_issue_writer_repair_deferred_edges_.fetch_add(
            static_cast<uint64_t>(edge_count), std::memory_order_relaxed);
        const uint64_t queue_size =
            static_cast<uint64_t>(tail_issue_deferred_repair_queue_.size());
        uint64_t prev =
            tail_issue_writer_repair_queue_max_.load(std::memory_order_relaxed);
        while (queue_size > prev &&
               !tail_issue_writer_repair_queue_max_.compare_exchange_weak(
                   prev, queue_size, std::memory_order_relaxed)) {
        }
        return true;
    }

    bool tryPopTailIssueDeferredRepair(tableint *internal_id) const {
        std::lock_guard<std::mutex> guard(tail_issue_deferred_repair_queue_lock_);
        if (tail_issue_deferred_repair_queue_.empty()) {
            return false;
        }
        *internal_id = tail_issue_deferred_repair_queue_.front();
        tail_issue_deferred_repair_queue_.pop_front();
        tail_issue_deferred_repair_set_.erase(*internal_id);
        return true;
    }

    bool applyTailIssueReverseRepairOnly(tableint cur_c) {
        if (cur_c < 0 || cur_c >= cur_element_count ||
            isMarkedDeleted(cur_c) ||
            !insertion_complete_[cur_c].load(std::memory_order_acquire)) {
            return false;
        }
        std::vector<tableint> selected_neighbors = getConnectionsWithLock(cur_c, 0);
        if (selected_neighbors.empty()) {
            return false;
        }

        uint64_t local_link_updates = 0;
        const size_t mcurmax = maxM0_;
        const bool disable_existing_undo_record_for_point =
            disable_existing_neighbor_undo_record_ &&
            (disable_existing_neighbor_undo_record_from_id_ < 0 ||
             cur_c >= static_cast<tableint>(disable_existing_neighbor_undo_record_from_id_));

        const auto existing_neighbor_update_loop_start =
            std::chrono::steady_clock::now();
        ActivePhaseGuard existing_update_phase(active_phase_existing_update_);
        for (tableint neighbor : selected_neighbors) {
            if (neighbor < 0 || neighbor >= cur_element_count ||
                neighbor == cur_c || isMarkedDeleted(neighbor)) {
                continue;
            }
            metric_graph_existing_neighbor_visits.fetch_add(
                1, std::memory_order_relaxed);
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

            const auto load_scan_start = std::chrono::steady_clock::now();
            linklistsizeint *ll_other = get_linklist0(neighbor);
            size_t sz_link_list_other = getListCount(ll_other);
            if (sz_link_list_other > mcurmax) {
                tls_in_link_critical_section = false;
                throw std::runtime_error("Bad value of sz_link_list_other");
            }
            tableint *data = (tableint *)(ll_other + 1);
            bool is_cur_c_present = false;
            {
                ActivePhaseGuard load_scan_phase(active_phase_load_scan_);
                for (size_t j = 0; j < sz_link_list_other; ++j) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
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

            if (!is_cur_c_present) {
                if (sz_link_list_other < mcurmax) {
                    const auto append_start = std::chrono::steady_clock::now();
                    ActivePhaseGuard append_phase(active_phase_append_);
                    metric_graph_existing_neighbor_appends.fetch_add(
                        1, std::memory_order_relaxed);
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                    const auto append_end = std::chrono::steady_clock::now();
                    metric_graph_existing_neighbor_append_ns.fetch_add(
                        static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                append_end - append_start)
                                .count()),
                        std::memory_order_relaxed);
                    metric_graph_existing_neighbor_edges_written.fetch_add(
                        1, std::memory_order_relaxed);
                    local_link_updates += 2;
                    bumpNodeLinkVersion(neighbor);
                } else {
                    metric_graph_existing_neighbor_prunes.fetch_add(
                        1, std::memory_order_relaxed);
                    metric_graph_existing_neighbor_prune_candidates.fetch_add(
                        sz_link_list_other + 1, std::memory_order_relaxed);
                    const auto prune_heuristic_start =
                        std::chrono::steady_clock::now();
                    ActivePhaseGuard prune_phase(active_phase_prune_);
                    uint64_t prune_dist_comps = 0;
                    dist_t d_max = fstdistfunc_(
                        getDataByInternalId(cur_c),
                        getDataByInternalId(neighbor), dist_func_param_);
                    ++prune_dist_comps;
                    std::priority_queue<
                        std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
                        candidates;
                    candidates.emplace(d_max, cur_c);
                    for (size_t j = 0; j < sz_link_list_other; ++j) {
                        candidates.emplace(
                            fstdistfunc_(
                                getDataByInternalId(data[j]),
                                getDataByInternalId(neighbor),
                                dist_func_param_),
                            data[j]);
                        ++prune_dist_comps;
                    }
                    uint64_t prune_heuristic_dist_comps = 0;
                    getNeighborsByHeuristic(candidates, mcurmax,
                                            &prune_heuristic_dist_comps);
                    prune_dist_comps += prune_heuristic_dist_comps;
                    const auto prune_heuristic_end =
                        std::chrono::steady_clock::now();
                    metric_graph_existing_neighbor_prune_ns.fetch_add(
                        static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                prune_heuristic_end - prune_heuristic_start)
                                .count()),
                        std::memory_order_relaxed);
                    metric_graph_existing_neighbor_prune_dist_comps.fetch_add(
                        prune_dist_comps, std::memory_order_relaxed);

                    std::vector<tableint> old_edges(
                        data, data + sz_link_list_other);
                    std::vector<tableint> new_edges;
                    auto candidates_copy = candidates;
                    while (!candidates_copy.empty()) {
                        new_edges.push_back(candidates_copy.top().second);
                        candidates_copy.pop();
                    }

                    std::vector<tableint> old_edges_sorted = old_edges;
                    std::vector<tableint> new_edges_sorted = new_edges;
                    std::sort(old_edges_sorted.begin(), old_edges_sorted.end());
                    std::sort(new_edges_sorted.begin(), new_edges_sorted.end());
                    const bool linkset_changed =
                        old_edges_sorted != new_edges_sorted;

                    uint64_t edges_pruned = 0;
                    const bool record_undo =
                        !disable_existing_undo_record_for_point &&
                        enable_mvcc_ && undo_log_;
                    const auto undo_record_start =
                        std::chrono::steady_clock::now();
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
                                long batch_ts =
                                    global_watermark_.load(std::memory_order_acquire);
                                float dist_node_pruned = fstdistfunc_(
                                    getDataByInternalId(neighbor),
                                    getDataByInternalId(old_edge),
                                    dist_func_param_);
                                float dist_cause_pruned = fstdistfunc_(
                                    getDataByInternalId(cur_c),
                                    getDataByInternalId(old_edge),
                                    dist_func_param_);
                                undo_log_->recordPrune(
                                    neighbor, old_edge, cur_c, batch_ts,
                                    dist_node_pruned, dist_cause_pruned, nullptr);
                                metric_graph_existing_neighbor_pruned_edges_recorded
                                    .fetch_add(1, std::memory_order_relaxed);
                                markNodeModified(neighbor);

                                long horizon =
                                    gc_horizon_.load(std::memory_order_relaxed);
                                if (horizon > 0) {
                                    undo_log_->advanceGcStart(neighbor, horizon);
                                }
                            }
                        }
                    }
                    metric_graph_existing_neighbor_edges_pruned.fetch_add(
                        edges_pruned, std::memory_order_relaxed);
                    if (record_undo) {
                        const auto undo_record_end =
                            std::chrono::steady_clock::now();
                        metric_graph_existing_neighbor_undo_record_ns.fetch_add(
                            static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    undo_record_end - undo_record_start)
                                    .count()),
                            std::memory_order_relaxed);
                    }
                    undo_phase.reset();

                    const auto rewrite_start = std::chrono::steady_clock::now();
                    ActivePhaseGuard rewrite_phase(active_phase_rewrite_);
                    int indx = 0;
                    while (!candidates.empty()) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        ++indx;
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
                        static_cast<uint64_t>(indx),
                        std::memory_order_relaxed);
                    ++local_link_updates;
                    if (linkset_changed) {
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
        }
        const auto existing_neighbor_update_loop_end =
            std::chrono::steady_clock::now();
        metric_graph_existing_neighbor_update_loop_ns.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    existing_neighbor_update_loop_end -
                    existing_neighbor_update_loop_start)
                    .count()),
            std::memory_order_relaxed);
        metric_graph_link_updates.fetch_add(
            local_link_updates, std::memory_order_relaxed);
        return true;
    }

    size_t drainTailIssueDeferredRepairs(size_t budget) {
        if (budget == 0 || tail_issue_tls.writer_repair_drain_active) {
            return 0;
        }
        std::unique_lock<std::mutex> drain_lock(
            tail_issue_deferred_repair_drain_lock_, std::try_to_lock);
        if (!drain_lock.owns_lock()) {
            return 0;
        }
        tail_issue_tls.writer_repair_drain_active = true;
        size_t drained = 0;
        while (drained < budget) {
            tableint deferred_id = static_cast<tableint>(-1);
            if (!tryPopTailIssueDeferredRepair(&deferred_id)) {
                break;
            }
            if (drained == 0) {
                tail_issue_writer_repair_drain_runs_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (deferred_id < 0 || deferred_id >= cur_element_count ||
                isMarkedDeleted(deferred_id)) {
                continue;
            }
            if (applyTailIssueReverseRepairOnly(deferred_id)) {
                tail_issue_writer_repair_drained_.fetch_add(
                    1, std::memory_order_relaxed);
                ++drained;
            }
        }
        tail_issue_tls.writer_repair_drain_active = false;
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
                    tableint trace_point_id = std::numeric_limits<tableint>::max(),
                    size_t ef_construction_override = 0) {
        const uint64_t base_trace_start =
            insertPhaseTraceEnabled() ? monotonicRawNs() : 0;
        const auto base_search_start = std::chrono::steady_clock::now();
        const size_t search_ef_construction =
            ef_construction_override > 0
                ? std::max<size_t>(M_, ef_construction_override)
                : ef_construction_;
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
            dist_t dist =
                distanceForCurrentRole(data_point, getDataByInternalId(ep_id));
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
                top_candidates.size() == search_ef_construction) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;
            ++local_expansions;
            tailIssueReaderExpansionCheckpoint();

            const bool use_shared_scan_lock = shouldUseWriterSharedScanLock();
            std::unique_lock<std::shared_mutex> lock;
            std::shared_lock<std::shared_mutex> shared_scan_lock;
            if (use_shared_scan_lock) {
                shared_scan_lock =
                    std::shared_lock<std::shared_mutex>(link_list_locks_[curNodeNum]);
            } else {
                lock = std::unique_lock<std::shared_mutex>(
                    link_list_locks_[curNodeNum]);
            }

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
            const size_t prefetch_stride = prefetchVectorStride();
	            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
	            _mm_prefetch((char *)(visited_array + *(data + 1) + 64),
	                         _MM_HINT_T0);
	            if (effective_size > 0) prefetchVectorPayload(*datal);
	            if (effective_size > prefetch_stride) {
	                prefetchVectorPayloadLookahead(*(datal + prefetch_stride));
	            }
#endif

	            for (size_t j = 0; j < effective_size; j++) {
	                tableint candidate_id = *(datal + j);
#ifdef USE_SSE
	                const size_t prefetch_j = j + prefetch_stride;
	                if (prefetch_j < effective_size) {
	                    _mm_prefetch((char *)(visited_array + *(datal + prefetch_j)),
	                                 _MM_HINT_T0);
                    if (!maybeStageReaderLookahead(*(datal + prefetch_j))) {
                        prefetchVectorPayloadLookahead(*(datal + prefetch_j));
                    }
	                }
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));
                maybePrefetchReaderCurrent(currObj1);

                const bool exact_required =
                    top_candidates.size() < search_ef_construction;
                dist_t dist1 = distanceForCurrentRoleBounded(
                    data_point, currObj1, lowerBound, exact_required);
                issueDeferredReaderLookahead();
                ++local_dist_comps;
                if (top_candidates.size() < search_ef_construction ||
                    lowerBound > dist1) {
                    const bool skip_frontier_expansion =
                        shouldSkipWriterFrontierExpansion(
                            dist1, lowerBound, exact_required);
                    if (!skip_frontier_expansion) {
                        candidateSet.emplace(-dist1, candidate_id);
                    }
#ifdef USE_SSE
                    if (!skip_frontier_expansion) {
                        prefetchVectorPayloadLookahead(candidateSet.top().second);
                    }
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > search_ef_construction)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
            // Release node lock before yield — safe point
            if (use_shared_scan_lock) {
                shared_scan_lock.unlock();
            } else {
                lock.unlock();
            }
            if (use_yield && fine_grain_yield_active_.load(std::memory_order_relaxed) && fine_grain_yield_cb_) {
                last_fine_grain_yield_ctx_.expanded_neighbors = static_cast<int>(size);
                last_fine_grain_yield_ctx_.candidate_queue_size = static_cast<int>(candidateSet.size());
                last_fine_grain_yield_ctx_.top_candidate_size = static_cast<int>(top_candidates.size());
                last_fine_grain_yield_ctx_.lower_bound = lowerBound;
                last_fine_grain_yield_ctx_.best_frontier_dist =
                    candidateSet.empty() ? lowerBound : -candidateSet.top().first;
                last_fine_grain_yield_ctx_.top_candidates_full =
                    top_candidates.size() == search_ef_construction;
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
        // Instruction-level coherence mitigation (read once): prefetch this many
        // 64B lines of an enqueued candidate's ADJACENCY (the HitM'd data) so the
        // cross-core transfer overlaps the beam's distance computations. 0 = off.
        static const int ti_adj_prefetch_lines = []{
            const char* e = std::getenv("ANNCHOR_TI_ADJ_PREFETCH");
            return e ? std::atoi(e) : 0;
        }();
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
        const size_t full_ef_runtime = ef;

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
        size_t effective_ef_runtime = ef;
        bool tail_issue_reader_clamped_this_search = false;
        bool tail_issue_reader_expansion_stop_recorded = false;
        uint32_t tail_issue_reader_expansions_after_urgent = 0;
        uint32_t tail_issue_reader_frontier_continue_left =
            tail_issue_reader_frontier_continue_budget_;
        uint32_t tail_issue_reader_stagnant_hops = 0;
        dist_t tail_issue_reader_prev_lower_bound = lowerBound;

        auto refresh_tail_issue_reader_ef = [&]() {
            if (tail_issue_reader_stag_clamp_hops_ > 0) {
                const size_t prev_ef = effective_ef_runtime;
                const size_t proposed_ef = tailIssueReaderStagnationEfLimit(
                    effective_ef_runtime, tail_issue_reader_stagnant_hops,
                    tail_issue_reader_expansions_after_urgent);
                if (proposed_ef < prev_ef &&
                    tailIssueReaderClampConfidence(
                        top_candidates, proposed_ef)) {
                    effective_ef_runtime = proposed_ef;
                }
                if (effective_ef_runtime < prev_ef &&
                    !tail_issue_reader_clamped_this_search) {
                    tail_issue_reader_clamped_this_search = true;
                    tail_issue_reader_beam_clamps_.fetch_add(
                        1, std::memory_order_relaxed);
                    tail_issue_reader_beam_trimmed_.fetch_add(
                        prev_ef - effective_ef_runtime,
                        std::memory_order_relaxed);
                }
                return;
            }
            if (tail_issue_reader_limit_mode_ != 0) {
                return;
            }
            const size_t prev_ef = effective_ef_runtime;
            const size_t proposed_ef =
                tailIssueReaderEfLimit(effective_ef_runtime);
            if (proposed_ef < prev_ef &&
                tailIssueReaderClampConfidence(
                    top_candidates, proposed_ef)) {
                effective_ef_runtime = proposed_ef;
            }
            if (effective_ef_runtime < prev_ef &&
                !tail_issue_reader_clamped_this_search) {
                tail_issue_reader_clamped_this_search = true;
                tail_issue_reader_beam_clamps_.fetch_add(
                    1, std::memory_order_relaxed);
                tail_issue_reader_beam_trimmed_.fetch_add(
                    prev_ef - effective_ef_runtime, std::memory_order_relaxed);
            }
        };
        auto tail_issue_reader_margin_stop = [&](dist_t candidate_dist) {
            if (tail_issue_reader_stop_alpha_permille_ >= 1000 ||
                top_candidates.size() != effective_ef_runtime ||
                !maybeActivateReaderTailIssue(monotonicRawNs())) {
                return false;
            }
            const double stop_dist =
                static_cast<double>(lowerBound) *
                static_cast<double>(tail_issue_reader_stop_alpha_permille_) /
                1000.0;
            if (static_cast<double>(candidate_dist) > stop_dist) {
                tail_issue_reader_margin_stops_.fetch_add(
                    1, std::memory_order_relaxed);
                return true;
            }
            return false;
        };
        auto tail_issue_reader_continue_frontier = [&]() {
            if (tail_issue_reader_frontier_continue_budget_ == 0 ||
                !tail_issue_tls.reader_clamped ||
                tail_issue_tls.reader_force_full_ef ||
                !tail_issue_tls.urgent) {
                return false;
            }
            tail_issue_reader_frontier_continue_checks_.fetch_add(
                1, std::memory_order_relaxed);
            if (tail_issue_reader_frontier_continue_left == 0) {
                return false;
            }
            --tail_issue_reader_frontier_continue_left;
            tail_issue_reader_frontier_continue_grants_.fetch_add(
                1, std::memory_order_relaxed);
            return true;
        };

        while (!candidate_set.empty()) {
            refresh_tail_issue_reader_ef();
            if (tail_issue_reader_expansion_budget_ > 0) {
                const bool urgent =
                    maybeActivateReaderTailIssue(monotonicRawNs());
                if (urgent &&
                    tail_issue_reader_expansions_after_urgent >=
                        tail_issue_reader_expansion_budget_) {
                    if (!tail_issue_reader_expansion_stop_recorded) {
                        tail_issue_reader_expansion_stop_recorded = true;
                        tail_issue_reader_expansion_stops_.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                    break;
                }
            }
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (effective_bare) {
                flag_stop_search = candidate_dist > lowerBound;
                if (!flag_stop_search) {
                    flag_stop_search =
                        tail_issue_reader_margin_stop(candidate_dist);
                }
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(
                        candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound &&
                                       top_candidates.size() == effective_ef_runtime;
                    if (!flag_stop_search) {
                        flag_stop_search =
                            tail_issue_reader_margin_stop(candidate_dist);
                    }
                }
            }
            if (!flag_stop_search) {
                flag_stop_search = tailIssueReaderFrontierStop(
                    top_candidates, candidate_dist,
                    tail_issue_reader_stagnant_hops,
                    tail_issue_reader_expansions_after_urgent);
            }
            if (flag_stop_search) {
                if (tailIssueReaderContinueNeeded(top_candidates, candidate_dist,
                                                  full_ef_runtime)) {
                    effective_ef_runtime = full_ef_runtime;
                    flag_stop_search = false;
                } else if (tail_issue_reader_continue_frontier()) {
                    flag_stop_search = false;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();
            if (tail_issue_tls.urgent &&
                (tail_issue_reader_expansion_budget_ > 0 ||
                 tail_issue_reader_stag_clamp_hops_ > 0 ||
                 tail_issue_reader_frontier_ef_ > 0)) {
                ++tail_issue_reader_expansions_after_urgent;
            }
            if (work_stats) {
                work_stats->candidate_pops++;
                work_stats->level0_expansions++;
            }
            m2_expand_step++;
            tailIssueReaderExpansionCheckpoint();
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
            const size_t prefetch_stride = prefetchVectorStride();
		            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
	            _mm_prefetch((char *)(visited_array + *(data + 1) + 64),
	                         _MM_HINT_T0);
	            if (size > 0) {
	                prefetchVectorPayloadFromLevel0Base(*(data + 1));
	            }
            if (size > prefetch_stride) {
                prefetchVectorPayloadFromLevel0BaseLookahead(*(data + 1 + prefetch_stride));
            }
            if (tail_issue_reader_window_extra_ > 0 && tail_issue_tls.urgent) {
                const size_t window_end =
                    std::min<size_t>(size, prefetch_stride + 1 + tail_issue_reader_window_extra_);
                for (size_t prefetch_j = prefetch_stride + 2; prefetch_j <= window_end; ++prefetch_j) {
                    maybePrefetchReaderWindow(*(data + prefetch_j));
                }
            }
            _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

		            for (size_t j = 1; j <= size; j++) {
		                int candidate_id = *(data + j);
#ifdef USE_SSE
                const size_t prefetch_j = j + prefetch_stride;
                if (prefetch_j <= size) {
                    _mm_prefetch((char *)(visited_array + *(data + prefetch_j)),
                                 _MM_HINT_T0);
                    if (!maybeStageReaderLookahead(*(data + prefetch_j))) {
                        prefetchVectorPayloadFromLevel0BaseLookahead(*(data + prefetch_j));
                    }
                }
                if (tail_issue_reader_batch_size_ > 0 &&
                    (tail_issue_tls.urgent ||
                     maybeActivateReaderTailIssue(monotonicRawNs())) &&
                    ((j - 1) % tail_issue_reader_batch_size_ == 0)) {
                    const size_t batch_end =
                        std::min<size_t>(size, j + tail_issue_reader_batch_size_ - 1);
                    for (size_t batch_j = j; batch_j <= batch_end; ++batch_j) {
                        const tableint batch_id = *(data + batch_j);
                        if (visited_array[batch_id] != visited_array_tag) {
                            prefetchVectorPayloadFromLevel0Base(batch_id);
                            tail_issue_reader_batch_prefetches_.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                    }
                }
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;
                    if (work_stats) work_stats->visited_nodes++;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    maybePrefetchReaderCurrent(currObj1);
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
                    const bool exact_required =
                        top_candidates.size() < effective_ef_runtime;
                    dist_t dist = readerDistanceForTail(
                        data_point, currObj1, lowerBound, exact_required);
                    issueDeferredReaderLookaheadFromLevel0Base();
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
                            top_candidates.size() < effective_ef_runtime ||
                            lowerBound > dist;
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
#ifdef USE_SSE
                        if (ti_adj_prefetch_lines > 0) {
                            const char* adj = reinterpret_cast<const char*>(
                                get_linklist0(candidate_id));
                            for (int pl = 0; pl < ti_adj_prefetch_lines; ++pl)
                                _mm_prefetch(adj + pl * 64, _MM_HINT_T0);
                        }
#endif
                        if (enforce_view_limit && !candidate_visible) {
                            metric_search_invisible_candidate_enqueues.fetch_add(
                                1, std::memory_order_relaxed);
                            if (work_stats) work_stats->invisible_candidate_enqueues++;
                        }
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ +
                                         candidate_set.top().second *
                                             size_data_per_element_ +
                                         offsetLevel0_,
                                     _MM_HINT_T0);
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
                            flag_remove_extra =
                                top_candidates.size() > effective_ef_runtime;
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
                                flag_remove_extra =
                                    top_candidates.size() > effective_ef_runtime;
                            }
                        }

                        if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                    }
                }
            }
            if (tail_issue_tls.urgent &&
                (tail_issue_reader_stag_clamp_hops_ > 0 ||
                 tail_issue_reader_frontier_ef_ > 0)) {
                if (lowerBound < tail_issue_reader_prev_lower_bound) {
                    tail_issue_reader_prev_lower_bound = lowerBound;
                    tail_issue_reader_stagnant_hops = 0;
                } else if (tail_issue_reader_stagnant_hops <
                           std::numeric_limits<uint32_t>::max()) {
                    ++tail_issue_reader_stagnant_hops;
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

                            bool flag_consider =
                                top_candidates.size() < effective_ef_runtime ||
                                lowerBound > rec_dist;
                            if (flag_consider) {
                                candidate_set.emplace(-rec_dist, rec_id);
                                bool rec_allowed = effective_bare ||
                                    (!isMarkedDeleted(rec_id) &&
                                     ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(rec_id))));
                                if (rec_allowed && is_visible(rec_id)) {
                                    top_candidates.emplace(rec_dist, rec_id);
                                    m2_useful_edges_.fetch_add(1, std::memory_order_relaxed);
                                }
                                while (top_candidates.size() >
                                       effective_ef_runtime) {
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
        if (work_stats) {
            work_stats->phase_existing_update_hit = phase_existing_update_hit_query;
            work_stats->phase_link_critical_hit = phase_link_critical_hit_query;
            work_stats->phase_load_scan_hit = phase_load_scan_hit_query;
            work_stats->phase_append_hit = phase_append_hit_query;
            work_stats->phase_prune_hit = phase_prune_hit_query;
            work_stats->phase_undo_record_hit = phase_undo_record_hit_query;
            work_stats->phase_rewrite_hit = phase_rewrite_hit_query;
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

        tailIssueReaderLocalRefine(top_candidates, candidate_set, data_point,
                                   effective_ef_runtime, full_ef_runtime,
                                   visited_array, visited_array_tag,
                                   effective_bare, isIdAllowed, view_limit,
                                   lowerBound);

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

    bool tryTailIssueOfflockOldNeighborPrune(
        tableint cur_c, tableint neighbor, size_t mcurmax,
        uint64_t &local_link_updates,
        bool disable_existing_undo_record_for_point) {
        if (!tail_issue_writer_policy_enabled_ ||
            tail_issue_tls.role != TailIssueRole::Writer ||
            tail_issue_writer_offlock_prune_budget_ == 0 ||
            tail_issue_writer_offlock_prune_global_.load(
                std::memory_order_acquire) == 0) {
            return false;
        }
        const uint64_t now_ns = monotonicRawNs();
        if (!tailIssueActive(now_ns)) {
            tail_issue_writer_offlock_prune_global_.store(
                0, std::memory_order_release);
            return false;
        }

        std::vector<tableint> old_edges;
        uint64_t version_before = 0;
        {
            const auto load_scan_start = std::chrono::steady_clock::now();
            ActivePhaseGuard load_scan_phase(active_phase_load_scan_);
            std::shared_lock<std::shared_mutex> scan_lock(
                link_list_locks_[neighbor]);
            linklistsizeint *ll_other = get_linklist0(neighbor);
            const size_t sz_link_list_other = getListCount(ll_other);
            if (sz_link_list_other > mcurmax) {
                throw std::runtime_error("Bad value of sz_link_list_other");
            }
            if (neighbor == cur_c) {
                throw std::runtime_error(
                    "Trying to connect an element to itself");
            }
            tableint *data = (tableint *)(ll_other + 1);
            bool is_cur_c_present = false;
            for (size_t j = 0; j < sz_link_list_other; ++j) {
                if (data[j] == cur_c) {
                    is_cur_c_present = true;
                    break;
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
            if (is_cur_c_present || sz_link_list_other < mcurmax) {
                tail_issue_writer_offlock_prune_skipped_.fetch_add(
                    1, std::memory_order_relaxed);
                return false;
            }
            old_edges.assign(data, data + sz_link_list_other);
            version_before = getNodeLinkVersion(neighbor);
        }

        if (!consumeTailIssueWriterOfflockPruneToken()) {
            return false;
        }
        tail_issue_writer_offlock_prune_attempts_.fetch_add(
            1, std::memory_order_relaxed);
        metric_graph_existing_neighbor_prunes.fetch_add(
            1, std::memory_order_relaxed);
        metric_graph_existing_neighbor_prune_candidates.fetch_add(
            old_edges.size() + 1, std::memory_order_relaxed);

        const uint64_t prune_trace_start =
            insertPhaseTraceEnabled() ? monotonicRawNs() : 0;
        const auto prune_heuristic_start = std::chrono::steady_clock::now();
        ActivePhaseGuard prune_phase(active_phase_prune_);
        uint64_t prune_dist_comps = 0;
        dist_t d_max = fstdistfunc_(
            getDataByInternalId(cur_c), getDataByInternalId(neighbor),
            dist_func_param_);
        ++prune_dist_comps;
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            candidates;
        candidates.emplace(d_max, cur_c);
        for (tableint old_edge : old_edges) {
            candidates.emplace(
                fstdistfunc_(getDataByInternalId(old_edge),
                             getDataByInternalId(neighbor),
                             dist_func_param_),
                old_edge);
            ++prune_dist_comps;
        }
        uint64_t prune_heuristic_dist_comps = 0;
        getNeighborsByHeuristic(candidates, mcurmax,
                                &prune_heuristic_dist_comps);
        prune_dist_comps += prune_heuristic_dist_comps;

        std::vector<tableint> new_edges;
        new_edges.reserve(candidates.size());
        auto candidates_copy = candidates;
        while (!candidates_copy.empty()) {
            new_edges.push_back(candidates_copy.top().second);
            candidates_copy.pop();
        }

        std::vector<tableint> old_edges_sorted = old_edges;
        std::vector<tableint> new_edges_sorted = new_edges;
        std::sort(old_edges_sorted.begin(), old_edges_sorted.end());
        std::sort(new_edges_sorted.begin(), new_edges_sorted.end());
        const bool linkset_changed = old_edges_sorted != new_edges_sorted;

        struct UndoRecord {
            tableint old_edge;
            float dist_node_pruned;
            float dist_cause_pruned;
        };
        std::vector<UndoRecord> undo_records;
        uint64_t edges_pruned = 0;
        const bool record_undo =
            !disable_existing_undo_record_for_point && enable_mvcc_ && undo_log_;
        std::unique_ptr<ActivePhaseGuard> undo_dist_phase;
        const auto undo_dist_start = std::chrono::steady_clock::now();
        if (record_undo) {
            undo_dist_phase = std::make_unique<ActivePhaseGuard>(
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
                    undo_records.push_back(
                        {old_edge,
                         static_cast<float>(fstdistfunc_(
                             getDataByInternalId(neighbor),
                             getDataByInternalId(old_edge),
                             dist_func_param_)),
                         static_cast<float>(fstdistfunc_(
                             getDataByInternalId(cur_c),
                             getDataByInternalId(old_edge),
                             dist_func_param_))});
                }
            }
        }
        undo_dist_phase.reset();
        const auto prune_heuristic_end = std::chrono::steady_clock::now();
        metric_graph_existing_neighbor_prune_ns.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    prune_heuristic_end - prune_heuristic_start)
                    .count()),
            std::memory_order_relaxed);
        metric_graph_existing_neighbor_prune_dist_comps.fetch_add(
            prune_dist_comps, std::memory_order_relaxed);
        if (record_undo) {
            metric_graph_existing_neighbor_undo_record_ns.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        prune_heuristic_end - undo_dist_start)
                        .count()),
                std::memory_order_relaxed);
        }
        const uint64_t prune_trace_end = prune_trace_start ? monotonicRawNs() : 0;
        traceInsertPhase("existing_neighbor_prune_offlock", prune_trace_start,
                         prune_trace_end, cur_c, 0, neighbor,
                         old_edges.size(), 0, old_edges.size() + 1,
                         prune_dist_comps, 0, 0);

        tls_in_link_critical_section = true;
        const auto wait_start = std::chrono::steady_clock::now();
        std::unique_lock<std::shared_mutex> lock(link_list_locks_[neighbor],
                                                 std::defer_lock);
        lock.lock();
        const auto critical_start = std::chrono::steady_clock::now();
        metric_graph_unique_lock_wait_ns.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    critical_start - wait_start)
                    .count()),
            std::memory_order_relaxed);
        ActivePhaseGuard link_critical_phase(active_phase_link_critical_);

        linklistsizeint *ll_other = get_linklist0(neighbor);
        const size_t sz_link_list_other = getListCount(ll_other);
        tableint *data = (tableint *)(ll_other + 1);
        bool validate_ok =
            sz_link_list_other == old_edges.size() &&
            getNodeLinkVersion(neighbor) == version_before;
        if (validate_ok) {
            for (size_t j = 0; j < old_edges.size(); ++j) {
                if (data[j] != old_edges[j]) {
                    validate_ok = false;
                    break;
                }
            }
        }
        if (!validate_ok) {
            tail_issue_writer_offlock_prune_validate_fail_.fetch_add(
                1, std::memory_order_relaxed);
            const auto critical_end = std::chrono::steady_clock::now();
            metric_graph_link_critical_ns.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        critical_end - critical_start)
                        .count()),
                std::memory_order_relaxed);
            tls_in_link_critical_section = false;
            return false;
        }

        const bool did_record_undo = record_undo && !undo_records.empty();
        const uint64_t undo_trace_start =
            (did_record_undo && insertPhaseTraceEnabled()) ? monotonicRawNs() : 0;
        const auto undo_record_start = std::chrono::steady_clock::now();
        std::unique_ptr<ActivePhaseGuard> undo_phase;
        if (did_record_undo) {
            undo_phase = std::make_unique<ActivePhaseGuard>(
                active_phase_undo_record_);
        }
        const long batch_ts = global_watermark_.load(std::memory_order_acquire);
        for (const UndoRecord &record : undo_records) {
            undo_log_->recordPrune(neighbor, record.old_edge, cur_c, batch_ts,
                                   record.dist_node_pruned,
                                   record.dist_cause_pruned, nullptr);
            metric_graph_existing_neighbor_pruned_edges_recorded.fetch_add(
                1, std::memory_order_relaxed);
            markNodeModified(neighbor);

            long horizon = gc_horizon_.load(std::memory_order_relaxed);
            if (horizon > 0) {
                undo_log_->advanceGcStart(neighbor, horizon);
            }
        }
        metric_graph_existing_neighbor_edges_pruned.fetch_add(
            edges_pruned, std::memory_order_relaxed);
        if (did_record_undo) {
            const auto undo_record_end = std::chrono::steady_clock::now();
            metric_graph_existing_neighbor_undo_record_ns.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        undo_record_end - undo_record_start)
                        .count()),
                std::memory_order_relaxed);
            const uint64_t undo_trace_end =
                undo_trace_start ? monotonicRawNs() : 0;
            traceInsertPhase("existing_neighbor_undo_record_offlock",
                             undo_trace_start, undo_trace_end, cur_c, 0,
                             neighbor, old_edges.size(), 0, 0, 0, 0,
                             edges_pruned);
        }
        undo_phase.reset();

        const uint64_t rewrite_trace_start =
            insertPhaseTraceEnabled() ? monotonicRawNs() : 0;
        const auto rewrite_start = std::chrono::steady_clock::now();
        ActivePhaseGuard rewrite_phase(active_phase_rewrite_);
        int indx = 0;
        for (tableint new_edge : new_edges) {
            data[indx] = new_edge;
            ++indx;
            ++local_link_updates;
        }
        setListCount(ll_other, indx);
        tailIssueWriterDemoteLines(
            ll_other,
            (sizeof(linklistsizeint) +
             static_cast<size_t>(indx) * sizeof(tableint) + 63) /
                64);
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
        traceInsertPhase("existing_neighbor_rewrite_offlock",
                         rewrite_trace_start, rewrite_trace_end, cur_c, 0,
                         neighbor, old_edges.size(), 0, 0, 0,
                         static_cast<uint64_t>(indx), edges_pruned);
        ++local_link_updates;
        if (linkset_changed) {
            bumpNodeLinkVersion(neighbor);
        }

        const auto critical_end = std::chrono::steady_clock::now();
        metric_graph_link_critical_ns.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    critical_end - critical_start)
                    .count()),
            std::memory_order_relaxed);
        tls_in_link_critical_section = false;
        tail_issue_writer_offlock_prune_success_.fetch_add(
            1, std::memory_order_relaxed);
        return true;
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

        const bool tail_repair_deferred =
            level == 0 && !isUpdate && consumeTailIssueWriterRepairDeferToken() &&
            tryEnqueueTailIssueDeferredRepair(cur_c, selectedNeighbors.size());
        if (tail_repair_deferred) {
            tail_issue_tls.writer_repair_deferred_pending = true;
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
            if (level == 0 && !isUpdate &&
                !existing_update_read_scan_only_for_point &&
                tryTailIssueOfflockOldNeighborPrune(
                    cur_c, neighbor, Mcurmax, local_link_updates,
                    disable_existing_undo_record_for_point)) {
                continue;
            }
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
                            tailIssueWriterDemoteLines(
                                ll_other,
                                (sizeof(linklistsizeint) +
                                 (sz_link_list_other + 1) * sizeof(tableint) +
                                 63) /
                                    64);
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
                    dist_t worst_old_dist = static_cast<dist_t>(0);
	                    // Heuristic:
                    std::priority_queue<
                        std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
                        candidates;
                    candidates.emplace(d_max, cur_c);

	                    for (size_t j = 0; j < sz_link_list_other; j++) {
                            const dist_t old_dist = fstdistfunc_(
                                getDataByInternalId(data[j]),
                                getDataByInternalId(neighbor),
                                dist_func_param_);
                            if (old_dist > worst_old_dist) {
                                worst_old_dist = old_dist;
                            }
	                        candidates.emplace(old_dist, data[j]);
                        ++prune_dist_comps;
	                    }

                    if (level == 0 && !isUpdate &&
                        shouldSkipTailIssueLowvalueReverseLink(d_max, worst_old_dist)) {
                        metric_graph_existing_neighbor_prune_dist_comps.fetch_add(
                            prune_dist_comps, std::memory_order_relaxed);
                        const auto prune_heuristic_end = std::chrono::steady_clock::now();
                        metric_graph_existing_neighbor_prune_ns.fetch_add(
                            static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    prune_heuristic_end - prune_heuristic_start)
                                    .count()),
                            std::memory_order_relaxed);
                        const uint64_t prune_trace_end =
                            prune_trace_start ? monotonicRawNs() : 0;
                        traceInsertPhase("existing_neighbor_lowvalue_skip",
                                         prune_trace_start, prune_trace_end,
                                         cur_c, level, neighbor,
                                         sz_link_list_other, 0,
                                         sz_link_list_other + 1,
                                         prune_dist_comps, 0, 0);
                        const auto critical_end = std::chrono::steady_clock::now();
                        metric_graph_link_critical_ns.fetch_add(
                            static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    critical_end - critical_start)
                                    .count()),
                            std::memory_order_relaxed);
                        tls_in_link_critical_section = false;
                        if (rewrite_active_diag_enabled_ && node_rewrite_active_ &&
                            static_cast<size_t>(active_rewrite_node) < max_elements_) {
                            node_rewrite_active_[active_rewrite_node].store(
                                0, std::memory_order_release);
                        }
                        continue;
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
                        tailIssueWriterDemoteLines(
                            ll_other,
                            (sizeof(linklistsizeint) +
                             static_cast<size_t>(indx) * sizeof(tableint) +
                             63) /
                                64);
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
	                    const size_t prefetch_stride = prefetchVectorStride();
	                    if (size > 0) prefetchVectorPayload(*datal);
	                    if (static_cast<size_t>(size) > prefetch_stride) {
	                        prefetchVectorPayloadLookahead(*(datal + prefetch_stride));
	                    }
#endif
	                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
	                        const int prefetch_i =
	                            i + static_cast<int>(prefetch_stride);
	                        if (prefetch_i < size) {
	                            prefetchVectorPayloadLookahead(*(datal + prefetch_i));
	                        }
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
        TailIssueScope tail_issue_scope(TailIssueRole::Writer, tail_issue_writer_policy_enabled_);
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
        if (curlevel > 0 && consumeTailIssueWriterLevel0Token()) {
            curlevel = 0;
            tail_issue_writer_level0_forced_.fetch_add(
                1, std::memory_order_relaxed);
        }

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
                dist_t curdist =
                    distanceForCurrentRole(data_point, getDataByInternalId(currObj));
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
		                            dist_t d =
                                        distanceForCurrentRoleBounded(
                                            data_point, getDataByInternalId(cand),
                                            curdist, false);
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
                    const size_t writer_ef = tailIssueWriterEffectiveEfConstruction();
	                    std::priority_queue<std::pair<dist_t, tableint>,
	                                        std::vector<std::pair<dist_t, tableint>>,
	                                        CompareByFirst>
	                        tc = searchBaseLayer(currObj, data_point, level,
	                                             fine_grain_yield_cb_ ? true : false,
                                                 cur_c, writer_ef);
                    if (epDeleted) {
                        tc.emplace(
                            fstdistfunc_(data_point,
                                         getDataByInternalId(enterpoint_copy),
                                         dist_func_param_),
                            enterpoint_copy);
                        if (tc.size() > writer_ef)
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
        if (lock_el.owns_lock()) {
            lock_el.unlock();
        }
        if (!defer_watermark) {
            markCompleted(cur_c, -1);
            if (enqueue_deferred_repair) {
                tryEnqueueMidInsertDeferredRepair(cur_c);
            }
            const bool tail_repair_deferred_this_insert =
                tail_issue_tls.writer_repair_deferred_pending;
            if (tail_repair_deferred_this_insert) {
                tail_issue_tls.writer_repair_deferred_pending = false;
            }
            if (!tail_repair_deferred_this_insert &&
                tail_issue_writer_repair_drain_budget_ > 0) {
                drainTailIssueDeferredRepairs(
                    static_cast<size_t>(tail_issue_writer_repair_drain_budget_));
            }
            if (allow_deferred_drain) {
                drainMidInsertDeferredRepairs(
                    static_cast<size_t>(mid_insert_harm_defer_drain_budget_));
            }
        } else if (enqueue_deferred_repair) {
            markMidInsertDeferredPending(cur_c, true);
        } else if (tail_issue_tls.writer_repair_deferred_pending) {
            tail_issue_tls.writer_repair_deferred_pending = false;
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
        if (tail_issue_writer_repair_drain_budget_ > 0) {
            drainTailIssueDeferredRepairs(
                static_cast<size_t>(tail_issue_writer_repair_drain_budget_));
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
        TailIssueScope tail_issue_scope(TailIssueRole::Reader, tail_issue_reader_policy_enabled_);
        tail_issue_tls.reader_k = k;
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
        const size_t base_ef_requested = std::max(ef_, k);
        const size_t base_ef =
            tail_issue_reader_continue_enabled_
                ? base_ef_requested
                : tailIssueReaderEffectiveEf(base_ef_requested, k);
        if (bare_bone_search) {
            top_candidates =
                searchBaseLayerST<true>(currObj, query_data, base_ef,
                                        isIdAllowed, nullptr, base_view_limit,
                                        work_stats);
        } else {
            top_candidates =
                searchBaseLayerST<false>(currObj, query_data, base_ef,
                                         isIdAllowed, nullptr, base_view_limit,
                                         work_stats);
        }
        if (work_stats) {
            work_stats->base_search_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - base_start)
                    .count());
        }
        if (tailIssueReaderRetryNeeded(top_candidates, k)) {
            tail_issue_reader_retry_runs_.fetch_add(1, std::memory_order_relaxed);
            const auto retry_start = std::chrono::steady_clock::now();
            TailIssueScope retry_scope(TailIssueRole::None, true);
            if (bare_bone_search) {
                top_candidates =
                    searchBaseLayerST<true>(currObj, query_data, base_ef_requested,
                                            isIdAllowed, nullptr, base_view_limit,
                                            work_stats);
            } else {
                top_candidates =
                    searchBaseLayerST<false>(currObj, query_data, base_ef_requested,
                                             isIdAllowed, nullptr, base_view_limit,
                                             work_stats);
            }
            if (work_stats) {
                work_stats->base_search_ns += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - retry_start)
                        .count());
            }
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
