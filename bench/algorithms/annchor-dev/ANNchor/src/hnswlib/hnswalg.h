#pragma once

#include <assert.h>
#include <stdlib.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <future>
#include <limits>
#include <list>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <shared_mutex>
#include <thread>
#include <unordered_set>
#include <vector>

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
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

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

    bool allow_replace_deleted_ = false;

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint>
        deleted_elements;  // contains internal ids of deleted elements

    std::unique_ptr<std::atomic<bool>[]> insertion_complete_;
    std::atomic<long> global_watermark_{-1};

    std::unique_ptr<ThreadPool> thread_pool_;

    // MVCC Mechanism 1 components
    bool enable_mvcc_{true};
    std::unique_ptr<UndoLogManager> undo_log_;
    mutable SnapshotTracker snapshot_tracker_;
    std::atomic<long> gc_horizon_{-1};
    std::atomic<long> gc_safe_snapshot_{-1};  
    bool enable_online_gc_{true};
    std::atomic<uint8_t>* node_modified_bitmap_{nullptr};

    // CNR (Counterfactual Neighborhood Repair) config
    bool enable_cnr_{false};
    float cnr_degree_threshold_{0.5f};  // trigger when visible_degree < threshold * maxM0_
    int cnr_max_recover_{4};            // max edges to recover per node per trigger
    int cnr_stagnation_hops_{3};        // beam stagnation detection window

    // CNR stats
    mutable std::atomic<long> cnr_triggered_{0};
    mutable std::atomic<long> cnr_recovered_edges_{0};
    mutable std::atomic<long> cnr_useful_edges_{0};
    mutable std::atomic<long> cnr_degree_deficit_triggers_{0};
    mutable std::atomic<long> cnr_stagnation_triggers_{0};

    // M2-v1: Risk-triggered dual-path self-healing
    bool enable_m2_dual_path_{false};
    int m2_risk_hops_{4};      // trigger when lowerBound stagnates for this many hops
    int m2_assist_budget_{2};  // max backup assists per query

    // M2 stats
    mutable std::atomic<long> m2_triggered_{0};
    mutable std::atomic<long> m2_backup_injected_{0};
    mutable std::atomic<long> m2_backup_useful_{0};
    mutable std::atomic<long> m2_budget_exhausted_{0};

    // Preempt experiment: configurable sleep at any stage, staleness measured at stage 2.
    // Stages: 0=after_init, 1=after_route, 2=after_search, 3=after_connect
    //
    // SleepHook: called at every stage, decides whether to sleep.
    //   args: (stage, node_id) → returns true if this node should be preempted
    using SleepHook = std::function<bool(int stage, tableint node_id)>;
    SleepHook sleep_hook_{nullptr};
    void setSleepHook(SleepHook hook) { sleep_hook_ = std::move(hook); }

    // Staleness report: always measured at stage 2, regardless of where sleep happened
    struct StalenessReport {
        tableint node_id;
        int level;
        std::vector<tableint> candidates_before;
        std::vector<tableint> candidates_after;
        std::vector<tableint> neighbors_before;
        std::vector<tableint> neighbors_after;
        int vicinity_churn;  // total global inserts during ALL sleeps for this node
        int sleep_stage;     // which stage this node slept at (-1 = none)
    };
    using StalenessCallback = std::function<void(const StalenessReport& rpt)>;
    StalenessCallback staleness_cb_{nullptr};
    void setStalenessCallback(StalenessCallback cb) { staleness_cb_ = std::move(cb); }

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
        global_watermark_.store(-1, std::memory_order_relaxed);
        
        // Clean up MVCC components
        if (node_modified_bitmap_) {
            delete[] node_modified_bitmap_;
            node_modified_bitmap_ = nullptr;
        }
        undo_log_.reset();
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

    void setEnableMvcc(bool enabled) { enable_mvcc_ = enabled; }
    bool isMvccEnabled() const { return enable_mvcc_; }
    
    // CNR setters
    void setEnableCnr(bool enable) { enable_cnr_ = enable; }
    bool isCnrEnabled() const { return enable_cnr_; }
    void setCnrDegreeThreshold(float t) { cnr_degree_threshold_ = t; }
    void setCnrMaxRecover(int m) { cnr_max_recover_ = m; }
    void setCnrStagnationHops(int h) { cnr_stagnation_hops_ = h; }

    // M2 setters
    void setEnableM2DualPath(bool enable) { enable_m2_dual_path_ = enable; }
    bool isM2DualPathEnabled() const { return enable_m2_dual_path_; }
    void setM2RiskHops(int h) { m2_risk_hops_ = std::max(1, h); }
    void setM2AssistBudget(int b) { m2_assist_budget_ = std::max(0, b); }
    
    void initModifiedBitmap(size_t max_elements) {
        if (node_modified_bitmap_) {
            delete[] node_modified_bitmap_;
        }
        node_modified_bitmap_ = new std::atomic<uint8_t>[max_elements];
        for (size_t i = 0; i < max_elements; i++) {
            node_modified_bitmap_[i].store(0, std::memory_order_relaxed);
        }
    }
    
    void markNodeModified(tableint node_id) {
        if (node_id < max_elements_ && node_modified_bitmap_) {
            node_modified_bitmap_[node_id].store(1, std::memory_order_release);
        }
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
        size_t view_limit = std::numeric_limits<size_t>::max()) const {
        
        if (enable_mvcc_ && tls_snapshot_slot == SnapshotTracker::MAX_SLOTS) {
            tls_snapshot_slot = const_cast<SnapshotTracker&>(snapshot_tracker_).allocate_slot();
        }
        SnapshotTracker::Guard snap_guard(
            const_cast<SnapshotTracker*>(&snapshot_tracker_),
            tls_snapshot_slot, static_cast<long>(view_limit));
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
        size_t future_skip_hops = 0;

        auto is_visible = [&](tableint internal_id) -> bool {
            return !enforce_view_limit ||
                   static_cast<size_t>(internal_id) < view_limit;
        };

        dist_t lowerBound;
        bool ep_allowed =
            effective_bare ||
            (!isMarkedDeleted(ep_id) &&
             ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))));
        if (ep_allowed) {
            char *ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            if (is_visible(ep_id)) {
                top_candidates.emplace(dist, ep_id);
            } else if (enforce_view_limit) {
                future_skip_hops++;
            }
            if (!effective_bare && stop_condition && is_visible(ep_id)) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id),
                                                    ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        // CNR: beam stagnation tracking
        dist_t cnr_prev_lower_bound = lowerBound;
        int cnr_stagnation_count = 0;

        // M2-v2: writer-paid assist edge consumption uses per-node undo-log entries.

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
            int *data = (int *)get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint *)data);

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
                _mm_prefetch((char *)(visited_array + *(data + j + 1)),
                             _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ +
                                 (*(data + j + 1)) * size_data_per_element_ +
                                 offsetData_,
                             _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist =
                        fstdistfunc_(data_point, currObj1, dist_func_param_);

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
                        candidate_set.emplace(-dist, candidate_id);
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
                            if (is_visible(candidate_id)) {
                                top_candidates.emplace(dist, candidate_id);
                                if (!effective_bare && stop_condition) {
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

            // ── CNR: Counterfactual Neighborhood Repair ──────────────────
            if (enable_cnr_ && undo_log_) {
                // 1) Count visible neighbors for degree deficit detection
                size_t visible_count = 0;
                for (size_t j = 1; j <= size; j++) {
                    tableint nid = *(data + j);
                    if (is_visible(nid)) visible_count++;
                }

                // 2) Update stagnation tracking
                if (lowerBound < cnr_prev_lower_bound) {
                    cnr_stagnation_count = 0;
                    cnr_prev_lower_bound = lowerBound;
                } else {
                    cnr_stagnation_count++;
                }

                bool degree_deficit = visible_count < static_cast<size_t>(cnr_degree_threshold_ * maxM0_);
                bool beam_stagnation = cnr_stagnation_count >= cnr_stagnation_hops_;

                if (degree_deficit || beam_stagnation) {
                    // 3) Get recoverable edges from undo log
                    const uint32_t* rec_edges = nullptr;
                    const uint32_t* rec_causes = nullptr;
                    const float* rec_dist_node = nullptr;
                    const float* rec_dist_cause = nullptr;
                    size_t rec_count = undo_log_->getRecoverPointersEUL(
                        current_node_id, static_cast<uint32_t>(view_limit),
                        &rec_edges, &rec_causes, &rec_dist_node, &rec_dist_cause);

                    if (rec_count > 0) {
                        if (collect_metrics) {
                            cnr_triggered_.fetch_add(1, std::memory_order_relaxed);
                            if (degree_deficit)
                                cnr_degree_deficit_triggers_.fetch_add(1, std::memory_order_relaxed);
                            if (beam_stagnation)
                                cnr_stagnation_triggers_.fetch_add(1, std::memory_order_relaxed);
                        }

                        // 4) Coarse filter + exact distance for promising edges
                        int recovered = 0;
                        for (size_t ri = 0; ri < rec_count && recovered < cnr_max_recover_; ri++) {
                            tableint rec_id = rec_edges[ri];

                            // Skip invisible, already visited, or out-of-range
                            if (!is_visible(rec_id)) continue;
                            if (visited_array[rec_id] == visited_array_tag) continue;

                            // Triangle inequality coarse filter:
                            // dist(q, rec) >= |dist(q, owner) - dist(owner, rec)|
                            float d_owner_rec = rec_dist_node[ri];
                            dist_t lb_dist = std::abs(candidate_dist - d_owner_rec);
                            if (top_candidates.size() >= ef && lb_dist > lowerBound) continue;

                            // Exact distance
                            visited_array[rec_id] = visited_array_tag;
                            char *rec_data = getDataByInternalId(rec_id);
                            dist_t rec_dist = fstdistfunc_(data_point, rec_data, dist_func_param_);
                            recovered++;

                            if (collect_metrics) {
                                cnr_recovered_edges_.fetch_add(1, std::memory_order_relaxed);
                                metric_distance_computations++;
                            }

                            bool flag_consider = top_candidates.size() < ef || lowerBound > rec_dist;
                            if (flag_consider) {
                                candidate_set.emplace(-rec_dist, rec_id);

                                bool rec_allowed = effective_bare ||
                                    (!isMarkedDeleted(rec_id) &&
                                     ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(rec_id))));
                                if (rec_allowed && is_visible(rec_id)) {
                                    top_candidates.emplace(rec_dist, rec_id);
                                    if (collect_metrics)
                                        cnr_useful_edges_.fetch_add(1, std::memory_order_relaxed);
                                }

                                while (top_candidates.size() > ef) {
                                    top_candidates.pop();
                                }
                                if (!top_candidates.empty())
                                    lowerBound = top_candidates.top().first;
                            }
                        }
                        // Reset stagnation counter after repair attempt
                        if (recovered > 0) cnr_stagnation_count = 0;
                    }
                }
            }
            // ── end CNR ──────────────────────────────────────────────────

            // ── M2-v2: writer-paid assist edge consumption ───────────
            if (enable_m2_dual_path_ && m2_assist_budget_ > 0 &&
                undo_log_ && isNodeModified(current_node_id)) {
                const uint32_t* rec_edges = nullptr;
                const uint32_t* rec_causes = nullptr;
                const float* rec_dist_node = nullptr;
                const float* rec_dist_cause = nullptr;
                uint32_t effective_view_limit = enforce_view_limit
                    ? static_cast<uint32_t>(view_limit)
                    : static_cast<uint32_t>(cur_element_count);
                size_t rec_count = undo_log_->getRecoverPointersEUL(
                    current_node_id, effective_view_limit,
                    &rec_edges, &rec_causes, &rec_dist_node, &rec_dist_cause);

                if (rec_count > 0) {
                    if (collect_metrics) {
                        m2_triggered_.fetch_add(1, std::memory_order_relaxed);
                    }

                    int injected = 0;
                    for (size_t ri = 0; ri < rec_count && injected < m2_assist_budget_; ri++) {
                        tableint rec_id = rec_edges[ri];
                        if (!is_visible(rec_id)) continue;
                        if (visited_array[rec_id] == visited_array_tag) continue;

                        float d_owner_rec = rec_dist_node ? rec_dist_node[ri] : 0.0f;
                        dist_t lb_dist = std::abs(candidate_dist - d_owner_rec);
                        if (top_candidates.size() >= ef && lb_dist > lowerBound) continue;

                        visited_array[rec_id] = visited_array_tag;
                        char *rec_data = getDataByInternalId(rec_id);
                        dist_t rec_dist = fstdistfunc_(data_point, rec_data, dist_func_param_);
                        injected++;

                        if (collect_metrics) {
                            m2_backup_injected_.fetch_add(1, std::memory_order_relaxed);
                            metric_distance_computations++;
                        }

                        bool flag_consider = top_candidates.size() < ef || lowerBound > rec_dist;
                        if (flag_consider) {
                            candidate_set.emplace(-rec_dist, rec_id);

                            bool rec_allowed = effective_bare ||
                                (!isMarkedDeleted(rec_id) &&
                                 ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(rec_id))));
                            if (rec_allowed && is_visible(rec_id)) {
                                top_candidates.emplace(rec_dist, rec_id);
                                if (collect_metrics) {
                                    m2_backup_useful_.fetch_add(1, std::memory_order_relaxed);
                                }
                            }

                            while (top_candidates.size() > ef) {
                                top_candidates.pop();
                            }
                            if (!top_candidates.empty()) {
                                lowerBound = top_candidates.top().first;
                            }
                        }
                    }

                    if (collect_metrics && rec_count > static_cast<size_t>(m2_assist_budget_)) {
                        m2_budget_exhausted_.fetch_add(1, std::memory_order_relaxed);
                    }
                }
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

    tableint mutuallyConnectNewElement(
        const void *data_point, tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> &top_candidates,
        int level, bool isUpdate) {
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
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
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

                    // Record pruned edges in undo log before replacement
                    if (enable_mvcc_ && undo_log_) {
                        std::vector<tableint> old_edges(data, data + sz_link_list_other);
                        std::vector<tableint> new_edges;
                        
                        // Collect new edges
                        auto candidates_copy = candidates;
                        while (candidates_copy.size() > 0) {
                            new_edges.push_back(candidates_copy.top().second);
                            candidates_copy.pop();
                        }
                        
                        // Find pruned edges (in old but not in new)
                        for (tableint old_edge : old_edges) {
                            bool found = false;
                            for (tableint new_edge : new_edges) {
                                if (old_edge == new_edge) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                // This edge was pruned, record it
                                long batch_ts = global_watermark_.load(std::memory_order_acquire);
                                float dist_node_pruned = fstdistfunc_(
                                    getDataByInternalId(selectedNeighbors[idx]),
                                    getDataByInternalId(old_edge),
                                    dist_func_param_);
                                float dist_cause_pruned = fstdistfunc_(
                                    getDataByInternalId(cur_c),
                                    getDataByInternalId(old_edge),
                                    dist_func_param_);
                                    
                                undo_log_->recordPrune(selectedNeighbors[idx], old_edge, cur_c, 
                                    batch_ts, dist_node_pruned, dist_cause_pruned, nullptr);
                                markNodeModified(selectedNeighbors[idx]);
                                
                                // Inline GC
                                long horizon = gc_horizon_.load(std::memory_order_relaxed);
                                if (horizon > 0) {
                                    undo_log_->advanceGcStart(selectedNeighbors[idx], horizon);
                                }
                            }
                        }
                    }

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
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

    tableint addPoint(const void *data_point, labeltype label, int level,
                      bool defer_watermark = false) {
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

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error(
                    "The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count.fetch_add(1, std::memory_order_acq_rel);
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

        if (curlevel) {
            linkLists_[cur_c] =
                (char *)malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error(
                    "Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0,
                   size_links_per_element_ * curlevel + 1);
        }

        // Track whether this node was preempted and at which stage
        thread_local int tl_sleep_stage = -1;
        thread_local size_t tl_count_before_sleep = 0;
        tl_sleep_stage = -1;

        // Stage 0: after init, before route
        if (sleep_hook_ && sleep_hook_(0, cur_c)) {
            tl_sleep_stage = 0;
            tl_count_before_sleep = cur_element_count.load(std::memory_order_acquire);
        }

        if ((signed)currObj != -1) {
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
                                continue;
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
                }
            }

            // Stage 1: after route, before layer-0 search
            if (sleep_hook_ && sleep_hook_(1, cur_c)) {
                if (tl_sleep_stage < 0) {
                    tl_sleep_stage = 1;
                    tl_count_before_sleep = cur_element_count.load(std::memory_order_acquire);
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0;
                 level--) {
                if (level > maxlevelcopy || level < 0)
                    throw std::runtime_error("Level error");

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

                // Stage 2: after search, before connect
                if (sleep_hook_ && sleep_hook_(2, cur_c) && level == 0) {
                    if (tl_sleep_stage < 0) {
                        tl_sleep_stage = 2;
                        tl_count_before_sleep = cur_element_count.load(std::memory_order_acquire);
                    }
                }

                // Staleness measurement at layer 0 (if this node was preempted at ANY stage)
                if (staleness_cb_ && level == 0 && tl_sleep_stage >= 0) {
                    StalenessReport rpt;
                    rpt.node_id = cur_c;
                    rpt.level = level;
                    rpt.sleep_stage = tl_sleep_stage;

                    // Snapshot old candidates
                    {
                        auto tc = top_candidates;
                        while (!tc.empty()) { rpt.candidates_before.push_back(tc.top().second); tc.pop(); }
                    }
                    // Shadow neighbor selection on old candidates
                    {
                        auto tc = top_candidates;
                        getNeighborsByHeuristic(tc, M_);
                        while (!tc.empty()) { rpt.neighbors_before.push_back(tc.top().second); tc.pop(); }
                    }

                    // Re-search on current (post-sleep) graph
                    auto fresh = searchBaseLayer(currObj, data_point, level);
                    {
                        auto tc = fresh;
                        while (!tc.empty()) { rpt.candidates_after.push_back(tc.top().second); tc.pop(); }
                    }
                    {
                        auto tc = fresh;
                        getNeighborsByHeuristic(tc, M_);
                        while (!tc.empty()) { rpt.neighbors_after.push_back(tc.top().second); tc.pop(); }
                    }

                    size_t count_now = cur_element_count.load(std::memory_order_acquire);
                    rpt.vicinity_churn = (int)(count_now - tl_count_before_sleep);

                    staleness_cb_(rpt);
                }

                currObj = mutuallyConnectNewElement(
                    data_point, cur_c, top_candidates, level, false);

                // Stage 3: after connect
                if (sleep_hook_ && sleep_hook_(3, cur_c) && level == 0) {
                    // post-connect sleep (for completeness; staleness already measured)
                }
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
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;

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
                unsigned int *data;

                data = (unsigned int *)get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;

                tableint *datal = (tableint *)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        continue;  // skip dirty candidate (concurrent insert)
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
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates =
                searchBaseLayerST<true, true>(currObj, query_data, std::max(ef_, k),
                                              isIdAllowed, nullptr, view_limit);
        } else {
            top_candidates =
                searchBaseLayerST<false, true>(currObj, query_data, std::max(ef_, k),
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
                        continue;  // skip dirty candidate (concurrent insert)
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