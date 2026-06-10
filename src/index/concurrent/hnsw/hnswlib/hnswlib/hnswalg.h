#pragma once

#include <assert.h>
#include <stdlib.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <thread>
#include <time.h>
#include <type_traits>
#include <unordered_set>
#include <vector>
#include "hnswlib.h"
#include "visited_list_pool.h"

#if defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

#ifndef HNSW_SEARCH_PREFETCH_AHEAD
#define HNSW_SEARCH_PREFETCH_AHEAD 1
#endif

#ifndef HNSW_SEARCH_PREFETCH_HINT
#define HNSW_SEARCH_PREFETCH_HINT 0
#endif

#ifndef HNSW_INSERT_SEARCH_PREFETCH_AHEAD
#define HNSW_INSERT_SEARCH_PREFETCH_AHEAD 1
#endif

#ifndef HNSW_INSERT_SEARCH_PREFETCH_HINT
#define HNSW_INSERT_SEARCH_PREFETCH_HINT 0
#endif

#ifndef HNSW_INSERT_SEARCH_ADDR_BATCH
#define HNSW_INSERT_SEARCH_ADDR_BATCH 0
#endif

#ifndef HNSW_INSERT_SEARCH_GLOBAL_METRICS
#define HNSW_INSERT_SEARCH_GLOBAL_METRICS 0
#endif

#ifndef HNSW_INSERT_SEARCH_NT_DISTANCE
#define HNSW_INSERT_SEARCH_NT_DISTANCE 0
#endif

#ifndef HNSW_INSERT_SEARCH_NT_LEVEL0_ONLY
#define HNSW_INSERT_SEARCH_NT_LEVEL0_ONLY 0
#endif

#ifndef HNSW_INSERT_SEARCH_CACHELINE_TIEBREAK
#define HNSW_INSERT_SEARCH_CACHELINE_TIEBREAK 0
#endif

#ifndef HNSW_DISTANCE_METADATA_CACHELINE_SPLIT
#define HNSW_DISTANCE_METADATA_CACHELINE_SPLIT 0
#endif

#ifndef HNSW_M3_SEARCH_WRITE_METADATA_SPLIT
#define HNSW_M3_SEARCH_WRITE_METADATA_SPLIT 0
#endif

#ifndef HNSW_LINK_LOCK_CACHELINE_PAD
#define HNSW_LINK_LOCK_CACHELINE_PAD 0
#endif

#ifndef HNSW_ALIGN_DATA_TO_CACHELINE
#define HNSW_ALIGN_DATA_TO_CACHELINE 0
#endif

#ifndef HNSW_SEARCH_VECTOR_SHADOW
#define HNSW_SEARCH_VECTOR_SHADOW 0
#endif

#ifndef HNSW_INSERT_VECTOR_SHADOW
#define HNSW_INSERT_VECTOR_SHADOW 0
#endif

#ifndef HNSW_INSERT_VECTOR_PARTIAL_SHADOW
#define HNSW_INSERT_VECTOR_PARTIAL_SHADOW 0
#endif

#ifndef HNSW_INSERT_VECTOR_HOT_SHADOW
#define HNSW_INSERT_VECTOR_HOT_SHADOW 0
#endif

#ifndef HNSW_INSERT_VECTOR_HOT_ONLINE
#define HNSW_INSERT_VECTOR_HOT_ONLINE 0
#endif

#ifndef HNSW_INSERT_VECTOR_HOT_ONLINE_BUDGET
#define HNSW_INSERT_VECTOR_HOT_ONLINE_BUDGET 500000
#endif

#ifndef HNSW_INSERT_VECTOR_HOT_ONLINE_THRESHOLD
#define HNSW_INSERT_VECTOR_HOT_ONLINE_THRESHOLD 8
#endif

#ifndef HNSW_INSERT_VECTOR_HOT_ONLINE_EPOCH_SIZE
#define HNSW_INSERT_VECTOR_HOT_ONLINE_EPOCH_SIZE 0
#endif

#ifndef HNSW_INSERT_VECTOR_HOT_ONLINE_STABILITY
#define HNSW_INSERT_VECTOR_HOT_ONLINE_STABILITY 1
#endif

#ifndef HNSW_INSERT_VECTOR_HOT_ONLINE_SAMPLE_MASK
#define HNSW_INSERT_VECTOR_HOT_ONLINE_SAMPLE_MASK 15
#endif

#ifndef HNSW_INSERT_VECTOR_GRAPH_HOT
#define HNSW_INSERT_VECTOR_GRAPH_HOT 0
#endif

#ifndef HNSW_INSERT_VECTOR_GRAPH_HOT_BUDGET
#define HNSW_INSERT_VECTOR_GRAPH_HOT_BUDGET 500000
#endif

#ifndef HNSW_INSERT_VECTOR_GRAPH_HOT_MODE
#define HNSW_INSERT_VECTOR_GRAPH_HOT_MODE 0
#endif

#ifndef HNSW_INSERT_VECTOR_READ_TRACE
#define HNSW_INSERT_VECTOR_READ_TRACE 0
#endif

#ifndef HNSW_SEARCH_VECTOR_READ_TRACE
#define HNSW_SEARCH_VECTOR_READ_TRACE 0
#endif

#ifndef HNSW_INSERT_ADJ_LOOKAHEAD_PREFETCH
#define HNSW_INSERT_ADJ_LOOKAHEAD_PREFETCH 0
#endif
#ifndef HNSW_SEARCH_ADJ_LOOKAHEAD_PREFETCH
#define HNSW_SEARCH_ADJ_LOOKAHEAD_PREFETCH 0
#endif

// Hot-vector ID-range encoding: after initial-graph build, renumber so that
// hot nodes (by HNSW level) occupy IDs [0, HOT_THRESHOLD). Dispatch becomes
// a single integer compare — no atomic load, no slot table.
#ifndef HNSW_HOT_VEC_ID_RANGE
#define HNSW_HOT_VEC_ID_RANGE 0
#endif
#ifndef HNSW_HOT_VEC_ID_RANGE_THRESHOLD
#define HNSW_HOT_VEC_ID_RANGE_THRESHOLD 500000
#endif
// When HNSW_HOT_VEC_LEVEL_RENUMBER is on, the post-init renumber pass assigns
// level>=1 nodes to IDs 0..N_upper-1 and the rest to N_upper..cur_element_count.
// HNSW_HOT_VEC_ID_RANGE_THRESHOLD becomes the *budget* for the hot pool;
// the actual hot count = number of level>=1 nodes (typically ~6.5% of total).
#ifndef HNSW_HOT_VEC_LEVEL_RENUMBER
#define HNSW_HOT_VEC_LEVEL_RENUMBER 0
#endif
#ifndef HNSW_HOT_VEC_LEVEL_RENUMBER_TRIGGER
#define HNSW_HOT_VEC_LEVEL_RENUMBER_TRIGGER 1000000
#endif

#ifndef HNSW_INSERT_VECTOR_REGION_SHADOW
#define HNSW_INSERT_VECTOR_REGION_SHADOW 0
#endif

#ifndef HNSW_INSERT_VECTOR_REGION_SHADOW_BYTES
#define HNSW_INSERT_VECTOR_REGION_SHADOW_BYTES 268435456ULL
#endif

#ifndef HNSW_INSERT_VECTOR_REGION_SHIFT
#define HNSW_INSERT_VECTOR_REGION_SHIFT 21
#endif

#ifndef HNSW_INSERT_VECTOR_REGION_THRESHOLD
#define HNSW_INSERT_VECTOR_REGION_THRESHOLD 16
#endif

#ifndef HNSW_INSERT_VECTOR_REGION_SAMPLE_MASK
#define HNSW_INSERT_VECTOR_REGION_SAMPLE_MASK 31
#endif

#ifndef HNSW_INSERT_VECTOR_REGION_STATS
#define HNSW_INSERT_VECTOR_REGION_STATS 0
#endif

#ifndef HNSW_INSERT_VECTOR_REGION_STATS_SAMPLE_MASK
#define HNSW_INSERT_VECTOR_REGION_STATS_SAMPLE_MASK 1023
#endif

#ifndef HNSW_INSERT_VECTOR_THREAD_CACHE
#define HNSW_INSERT_VECTOR_THREAD_CACHE 0
#endif

#ifndef HNSW_INSERT_VECTOR_THREAD_CACHE_SLOTS
#define HNSW_INSERT_VECTOR_THREAD_CACHE_SLOTS 256
#endif

#ifndef HNSW_INSERT_VECTOR_THREAD_FILTER_SLOTS
#define HNSW_INSERT_VECTOR_THREAD_FILTER_SLOTS 4096
#endif

#ifndef HNSW_INSERT_VECTOR_THREAD_CACHE_THRESHOLD
#define HNSW_INSERT_VECTOR_THREAD_CACHE_THRESHOLD 2
#endif

#ifndef HNSW_INSERT_VECTOR_THREAD_CACHE_STATS
#define HNSW_INSERT_VECTOR_THREAD_CACHE_STATS 0
#endif

#ifndef HNSW_INSERT_VECTOR_THREAD_CACHE_STATS_SAMPLE_MASK
#define HNSW_INSERT_VECTOR_THREAD_CACHE_STATS_SAMPLE_MASK 1023
#endif

#ifndef HNSW_INSERT_VECTOR_THREAD_CACHE_STREAM_FILL
#define HNSW_INSERT_VECTOR_THREAD_CACHE_STREAM_FILL 0
#endif

#ifndef HNSW_INSERT_VECTOR_PARTIAL_SHADOW_NUM
#define HNSW_INSERT_VECTOR_PARTIAL_SHADOW_NUM 1
#endif

#ifndef HNSW_INSERT_VECTOR_PARTIAL_SHADOW_DEN
#define HNSW_INSERT_VECTOR_PARTIAL_SHADOW_DEN 10
#endif

#ifndef HNSW_INSERT_VECTOR_PARTIAL_SHADOW_MODE
#define HNSW_INSERT_VECTOR_PARTIAL_SHADOW_MODE 0
#endif

#ifndef HNSW_INSERT_VECTOR_SHADOW_OFFSET_BYTES
#define HNSW_INSERT_VECTOR_SHADOW_OFFSET_BYTES 64
#endif

#ifndef HNSW_INSERT_VECTOR_COLD_PREFETCH
#define HNSW_INSERT_VECTOR_COLD_PREFETCH 0
#endif

#if (HNSW_INSERT_VECTOR_SHADOW + HNSW_INSERT_VECTOR_PARTIAL_SHADOW + \
     HNSW_INSERT_VECTOR_HOT_SHADOW + HNSW_INSERT_VECTOR_REGION_SHADOW + \
     HNSW_INSERT_VECTOR_THREAD_CACHE) > 1
#error "Use only one insert vector shadow mode"
#endif

#if HNSW_INSERT_VECTOR_HOT_ONLINE && !HNSW_INSERT_VECTOR_HOT_SHADOW
#error "HNSW_INSERT_VECTOR_HOT_ONLINE requires HNSW_INSERT_VECTOR_HOT_SHADOW"
#endif

#if HNSW_INSERT_VECTOR_GRAPH_HOT && !HNSW_INSERT_VECTOR_HOT_SHADOW
#error "HNSW_INSERT_VECTOR_GRAPH_HOT requires HNSW_INSERT_VECTOR_HOT_SHADOW"
#endif

#if HNSW_INSERT_VECTOR_GRAPH_HOT && HNSW_INSERT_VECTOR_HOT_ONLINE
#error "Use either graph-hot or online-hot insert vector admission, not both"
#endif

#if HNSW_INSERT_VECTOR_HOT_ONLINE && HNSW_INSERT_VECTOR_HOT_ONLINE_BUDGET <= 0
#error "HNSW_INSERT_VECTOR_HOT_ONLINE_BUDGET must be positive"
#endif

#if HNSW_INSERT_VECTOR_HOT_ONLINE && HNSW_INSERT_VECTOR_HOT_ONLINE_THRESHOLD <= 0
#error "HNSW_INSERT_VECTOR_HOT_ONLINE_THRESHOLD must be positive"
#endif

#if HNSW_INSERT_VECTOR_GRAPH_HOT && HNSW_INSERT_VECTOR_GRAPH_HOT_BUDGET <= 0
#error "HNSW_INSERT_VECTOR_GRAPH_HOT_BUDGET must be positive"
#endif

#if HNSW_INSERT_VECTOR_GRAPH_HOT && \
    (HNSW_INSERT_VECTOR_GRAPH_HOT_MODE < 0 || HNSW_INSERT_VECTOR_GRAPH_HOT_MODE > 1)
#error "Invalid HNSW_INSERT_VECTOR_GRAPH_HOT_MODE; use 0=in-degree, 1=level-first"
#endif

#if HNSW_INSERT_VECTOR_REGION_SHADOW && HNSW_INSERT_VECTOR_REGION_SHIFT <= 0
#error "HNSW_INSERT_VECTOR_REGION_SHIFT must be positive"
#endif

#if HNSW_INSERT_VECTOR_REGION_SHADOW && HNSW_INSERT_VECTOR_REGION_THRESHOLD <= 0
#error "HNSW_INSERT_VECTOR_REGION_THRESHOLD must be positive"
#endif

#if HNSW_INSERT_VECTOR_THREAD_CACHE && \
    (HNSW_INSERT_VECTOR_THREAD_CACHE_SLOTS <= 0 || \
     (HNSW_INSERT_VECTOR_THREAD_CACHE_SLOTS & \
      (HNSW_INSERT_VECTOR_THREAD_CACHE_SLOTS - 1)) != 0)
#error "HNSW_INSERT_VECTOR_THREAD_CACHE_SLOTS must be a positive power of two"
#endif

#if HNSW_INSERT_VECTOR_THREAD_CACHE && \
    (HNSW_INSERT_VECTOR_THREAD_FILTER_SLOTS <= 0 || \
     (HNSW_INSERT_VECTOR_THREAD_FILTER_SLOTS & \
      (HNSW_INSERT_VECTOR_THREAD_FILTER_SLOTS - 1)) != 0)
#error "HNSW_INSERT_VECTOR_THREAD_FILTER_SLOTS must be a positive power of two"
#endif

#if HNSW_INSERT_VECTOR_THREAD_CACHE && \
    HNSW_INSERT_VECTOR_THREAD_CACHE_THRESHOLD <= 0
#error "HNSW_INSERT_VECTOR_THREAD_CACHE_THRESHOLD must be positive"
#endif

#if HNSW_INSERT_VECTOR_PARTIAL_SHADOW && \
    (HNSW_INSERT_VECTOR_PARTIAL_SHADOW_DEN <= 0 || \
     HNSW_INSERT_VECTOR_PARTIAL_SHADOW_NUM <= 0 || \
     HNSW_INSERT_VECTOR_PARTIAL_SHADOW_NUM > HNSW_INSERT_VECTOR_PARTIAL_SHADOW_DEN)
#error "Invalid partial insert vector shadow fraction"
#endif

#if HNSW_INSERT_VECTOR_PARTIAL_SHADOW && \
    (HNSW_INSERT_VECTOR_PARTIAL_SHADOW_MODE < 0 || \
     HNSW_INSERT_VECTOR_PARTIAL_SHADOW_MODE > 1)
#error "Invalid partial insert vector shadow mode; use 0=stripe, 1=prefix"
#endif

#if HNSW_SEARCH_PREFETCH_HINT == 0
#define HNSW_SEARCH_PREFETCH_HINT_VALUE _MM_HINT_T0
#elif HNSW_SEARCH_PREFETCH_HINT == 1
#define HNSW_SEARCH_PREFETCH_HINT_VALUE _MM_HINT_T1
#elif HNSW_SEARCH_PREFETCH_HINT == 2
#define HNSW_SEARCH_PREFETCH_HINT_VALUE _MM_HINT_T2
#elif HNSW_SEARCH_PREFETCH_HINT == 3
#define HNSW_SEARCH_PREFETCH_HINT_VALUE _MM_HINT_NTA
#else
#error "Unsupported HNSW_SEARCH_PREFETCH_HINT; use 0=T0, 1=T1, 2=T2, 3=NTA"
#endif

#if HNSW_INSERT_SEARCH_PREFETCH_HINT == 0
#define HNSW_INSERT_SEARCH_PREFETCH_HINT_VALUE _MM_HINT_T0
#elif HNSW_INSERT_SEARCH_PREFETCH_HINT == 1
#define HNSW_INSERT_SEARCH_PREFETCH_HINT_VALUE _MM_HINT_T1
#elif HNSW_INSERT_SEARCH_PREFETCH_HINT == 2
#define HNSW_INSERT_SEARCH_PREFETCH_HINT_VALUE _MM_HINT_T2
#elif HNSW_INSERT_SEARCH_PREFETCH_HINT == 3
#define HNSW_INSERT_SEARCH_PREFETCH_HINT_VALUE _MM_HINT_NTA
#else
#error "Unsupported HNSW_INSERT_SEARCH_PREFETCH_HINT; use 0=T0, 1=T1, 2=T2, 3=NTA"
#endif

#ifndef HNSW_INSERT_REPAIR_ORDER_BY_ADDR
#define HNSW_INSERT_REPAIR_ORDER_BY_ADDR 0
#endif

#ifndef HNSW_SEARCH_ACTIVE_NODE_TRACKING
#define HNSW_SEARCH_ACTIVE_NODE_TRACKING 0
#endif

#ifndef HNSW_SEARCH_AWARE_REPAIR_REORDER
#define HNSW_SEARCH_AWARE_REPAIR_REORDER 0
#endif

#ifndef HNSW_SEARCH_ACTIVE_CACHE_BUCKET_TRACKING
#define HNSW_SEARCH_ACTIVE_CACHE_BUCKET_TRACKING 0
#endif

#ifndef HNSW_SEARCH_ACTIVE_CACHE_BUCKET_STORE_ONLY
#define HNSW_SEARCH_ACTIVE_CACHE_BUCKET_STORE_ONLY 0
#endif

#ifndef HNSW_CACHE_AWARE_REPAIR_REORDER
#define HNSW_CACHE_AWARE_REPAIR_REORDER 0
#endif

#ifndef HNSW_CACHE_AWARE_INSERT_SEARCH_ORDER
#define HNSW_CACHE_AWARE_INSERT_SEARCH_ORDER 0
#endif

#ifndef HNSW_SEARCH_ACTIVE_CACHE_BUCKETS
#define HNSW_SEARCH_ACTIVE_CACHE_BUCKETS 4096
#endif

#ifndef HNSW_SEARCH_ACTIVE_CACHE_GUARD_MAX_BUCKETS
#define HNSW_SEARCH_ACTIVE_CACHE_GUARD_MAX_BUCKETS 4
#endif

#ifndef HNSW_INSERT_REPAIR_BULK_REWRITE
#define HNSW_INSERT_REPAIR_BULK_REWRITE 0
#endif

#ifndef HNSW_L0_ADJ_COW_PUBLISH
#define HNSW_L0_ADJ_COW_PUBLISH 0
#endif

#ifndef HNSW_L0_REPAIR_NT_STORE
#define HNSW_L0_REPAIR_NT_STORE 0
#endif

#if HNSW_L0_REPAIR_NT_STORE &&                                      \
    (defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64) || \
     (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#define HNSW_L0_REPAIR_NT_STORE_X86 1
#else
#define HNSW_L0_REPAIR_NT_STORE_X86 0
#endif

#ifndef HNSW_INSERT_ZERO_LINKS_ONLY
#define HNSW_INSERT_ZERO_LINKS_ONLY 0
#endif

#ifndef HNSW_INSERT_VECTOR_CLDEMOTE
#define HNSW_INSERT_VECTOR_CLDEMOTE 0
#endif

#ifndef HNSW_INSERT_VECTOR_CLDEMOTE_AFTER_REPAIR
#define HNSW_INSERT_VECTOR_CLDEMOTE_AFTER_REPAIR 0
#endif

#ifndef HNSW_INSERT_REPAIR_PHASE_MAX_ACTIVE
#define HNSW_INSERT_REPAIR_PHASE_MAX_ACTIVE 0
#endif

#ifndef HNSW_INSERT_REPAIR_PHASE_WAIT_NS
#define HNSW_INSERT_REPAIR_PHASE_WAIT_NS 1000
#endif

#ifndef HNSW_INSERT_VECTOR_READ_MAX_ACTIVE
#define HNSW_INSERT_VECTOR_READ_MAX_ACTIVE 0
#endif

#ifndef HNSW_INSERT_VECTOR_READ_WAIT_NS
#define HNSW_INSERT_VECTOR_READ_WAIT_NS 1000
#endif

#ifndef HNSW_M3_ALLOW_RETIRED_MECHANISMS
#define HNSW_M3_ALLOW_RETIRED_MECHANISMS 0
#endif

#if !HNSW_M3_ALLOW_RETIRED_MECHANISMS &&                              \
    (HNSW_DISTANCE_METADATA_CACHELINE_SPLIT || HNSW_LINK_LOCK_CACHELINE_PAD || \
     HNSW_INSERT_SEARCH_NT_DISTANCE || HNSW_INSERT_SEARCH_NT_LEVEL0_ONLY || \
     HNSW_INSERT_SEARCH_CACHELINE_TIEBREAK ||                          \
     HNSW_INSERT_SEARCH_ADDR_BATCH || HNSW_INSERT_VECTOR_SHADOW ||      \
     HNSW_INSERT_VECTOR_PARTIAL_SHADOW || HNSW_INSERT_VECTOR_HOT_SHADOW || \
     HNSW_INSERT_VECTOR_HOT_ONLINE || HNSW_INSERT_VECTOR_GRAPH_HOT ||  \
     HNSW_INSERT_VECTOR_REGION_SHADOW || HNSW_INSERT_VECTOR_THREAD_CACHE || \
     HNSW_INSERT_VECTOR_COLD_PREFETCH || HNSW_HOT_VEC_ID_RANGE ||      \
     HNSW_HOT_VEC_LEVEL_RENUMBER || HNSW_SEARCH_ACTIVE_NODE_TRACKING || \
     HNSW_SEARCH_AWARE_REPAIR_REORDER ||                               \
     HNSW_SEARCH_ACTIVE_CACHE_BUCKET_TRACKING ||                        \
     HNSW_CACHE_AWARE_REPAIR_REORDER || HNSW_CACHE_AWARE_INSERT_SEARCH_ORDER || \
     HNSW_INSERT_REPAIR_BULK_REWRITE || HNSW_L0_ADJ_COW_PUBLISH ||     \
     HNSW_L0_REPAIR_NT_STORE || HNSW_INSERT_ZERO_LINKS_ONLY ||          \
     HNSW_INSERT_VECTOR_CLDEMOTE || HNSW_INSERT_VECTOR_CLDEMOTE_AFTER_REPAIR || \
     HNSW_INSERT_REPAIR_PHASE_MAX_ACTIVE || HNSW_INSERT_VECTOR_READ_MAX_ACTIVE)
#error "Retired M3 cache/layout experiments are disabled. Define HNSW_M3_ALLOW_RETIRED_MECHANISMS=1 only to reproduce archived runs."
#endif

template <typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
   public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;
    static constexpr size_t kLayoutCacheLineBytes = 64;

    struct alignas(kLayoutCacheLineBytes) CachelineSharedMutex {
        std::shared_mutex mutex;

        operator std::shared_mutex&() noexcept {
            return mutex;
        }
        operator const std::shared_mutex&() const noexcept {
            return mutex;
        }
    };

#if HNSW_LINK_LOCK_CACHELINE_PAD
    using LinkListLock = CachelineSharedMutex;
#else
    using LinkListLock = std::shared_mutex;
#endif

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
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

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    mutable std::vector<LinkListLock> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

    char* data_level0_memory_{nullptr};
	    char* search_vector_shadow_memory_{nullptr};
	    char* insert_vector_shadow_alloc_{nullptr};
	    char* insert_vector_shadow_memory_{nullptr};
	    size_t insert_vector_shadow_slots_{0};
	    char* insert_vector_region_shadow_alloc_{nullptr};
	    char* insert_vector_region_shadow_memory_{nullptr};
	    size_t insert_vector_region_shadow_slots_{0};
	    size_t insert_vector_region_shadow_bytes_{0};
	    size_t insert_vector_region_shadow_region_bytes_{0};
	    size_t insert_vector_region_count_{0};
	    size_t insert_vector_region_shadow_max_id_{0};
	    std::atomic<int32_t>* insert_vector_region_shadow_slot_table_{nullptr};
	    mutable std::atomic<uint32_t>* insert_vector_region_shadow_counts_{nullptr};
	    mutable std::atomic<uint32_t> insert_vector_region_shadow_next_slot_{0};
	    mutable std::atomic<bool> insert_vector_region_shadow_budget_exhausted_{false};
	    mutable std::atomic<uint64_t> metric_insert_vector_region_shadow_samples{0};
	    mutable std::atomic<uint64_t> metric_insert_vector_region_shadow_promotions{0};
	    mutable std::atomic<uint64_t> metric_insert_vector_region_shadow_hits{0};
	    mutable std::atomic<uint64_t> metric_insert_vector_region_shadow_fallbacks{0};
	    mutable std::atomic<uint64_t> metric_insert_vector_region_shadow_cross_region{0};
	    mutable std::atomic<uint64_t> metric_insert_vector_region_shadow_budget_full{0};
	    mutable std::atomic<uint64_t> metric_insert_vector_region_shadow_copy_ns{0};
	    mutable std::atomic<uint64_t> metric_insert_vector_thread_cache_hits{0};
	    mutable std::atomic<uint64_t> metric_insert_vector_thread_cache_fallbacks{0};
	    mutable std::atomic<uint64_t> metric_insert_vector_thread_cache_filter_hits{0};
	    mutable std::atomic<uint64_t> metric_insert_vector_thread_cache_admissions{0};
	    mutable std::atomic<uint64_t> metric_insert_vector_thread_cache_copies{0};
	    std::atomic<int32_t>* insert_vector_hot_slots_{nullptr};
    size_t insert_vector_hot_slot_table_size_{0};
    mutable std::atomic<uint32_t>* insert_vector_hot_online_counts_{nullptr};
    mutable std::atomic<uint32_t>* insert_vector_hot_online_prev_counts_{nullptr};
    mutable std::atomic<uint64_t> insert_vector_hot_epoch_reads_{0};
    mutable std::atomic<uint64_t> insert_vector_hot_epoch_id_{0};
    mutable std::mutex insert_vector_hot_epoch_mu_;
    mutable std::atomic<uint32_t> insert_vector_hot_online_next_slot_{0};
    mutable std::atomic<bool> insert_vector_hot_online_budget_exhausted_{false};
    mutable std::atomic<uint64_t> metric_insert_vector_hot_online_samples{0};
    mutable std::atomic<uint64_t> metric_insert_vector_hot_online_promotions{0};
    mutable std::atomic<uint64_t> metric_insert_vector_hot_online_budget_full{0};
    mutable std::atomic<uint64_t> metric_insert_vector_hot_epoch_swaps{0};
    mutable std::atomic<uint64_t> metric_insert_vector_hot_epoch_skipped_stability{0};
    mutable std::atomic<uint64_t> metric_insert_vector_graph_hot_select_ns{0};
    mutable std::atomic<uint64_t> metric_insert_vector_graph_hot_copy_ns{0};
    mutable std::atomic<uint64_t> metric_insert_vector_graph_hot_scanned_edges{0};
    mutable std::atomic<uint64_t> metric_insert_vector_graph_hot_selected{0};
    mutable std::atomic<uint64_t> metric_insert_vector_graph_hot_candidates{0};
    mutable std::atomic<uint64_t>* insert_vector_read_counts_{nullptr};
    mutable std::atomic<bool> insert_vector_read_trace_enabled_{true};
    mutable std::atomic<uint64_t> metric_insert_vector_read_trace_reads{0};
    mutable std::atomic<uint64_t>* search_vector_read_counts_{nullptr};
    mutable std::atomic<uint64_t> metric_search_vector_read_trace_reads{0};
    char* hot_vec_pool_{nullptr};
    char* cold_vec_pool_{nullptr};
    size_t hot_vec_pool_count_{0};
    bool hot_id_renumber_done_{false};
    mutable std::atomic<uint64_t> metric_hot_id_range_renumber_ns{0};
    mutable std::atomic<uint64_t> metric_hot_id_range_hot_reads{0};
    mutable std::atomic<uint64_t> metric_hot_id_range_cold_reads{0};
    std::atomic<char*>* level0_linklist_overrides_{nullptr};
    mutable std::mutex level0_cow_blocks_lock_;
    mutable std::vector<char*> level0_cow_blocks_;
    char** linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

#if HNSW_DISTANCE_METADATA_CACHELINE_SPLIT
    alignas(kLayoutCacheLineBytes) DISTFUNC<dist_t> fstdistfunc_;
    void* dist_func_param_{nullptr};
    alignas(kLayoutCacheLineBytes) mutable std::mutex label_lookup_lock;
#elif HNSW_M3_SEARCH_WRITE_METADATA_SPLIT
    DISTFUNC<dist_t> fstdistfunc_;
    void* dist_func_param_{nullptr};
    alignas(kLayoutCacheLineBytes) mutable std::mutex label_lookup_lock;
#else
    DISTFUNC<dist_t> fstdistfunc_;
    void* dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
#endif
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};
    mutable std::atomic<uint64_t> metric_graph_connect_calls{0};
    mutable std::atomic<uint64_t> metric_graph_connect_ns{0};
    mutable std::atomic<uint64_t> metric_graph_link_critical_ns{0};
    mutable std::atomic<uint64_t> metric_graph_unique_lock_wait_ns{0};
    // search-during-insert: searchBaseLayer takes UNIQUE_LOCK on each visited
    // node (line 812 in this file). These were not previously timed.
    mutable std::atomic<uint64_t> metric_graph_search_unique_lock_wait_ns{0};
    mutable std::atomic<uint64_t> metric_graph_search_unique_lock_acqs{0};
    mutable std::atomic<uint64_t> metric_graph_search_critical_ns{0};
    mutable std::atomic<uint64_t> metric_graph_link_updates{0};
    mutable std::atomic<uint64_t> metric_graph_upper_search_ns{0};
    mutable std::atomic<uint64_t> metric_graph_upper_search_dist_comps{0};
    mutable std::atomic<uint64_t> metric_graph_upper_search_edges_scanned{0};
    mutable std::atomic<uint64_t> metric_graph_base_search_ns{0};
    mutable std::atomic<uint64_t> metric_graph_base_search_expansions{0};
    mutable std::atomic<uint64_t> metric_graph_base_search_edges_scanned{0};
    mutable std::atomic<uint64_t> metric_graph_base_search_dist_comps{0};
    mutable std::atomic<uint64_t> metric_graph_base_search_addr_batch_candidates{0};
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
    mutable std::atomic<uint64_t> metric_l0_adj_cow_nodes{0};
    mutable std::atomic<uint64_t> metric_l0_adj_cow_publishes{0};
    mutable std::atomic<uint64_t> metric_l0_adj_cow_append_publishes{0};
    mutable std::atomic<uint64_t> metric_l0_adj_cow_prune_publishes{0};
    mutable std::atomic<uint64_t> metric_l0_adj_cow_bytes{0};
    mutable std::atomic<uint64_t> metric_l0_adj_cow_partial_fallbacks{0};
    mutable std::atomic<uint64_t> metric_l0_repair_nt_edges{0};
    mutable std::atomic<uint64_t> metric_l0_repair_nt_rewrites{0};
    mutable std::atomic<uint64_t> metric_l0_repair_nt_appends{0};
    mutable std::atomic<uint64_t> metric_l0_repair_nt_partial_fallbacks{0};
    mutable std::atomic<uint32_t> active_insert_vector_readers{0};
    mutable std::atomic<uint64_t> metric_insert_vector_read_admissions{0};
    mutable std::atomic<uint64_t> metric_insert_vector_read_waits{0};
    mutable std::atomic<uint64_t> metric_insert_vector_read_wait_ns{0};
    mutable std::atomic<uint16_t>* search_active_node_counts_{nullptr};
    mutable std::atomic<uint64_t> metric_search_active_node_marks{0};
    mutable std::atomic<uint64_t> metric_insert_search_hot_reorder_lists{0};
    mutable std::atomic<uint64_t> metric_insert_search_hot_reorder_nodes{0};
    mutable std::atomic<uint16_t>* search_active_cache_buckets_{nullptr};
    mutable std::atomic<uint64_t> metric_search_active_cache_bucket_marks{0};
    mutable std::atomic<uint64_t> metric_insert_cache_hot_reorder_lists{0};
    mutable std::atomic<uint64_t> metric_insert_cache_hot_reorder_nodes{0};
    mutable std::atomic<uint64_t> metric_insert_search_cache_defer_lists{0};
    mutable std::atomic<uint64_t> metric_insert_search_cache_defer_nodes{0};

    using ActiveCacheBucketArray =
        std::array<uint32_t, HNSW_SEARCH_ACTIVE_CACHE_GUARD_MAX_BUCKETS>;

    static size_t alignToLayoutCacheLine(size_t bytes) {
        return ((bytes + kLayoutCacheLineBytes - 1) / kLayoutCacheLineBytes) *
               kLayoutCacheLineBytes;
    }

    static size_t computeLevel0LinksBytes(size_t maxM0) {
        return maxM0 * sizeof(tableint) + sizeof(linklistsizeint);
    }

    static size_t computeLevel0DataOffset(size_t size_links_level0) {
#if HNSW_ALIGN_DATA_TO_CACHELINE
        return alignToLayoutCacheLine(size_links_level0);
#else
        return size_links_level0;
#endif
    }

    void configureLevel0Layout() {
        size_links_level0_ = computeLevel0LinksBytes(maxM0_);
        offsetData_ = computeLevel0DataOffset(size_links_level0_);
        label_offset_ = offsetData_ + data_size_;
        size_data_per_element_ = label_offset_ + sizeof(labeltype);
        offsetLevel0_ = 0;
    }

    void validateLevel0Layout(const char* context) const {
        if (offsetData_ < size_links_level0_) {
            throw std::runtime_error(std::string(context) + ": offsetData before level-0 links");
        }
        if (label_offset_ != offsetData_ + data_size_) {
            throw std::runtime_error(std::string(context) + ": label offset does not follow data");
        }
        if (size_data_per_element_ != label_offset_ + sizeof(labeltype)) {
            throw std::runtime_error(std::string(context) +
                                     ": element size does not follow label");
        }
    }

    size_t level0LayoutOverheadBytes() const {
        return offsetData_ >= size_links_level0_ ? offsetData_ - size_links_level0_ : 0;
    }

    void allocateSearchVectorShadow(size_t max_elements) {
#if HNSW_SEARCH_VECTOR_SHADOW
        if (data_size_ == 0) {
            return;
        }
        search_vector_shadow_memory_ = (char*)malloc(max_elements * data_size_);
        if (search_vector_shadow_memory_ == nullptr) {
            throw std::runtime_error("Not enough memory: failed to allocate search vector shadow");
        }
#else
        (void)max_elements;
#endif
    }

    static bool insertVectorReplicaEnabled() {
#if HNSW_INSERT_VECTOR_SHADOW || HNSW_INSERT_VECTOR_PARTIAL_SHADOW || \
    HNSW_INSERT_VECTOR_HOT_SHADOW
        return true;
#else
        return false;
#endif
    }

    bool insertVectorHasReplicaSlot(tableint internal_id) const {
#if HNSW_INSERT_VECTOR_SHADOW
        (void)internal_id;
        return true;
#elif HNSW_INSERT_VECTOR_HOT_SHADOW
        return insert_vector_hot_slots_ &&
               internal_id < insert_vector_hot_slot_table_size_ &&
               insert_vector_hot_slots_[internal_id].load(std::memory_order_acquire) >= 0;
#elif HNSW_INSERT_VECTOR_PARTIAL_SHADOW
#if HNSW_INSERT_VECTOR_PARTIAL_SHADOW_MODE == 1
        return internal_id < insert_vector_shadow_slots_;
#else
        return (internal_id % HNSW_INSERT_VECTOR_PARTIAL_SHADOW_DEN) <
               HNSW_INSERT_VECTOR_PARTIAL_SHADOW_NUM;
#endif
#else
        (void)internal_id;
        return false;
#endif
    }

    size_t insertVectorReplicaSlot(tableint internal_id) const {
#if HNSW_INSERT_VECTOR_SHADOW
        return internal_id;
#elif HNSW_INSERT_VECTOR_HOT_SHADOW
        return static_cast<size_t>(
            insert_vector_hot_slots_[internal_id].load(std::memory_order_acquire));
#elif HNSW_INSERT_VECTOR_PARTIAL_SHADOW
#if HNSW_INSERT_VECTOR_PARTIAL_SHADOW_MODE == 1
        return internal_id;
#else
        const size_t den = HNSW_INSERT_VECTOR_PARTIAL_SHADOW_DEN;
        const size_t num = HNSW_INSERT_VECTOR_PARTIAL_SHADOW_NUM;
        return (internal_id / den) * num + std::min<size_t>(internal_id % den, num);
#endif
#else
        (void)internal_id;
        return 0;
#endif
    }

    static size_t insertVectorReplicaSlotsForCount(size_t count) {
#if HNSW_INSERT_VECTOR_SHADOW
        return count;
#elif HNSW_INSERT_VECTOR_HOT_SHADOW
#if HNSW_INSERT_VECTOR_HOT_ONLINE
        return std::min<size_t>(count, HNSW_INSERT_VECTOR_HOT_ONLINE_BUDGET);
#elif HNSW_INSERT_VECTOR_GRAPH_HOT
        return std::min<size_t>(count, HNSW_INSERT_VECTOR_GRAPH_HOT_BUDGET);
#else
        (void)count;
        return 0;
#endif
#elif HNSW_INSERT_VECTOR_PARTIAL_SHADOW
        const size_t den = HNSW_INSERT_VECTOR_PARTIAL_SHADOW_DEN;
        const size_t num = HNSW_INSERT_VECTOR_PARTIAL_SHADOW_NUM;
        return (count / den) * num + std::min<size_t>(count % den, num);
#else
        (void)count;
        return 0;
#endif
    }

    size_t loadInsertVectorHotSlots(size_t max_elements) {
#if HNSW_INSERT_VECTOR_HOT_SHADOW
        delete[] insert_vector_hot_slots_;
        insert_vector_hot_slots_ = new std::atomic<int32_t>[max_elements];
        insert_vector_hot_slot_table_size_ = max_elements;
        for (size_t i = 0; i < max_elements; i++) {
            insert_vector_hot_slots_[i].store(-1, std::memory_order_relaxed);
        }
#if HNSW_INSERT_VECTOR_HOT_ONLINE
        delete[] insert_vector_hot_online_counts_;
        insert_vector_hot_online_counts_ = new std::atomic<uint32_t>[max_elements];
        for (size_t i = 0; i < max_elements; i++) {
            insert_vector_hot_online_counts_[i].store(0, std::memory_order_relaxed);
        }
#if HNSW_INSERT_VECTOR_HOT_ONLINE_STABILITY > 1
        delete[] insert_vector_hot_online_prev_counts_;
        insert_vector_hot_online_prev_counts_ = new std::atomic<uint32_t>[max_elements];
        for (size_t i = 0; i < max_elements; i++) {
            insert_vector_hot_online_prev_counts_[i].store(0, std::memory_order_relaxed);
        }
        insert_vector_hot_epoch_reads_.store(0, std::memory_order_relaxed);
        insert_vector_hot_epoch_id_.store(0, std::memory_order_relaxed);
#endif
        insert_vector_hot_online_next_slot_.store(0, std::memory_order_relaxed);
        insert_vector_hot_online_budget_exhausted_.store(false, std::memory_order_relaxed);
        return insertVectorReplicaSlotsForCount(max_elements);
#elif HNSW_INSERT_VECTOR_GRAPH_HOT
        return insertVectorReplicaSlotsForCount(max_elements);
#else
        const char* path = std::getenv("HNSW_INSERT_VECTOR_HOT_IDS_PATH");
        if (!path || path[0] == '\0') {
            return 0;
        }
        std::ifstream input(path);
        if (!input.good()) {
            throw std::runtime_error(
                std::string("Failed to open HNSW_INSERT_VECTOR_HOT_IDS_PATH=") + path);
        }
        size_t slots = 0;
        std::string line;
        while (std::getline(input, line)) {
            for (char& ch : line) {
                if (ch == ',') {
                    ch = ' ';
                }
            }
            std::istringstream iss(line);
            size_t id = 0;
            if (!(iss >> id)) {
                continue;
            }
            if (id >= max_elements ||
                insert_vector_hot_slots_[id].load(std::memory_order_relaxed) >= 0) {
                continue;
            }
            insert_vector_hot_slots_[id].store(static_cast<int32_t>(slots++),
                                               std::memory_order_relaxed);
        }
        return slots;
#endif
#else
        (void)max_elements;
        return 0;
#endif
    }

    void freeInsertVectorShadow() {
#if HNSW_INSERT_VECTOR_SHADOW || HNSW_INSERT_VECTOR_PARTIAL_SHADOW || \
    HNSW_INSERT_VECTOR_HOT_SHADOW
        free(insert_vector_shadow_alloc_);
        insert_vector_shadow_alloc_ = nullptr;
        insert_vector_shadow_memory_ = nullptr;
        insert_vector_shadow_slots_ = 0;
        delete[] insert_vector_hot_slots_;
        insert_vector_hot_slots_ = nullptr;
        insert_vector_hot_slot_table_size_ = 0;
        delete[] insert_vector_hot_online_counts_;
        insert_vector_hot_online_counts_ = nullptr;
        delete[] insert_vector_hot_online_prev_counts_;
        insert_vector_hot_online_prev_counts_ = nullptr;
        insert_vector_hot_epoch_reads_.store(0, std::memory_order_relaxed);
        insert_vector_hot_epoch_id_.store(0, std::memory_order_relaxed);
        insert_vector_hot_online_next_slot_.store(0, std::memory_order_relaxed);
        insert_vector_hot_online_budget_exhausted_.store(false, std::memory_order_relaxed);
#endif
    }

	    void allocateInsertVectorShadow(size_t max_elements) {
#if HNSW_INSERT_VECTOR_SHADOW || HNSW_INSERT_VECTOR_PARTIAL_SHADOW || \
	    HNSW_INSERT_VECTOR_HOT_SHADOW
        if (data_size_ == 0) {
            return;
        }
#if HNSW_INSERT_VECTOR_HOT_SHADOW
        const size_t slots = loadInsertVectorHotSlots(max_elements);
#else
        const size_t slots = insertVectorReplicaSlotsForCount(max_elements);
#endif
        if (slots == 0) {
            return;
        }
        const size_t bytes = slots * data_size_;
        const size_t extra =
            kLayoutCacheLineBytes + HNSW_INSERT_VECTOR_SHADOW_OFFSET_BYTES;
        char* new_alloc = (char*)malloc(bytes + extra);
        if (new_alloc == nullptr) {
            throw std::runtime_error("Not enough memory: failed to allocate insert vector shadow");
        }
        uintptr_t aligned =
            (reinterpret_cast<uintptr_t>(new_alloc) + kLayoutCacheLineBytes - 1) &
            ~(uintptr_t)(kLayoutCacheLineBytes - 1);
        free(insert_vector_shadow_alloc_);
        insert_vector_shadow_alloc_ = new_alloc;
        insert_vector_shadow_slots_ = slots;
        insert_vector_shadow_memory_ =
            reinterpret_cast<char*>(aligned + HNSW_INSERT_VECTOR_SHADOW_OFFSET_BYTES);
#else
        (void)max_elements;
#endif
	    }

	    size_t insertVectorRegionShadowRegionBytes() const {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        return size_t{1} << HNSW_INSERT_VECTOR_REGION_SHIFT;
#else
	        return 0;
#endif
	    }

	    size_t insertVectorRegionShadowDataBytes() const {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        return max_elements_ * size_data_per_element_;
#else
	        return 0;
#endif
	    }

	    void freeInsertVectorRegionShadow() {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        free(insert_vector_region_shadow_alloc_);
	        insert_vector_region_shadow_alloc_ = nullptr;
	        insert_vector_region_shadow_memory_ = nullptr;
	        insert_vector_region_shadow_slots_ = 0;
	        insert_vector_region_shadow_bytes_ = 0;
	        insert_vector_region_shadow_region_bytes_ = 0;
	        insert_vector_region_count_ = 0;
	        insert_vector_region_shadow_max_id_ = 0;
	        delete[] insert_vector_region_shadow_slot_table_;
	        insert_vector_region_shadow_slot_table_ = nullptr;
	        delete[] insert_vector_region_shadow_counts_;
	        insert_vector_region_shadow_counts_ = nullptr;
	        insert_vector_region_shadow_next_slot_.store(0, std::memory_order_relaxed);
	        insert_vector_region_shadow_budget_exhausted_.store(false,
	                                                            std::memory_order_relaxed);
#endif
	    }

	    void allocateInsertVectorRegionShadow(size_t max_elements) {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        if (data_size_ == 0 || size_data_per_element_ == 0) {
	            return;
	        }
	        freeInsertVectorRegionShadow();
	        const size_t region_bytes = insertVectorRegionShadowRegionBytes();
	        const size_t data_bytes = max_elements * size_data_per_element_;
	        const size_t region_count = (data_bytes + region_bytes - 1) / region_bytes;
	        const size_t requested_slots =
	            HNSW_INSERT_VECTOR_REGION_SHADOW_BYTES / region_bytes;
	        const size_t slots = std::min(region_count, requested_slots);
	        if (slots == 0 || region_count == 0) {
	            return;
	        }

	        const size_t bytes = slots * region_bytes;
	        char* new_alloc = (char*)malloc(bytes + kLayoutCacheLineBytes);
	        if (new_alloc == nullptr) {
	            throw std::runtime_error(
	                "Not enough memory: failed to allocate insert vector region shadow");
	        }
	        uintptr_t aligned =
	            (reinterpret_cast<uintptr_t>(new_alloc) + kLayoutCacheLineBytes - 1) &
	            ~(uintptr_t)(kLayoutCacheLineBytes - 1);
	        insert_vector_region_shadow_alloc_ = new_alloc;
	        insert_vector_region_shadow_memory_ = reinterpret_cast<char*>(aligned);
	        insert_vector_region_shadow_slots_ = slots;
	        insert_vector_region_shadow_bytes_ = bytes;
	        insert_vector_region_shadow_region_bytes_ = region_bytes;
	        insert_vector_region_count_ = region_count;
	        insert_vector_region_shadow_max_id_ = max_elements;

	        insert_vector_region_shadow_slot_table_ = new std::atomic<int32_t>[region_count];
	        insert_vector_region_shadow_counts_ = new std::atomic<uint32_t>[region_count];
	        for (size_t i = 0; i < region_count; i++) {
	            insert_vector_region_shadow_slot_table_[i].store(-1, std::memory_order_relaxed);
	            insert_vector_region_shadow_counts_[i].store(0, std::memory_order_relaxed);
	        }
	        insert_vector_region_shadow_next_slot_.store(0, std::memory_order_relaxed);
	        insert_vector_region_shadow_budget_exhausted_.store(false,
	                                                            std::memory_order_relaxed);
#else
	        (void)max_elements;
#endif
	    }

	    inline size_t insertVectorPayloadOffset(tableint internal_id) const {
	        return static_cast<size_t>(internal_id) * size_data_per_element_ + offsetData_;
	    }

	    inline bool insertVectorRegionForId(tableint internal_id, size_t* region,
	                                        size_t* offset) const {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        if (!insert_vector_region_shadow_memory_ || internal_id >= max_elements_ ||
	            internal_id >= insert_vector_region_shadow_max_id_) {
	            return false;
	        }
	        const size_t payload_offset = insertVectorPayloadOffset(internal_id);
	        const size_t region_bytes = insert_vector_region_shadow_region_bytes_;
	        const size_t region_id = payload_offset >> HNSW_INSERT_VECTOR_REGION_SHIFT;
	        if (region_id >= insert_vector_region_count_) {
	            return false;
	        }
	        *region = region_id;
	        *offset = payload_offset & (region_bytes - 1);
	        return true;
#else
	        (void)internal_id;
	        (void)region;
	        (void)offset;
	        return false;
#endif
	    }

	    void copyInsertVectorRegion(tableint internal_id, const void* data_point) const {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        size_t region = 0;
	        size_t offset = 0;
	        if (!insertVectorRegionForId(internal_id, &region, &offset) ||
	            offset + data_size_ > insert_vector_region_shadow_region_bytes_) {
	            return;
	        }
	        const int32_t slot =
	            insert_vector_region_shadow_slot_table_[region].load(
	                std::memory_order_acquire);
	        if (slot < 0) {
	            return;
	        }
	        memcpy(insert_vector_region_shadow_memory_ +
	                   static_cast<size_t>(slot) *
	                       insert_vector_region_shadow_region_bytes_ +
	                   offset,
	               data_point, data_size_);
#else
	        (void)internal_id;
	        (void)data_point;
#endif
	    }

	    void promoteInsertVectorRegion(size_t region) const {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        if (insert_vector_region_shadow_budget_exhausted_.load(
	                std::memory_order_relaxed)) {
	            return;
	        }
	        int32_t expected = -1;
	        if (!insert_vector_region_shadow_slot_table_[region].compare_exchange_strong(
	                expected, -2, std::memory_order_acq_rel, std::memory_order_acquire)) {
	            return;
	        }
	        const uint32_t slot =
	            insert_vector_region_shadow_next_slot_.fetch_add(1, std::memory_order_relaxed);
	        if (slot >= insert_vector_region_shadow_slots_) {
	            metric_insert_vector_region_shadow_budget_full.fetch_add(
	                1, std::memory_order_relaxed);
	            insert_vector_region_shadow_budget_exhausted_.store(
	                true, std::memory_order_relaxed);
	            insert_vector_region_shadow_slot_table_[region].store(
	                -1, std::memory_order_release);
	            return;
	        }
	        const auto copy_start = std::chrono::steady_clock::now();
	        const size_t region_bytes = insert_vector_region_shadow_region_bytes_;
	        const size_t src_offset = region * region_bytes;
	        const size_t total_bytes = insertVectorRegionShadowDataBytes();
	        const size_t copy_bytes = std::min(region_bytes, total_bytes - src_offset);
	        char* dst = insert_vector_region_shadow_memory_ +
	                    static_cast<size_t>(slot) * region_bytes;
	        memcpy(dst, data_level0_memory_ + src_offset, copy_bytes);
	        if (copy_bytes < region_bytes) {
	            memset(dst + copy_bytes, 0, region_bytes - copy_bytes);
	        }
	        const auto copy_end = std::chrono::steady_clock::now();
	        metric_insert_vector_region_shadow_copy_ns.fetch_add(
	            static_cast<uint64_t>(
	                std::chrono::duration_cast<std::chrono::nanoseconds>(
	                    copy_end - copy_start)
	                    .count()),
	            std::memory_order_relaxed);
	        insert_vector_region_shadow_slot_table_[region].store(
	            static_cast<int32_t>(slot), std::memory_order_release);
	        metric_insert_vector_region_shadow_promotions.fetch_add(
	            1, std::memory_order_relaxed);
#else
	        (void)region;
#endif
	    }

	    inline void observeInsertVectorRegionShadow(tableint internal_id) const {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        if (!insert_vector_read_admission_enabled_.load(std::memory_order_relaxed)) {
	            return;
	        }
	        size_t region = 0;
	        size_t offset = 0;
	        if (!insertVectorRegionForId(internal_id, &region, &offset)) {
	            return;
	        }
	        if (offset + data_size_ > insert_vector_region_shadow_region_bytes_) {
	            metric_insert_vector_region_shadow_cross_region.fetch_add(
	                1, std::memory_order_relaxed);
	            return;
	        }
	        if (insert_vector_region_shadow_slot_table_[region].load(
	                std::memory_order_acquire) >= 0) {
	            return;
	        }
	        static thread_local uint64_t sample_seq = 0;
	        if ((++sample_seq & HNSW_INSERT_VECTOR_REGION_SAMPLE_MASK) != 0) {
	            return;
	        }
	        metric_insert_vector_region_shadow_samples.fetch_add(1,
	                                                            std::memory_order_relaxed);
	        const uint32_t count =
	            insert_vector_region_shadow_counts_[region].fetch_add(
	                1, std::memory_order_relaxed) +
	            1;
	        if (count >= HNSW_INSERT_VECTOR_REGION_THRESHOLD) {
	            promoteInsertVectorRegion(region);
	        }
#else
	        (void)internal_id;
#endif
	    }

	    inline char* getInsertVectorRegionShadowData(tableint internal_id) const {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
#if HNSW_INSERT_VECTOR_REGION_STATS
	        static thread_local uint64_t region_shadow_stats_seq = 0;
	        const bool record_region_lookup =
	            ((++region_shadow_stats_seq &
	              HNSW_INSERT_VECTOR_REGION_STATS_SAMPLE_MASK) == 0);
#endif
	        size_t region = 0;
	        size_t offset = 0;
	        if (!insertVectorRegionForId(internal_id, &region, &offset) ||
	            offset + data_size_ > insert_vector_region_shadow_region_bytes_) {
#if HNSW_INSERT_VECTOR_REGION_STATS
	            if (record_region_lookup) {
	                metric_insert_vector_region_shadow_fallbacks.fetch_add(
	                    1, std::memory_order_relaxed);
	            }
#endif
	            return nullptr;
	        }
	        const int32_t slot =
	            insert_vector_region_shadow_slot_table_[region].load(
	                std::memory_order_acquire);
	        if (slot < 0) {
#if HNSW_INSERT_VECTOR_REGION_STATS
	            if (record_region_lookup) {
	                metric_insert_vector_region_shadow_fallbacks.fetch_add(
	                    1, std::memory_order_relaxed);
	            }
#endif
	            return nullptr;
	        }
#if HNSW_INSERT_VECTOR_REGION_STATS
	        if (record_region_lookup) {
	            metric_insert_vector_region_shadow_hits.fetch_add(
	                1, std::memory_order_relaxed);
	        }
#endif
	        return insert_vector_region_shadow_memory_ +
	               static_cast<size_t>(slot) * insert_vector_region_shadow_region_bytes_ +
	               offset;
#else
	        (void)internal_id;
	        return nullptr;
#endif
	    }

    void rebuildInsertVectorShadow(size_t count) {
#if HNSW_INSERT_VECTOR_SHADOW || HNSW_INSERT_VECTOR_PARTIAL_SHADOW || \
    HNSW_INSERT_VECTOR_HOT_SHADOW
        if (!insert_vector_shadow_memory_) {
            return;
        }
        for (size_t i = 0; i < count; i++) {
            if (insertVectorHasReplicaSlot(static_cast<tableint>(i))) {
                const size_t slot = insertVectorReplicaSlot(static_cast<tableint>(i));
                memcpy(insert_vector_shadow_memory_ + slot * data_size_,
                       data_level0_memory_ + i * size_data_per_element_ + offsetData_,
                       data_size_);
            }
        }
#else
        (void)count;
#endif
    }

    void buildInsertVectorGraphHotReplica() {
#if HNSW_INSERT_VECTOR_HOT_SHADOW && HNSW_INSERT_VECTOR_GRAPH_HOT
        if (!insert_vector_shadow_memory_ || !insert_vector_hot_slots_) {
            return;
        }
        const auto select_start = std::chrono::steady_clock::now();
        const size_t n = cur_element_count.load(std::memory_order_acquire);
        const size_t budget = std::min(insert_vector_shadow_slots_, n);
        if (budget == 0) {
            return;
        }

        std::vector<uint32_t> in0(n, 0);
        std::vector<uint32_t> upper_in(n, 0);
        std::vector<uint16_t> out0(n, 0);
        std::vector<uint16_t> upper_out(n, 0);
        uint64_t scanned_edges = 0;

        for (tableint i = 0; i < static_cast<tableint>(n); i++) {
            linklistsizeint* ll0 = get_linklist0(i);
            const size_t sz0 = getListCount(ll0);
            out0[i] = static_cast<uint16_t>(std::min<size_t>(sz0, UINT16_MAX));
            tableint* data = reinterpret_cast<tableint*>(ll0 + 1);
            for (size_t j = 0; j < sz0; j++) {
                const tableint neighbor = data[j];
                if (neighbor < n) {
                    in0[neighbor]++;
                }
                scanned_edges++;
            }
            const int level = i < element_levels_.size() ? element_levels_[i] : 0;
            if (level <= 0 || linkLists_[i] == nullptr) {
                continue;
            }
            uint32_t upper_degree = 0;
            for (int l = 1; l <= level; l++) {
                linklistsizeint* ll = get_linklist(i, l);
                const size_t sz = getListCount(ll);
                upper_degree += static_cast<uint32_t>(std::min<size_t>(sz, UINT32_MAX));
                tableint* upper_data = reinterpret_cast<tableint*>(ll + 1);
                for (size_t j = 0; j < sz; j++) {
                    const tableint neighbor = upper_data[j];
                    if (neighbor < n) {
                        upper_in[neighbor]++;
                    }
                    scanned_edges++;
                }
            }
            upper_out[i] =
                static_cast<uint16_t>(std::min<uint32_t>(upper_degree, UINT16_MAX));
        }

        struct GraphHotRank {
            tableint id;
            uint32_t in0;
            uint32_t upper_in;
            uint16_t out0;
            uint16_t upper_out;
            int level;
        };
        std::vector<GraphHotRank> ranks;
        ranks.reserve(n);
        for (tableint i = 0; i < static_cast<tableint>(n); i++) {
            ranks.push_back(GraphHotRank{
                i,
                in0[i],
                upper_in[i],
                out0[i],
                upper_out[i],
                i < element_levels_.size() ? element_levels_[i] : 0});
        }
        std::sort(ranks.begin(), ranks.end(), [](const GraphHotRank& a,
                                                 const GraphHotRank& b) {
#if HNSW_INSERT_VECTOR_GRAPH_HOT_MODE == 1
            if (a.level != b.level) return a.level > b.level;
#endif
            if (a.in0 != b.in0) return a.in0 > b.in0;
            if (a.upper_in != b.upper_in) return a.upper_in > b.upper_in;
            if (a.level != b.level) return a.level > b.level;
            if (a.out0 != b.out0) return a.out0 > b.out0;
            if (a.upper_out != b.upper_out) return a.upper_out > b.upper_out;
            return a.id < b.id;
        });
        const auto copy_start = std::chrono::steady_clock::now();
        size_t selected = 0;
        for (size_t slot = 0; slot < budget; slot++) {
            const tableint id = ranks[slot].id;
            memcpy(insert_vector_shadow_memory_ + slot * data_size_,
                   data_level0_memory_ + static_cast<size_t>(id) *
                                             size_data_per_element_ +
                       offsetData_,
                   data_size_);
            insert_vector_hot_slots_[id].store(static_cast<int32_t>(slot),
                                               std::memory_order_release);
            selected++;
        }
        const auto end = std::chrono::steady_clock::now();
        metric_insert_vector_graph_hot_select_ns.store(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    copy_start - select_start)
                    .count()),
            std::memory_order_relaxed);
        metric_insert_vector_graph_hot_copy_ns.store(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - copy_start)
                    .count()),
            std::memory_order_relaxed);
        metric_insert_vector_graph_hot_scanned_edges.store(scanned_edges,
                                                           std::memory_order_relaxed);
        metric_insert_vector_graph_hot_selected.store(selected,
                                                      std::memory_order_relaxed);
        metric_insert_vector_graph_hot_candidates.store(n, std::memory_order_relaxed);
#endif
	    }

	    struct InsertVectorThreadCacheState {
	        const void* owner{nullptr};
	        size_t data_size{0};
	        std::vector<char> data;
	        std::vector<tableint> tags;
	        std::vector<tableint> filter_tags;
	        std::vector<uint8_t> filter_counts;
	        bool pending{false};
	        tableint pending_id{0};
	        uint64_t stats_seq{0};
	    };

	    static constexpr tableint insertVectorThreadCacheInvalidTag() {
	        return std::numeric_limits<tableint>::max();
	    }

	    static inline size_t insertVectorThreadCacheHash(tableint internal_id) {
	        uint64_t x = static_cast<uint64_t>(internal_id);
	        x ^= x >> 33;
	        x *= 0xff51afd7ed558ccdULL;
	        x ^= x >> 33;
	        x *= 0xc4ceb9fe1a85ec53ULL;
	        x ^= x >> 33;
	        return static_cast<size_t>(x);
	    }

	    inline InsertVectorThreadCacheState& ensureInsertVectorThreadCache() const {
	        static thread_local InsertVectorThreadCacheState* state_ptr = nullptr;
	        if (state_ptr == nullptr) {
	            state_ptr = new InsertVectorThreadCacheState();
	        }
	        auto& state = *state_ptr;
	        if (state.owner != this || state.data_size != data_size_) {
	            state.owner = this;
	            state.data_size = data_size_;
	            state.data.assign(
	                static_cast<size_t>(HNSW_INSERT_VECTOR_THREAD_CACHE_SLOTS) *
	                    data_size_,
	                0);
	            state.tags.assign(HNSW_INSERT_VECTOR_THREAD_CACHE_SLOTS,
	                              insertVectorThreadCacheInvalidTag());
	            state.filter_tags.assign(HNSW_INSERT_VECTOR_THREAD_FILTER_SLOTS,
	                                     insertVectorThreadCacheInvalidTag());
	            state.filter_counts.assign(HNSW_INSERT_VECTOR_THREAD_FILTER_SLOTS, 0);
	            state.pending = false;
	            state.pending_id = 0;
	            state.stats_seq = 0;
	        }
	        return state;
	    }

	    inline void flushInsertVectorThreadCachePending(
	        InsertVectorThreadCacheState& state) const {
#if HNSW_INSERT_VECTOR_THREAD_CACHE
	        if (!state.pending) {
	            return;
	        }
	        const tableint pending_id = state.pending_id;
	        state.pending = false;
	        if (pending_id >= max_elements_) {
	            return;
	        }
	        const size_t cache_slot =
	            insertVectorThreadCacheHash(pending_id) &
	            (HNSW_INSERT_VECTOR_THREAD_CACHE_SLOTS - 1);
	        char* dst = state.data.data() + cache_slot * data_size_;
	        char* src = getDataByInternalId(pending_id);
#if HNSW_INSERT_VECTOR_THREAD_CACHE_STREAM_FILL && defined(USE_SSE)
	        if ((reinterpret_cast<uintptr_t>(dst) & 15U) == 0) {
	            size_t offset = 0;
	            for (; offset + 16 <= data_size_; offset += 16) {
	                __m128i value = _mm_loadu_si128(
	                    reinterpret_cast<const __m128i*>(src + offset));
	                _mm_stream_si128(reinterpret_cast<__m128i*>(dst + offset),
	                                 value);
	            }
	            if (offset < data_size_) {
	                memcpy(dst + offset, src + offset, data_size_ - offset);
	            }
	            _mm_sfence();
	        } else {
	            memcpy(dst, src, data_size_);
	        }
#else
	        memcpy(dst, src, data_size_);
#endif
	        state.tags[cache_slot] = pending_id;
#if HNSW_INSERT_VECTOR_THREAD_CACHE_STATS
	        metric_insert_vector_thread_cache_copies.fetch_add(
	            1, std::memory_order_relaxed);
#endif
#else
	        (void)state;
#endif
	    }

	    inline char* getInsertVectorThreadCacheData(tableint internal_id) const {
#if HNSW_INSERT_VECTOR_THREAD_CACHE
	        if (!insert_vector_read_admission_enabled_.load(std::memory_order_relaxed)) {
	            return nullptr;
	        }
	        auto& state = ensureInsertVectorThreadCache();
	        flushInsertVectorThreadCachePending(state);
	        const bool record_stats =
#if HNSW_INSERT_VECTOR_THREAD_CACHE_STATS
	            ((++state.stats_seq &
	              HNSW_INSERT_VECTOR_THREAD_CACHE_STATS_SAMPLE_MASK) == 0);
#else
	            false;
#endif
	        const size_t cache_slot =
	            insertVectorThreadCacheHash(internal_id) &
	            (HNSW_INSERT_VECTOR_THREAD_CACHE_SLOTS - 1);
	        if (state.tags[cache_slot] == internal_id) {
#if HNSW_INSERT_VECTOR_THREAD_CACHE_STATS
	            if (record_stats) {
	                metric_insert_vector_thread_cache_hits.fetch_add(
	                    1, std::memory_order_relaxed);
	            }
#endif
	            return state.data.data() + cache_slot * data_size_;
	        }
#if HNSW_INSERT_VECTOR_THREAD_CACHE_STATS
	        if (record_stats) {
	            metric_insert_vector_thread_cache_fallbacks.fetch_add(
	                1, std::memory_order_relaxed);
	        }
#endif
	        const size_t filter_slot =
	            insertVectorThreadCacheHash(internal_id) &
	            (HNSW_INSERT_VECTOR_THREAD_FILTER_SLOTS - 1);
	        if (state.filter_tags[filter_slot] == internal_id) {
	            uint8_t& count = state.filter_counts[filter_slot];
	            if (count < HNSW_INSERT_VECTOR_THREAD_CACHE_THRESHOLD) {
	                ++count;
	            }
#if HNSW_INSERT_VECTOR_THREAD_CACHE_STATS
	            if (record_stats) {
	                metric_insert_vector_thread_cache_filter_hits.fetch_add(
	                    1, std::memory_order_relaxed);
	            }
#endif
	            if (count == HNSW_INSERT_VECTOR_THREAD_CACHE_THRESHOLD) {
	                state.pending = true;
	                state.pending_id = internal_id;
	                if (count < 255) {
	                    ++count;
	                }
#if HNSW_INSERT_VECTOR_THREAD_CACHE_STATS
	                metric_insert_vector_thread_cache_admissions.fetch_add(
	                    1, std::memory_order_relaxed);
#endif
	            }
	        } else {
	            state.filter_tags[filter_slot] = internal_id;
	            state.filter_counts[filter_slot] = 1;
	        }
	        return nullptr;
#else
	        (void)internal_id;
	        return nullptr;
#endif
	    }

	    inline void copyInsertVectorShadow(tableint internal_id, const void* data_point) const {
#if HNSW_INSERT_VECTOR_SHADOW || HNSW_INSERT_VECTOR_PARTIAL_SHADOW || \
	    HNSW_INSERT_VECTOR_HOT_SHADOW
	        if (insert_vector_shadow_memory_ && insertVectorHasReplicaSlot(internal_id)) {
	            const size_t slot = insertVectorReplicaSlot(internal_id);
	            memcpy(insert_vector_shadow_memory_ + slot * data_size_, data_point, data_size_);
	        }
#endif
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        copyInsertVectorRegion(internal_id, data_point);
#else
	        (void)internal_id;
	        (void)data_point;
#endif
	    }

	    inline char* getInsertDataByInternalId(tableint internal_id) const {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        if (char* region_data = getInsertVectorRegionShadowData(internal_id)) {
	            return region_data;
	        }
#endif
#if HNSW_INSERT_VECTOR_SHADOW || HNSW_INSERT_VECTOR_PARTIAL_SHADOW || \
	    HNSW_INSERT_VECTOR_HOT_SHADOW
	        if (insert_vector_shadow_memory_ && insertVectorHasReplicaSlot(internal_id)) {
            return insert_vector_shadow_memory_ + insertVectorReplicaSlot(internal_id) * data_size_;
        }
#endif
        return getDataByInternalId(internal_id);
    }

    inline bool insertDataUsesReplica(tableint internal_id) const {
#if HNSW_INSERT_VECTOR_SHADOW || HNSW_INSERT_VECTOR_PARTIAL_SHADOW || \
    HNSW_INSERT_VECTOR_HOT_SHADOW
        return insert_vector_shadow_memory_ && insertVectorHasReplicaSlot(internal_id);
#else
        (void)internal_id;
        return false;
#endif
    }

    inline void observeInsertVectorHotOnline(tableint internal_id) const {
#if HNSW_INSERT_VECTOR_HOT_SHADOW && HNSW_INSERT_VECTOR_HOT_ONLINE
        if (!insert_vector_shadow_memory_ || !insert_vector_hot_slots_ ||
            !insert_vector_hot_online_counts_ ||
            internal_id >= insert_vector_hot_slot_table_size_) {
            return;
        }
        if (insert_vector_hot_slots_[internal_id].load(std::memory_order_acquire) >= 0) {
            return;
        }
        static thread_local uint64_t sample_seq = 0;
        if ((++sample_seq & HNSW_INSERT_VECTOR_HOT_ONLINE_SAMPLE_MASK) != 0) {
            return;
        }
        metric_insert_vector_hot_online_samples.fetch_add(1, std::memory_order_relaxed);
        const uint32_t count =
            insert_vector_hot_online_counts_[internal_id].fetch_add(
                1, std::memory_order_relaxed) +
            1;
        if (count < HNSW_INSERT_VECTOR_HOT_ONLINE_THRESHOLD ||
            insert_vector_hot_online_budget_exhausted_.load(std::memory_order_relaxed)) {
#if HNSW_INSERT_VECTOR_HOT_ONLINE_EPOCH_SIZE > 0
            maybeRunHotEpochTransition();
#endif
            return;
        }
#if HNSW_INSERT_VECTOR_HOT_ONLINE_STABILITY > 1
        if (!insert_vector_hot_online_prev_counts_ ||
            insert_vector_hot_online_prev_counts_[internal_id].load(
                std::memory_order_relaxed) <
                HNSW_INSERT_VECTOR_HOT_ONLINE_THRESHOLD) {
            metric_insert_vector_hot_epoch_skipped_stability.fetch_add(
                1, std::memory_order_relaxed);
#if HNSW_INSERT_VECTOR_HOT_ONLINE_EPOCH_SIZE > 0
            maybeRunHotEpochTransition();
#endif
            return;
        }
#endif
        int32_t expected = -1;
        if (!insert_vector_hot_slots_[internal_id].compare_exchange_strong(
                expected, -2, std::memory_order_acq_rel, std::memory_order_acquire)) {
            return;
        }
        const uint32_t slot = insert_vector_hot_online_next_slot_.fetch_add(
            1, std::memory_order_relaxed);
        if (slot >= insert_vector_shadow_slots_) {
            metric_insert_vector_hot_online_budget_full.fetch_add(
                1, std::memory_order_relaxed);
            insert_vector_hot_online_budget_exhausted_.store(
                true, std::memory_order_relaxed);
            insert_vector_hot_slots_[internal_id].store(-1, std::memory_order_release);
            return;
        }
        memcpy(insert_vector_shadow_memory_ + static_cast<size_t>(slot) * data_size_,
               data_level0_memory_ + static_cast<size_t>(internal_id) *
                                         size_data_per_element_ +
                   offsetData_,
               data_size_);
        insert_vector_hot_slots_[internal_id].store(static_cast<int32_t>(slot),
                                                    std::memory_order_release);
        metric_insert_vector_hot_online_promotions.fetch_add(
            1, std::memory_order_relaxed);
#if HNSW_INSERT_VECTOR_HOT_ONLINE_EPOCH_SIZE > 0
        maybeRunHotEpochTransition();
#endif
#else
        (void)internal_id;
#endif
    }

#if HNSW_INSERT_VECTOR_HOT_SHADOW && HNSW_INSERT_VECTOR_HOT_ONLINE && \
    HNSW_INSERT_VECTOR_HOT_ONLINE_EPOCH_SIZE > 0
    inline void maybeRunHotEpochTransition() const {
        uint64_t reads = insert_vector_hot_epoch_reads_.fetch_add(
            1, std::memory_order_relaxed) + 1;
        if (reads < static_cast<uint64_t>(HNSW_INSERT_VECTOR_HOT_ONLINE_EPOCH_SIZE)) {
            return;
        }
        std::unique_lock<std::mutex> lk(insert_vector_hot_epoch_mu_, std::try_to_lock);
        if (!lk.owns_lock()) {
            return;
        }
        if (insert_vector_hot_epoch_reads_.load(std::memory_order_relaxed) <
            static_cast<uint64_t>(HNSW_INSERT_VECTOR_HOT_ONLINE_EPOCH_SIZE)) {
            return;
        }
        // Promote curr -> prev, zero curr.
        if (insert_vector_hot_online_prev_counts_ &&
            insert_vector_hot_online_counts_) {
            const size_t n = insert_vector_hot_slot_table_size_;
            for (size_t i = 0; i < n; i++) {
                uint32_t c = insert_vector_hot_online_counts_[i].load(
                    std::memory_order_relaxed);
                insert_vector_hot_online_prev_counts_[i].store(
                    c, std::memory_order_relaxed);
                insert_vector_hot_online_counts_[i].store(
                    0, std::memory_order_relaxed);
            }
        }
        insert_vector_hot_epoch_id_.fetch_add(1, std::memory_order_relaxed);
        insert_vector_hot_epoch_reads_.store(0, std::memory_order_relaxed);
        metric_insert_vector_hot_epoch_swaps.fetch_add(
            1, std::memory_order_relaxed);
    }
#endif

    inline void prefetchInsertDataByInternalId(tableint internal_id) const {
#if defined(USE_SSE)
        char* data = getInsertDataByInternalId(internal_id);
#if HNSW_INSERT_VECTOR_COLD_PREFETCH
        if (insertDataUsesReplica(internal_id)) {
            _mm_prefetch(data, HNSW_INSERT_SEARCH_PREFETCH_HINT_VALUE);
        } else {
            _mm_prefetch(data, _MM_HINT_NTA);
        }
#else
        _mm_prefetch(data, HNSW_INSERT_SEARCH_PREFETCH_HINT_VALUE);
#endif
#if HNSW_INSERT_ADJ_LOOKAHEAD_PREFETCH
        // Lookahead: prefetch the adjacency block too so next outer-loop
        // iteration's get_linklist0 read finds it warm.
        char* adj = (char*)(data_level0_memory_ +
                            static_cast<size_t>(internal_id) *
                                size_data_per_element_ +
                            offsetLevel0_);
        _mm_prefetch(adj, HNSW_INSERT_SEARCH_PREFETCH_HINT_VALUE);
        if (size_links_level0_ > 64) {
            _mm_prefetch(adj + 64, HNSW_INSERT_SEARCH_PREFETCH_HINT_VALUE);
        }
        if (size_links_level0_ > 128) {
            _mm_prefetch(adj + 128, HNSW_INSERT_SEARCH_PREFETCH_HINT_VALUE);
        }
#endif
#else
        (void)internal_id;
#endif
    }

	    inline char* getInsertDataForDistance(tableint internal_id) const {
#if HNSW_INSERT_VECTOR_READ_TRACE
	        recordInsertVectorRead(internal_id);
#endif
	        if (char* thread_cache_data =
	                getInsertVectorThreadCacheData(internal_id)) {
	            return thread_cache_data;
	        }
	        observeInsertVectorRegionShadow(internal_id);
	        observeInsertVectorHotOnline(internal_id);
#if HNSW_INSERT_VECTOR_COLD_PREFETCH
	        prefetchInsertDataByInternalId(internal_id);
#endif
        return getInsertDataByInternalId(internal_id);
    }

    inline dist_t insertConstructionDistance(const void* data_point,
                                             tableint internal_id) const {
        char* data = getInsertDataForDistance(internal_id);
#if HNSW_INSERT_SEARCH_NT_DISTANCE && defined(__AVX2__)
        if constexpr (std::is_same<dist_t, float>::value) {
            if (shouldUseInsertNtDistance(internal_id)) {
                return static_cast<dist_t>(
                    l2SqrInsertNtAligned(data_point, data, dist_func_param_));
            }
        }
#endif
        return fstdistfunc_(data_point, data, dist_func_param_);
    }

#if HNSW_INSERT_SEARCH_NT_DISTANCE && defined(__AVX2__)
    inline bool shouldUseInsertNtDistance(tableint internal_id) const {
#if HNSW_INSERT_SEARCH_NT_LEVEL0_ONLY
        if (internal_id < element_levels_.size() && element_levels_[internal_id] > 0) {
            return false;
        }
#else
        (void)internal_id;
#endif
        return true;
    }

    static float l2SqrInsertNtAligned(const void* pVect1v, const void* pVect2v,
                                      const void* qty_ptr) {
        const float* pVect1 = static_cast<const float*>(pVect1v);
        const float* pVect2 = static_cast<const float*>(pVect2v);
        const size_t qty = *static_cast<const size_t*>(qty_ptr);
        const uintptr_t p2_addr = reinterpret_cast<uintptr_t>(pVect2);

        float PORTABLE_ALIGN32 tmp[8];
        const size_t qty16 = qty >> 4;
        const float* pEnd1 = pVect1 + (qty16 << 4);
        __m256 sum = _mm256_setzero_ps();
        if ((p2_addr & 31U) == 0) {
            while (pVect1 < pEnd1) {
                __m256 v1 = _mm256_loadu_ps(pVect1);
                __m256 v2 = _mm256_castsi256_ps(_mm256_stream_load_si256(
                    reinterpret_cast<const __m256i*>(pVect2)));
                __m256 diff = _mm256_sub_ps(v1, v2);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
                pVect1 += 8;
                pVect2 += 8;

                v1 = _mm256_loadu_ps(pVect1);
                v2 = _mm256_castsi256_ps(_mm256_stream_load_si256(
                    reinterpret_cast<const __m256i*>(pVect2)));
                diff = _mm256_sub_ps(v1, v2);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
                pVect1 += 8;
                pVect2 += 8;
            }
        } else {
            while (pVect1 < pEnd1) {
                __m256 v1 = _mm256_loadu_ps(pVect1);
                __m256 v2 = _mm256_loadu_ps(pVect2);
                __m256 diff = _mm256_sub_ps(v1, v2);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
                pVect1 += 8;
                pVect2 += 8;

                v1 = _mm256_loadu_ps(pVect1);
                v2 = _mm256_loadu_ps(pVect2);
                diff = _mm256_sub_ps(v1, v2);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
                pVect1 += 8;
                pVect2 += 8;
            }
        }
        _mm256_store_ps(tmp, sum);
        float res = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] +
                    tmp[6] + tmp[7];
        const float* pEnd = static_cast<const float*>(pVect1v) + qty;
        while (pVect1 < pEnd) {
            const float diff = *pVect1 - *pVect2;
            res += diff * diff;
            ++pVect1;
            ++pVect2;
        }
        return res;
    }
#endif

    void allocateInsertVectorReadTrace(size_t max_elements) {
#if HNSW_INSERT_VECTOR_READ_TRACE
        insert_vector_read_counts_ = new std::atomic<uint64_t>[max_elements];
        for (size_t i = 0; i < max_elements; i++) {
            insert_vector_read_counts_[i].store(0, std::memory_order_relaxed);
        }
#else
        (void)max_elements;
#endif
    }

    void freeInsertVectorReadTrace() {
#if HNSW_INSERT_VECTOR_READ_TRACE
        delete[] insert_vector_read_counts_;
        insert_vector_read_counts_ = nullptr;
#endif
    }

    void setInsertVectorReadTraceEnabled(bool enabled) const {
#if HNSW_INSERT_VECTOR_READ_TRACE
        insert_vector_read_trace_enabled_.store(enabled, std::memory_order_release);
#else
        (void)enabled;
#endif
    }

    void recordInsertVectorRead(tableint internal_id) const {
#if HNSW_INSERT_VECTOR_READ_TRACE
        if (!insert_vector_read_trace_enabled_.load(std::memory_order_acquire)) {
            return;
        }
        if (insert_vector_read_counts_ && internal_id < max_elements_) {
            insert_vector_read_counts_[internal_id].fetch_add(
                1, std::memory_order_relaxed);
            metric_insert_vector_read_trace_reads.fetch_add(
                1, std::memory_order_relaxed);
        }
#else
        (void)internal_id;
#endif
    }

    uint64_t insertVectorReadTraceReads() const {
        return metric_insert_vector_read_trace_reads.load(std::memory_order_relaxed);
    }

    void resetInsertVectorReadTrace() const {
#if HNSW_INSERT_VECTOR_READ_TRACE
        if (!insert_vector_read_counts_) {
            return;
        }
        for (size_t i = 0; i < max_elements_; i++) {
            insert_vector_read_counts_[i].store(0, std::memory_order_relaxed);
        }
        metric_insert_vector_read_trace_reads.store(0, std::memory_order_relaxed);
#endif
    }

    void dumpInsertVectorReadTrace(const std::string& path, size_t top_k) const {
#if HNSW_INSERT_VECTOR_READ_TRACE
        if (!insert_vector_read_counts_ || path.empty()) {
            return;
        }
        std::vector<std::pair<uint64_t, tableint>> counts;
        counts.reserve(cur_element_count);
        uint64_t total = 0;
        for (tableint i = 0; i < cur_element_count; i++) {
            uint64_t c = insert_vector_read_counts_[i].load(std::memory_order_relaxed);
            if (c == 0) {
                continue;
            }
            total += c;
            counts.emplace_back(c, i);
        }
        std::sort(counts.begin(), counts.end(),
                  [](const auto& a, const auto& b) {
                      if (a.first != b.first) {
                          return a.first > b.first;
                      }
                      return a.second < b.second;
                  });
        if (top_k == 0 || top_k > counts.size()) {
            top_k = counts.size();
        }
        std::ofstream output(path);
        if (!output.good()) {
            throw std::runtime_error("Failed to open insert vector read trace output");
        }
        output << "id,count,cum_count,cum_frac,level\n";
        uint64_t cum = 0;
        for (size_t i = 0; i < top_k; i++) {
            cum += counts[i].first;
            const tableint id = counts[i].second;
            const int level = id < element_levels_.size() ? element_levels_[id] : -1;
            output << id << "," << counts[i].first << "," << cum << ",";
            output << (total == 0 ? 0.0 : static_cast<double>(cum) / total);
            output << "," << level << "\n";
        }
#else
        (void)path;
        (void)top_k;
#endif
    }

    void allocateSearchActiveNodeTable(size_t max_elements) {
#if HNSW_SEARCH_ACTIVE_NODE_TRACKING
        search_active_node_counts_ = new std::atomic<uint16_t>[max_elements];
        for (size_t i = 0; i < max_elements; i++) {
            search_active_node_counts_[i].store(0, std::memory_order_relaxed);
        }
#else
        (void)max_elements;
#endif
    }

    void resizeSearchActiveNodeTable(size_t new_max_elements) {
#if HNSW_SEARCH_ACTIVE_NODE_TRACKING
        std::atomic<uint16_t>* new_counts =
            new std::atomic<uint16_t>[new_max_elements];
        for (size_t i = 0; i < new_max_elements; i++) {
            new_counts[i].store(0, std::memory_order_relaxed);
        }
        delete[] search_active_node_counts_;
        search_active_node_counts_ = new_counts;
#else
        (void)new_max_elements;
#endif
    }

    void freeSearchActiveNodeTable() {
#if HNSW_SEARCH_ACTIVE_NODE_TRACKING
        delete[] search_active_node_counts_;
        search_active_node_counts_ = nullptr;
#endif
    }

    size_t searchActiveNodeTableBytes() const {
#if HNSW_SEARCH_ACTIVE_NODE_TRACKING
        return max_elements_ * sizeof(std::atomic<uint16_t>);
#else
        return 0;
#endif
    }

    inline bool markSearchActiveNode(tableint internal_id) const {
#if HNSW_SEARCH_ACTIVE_NODE_TRACKING
        if (!search_active_node_counts_ || internal_id >= max_elements_) {
            return false;
        }
        search_active_node_counts_[internal_id].fetch_add(1, std::memory_order_acquire);
        metric_search_active_node_marks.fetch_add(1, std::memory_order_relaxed);
        return true;
#else
        (void)internal_id;
        return false;
#endif
    }

    inline void unmarkSearchActiveNode(tableint internal_id, bool marked) const {
#if HNSW_SEARCH_ACTIVE_NODE_TRACKING
        if (marked && search_active_node_counts_) {
            search_active_node_counts_[internal_id].fetch_sub(1, std::memory_order_release);
        }
#else
        (void)internal_id;
        (void)marked;
#endif
    }

    inline bool isSearchActiveNode(tableint internal_id) const {
#if HNSW_SEARCH_ACTIVE_NODE_TRACKING
        return search_active_node_counts_ && internal_id < max_elements_ &&
               search_active_node_counts_[internal_id].load(std::memory_order_acquire) > 0;
#else
        (void)internal_id;
        return false;
#endif
    }

    void deferSearchActiveRepairTargets(std::vector<tableint>& repair_neighbors,
                                        int level) const {
#if HNSW_SEARCH_AWARE_REPAIR_REORDER
        if (level != 0 || repair_neighbors.size() <= 1) {
            return;
        }
        const auto first_active =
            std::stable_partition(repair_neighbors.begin(), repair_neighbors.end(),
                                  [this](tableint id) { return !isSearchActiveNode(id); });
        const auto moved =
            static_cast<uint64_t>(std::distance(first_active, repair_neighbors.end()));
        if (moved > 0) {
            metric_insert_search_hot_reorder_lists.fetch_add(1, std::memory_order_relaxed);
            metric_insert_search_hot_reorder_nodes.fetch_add(moved, std::memory_order_relaxed);
        }
#else
        (void)repair_neighbors;
        (void)level;
#endif
    }

    void allocateSearchActiveCacheBuckets() {
#if HNSW_SEARCH_ACTIVE_CACHE_BUCKET_TRACKING
        search_active_cache_buckets_ =
            new std::atomic<uint16_t>[HNSW_SEARCH_ACTIVE_CACHE_BUCKETS];
        for (size_t i = 0; i < HNSW_SEARCH_ACTIVE_CACHE_BUCKETS; i++) {
            search_active_cache_buckets_[i].store(0, std::memory_order_relaxed);
        }
#endif
    }

    void freeSearchActiveCacheBuckets() {
#if HNSW_SEARCH_ACTIVE_CACHE_BUCKET_TRACKING
        delete[] search_active_cache_buckets_;
        search_active_cache_buckets_ = nullptr;
#endif
    }

    size_t searchActiveCacheBucketBytes() const {
#if HNSW_SEARCH_ACTIVE_CACHE_BUCKET_TRACKING
        return HNSW_SEARCH_ACTIVE_CACHE_BUCKETS * sizeof(std::atomic<uint16_t>);
#else
        return 0;
#endif
    }

    static uintptr_t cacheLineIdOf(const void* ptr) {
        return reinterpret_cast<uintptr_t>(ptr) / kLayoutCacheLineBytes;
    }

    static uint32_t cacheBucketOfLine(uintptr_t line_id) {
        return static_cast<uint32_t>(line_id % HNSW_SEARCH_ACTIVE_CACHE_BUCKETS);
    }

    inline size_t appendCacheBucketForAddress(const void* ptr,
                                              ActiveCacheBucketArray& buckets,
                                              size_t count) const {
#if HNSW_SEARCH_ACTIVE_CACHE_BUCKET_TRACKING
        if (!ptr) {
            return count;
        }
        if (count < buckets.size()) {
            buckets[count++] = cacheBucketOfLine(cacheLineIdOf(ptr));
        }
#else
        (void)ptr;
        (void)buckets;
#endif
        return count;
    }

    inline size_t markSearchNodeCacheFootprint(tableint internal_id,
                                               ActiveCacheBucketArray& buckets) const {
#if HNSW_SEARCH_ACTIVE_CACHE_BUCKET_TRACKING
        if (!search_active_cache_buckets_ || internal_id >= max_elements_) {
            return 0;
        }
        size_t count = 0;
        count = appendCacheBucketForAddress(get_linklist0(internal_id), buckets, count);
        count = appendCacheBucketForAddress(getSearchDataByInternalId(internal_id),
                                            buckets, count);
        for (size_t i = 0; i < count; i++) {
#if HNSW_SEARCH_ACTIVE_CACHE_BUCKET_STORE_ONLY
            search_active_cache_buckets_[buckets[i]].store(1,
                                                           std::memory_order_release);
#else
            search_active_cache_buckets_[buckets[i]].fetch_add(1,
                                                               std::memory_order_acquire);
#endif
        }
        metric_search_active_cache_bucket_marks.fetch_add(count,
                                                          std::memory_order_relaxed);
        return count;
#else
        (void)internal_id;
        (void)buckets;
        return 0;
#endif
    }

    inline void unmarkSearchCacheBuckets(const ActiveCacheBucketArray& buckets,
                                         size_t count) const {
#if HNSW_SEARCH_ACTIVE_CACHE_BUCKET_TRACKING
        if (!search_active_cache_buckets_) {
            return;
        }
        for (size_t i = 0; i < count; i++) {
#if HNSW_SEARCH_ACTIVE_CACHE_BUCKET_STORE_ONLY
            search_active_cache_buckets_[buckets[i]].store(0,
                                                           std::memory_order_release);
#else
            search_active_cache_buckets_[buckets[i]].fetch_sub(1,
                                                               std::memory_order_release);
#endif
        }
#else
        (void)buckets;
        (void)count;
#endif
    }

    inline bool cacheAddressHasActiveBucket(const void* ptr) const {
#if HNSW_SEARCH_ACTIVE_CACHE_BUCKET_TRACKING
        if (!search_active_cache_buckets_ || !ptr) {
            return false;
        }
        return search_active_cache_buckets_[cacheBucketOfLine(cacheLineIdOf(ptr))].load(
                   std::memory_order_acquire) > 0;
#else
        (void)ptr;
        return false;
#endif
    }

    inline bool repairTargetTouchesActiveSearchCache(tableint internal_id,
                                                     int level) const {
#if HNSW_CACHE_AWARE_REPAIR_REORDER
        if (level == 0) {
            if (cacheAddressHasActiveBucket(get_linklist0(internal_id))) {
                return true;
            }
        }
        return cacheAddressHasActiveBucket(getInsertDataByInternalId(internal_id));
#else
        (void)internal_id;
        (void)level;
        return false;
#endif
    }

    void deferCacheHotRepairTargets(std::vector<tableint>& repair_neighbors,
                                    int level) const {
#if HNSW_CACHE_AWARE_REPAIR_REORDER
        if (repair_neighbors.size() <= 1) {
            return;
        }
        const auto first_hot =
            std::stable_partition(repair_neighbors.begin(), repair_neighbors.end(),
                                  [this, level](tableint id) {
                                      return !repairTargetTouchesActiveSearchCache(id, level);
                                  });
        const auto hot_nodes =
            static_cast<uint64_t>(std::distance(first_hot, repair_neighbors.end()));
        if (hot_nodes == 0) {
            return;
        }
        metric_insert_cache_hot_reorder_lists.fetch_add(1,
                                                        std::memory_order_relaxed);
        metric_insert_cache_hot_reorder_nodes.fetch_add(hot_nodes,
                                                        std::memory_order_relaxed);
#else
        (void)repair_neighbors;
        (void)level;
#endif
    }

    void rebuildSearchVectorShadow(size_t count) {
#if HNSW_SEARCH_VECTOR_SHADOW
        if (!search_vector_shadow_memory_) {
            return;
        }
        for (size_t i = 0; i < count; i++) {
            memcpy(search_vector_shadow_memory_ + i * data_size_,
                   data_level0_memory_ + i * size_data_per_element_ + offsetData_,
                   data_size_);
        }
#else
        (void)count;
#endif
    }

    inline void copySearchVectorShadow(tableint internal_id, const void* data_point) const {
#if HNSW_SEARCH_VECTOR_SHADOW
        if (search_vector_shadow_memory_) {
            memcpy(search_vector_shadow_memory_ + internal_id * data_size_, data_point,
                   data_size_);
        }
#else
        (void)internal_id;
        (void)data_point;
#endif
    }

    inline char* getSearchDataByInternalId(tableint internal_id) const {
#if HNSW_SEARCH_VECTOR_READ_TRACE
        recordSearchVectorRead(internal_id);
#endif
#if HNSW_SEARCH_VECTOR_SHADOW
        if (search_vector_shadow_memory_) {
            return search_vector_shadow_memory_ + internal_id * data_size_;
        }
#endif
        return getDataByInternalId(internal_id);
    }

    void allocateSearchVectorReadTrace(size_t max_elements) {
#if HNSW_SEARCH_VECTOR_READ_TRACE
        search_vector_read_counts_ = new std::atomic<uint64_t>[max_elements];
        for (size_t i = 0; i < max_elements; i++) {
            search_vector_read_counts_[i].store(0, std::memory_order_relaxed);
        }
#else
        (void)max_elements;
#endif
    }

    void freeSearchVectorReadTrace() {
#if HNSW_SEARCH_VECTOR_READ_TRACE
        delete[] search_vector_read_counts_;
        search_vector_read_counts_ = nullptr;
#endif
    }

    void recordSearchVectorRead(tableint internal_id) const {
#if HNSW_SEARCH_VECTOR_READ_TRACE
        if (search_vector_read_counts_ && internal_id < max_elements_) {
            search_vector_read_counts_[internal_id].fetch_add(
                1, std::memory_order_relaxed);
            metric_search_vector_read_trace_reads.fetch_add(
                1, std::memory_order_relaxed);
        }
#else
        (void)internal_id;
#endif
    }

    uint64_t searchVectorReadTraceReads() const {
        return metric_search_vector_read_trace_reads.load(std::memory_order_relaxed);
    }

    void resetSearchVectorReadTrace() const {
#if HNSW_SEARCH_VECTOR_READ_TRACE
        if (!search_vector_read_counts_) {
            return;
        }
        for (size_t i = 0; i < max_elements_; i++) {
            search_vector_read_counts_[i].store(0, std::memory_order_relaxed);
        }
        metric_search_vector_read_trace_reads.store(0, std::memory_order_relaxed);
#endif
    }

    void dumpSearchVectorReadTrace(const std::string& path, size_t top_k) const {
#if HNSW_SEARCH_VECTOR_READ_TRACE
        if (!search_vector_read_counts_ || path.empty()) {
            return;
        }
        std::vector<std::pair<uint64_t, tableint>> counts;
        counts.reserve(cur_element_count);
        uint64_t total = 0;
        for (tableint i = 0; i < cur_element_count; i++) {
            uint64_t c = search_vector_read_counts_[i].load(std::memory_order_relaxed);
            if (c == 0) {
                continue;
            }
            total += c;
            counts.emplace_back(c, i);
        }
        std::sort(counts.begin(), counts.end(),
                  [](const auto& a, const auto& b) {
                      if (a.first != b.first) {
                          return a.first > b.first;
                      }
                      return a.second < b.second;
                  });
        if (top_k == 0 || top_k > counts.size()) {
            top_k = counts.size();
        }
        std::ofstream output(path);
        if (!output.good()) {
            throw std::runtime_error("Failed to open search vector read trace output");
        }
        output << "id,count,cum_count,cum_frac,level\n";
        uint64_t cum = 0;
        for (size_t i = 0; i < top_k; i++) {
            cum += counts[i].first;
            const tableint id = counts[i].second;
            const int level = id < element_levels_.size() ? element_levels_[id] : -1;
            output << id << "," << counts[i].first << "," << cum << ",";
            output << (total == 0 ? 0.0 : static_cast<double>(cum) / total);
            output << "," << level << "\n";
        }
#else
        (void)path;
        (void)top_k;
#endif
    }

    static bool l0RepairNtStoreAvailable() {
#if HNSW_L0_REPAIR_NT_STORE_X86
        return sizeof(tableint) == sizeof(int);
#else
        return false;
#endif
    }

    bool shouldUseL0RepairNtStore(int level) const {
#if HNSW_L0_REPAIR_NT_STORE
        return level == 0 && partial_state_.limit == 0 && l0RepairNtStoreAvailable();
#else
        (void)level;
        return false;
#endif
    }

    bool shouldCountL0RepairNtPartialFallback(int level) const {
#if HNSW_L0_REPAIR_NT_STORE
        return level == 0 && partial_state_.limit > 0;
#else
        (void)level;
        return false;
#endif
    }

    void storeL0RepairNeighbor(tableint* dst, tableint value, bool use_nt) const {
#if HNSW_L0_REPAIR_NT_STORE_X86
        if (use_nt && sizeof(tableint) == sizeof(int)) {
            _mm_stream_si32(reinterpret_cast<int*>(dst), static_cast<int>(value));
            return;
        }
#endif
        (void)use_nt;
        *dst = value;
    }

    void fenceL0RepairNtStores(bool use_nt) const {
#if HNSW_L0_REPAIR_NT_STORE_X86
        if (use_nt && sizeof(tableint) == sizeof(int)) {
            _mm_sfence();
        }
#else
        (void)use_nt;
#endif
    }

    void allocateLevel0CowOverrides(size_t max_elements) {
#if HNSW_L0_ADJ_COW_PUBLISH
        level0_linklist_overrides_ = new std::atomic<char*>[max_elements];
        for (size_t i = 0; i < max_elements; i++) {
            level0_linklist_overrides_[i].store(nullptr, std::memory_order_relaxed);
        }
#else
        (void)max_elements;
#endif
    }

    void resizeLevel0CowOverrides(size_t new_max_elements) {
#if HNSW_L0_ADJ_COW_PUBLISH
        std::atomic<char*>* new_overrides = new std::atomic<char*>[new_max_elements];
        for (size_t i = 0; i < new_max_elements; i++) {
            char* ptr = nullptr;
            if (level0_linklist_overrides_ && i < max_elements_) {
                ptr = level0_linklist_overrides_[i].load(std::memory_order_acquire);
            }
            new_overrides[i].store(ptr, std::memory_order_relaxed);
        }
        delete[] level0_linklist_overrides_;
        level0_linklist_overrides_ = new_overrides;
#else
        (void)new_max_elements;
#endif
    }

    void freeLevel0CowStorage() {
#if HNSW_L0_ADJ_COW_PUBLISH
        for (char* block : level0_cow_blocks_) {
            free(block);
        }
        level0_cow_blocks_.clear();
        delete[] level0_linklist_overrides_;
        level0_linklist_overrides_ = nullptr;
#endif
    }

    size_t level0AdjCowBlockBytes() const {
#if HNSW_L0_ADJ_COW_PUBLISH
        return alignToLayoutCacheLine(size_links_level0_);
#else
        return 0;
#endif
    }

    char* allocateLevel0AdjCowBlock() const {
#if HNSW_L0_ADJ_COW_PUBLISH
        void* ptr = nullptr;
        const size_t bytes = level0AdjCowBlockBytes();
        if (posix_memalign(&ptr, kLayoutCacheLineBytes, bytes) != 0 || ptr == nullptr) {
            throw std::runtime_error(
                "Not enough memory: failed to allocate level-0 adjacency COW block");
        }
        memset(ptr, 0, bytes);
        return static_cast<char*>(ptr);
#else
        return nullptr;
#endif
    }

    void retainLevel0AdjCowBlock(char* block) const {
#if HNSW_L0_ADJ_COW_PUBLISH
        std::lock_guard<std::mutex> guard(level0_cow_blocks_lock_);
        level0_cow_blocks_.push_back(block);
#else
        (void)block;
#endif
    }

    linklistsizeint* getBaseLinklist0(tableint internal_id) const {
        return (linklistsizeint*)(data_level0_memory_ + internal_id * size_data_per_element_ +
                                  offsetLevel0_);
    }

    void publishLevel0AdjCow(tableint internal_id, char* block,
                             bool count_unique_node) const {
#if HNSW_L0_ADJ_COW_PUBLISH
        retainLevel0AdjCowBlock(block);
        if (count_unique_node) {
            metric_l0_adj_cow_nodes.fetch_add(1, std::memory_order_relaxed);
        }
        metric_l0_adj_cow_publishes.fetch_add(1, std::memory_order_relaxed);
        metric_l0_adj_cow_bytes.fetch_add(level0AdjCowBlockBytes(),
                                          std::memory_order_relaxed);
        level0_linklist_overrides_[internal_id].store(block, std::memory_order_release);
#else
        (void)internal_id;
        (void)block;
        (void)count_unique_node;
#endif
    }

    linklistsizeint* copyLevel0AdjForCow(tableint internal_id, bool* first_publish) const {
#if HNSW_L0_ADJ_COW_PUBLISH
        if (!level0_linklist_overrides_) {
            throw std::runtime_error("Level-0 adjacency COW override table is not allocated");
        }
        char* current =
            level0_linklist_overrides_[internal_id].load(std::memory_order_acquire);
        *first_publish = current == nullptr;
        linklistsizeint* source =
            current ? reinterpret_cast<linklistsizeint*>(current) : getBaseLinklist0(internal_id);
        char* block = allocateLevel0AdjCowBlock();
        memcpy(block, source, size_links_level0_);
        return reinterpret_cast<linklistsizeint*>(block);
#else
        (void)internal_id;
        (void)first_publish;
        return nullptr;
#endif
    }

    size_t searchVectorShadowBytes() const {
#if HNSW_SEARCH_VECTOR_SHADOW
        return max_elements_ * data_size_;
#else
        return 0;
#endif
    }

    size_t insertVectorShadowBytes() const {
#if HNSW_INSERT_VECTOR_SHADOW || HNSW_INSERT_VECTOR_PARTIAL_SHADOW || \
    HNSW_INSERT_VECTOR_HOT_SHADOW
        return insert_vector_shadow_slots_ * data_size_;
#else
        return 0;
#endif
    }

	    size_t insertVectorShadowSlots() const {
#if HNSW_INSERT_VECTOR_SHADOW || HNSW_INSERT_VECTOR_PARTIAL_SHADOW || \
	    HNSW_INSERT_VECTOR_HOT_SHADOW
	        return insert_vector_shadow_slots_;
#else
	        return 0;
#endif
	    }

	    size_t insertVectorRegionShadowSlots() const {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        return insert_vector_region_shadow_slots_;
#else
	        return 0;
#endif
	    }

	    size_t insertVectorRegionShadowBytes() const {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        return insert_vector_region_shadow_bytes_;
#else
	        return 0;
#endif
	    }

	    size_t insertVectorRegionShadowRegionBytesStat() const {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        return insert_vector_region_shadow_region_bytes_;
#else
	        return 0;
#endif
	    }

	    size_t insertVectorRegionCount() const {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        return insert_vector_region_count_;
#else
	        return 0;
#endif
	    }

	    void setInsertVectorRegionShadowMaxId(size_t count) {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        insert_vector_region_shadow_max_id_ = std::min(count, max_elements_);
#else
	        (void)count;
#endif
	    }

	    size_t insertVectorRegionShadowMaxId() const {
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        return insert_vector_region_shadow_max_id_;
#else
	        return 0;
#endif
	    }

	    uint64_t insertVectorRegionShadowSamples() const {
	        return metric_insert_vector_region_shadow_samples.load(std::memory_order_relaxed);
	    }

	    uint64_t insertVectorRegionShadowPromotions() const {
	        return metric_insert_vector_region_shadow_promotions.load(std::memory_order_relaxed);
	    }

	    uint64_t insertVectorRegionShadowHits() const {
	        return metric_insert_vector_region_shadow_hits.load(std::memory_order_relaxed);
	    }

	    uint64_t insertVectorRegionShadowFallbacks() const {
	        return metric_insert_vector_region_shadow_fallbacks.load(
	            std::memory_order_relaxed);
	    }

	    uint64_t insertVectorRegionShadowBudgetFull() const {
	        return metric_insert_vector_region_shadow_budget_full.load(
	            std::memory_order_relaxed);
	    }

	    double insertVectorRegionShadowCopyMs() const {
	        return static_cast<double>(
	                   metric_insert_vector_region_shadow_copy_ns.load(
	                       std::memory_order_relaxed)) /
	               1e6;
	    }

	    uint64_t insertVectorRegionShadowCrossRegion() const {
	        return metric_insert_vector_region_shadow_cross_region.load(
	            std::memory_order_relaxed);
	    }

	    uint64_t insertVectorThreadCacheHits() const {
	        return metric_insert_vector_thread_cache_hits.load(std::memory_order_relaxed);
	    }

	    uint64_t insertVectorThreadCacheFallbacks() const {
	        return metric_insert_vector_thread_cache_fallbacks.load(
	            std::memory_order_relaxed);
	    }

	    uint64_t insertVectorThreadCacheFilterHits() const {
	        return metric_insert_vector_thread_cache_filter_hits.load(
	            std::memory_order_relaxed);
	    }

	    uint64_t insertVectorThreadCacheAdmissions() const {
	        return metric_insert_vector_thread_cache_admissions.load(
	            std::memory_order_relaxed);
	    }

	    uint64_t insertVectorThreadCacheCopies() const {
	        return metric_insert_vector_thread_cache_copies.load(
	            std::memory_order_relaxed);
	    }

	    uint64_t insertVectorHotOnlineSamples() const {
	        return metric_insert_vector_hot_online_samples.load(std::memory_order_relaxed);
    }

    uint64_t insertVectorHotOnlinePromotions() const {
        return metric_insert_vector_hot_online_promotions.load(std::memory_order_relaxed);
    }

    uint64_t insertVectorHotOnlineBudgetFull() const {
        return metric_insert_vector_hot_online_budget_full.load(std::memory_order_relaxed);
    }

    double insertVectorGraphHotSelectMs() const {
        return static_cast<double>(
                   metric_insert_vector_graph_hot_select_ns.load(std::memory_order_relaxed)) /
               1e6;
    }

    double insertVectorGraphHotCopyMs() const {
        return static_cast<double>(
                   metric_insert_vector_graph_hot_copy_ns.load(std::memory_order_relaxed)) /
               1e6;
    }

    uint64_t insertVectorGraphHotScannedEdges() const {
        return metric_insert_vector_graph_hot_scanned_edges.load(std::memory_order_relaxed);
    }

    uint64_t insertVectorGraphHotSelected() const {
        return metric_insert_vector_graph_hot_selected.load(std::memory_order_relaxed);
    }

    uint64_t insertVectorGraphHotCandidates() const {
        return metric_insert_vector_graph_hot_candidates.load(std::memory_order_relaxed);
    }

    size_t level0AdjCowBytes() const {
#if HNSW_L0_ADJ_COW_PUBLISH
        return metric_l0_adj_cow_bytes.load(std::memory_order_relaxed);
#else
        return 0;
#endif
    }

    uint64_t level0AdjCowNodes() const {
        return metric_l0_adj_cow_nodes.load(std::memory_order_relaxed);
    }

    uint64_t level0AdjCowPublishes() const {
        return metric_l0_adj_cow_publishes.load(std::memory_order_relaxed);
    }

    uint64_t level0AdjCowAppendPublishes() const {
        return metric_l0_adj_cow_append_publishes.load(std::memory_order_relaxed);
    }

    uint64_t level0AdjCowPrunePublishes() const {
        return metric_l0_adj_cow_prune_publishes.load(std::memory_order_relaxed);
    }

    uint64_t level0AdjCowPartialFallbacks() const {
        return metric_l0_adj_cow_partial_fallbacks.load(std::memory_order_relaxed);
    }

    uint64_t l0RepairNtEdges() const {
        return metric_l0_repair_nt_edges.load(std::memory_order_relaxed);
    }

    uint64_t l0RepairNtRewrites() const {
        return metric_l0_repair_nt_rewrites.load(std::memory_order_relaxed);
    }

    uint64_t l0RepairNtAppends() const {
        return metric_l0_repair_nt_appends.load(std::memory_order_relaxed);
    }

    uint64_t l0RepairNtPartialFallbacks() const {
        return metric_l0_repair_nt_partial_fallbacks.load(std::memory_order_relaxed);
    }

    uint64_t insertVectorReadAdmissions() const {
        return metric_insert_vector_read_admissions.load(std::memory_order_relaxed);
    }

    uint64_t insertVectorReadWaits() const {
        return metric_insert_vector_read_waits.load(std::memory_order_relaxed);
    }

    uint64_t insertVectorReadWaitNs() const {
        return metric_insert_vector_read_wait_ns.load(std::memory_order_relaxed);
    }

    uint32_t activeInsertVectorReaders() const {
        return active_insert_vector_readers.load(std::memory_order_relaxed);
    }

    uint64_t searchActiveNodeMarks() const {
        return metric_search_active_node_marks.load(std::memory_order_relaxed);
    }

    uint64_t insertSearchHotReorderLists() const {
        return metric_insert_search_hot_reorder_lists.load(std::memory_order_relaxed);
    }

    uint64_t insertSearchHotReorderNodes() const {
        return metric_insert_search_hot_reorder_nodes.load(std::memory_order_relaxed);
    }

    uint64_t searchActiveCacheBucketMarks() const {
        return metric_search_active_cache_bucket_marks.load(std::memory_order_relaxed);
    }

    uint64_t insertCacheHotReorderLists() const {
        return metric_insert_cache_hot_reorder_lists.load(std::memory_order_relaxed);
    }

    uint64_t insertCacheHotReorderNodes() const {
        return metric_insert_cache_hot_reorder_nodes.load(std::memory_order_relaxed);
    }

    uint64_t insertSearchCacheDeferLists() const {
        return metric_insert_search_cache_defer_lists.load(std::memory_order_relaxed);
    }

    uint64_t insertSearchCacheDeferNodes() const {
        return metric_insert_search_cache_defer_nodes.load(std::memory_order_relaxed);
    }

    mutable std::atomic<uint32_t> active_existing_neighbor_repairs{0};

    bool allow_replace_deleted_ =
        false;  // flag to replace deleted elements (marked as deleted) during insertions
    bool use_node_lock_in_search_{true};  // skip internal shared locks during search when true

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;

    // ===== S3 (Insert byproduct serving Search) =====
    static inline uint64_t s3_rdtsc_ns() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    }

    struct SearchTelemetry {
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
    };

    struct Byproduct {
        tableint insert_id;
        std::vector<std::pair<dist_t, tableint>> candidates;  // S3: level-0 beam search results
        const void* insert_vector;
        uint64_t timestamp;
        mutable std::atomic<bool> consumed{false};

        // Path Skip: entry point at each layer during insert traversal
        std::vector<tableint> layer_entry_points;
        // Candidate Injection: level-0 beam search candidates (same as candidates for S3)
        std::vector<std::pair<dist_t, tableint>> level0_candidates;

        Byproduct() : insert_id(0), insert_vector(nullptr), timestamp(0) {}
        Byproduct(tableint id, std::vector<std::pair<dist_t, tableint>>&& cands,
                  const void* vec, uint64_t ts)
            : insert_id(id), candidates(std::move(cands)), insert_vector(vec), timestamp(ts) {}
        Byproduct(Byproduct&& other)
            : insert_id(other.insert_id),
              candidates(std::move(other.candidates)),
              insert_vector(other.insert_vector),
              timestamp(other.timestamp),
              consumed(other.consumed.load()),
              layer_entry_points(std::move(other.layer_entry_points)),
              level0_candidates(std::move(other.level0_candidates)) {}
        Byproduct& operator=(Byproduct&& other) {
            if (this != &other) {
                insert_id = other.insert_id;
                candidates = std::move(other.candidates);
                insert_vector = other.insert_vector;
                timestamp = other.timestamp;
                consumed.store(other.consumed.load());
                layer_entry_points = std::move(other.layer_entry_points);
                level0_candidates = std::move(other.level0_candidates);
            }
            return *this;
        }
        Byproduct(const Byproduct&) = delete;
        Byproduct& operator=(const Byproduct&) = delete;
    };

    struct S3ByproductStore {
        static const size_t DEFAULT_CAPACITY = 64;  // Reduced from 1024 to limit matching overhead in high-dim
        std::unique_ptr<Byproduct[]> buffer;
        std::atomic<size_t> head{0};
        std::atomic<size_t> tail{0};
        size_t capacity;
        dist_t proximity_threshold;

        // Adaptive locality detection
        static const size_t ADAPTIVE_WINDOW = 100;
        bool adaptive_enabled{true};
        dist_t adaptive_threshold{0.5};  // locality_ratio threshold
        std::atomic<double> sum_match_dist{0.0};
        std::atomic<double> sum_baseline_dist{0.0};
        std::atomic<size_t> match_count{0};
        std::atomic<size_t> baseline_count{0};
        mutable std::atomic<uint64_t> adaptive_accepts{0};
        mutable std::atomic<uint64_t> adaptive_rejects{0};

        S3ByproductStore(size_t cap = DEFAULT_CAPACITY, dist_t threshold = 0.1)
            : buffer(std::make_unique<Byproduct[]>(cap)), capacity(cap), proximity_threshold(threshold) {}

        void add(tableint insert_id, std::vector<std::pair<dist_t, tableint>>&& candidates,
                 const void* insert_vector, uint64_t timestamp,
                 std::vector<tableint>&& entry_points = {},
                 std::vector<std::pair<dist_t, tableint>>&& l0_candidates = {}) {
            size_t current_tail = tail.load(std::memory_order_relaxed);
            size_t next_tail = (current_tail + 1) % capacity;
            if (next_tail == head.load(std::memory_order_acquire)) {
                head.store((head.load(std::memory_order_relaxed) + 1) % capacity, std::memory_order_release);
            }
            new(&buffer[current_tail]) Byproduct(insert_id, std::move(candidates), insert_vector, timestamp);
            buffer[current_tail].layer_entry_points = std::move(entry_points);
            buffer[current_tail].level0_candidates = std::move(l0_candidates);
            tail.store(next_tail, std::memory_order_release);
        }

        const Byproduct* findByproduct(const void* query_vector,
                                       DISTFUNC<dist_t> dist_func,
                                       const void* dist_param,
                                       uint64_t max_age_ns = 1000000000ULL) {
            size_t current_head = head.load(std::memory_order_acquire);
            size_t current_tail = tail.load(std::memory_order_acquire);
            uint64_t now = s3_rdtsc_ns();
            for (size_t i = current_tail; i != current_head; i = (i == 0 ? capacity - 1 : i - 1)) {
                size_t idx = (i == 0 ? capacity - 1 : i - 1);
                const Byproduct& bp = buffer[idx];
                if (bp.insert_vector == nullptr) continue;
                if (now - bp.timestamp > max_age_ns) continue;
                if (bp.consumed.load(std::memory_order_acquire)) continue;
                dist_t dist = dist_func(query_vector, bp.insert_vector, dist_param);
                if (dist <= proximity_threshold) {
                    return &bp;
                }
            }
            return nullptr;
        }

        // Adaptive version: finds byproduct and checks locality ratio
        // Returns null if adaptive detector determines locality is absent
        const Byproduct* findByproductAdaptive(const void* query_vector,
                                                DISTFUNC<dist_t> dist_func,
                                                const void* dist_param,
                                                const void* entry_point_data,
                                                uint64_t max_age_ns = 1000000000ULL) {
            if (!adaptive_enabled) {
                return findByproduct(query_vector, dist_func, dist_param, max_age_ns);
            }

            // Update baseline: distance to entry point as proxy for random distance
            dist_t baseline_dist = dist_func(query_vector, entry_point_data, dist_param);
            {
                // Exponential moving average for baseline
                size_t bc = baseline_count.fetch_add(1, std::memory_order_relaxed);
                if (bc < ADAPTIVE_WINDOW) {
                    double old_sum = sum_baseline_dist.load(std::memory_order_relaxed);
                    sum_baseline_dist.store(old_sum + static_cast<double>(baseline_dist), std::memory_order_relaxed);
                } else {
                    // EMA update: new_avg = old_avg * (1 - 1/N) + new_val / N
                    double old_avg = sum_baseline_dist.load(std::memory_order_relaxed) / ADAPTIVE_WINDOW;
                    double new_avg = old_avg * (1.0 - 1.0 / ADAPTIVE_WINDOW) + static_cast<double>(baseline_dist) / ADAPTIVE_WINDOW;
                    sum_baseline_dist.store(new_avg * ADAPTIVE_WINDOW, std::memory_order_relaxed);
                }
            }

            // Adaptive: track total attempts and hit rate
            // After warmup, if hit rate is too low, skip scanning entirely
            size_t bc = baseline_count.load(std::memory_order_relaxed);
            size_t total_attempts = adaptive_accepts.load(std::memory_order_relaxed) +
                                    adaptive_rejects.load(std::memory_order_relaxed);
            size_t hits = adaptive_accepts.load(std::memory_order_relaxed);

            if (total_attempts >= ADAPTIVE_WINDOW) {
                // Periodic re-probe: every ADAPTIVE_WINDOW attempts, reset counters
                // to prevent death spiral where early misses permanently disable the filter
                if (total_attempts % ADAPTIVE_WINDOW == 0) {
                    adaptive_accepts.store(0, std::memory_order_relaxed);
                    adaptive_rejects.store(0, std::memory_order_relaxed);
                    // Fall through to try findByproduct
                } else {
                    double hit_rate = static_cast<double>(hits) / total_attempts;
                    if (hit_rate < 0.01) {
                        // Hit rate < 1% after warmup — sharing is useless, skip scanning
                        adaptive_rejects.fetch_add(1, std::memory_order_relaxed);
                        return nullptr;
                    }
                    // Also check locality ratio if we have enough matches
                    if (hits >= 10 && bc >= ADAPTIVE_WINDOW) {
                        double avg_match = sum_match_dist.load(std::memory_order_relaxed) / ADAPTIVE_WINDOW;
                        double avg_baseline = sum_baseline_dist.load(std::memory_order_relaxed) / ADAPTIVE_WINDOW;
                        if (avg_baseline > 0) {
                            double locality_ratio = avg_match / avg_baseline;
                            if (locality_ratio > static_cast<double>(adaptive_threshold)) {
                                adaptive_rejects.fetch_add(1, std::memory_order_relaxed);
                                return nullptr;
                            }
                        }
                    }
                }
            }

            const Byproduct* bp = findByproduct(query_vector, dist_func, dist_param, max_age_ns);
            if (bp != nullptr) {
                // Update match distance stats
                dist_t match_dist = dist_func(query_vector, bp->insert_vector, dist_param);
                size_t mc2 = match_count.fetch_add(1, std::memory_order_relaxed);
                if (mc2 < ADAPTIVE_WINDOW) {
                    double old_sum = sum_match_dist.load(std::memory_order_relaxed);
                    sum_match_dist.store(old_sum + static_cast<double>(match_dist), std::memory_order_relaxed);
                } else {
                    double old_avg = sum_match_dist.load(std::memory_order_relaxed) / ADAPTIVE_WINDOW;
                    double new_avg = old_avg * (1.0 - 1.0 / ADAPTIVE_WINDOW) + static_cast<double>(match_dist) / ADAPTIVE_WINDOW;
                    sum_match_dist.store(new_avg * ADAPTIVE_WINDOW, std::memory_order_relaxed);
                }
                adaptive_accepts.fetch_add(1, std::memory_order_relaxed);
            } else {
                // Count as attempt for hit rate tracking
                adaptive_rejects.fetch_add(1, std::memory_order_relaxed);
            }
            return bp;
        }
    };

    // ===== Search Frontier Pool (Search→Search byproduct sharing) =====
    struct SearchFrontier {
        const void* query_vector;
        std::vector<std::pair<dist_t, tableint>> current_beam;  // current best candidates
        std::atomic<bool> active{true};  // still searching
        uint64_t timestamp;
        
        SearchFrontier() : query_vector(nullptr), timestamp(0) {}
        SearchFrontier(const void* query, uint64_t ts) : query_vector(query), timestamp(ts) {}
    };

    struct SearchFrontierPool {
        static const size_t MAX_FRONTIERS = 64;
        SearchFrontier frontiers[MAX_FRONTIERS];
        std::atomic<size_t> count{0};

        // Register a new search (returns slot index, or -1 if full)
        int registerSearch(const void* query, uint64_t ts) {
            for (size_t i = 0; i < MAX_FRONTIERS; ++i) {
                bool expected = false;
                if (frontiers[i].active.compare_exchange_weak(expected, true, std::memory_order_acquire)) {
                    // Successfully claimed this slot
                    frontiers[i].query_vector = query;
                    frontiers[i].timestamp = ts;
                    frontiers[i].current_beam.clear();
                    count.fetch_add(1, std::memory_order_relaxed);
                    return static_cast<int>(i);
                }
            }
            return -1;  // Pool is full
        }

        // Update frontier with current beam state
        void updateFrontier(int slot, const std::vector<std::pair<dist_t, tableint>>& beam) {
            if (slot >= 0 && slot < static_cast<int>(MAX_FRONTIERS)) {
                if (frontiers[slot].active.load(std::memory_order_relaxed)) {
                    frontiers[slot].current_beam = beam;
                }
            }
        }

        // Find a frontier whose beam might help this query
        // Returns candidates from the closest active frontier
        std::vector<std::pair<dist_t, tableint>> findUsefulFrontier(
            const void* query, DISTFUNC<dist_t> dist_func, const void* dist_param, 
            dist_t max_dist, uint64_t max_age_ns = 1000000000ULL) {
            
            std::vector<std::pair<dist_t, tableint>> result;
            uint64_t now = s3_rdtsc_ns();
            dist_t best_query_dist = std::numeric_limits<dist_t>::max();
            int best_slot = -1;

            for (size_t i = 0; i < MAX_FRONTIERS; ++i) {
                if (!frontiers[i].active.load(std::memory_order_relaxed)) continue;
                if (frontiers[i].query_vector == nullptr) continue;
                if (now - frontiers[i].timestamp > max_age_ns) continue;
                if (frontiers[i].current_beam.empty()) continue;

                // Check similarity to this frontier's query
                dist_t query_dist = dist_func(query, frontiers[i].query_vector, dist_param);
                if (query_dist <= max_dist && query_dist < best_query_dist) {
                    best_query_dist = query_dist;
                    best_slot = static_cast<int>(i);
                }
            }

            if (best_slot >= 0) {
                // Return copy of the best frontier's beam
                result = frontiers[best_slot].current_beam;
            }

            return result;
        }

        // Deregister when search completes
        void deregister(int slot) {
            if (slot >= 0 && slot < static_cast<int>(MAX_FRONTIERS)) {
                frontiers[slot].active.store(false, std::memory_order_release);
                frontiers[slot].query_vector = nullptr;
                frontiers[slot].current_beam.clear();
                count.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    };

    std::unique_ptr<S3ByproductStore> s3_store_;
    bool s3_suspended_{false};
    mutable std::atomic<uint64_t> s3_hits_{0};
    mutable std::atomic<uint64_t> s3_misses_{0};
    mutable std::atomic<uint64_t> s3_dist_saved_{0};
    bool enable_s3_{false};
    
    // Warm start configuration and metrics
    mutable std::atomic<uint64_t> warm_start_hits_{0};
    mutable std::atomic<uint64_t> warm_start_improvement_{0};
    bool enable_warm_start_{false};

    void setEnableS3(bool enable) {
        enable_s3_ = enable;
        if (enable && !s3_store_) {
            s3_store_ = std::make_unique<S3ByproductStore>();
        }
    }
    bool isS3Enabled() const { return enable_s3_; }
    void setS3ProximityThreshold(dist_t threshold) {
        if (s3_store_) s3_store_->proximity_threshold = threshold;
    }
    uint64_t getS3Hits() const { return s3_hits_.load(); }
    uint64_t getS3Misses() const { return s3_misses_.load(); }
    uint64_t getS3DistSaved() const { return s3_dist_saved_.load(); }
    
    // Warm start methods
    void setEnableWarmStart(bool enable) {
        enable_warm_start_ = enable;
        if (enable && !s3_store_) {
            s3_store_ = std::make_unique<S3ByproductStore>();
        }
    }
    bool isWarmStartEnabled() const { return enable_warm_start_; }
    uint64_t getWarmStartHits() const { return warm_start_hits_.load(); }
    uint64_t getWarmStartImprovement() const { return warm_start_improvement_.load(); }

    // Temporarily suspend/resume all S3/PathSkip mechanisms (for recall evaluation)
    void suspendSharing() {
        s3_suspended_ = true;
    }
    void resumeSharing() {
        s3_suspended_ = false;
    }
    bool isSharingSuspended() const { return s3_suspended_; }

    // Adaptive locality detection config/stats
    void setS3AdaptiveEnabled(bool enabled) {
        if (s3_store_) s3_store_->adaptive_enabled = enabled;
    }
    bool isS3AdaptiveEnabled() const {
        return s3_store_ ? s3_store_->adaptive_enabled : true;
    }
    void setS3AdaptiveThreshold(dist_t threshold) {
        if (s3_store_) s3_store_->adaptive_threshold = threshold;
    }
    dist_t getS3AdaptiveThreshold() const {
        return s3_store_ ? s3_store_->adaptive_threshold : 0.5;
    }
    uint64_t getS3AdaptiveAccepts() const {
        return s3_store_ ? s3_store_->adaptive_accepts.load() : 0;
    }
    uint64_t getS3AdaptiveRejects() const {
        return s3_store_ ? s3_store_->adaptive_rejects.load() : 0;
    }

    // ===== Search Frontier Pool (Search→Search sharing) =====
    std::unique_ptr<SearchFrontierPool> search_frontier_pool_;
    bool enable_search_sharing_{false};
    int search_sharing_check_interval_{10};  // how often to check/publish
    mutable std::atomic<uint64_t> search_sharing_hits_{0};
    mutable std::atomic<uint64_t> search_sharing_misses_{0};

    void setEnableSearchSharing(bool enable) {
        enable_search_sharing_ = enable;
        if (enable && !search_frontier_pool_) {
            search_frontier_pool_ = std::make_unique<SearchFrontierPool>();
        }
    }
    
    bool isSearchSharingEnabled() const { return enable_search_sharing_; }
    
    void setSearchSharingCheckInterval(int interval) {
        search_sharing_check_interval_ = interval;
    }
    
    int getSearchSharingCheckInterval() const { return search_sharing_check_interval_; }
    
    uint64_t getSearchSharingHits() const { return search_sharing_hits_.load(); }
    uint64_t getSearchSharingMisses() const { return search_sharing_misses_.load(); }

    // ===== Path Skip & Candidate Injection =====
    bool enable_path_skip_{false};
    bool enable_candidate_injection_{false};

    void setEnablePathSkip(bool enable) {
        enable_path_skip_ = enable;
        if (enable && !s3_store_) {
            s3_store_ = std::make_unique<S3ByproductStore>();
        }
    }
    void setEnableCandidateInjection(bool enable) {
        enable_candidate_injection_ = enable;
        if (enable && !s3_store_) {
            s3_store_ = std::make_unique<S3ByproductStore>();
        }
    }
    bool isPathSkipEnabled() const { return enable_path_skip_; }
    bool isCandidateInjectionEnabled() const { return enable_candidate_injection_; }
    // ===== END Path Skip & Candidate Injection =====

    // ===== END S3 =====

    struct TelemetryState {
        bool collecting{false};
        size_t updates{0};
        size_t* output{nullptr};
    };
    inline static thread_local TelemetryState telemetry_state_;
    std::atomic<bool> telemetry_enabled_{false};
    mutable std::atomic<size_t> telemetry_last_updates_{0};
    std::atomic<bool> insert_vector_read_admission_enabled_{false};

    struct PartialState {
        bool active{false};
        size_t limit{0};
        size_t done{0};
    };
    inline static thread_local PartialState partial_state_;

    void enableInsertTelemetry(bool enabled) {
        telemetry_enabled_.store(enabled, std::memory_order_relaxed);
    }

    size_t getLastInsertUpdateCount() const {
        return telemetry_last_updates_.load(std::memory_order_relaxed);
    }

    void configurePartialInsert(size_t max_updates) {
        partial_state_.active = false;
        partial_state_.done = 0;
        partial_state_.limit = max_updates;
    }

    void disablePartialInsert() {
        partial_state_ = PartialState{};
    }

    void enableInsertVectorReadAdmission(bool enabled) {
        insert_vector_read_admission_enabled_.store(enabled, std::memory_order_relaxed);
    }

    void setTelemetryOutputTarget(size_t* target) const {
        telemetry_state_.output = target;
    }

    size_t isNSW_{0};

    HierarchicalNSW(SpaceInterface<dist_t>* s) {
    }

    HierarchicalNSW(SpaceInterface<dist_t>* s, const std::string& location, bool nmslib = false,
                    size_t max_elements = 0, bool allow_replace_deleted = false,
                    bool use_node_lock_in_search = true)
        : allow_replace_deleted_(allow_replace_deleted),
          use_node_lock_in_search_(use_node_lock_in_search) {
        loadIndex(location, s, max_elements);
    }

    HierarchicalNSW(SpaceInterface<dist_t>* s, size_t max_elements, size_t M = 16,
                    size_t ef_construction = 200, size_t isNSW = 0, size_t random_seed = 100,
                    bool allow_replace_deleted = false, bool use_node_lock_in_search = true)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS), isNSW_(isNSW), link_list_locks_(max_elements),
          element_levels_(max_elements), allow_replace_deleted_(allow_replace_deleted),
          use_node_lock_in_search_(use_node_lock_in_search) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if (M <= 10000) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects."
                    << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing."
                    << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        configureLevel0Layout();

	        data_level0_memory_ = (char*)malloc(max_elements_ * size_data_per_element_);
	        if (data_level0_memory_ == nullptr)
	            throw std::runtime_error("Not enough memory");
	        allocateHotVecIdRange(max_elements_);
	        allocateSearchVectorShadow(max_elements_);
	        allocateInsertVectorShadow(max_elements_);
	        allocateInsertVectorRegionShadow(max_elements_);
	        allocateInsertVectorReadTrace(max_elements_);
	        allocateSearchVectorReadTrace(max_elements_);
        allocateSearchActiveNodeTable(max_elements_);
        allocateSearchActiveCacheBuckets();
        allocateLevel0CowOverrides(max_elements_);

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char**)malloc(sizeof(void*) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error(
                "Not enough memory: HierarchicalNSW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }

    ~HierarchicalNSW() {
        clear();
    }

    class TelemetryGuard {
       public:
        TelemetryGuard(HierarchicalNSW* owner, bool enable) : owner_(owner) {
            active_ = enable && owner_->telemetry_enabled_.load(std::memory_order_relaxed);
            if (active_) {
                telemetry_state_.collecting = true;
                telemetry_state_.updates = 0;
            }
        }

        ~TelemetryGuard() {
            if (active_) {
                owner_->telemetry_last_updates_.store(telemetry_state_.updates,
                                                      std::memory_order_relaxed);
                if (telemetry_state_.output) {
                    *(telemetry_state_.output) = telemetry_state_.updates;
                }
                telemetry_state_.collecting = false;
                telemetry_state_.output = nullptr;
            }
        }

       private:
        HierarchicalNSW* owner_;
        bool active_;
    };

    inline bool recordEdgeUpdate() const {
        if (telemetry_state_.collecting)
            telemetry_state_.updates++;
        if (partial_state_.limit > 0) {
            if (!partial_state_.active) {
                partial_state_.active = true;
                partial_state_.done = 0;
            }
            if (++partial_state_.done >= partial_state_.limit)
                return false;
        }
        return true;
    }

    class PartialGuard {
       public:
        PartialGuard() = default;
        ~PartialGuard() {
            partial_state_ = PartialState{};
        }
    };

    void clear() {
        freeLevel0CowStorage();
	        free(search_vector_shadow_memory_);
	        search_vector_shadow_memory_ = nullptr;
	        freeInsertVectorShadow();
	        freeInsertVectorRegionShadow();
	        freeInsertVectorReadTrace();
	        freeSearchVectorReadTrace();
	        freeHotVecIdRange();
        freeSearchActiveNodeTable();
        freeSearchActiveCacheBuckets();
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }

    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
                                  std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    struct InsertSearchCandidateCompare {
        const HierarchicalNSW<dist_t>* index;

        bool operator()(std::pair<dist_t, tableint> const& a,
                        std::pair<dist_t, tableint> const& b) const noexcept {
            if (a.first != b.first) {
                return a.first < b.first;
            }
#if HNSW_INSERT_SEARCH_CACHELINE_TIEBREAK
            return index->vectorPayloadCacheLine(a.second) >
                   index->vectorPayloadCacheLine(b.second);
#else
            (void)index;
            return false;
#endif
        }
    };

    inline uintptr_t vectorPayloadCacheLine(tableint internal_id) const noexcept {
        return (reinterpret_cast<uintptr_t>(
                    data_level0_memory_ +
                    static_cast<size_t>(internal_id) * size_data_per_element_ +
                    offsetData_) >>
                6);
    }

    void setEf(size_t ef) {
        ef_ = ef;
    }

    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }

    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label,
               (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_),
               sizeof(labeltype));
        return return_label;
    }

    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label,
               sizeof(labeltype));
    }

    inline labeltype* getExternalLabeLp(tableint internal_id) const {
        return (labeltype*)(data_level0_memory_ + internal_id * size_data_per_element_ +
                            label_offset_);
    }

    inline char* getDataByInternalId(tableint internal_id) const {
#if HNSW_HOT_VEC_ID_RANGE || HNSW_HOT_VEC_LEVEL_RENUMBER
        if (hot_vec_pool_ != nullptr) {
            tableint thresh = (tableint)hot_vec_pool_count_;
            if (internal_id < thresh) {
                return hot_vec_pool_ + static_cast<size_t>(internal_id) * data_size_;
            } else {
                return cold_vec_pool_ +
                       static_cast<size_t>(internal_id - thresh) * data_size_;
            }
        }
#endif
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }

    void allocateHotVecIdRange(size_t max_elements) {
#if HNSW_HOT_VEC_ID_RANGE && !HNSW_HOT_VEC_LEVEL_RENUMBER
        // Static allocation with compile-time threshold (no renumber).
        if (max_elements == 0 || data_size_ == 0) {
            return;
        }
        size_t hot_count = HNSW_HOT_VEC_ID_RANGE_THRESHOLD;
        if (hot_count > max_elements) hot_count = max_elements;
        size_t cold_count = max_elements > hot_count ? max_elements - hot_count : 0;
        hot_vec_pool_count_ = hot_count;
        void* hp = nullptr;
        void* cp = nullptr;
        if (hot_count > 0) {
            if (posix_memalign(&hp, 64, hot_count * data_size_) != 0 || !hp) {
                throw std::runtime_error("hot_vec_pool alloc failed");
            }
            memset(hp, 0, hot_count * data_size_);
        }
        if (cold_count > 0) {
            if (posix_memalign(&cp, 64, cold_count * data_size_) != 0 || !cp) {
                throw std::runtime_error("cold_vec_pool alloc failed");
            }
            memset(cp, 0, cold_count * data_size_);
        }
        hot_vec_pool_ = (char*)hp;
        cold_vec_pool_ = (char*)cp;
#else
        (void)max_elements;
#endif
    }

    void freeHotVecIdRange() {
#if HNSW_HOT_VEC_ID_RANGE || HNSW_HOT_VEC_LEVEL_RENUMBER
        free(hot_vec_pool_);
        hot_vec_pool_ = nullptr;
        free(cold_vec_pool_);
        cold_vec_pool_ = nullptr;
        hot_vec_pool_count_ = 0;
        hot_id_renumber_done_ = false;
#endif
    }

#if HNSW_HOT_VEC_LEVEL_RENUMBER
    // One-time renumber pass: after the initial graph has been built, assign
    // level>=1 nodes to IDs 0..N_upper-1, and level==0 nodes to N_upper..n-1.
    // All adjacency-list ID references are rewritten in place. Vector data is
    // copied into the hot/cold pools. Single-threaded; caller must ensure no
    // other thread is accessing the index.
    void doHotVecLevelRenumber() {
        if (hot_id_renumber_done_) return;
        const size_t n = cur_element_count.load(std::memory_order_relaxed);
        if (n == 0) return;
        if (data_size_ == 0 || size_data_per_element_ == 0) return;

        const auto t0 = std::chrono::steady_clock::now();

        // Count hot nodes (level >= 1).
        size_t hot_count = 0;
        for (size_t i = 0; i < n; i++) {
            if (element_levels_[i] >= 1) hot_count++;
        }

        // Build old_to_new mapping.
        std::vector<tableint> old_to_new(n);
        {
            tableint next_hot = 0;
            tableint next_cold = (tableint)hot_count;
            for (size_t i = 0; i < n; i++) {
                if (element_levels_[i] >= 1) {
                    old_to_new[i] = next_hot++;
                } else {
                    old_to_new[i] = next_cold++;
                }
            }
        }

        // Allocate new data_level0_memory_ and permute blocks.
        char* new_memory =
            (char*)malloc(max_elements_ * size_data_per_element_);
        if (!new_memory) throw std::runtime_error("renumber: data alloc failed");
        memset(new_memory, 0, max_elements_ * size_data_per_element_);
        for (size_t old_id = 0; old_id < n; old_id++) {
            size_t new_id = old_to_new[old_id];
            memcpy(new_memory + new_id * size_data_per_element_,
                   data_level0_memory_ + old_id * size_data_per_element_,
                   size_data_per_element_);
        }

        // Rewrite level-0 adjacency IDs in new buffer.
        for (size_t new_id = 0; new_id < n; new_id++) {
            linklistsizeint* lnk =
                (linklistsizeint*)(new_memory +
                                   new_id * size_data_per_element_ +
                                   offsetLevel0_);
            unsigned short sz = *((unsigned short*)lnk);
            tableint* ids = (tableint*)(lnk + 1);
            for (unsigned short j = 0; j < sz; j++) {
                if (ids[j] < n) ids[j] = old_to_new[ids[j]];
            }
        }

        // Permute upper-layer linkLists_ pointers, then rewrite IDs.
        char** new_linkLists = (char**)malloc(max_elements_ * sizeof(char*));
        if (!new_linkLists) throw std::runtime_error("renumber: linklists alloc");
        memset(new_linkLists, 0, max_elements_ * sizeof(char*));
        for (size_t old_id = 0; old_id < n; old_id++) {
            if (linkLists_[old_id] != nullptr) {
                size_t new_id = old_to_new[old_id];
                new_linkLists[new_id] = linkLists_[old_id];
                int max_level = element_levels_[old_id];
                for (int level = 1; level <= max_level; level++) {
                    linklistsizeint* lnk =
                        (linklistsizeint*)(new_linkLists[new_id] +
                                           (level - 1) *
                                               size_links_per_element_);
                    unsigned short sz = *((unsigned short*)lnk);
                    tableint* ids = (tableint*)(lnk + 1);
                    for (unsigned short j = 0; j < sz; j++) {
                        if (ids[j] < n) ids[j] = old_to_new[ids[j]];
                    }
                }
            }
        }

        // Permute element_levels_.
        std::vector<int> new_levels(max_elements_, 0);
        for (size_t old_id = 0; old_id < n; old_id++) {
            new_levels[old_to_new[old_id]] = element_levels_[old_id];
        }

        // Rebuild label_lookup_ (external label -> internal id).
        {
            std::lock_guard<std::mutex> lk(label_lookup_lock);
            for (auto& kv : label_lookup_) {
                if (kv.second < n) kv.second = old_to_new[kv.second];
            }
        }

        // Update entry_point.
        if (enterpoint_node_ >= 0 && (size_t)enterpoint_node_ < n) {
            enterpoint_node_ = old_to_new[(size_t)enterpoint_node_];
        }

        // Swap pointers.
        free(data_level0_memory_);
        data_level0_memory_ = new_memory;
        free(linkLists_);
        linkLists_ = new_linkLists;
        element_levels_ = std::move(new_levels);

        // Allocate hot/cold pools and populate from canonical.
        size_t cold_count = max_elements_ > hot_count ? max_elements_ - hot_count : 0;
        hot_vec_pool_count_ = hot_count;
        void* hp = nullptr;
        void* cp = nullptr;
        if (hot_count > 0) {
            if (posix_memalign(&hp, 64, hot_count * data_size_) != 0 || !hp) {
                throw std::runtime_error("renumber: hot pool alloc");
            }
            memset(hp, 0, hot_count * data_size_);
        }
        if (cold_count > 0) {
            if (posix_memalign(&cp, 64, cold_count * data_size_) != 0 || !cp) {
                throw std::runtime_error("renumber: cold pool alloc");
            }
            memset(cp, 0, cold_count * data_size_);
        }
        hot_vec_pool_ = (char*)hp;
        cold_vec_pool_ = (char*)cp;
        for (size_t id = 0; id < n; id++) {
            char* src = data_level0_memory_ + id * size_data_per_element_ +
                        offsetData_;
            char* dst;
            if (id < hot_count) {
                dst = hot_vec_pool_ + id * data_size_;
            } else {
                dst = cold_vec_pool_ + (id - hot_count) * data_size_;
            }
            memcpy(dst, src, data_size_);
        }

        hot_id_renumber_done_ = true;
        const auto t1 = std::chrono::steady_clock::now();
        metric_hot_id_range_renumber_ns.store(
            (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                .count(),
            std::memory_order_relaxed);
    }

    inline void maybeTriggerHotVecLevelRenumber() {
        if (hot_id_renumber_done_) return;
        size_t cur = cur_element_count.load(std::memory_order_relaxed);
        if (cur < (size_t)HNSW_HOT_VEC_LEVEL_RENUMBER_TRIGGER) return;
        static std::mutex renumber_mu;
        std::lock_guard<std::mutex> lk(renumber_mu);
        if (hot_id_renumber_done_) return;
        doHotVecLevelRenumber();
    }
#endif

    inline void demoteInsertedVectorData(tableint internal_id) const {
#if HNSW_INSERT_VECTOR_CLDEMOTE && defined(__CLDEMOTE__)
        const char* data = getDataByInternalId(internal_id);
        constexpr size_t kCacheLineBytes = 64;
        for (size_t offset = 0; offset < data_size_; offset += kCacheLineBytes) {
            _cldemote(data + offset);
        }
#else
        (void)internal_id;
#endif
    }

    inline bool acquireExistingNeighborRepairAdmission(size_t level) const {
#if HNSW_INSERT_REPAIR_PHASE_MAX_ACTIVE > 0
        if (level != 0) {
            return false;
        }
        for (;;) {
            uint32_t active =
                active_existing_neighbor_repairs.load(std::memory_order_relaxed);
            if (active < HNSW_INSERT_REPAIR_PHASE_MAX_ACTIVE &&
                active_existing_neighbor_repairs.compare_exchange_weak(
                    active, active + 1, std::memory_order_acquire,
                    std::memory_order_relaxed)) {
                return true;
            }
#if HNSW_INSERT_REPAIR_PHASE_WAIT_NS > 0
            std::this_thread::sleep_for(
                std::chrono::nanoseconds(HNSW_INSERT_REPAIR_PHASE_WAIT_NS));
#else
            std::this_thread::yield();
#endif
        }
#else
        (void)level;
        return false;
#endif
    }

    inline void releaseExistingNeighborRepairAdmission(size_t level,
                                                       bool acquired) const {
#if HNSW_INSERT_REPAIR_PHASE_MAX_ACTIVE > 0
        if (acquired && level == 0) {
            active_existing_neighbor_repairs.fetch_sub(1, std::memory_order_release);
        }
#else
        (void)level;
        (void)acquired;
#endif
    }

    inline bool acquireInsertVectorReadAdmission() const {
#if HNSW_INSERT_VECTOR_READ_MAX_ACTIVE > 0
        if (!insert_vector_read_admission_enabled_.load(std::memory_order_relaxed)) {
            return false;
        }
        const auto wait_start = std::chrono::steady_clock::now();
        bool waited = false;
        for (;;) {
            uint32_t active =
                active_insert_vector_readers.load(std::memory_order_relaxed);
            if (active < HNSW_INSERT_VECTOR_READ_MAX_ACTIVE &&
                active_insert_vector_readers.compare_exchange_weak(
                    active, active + 1, std::memory_order_acquire,
                    std::memory_order_relaxed)) {
                break;
            }
            waited = true;
#if HNSW_INSERT_VECTOR_READ_WAIT_NS > 0
            std::this_thread::sleep_for(
                std::chrono::nanoseconds(HNSW_INSERT_VECTOR_READ_WAIT_NS));
#else
            std::this_thread::yield();
#endif
        }
        metric_insert_vector_read_admissions.fetch_add(1, std::memory_order_relaxed);
        if (waited) {
            metric_insert_vector_read_waits.fetch_add(1, std::memory_order_relaxed);
            metric_insert_vector_read_wait_ns.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - wait_start)
                        .count()),
                std::memory_order_relaxed);
        }
        return true;
#else
        return false;
#endif
    }

    inline void releaseInsertVectorReadAdmission(bool acquired) const {
#if HNSW_INSERT_VECTOR_READ_MAX_ACTIVE > 0
        if (acquired) {
            active_insert_vector_readers.fetch_sub(1, std::memory_order_release);
        }
#else
        (void)acquired;
#endif
    }

    class ExistingNeighborRepairAdmissionGuard {
       public:
        ExistingNeighborRepairAdmissionGuard(const HierarchicalNSW* index,
                                             size_t level)
            : index_(index),
              level_(level),
              acquired_(index->acquireExistingNeighborRepairAdmission(level)) {}

        ~ExistingNeighborRepairAdmissionGuard() {
            index_->releaseExistingNeighborRepairAdmission(level_, acquired_);
        }

       private:
        const HierarchicalNSW* index_;
        size_t level_;
        bool acquired_;
    };

    class InsertVectorReadAdmissionGuard {
       public:
        explicit InsertVectorReadAdmissionGuard(const HierarchicalNSW* index)
            : index_(index),
              acquired_(index->acquireInsertVectorReadAdmission()) {}

        ~InsertVectorReadAdmissionGuard() {
            index_->releaseInsertVectorReadAdmission(acquired_);
        }

       private:
        const HierarchicalNSW* index_;
        bool acquired_;
    };

    class SearchActiveNodeGuard {
       public:
        SearchActiveNodeGuard(const HierarchicalNSW* index, tableint internal_id)
            : index_(index),
              internal_id_(internal_id),
              marked_(index->markSearchActiveNode(internal_id)) {}

        ~SearchActiveNodeGuard() {
            index_->unmarkSearchActiveNode(internal_id_, marked_);
        }

       private:
        const HierarchicalNSW* index_;
        tableint internal_id_;
        bool marked_;
    };

    class SearchActiveCacheNodeGuard {
       public:
        SearchActiveCacheNodeGuard(const HierarchicalNSW* index, tableint internal_id)
            : index_(index),
              count_(index->markSearchNodeCacheFootprint(internal_id, buckets_)) {}

        ~SearchActiveCacheNodeGuard() {
            index_->unmarkSearchCacheBuckets(buckets_, count_);
        }

       private:
        const HierarchicalNSW* index_;
        ActiveCacheBucketArray buckets_{};
        size_t count_{0};
    };

    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    size_t getIndexMemory() const {
        size_t total_memory = 0;

        total_memory += max_elements_ * size_data_per_element_;
        total_memory += searchVectorShadowBytes();
        total_memory += insertVectorShadowBytes();
        total_memory += searchActiveNodeTableBytes();
        total_memory += searchActiveCacheBucketBytes();
        total_memory += level0AdjCowBytes();
#if HNSW_L0_ADJ_COW_PUBLISH
        total_memory += max_elements_ * sizeof(std::atomic<char*>);
#endif
        total_memory += max_elements_ * sizeof(char*);

        for (size_t i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0) {
                total_memory += size_links_per_element_ * element_levels_[i];
            }
        }

        total_memory += element_levels_.size() * sizeof(int);
        total_memory += link_list_locks_.size() * sizeof(LinkListLock);
        total_memory += label_op_locks_.size() * sizeof(std::mutex);
        total_memory += label_lookup_.size() * (sizeof(labeltype) + sizeof(tableint));
        total_memory += label_lookup_.bucket_count() * sizeof(void*);
        total_memory += label_lookup_.size() * sizeof(void*);
        total_memory += deleted_elements.size() * sizeof(tableint);

        if (visited_list_pool_) {
            total_memory += visited_list_pool_->getMemoryUsage();
        }

        return total_memory;
    }

    VisitedList* acquireVisitedListScratch() const {
        return visited_list_pool_->getFreeVisitedList();
    }

    void releaseVisitedListScratch(VisitedList* vl) const {
        visited_list_pool_->releaseVisitedList(vl);
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
    searchBaseLayer(tableint ep_id, const void* data_point, int layer) {
        InsertVectorReadAdmissionGuard vector_read_admission(this);
#if HNSW_INSERT_SEARCH_GLOBAL_METRICS
        const auto base_search_start = std::chrono::steady_clock::now();
#endif
        uint64_t local_expansions = 0;
        uint64_t local_edges_scanned = 0;
        uint64_t local_dist_comps = 0;
        VisitedList* vl = acquireVisitedListScratch();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            InsertSearchCandidateCompare>
            candidateSet{InsertSearchCandidateCompare{this}};

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = insertConstructionDistance(data_point, ep_id);
            local_dist_comps++;
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
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            // Insert construction search reads adjacency only; graph mutation
            // still takes the exclusive lock on the write path.
#if HNSW_INSERT_SEARCH_GLOBAL_METRICS
            const auto search_lock_wait_start = std::chrono::steady_clock::now();
#endif
            std::shared_lock<std::shared_mutex> lock(link_list_locks_[curNodeNum]);
#if HNSW_INSERT_SEARCH_GLOBAL_METRICS
            const auto search_lock_acquired = std::chrono::steady_clock::now();
            metric_graph_search_unique_lock_wait_ns.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        search_lock_acquired - search_lock_wait_start).count()),
                std::memory_order_relaxed);
            metric_graph_search_unique_lock_acqs.fetch_add(1, std::memory_order_relaxed);
#endif

            int* data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
                //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) *
                //                    size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            local_expansions++;
            local_edges_scanned += size;
            tableint* datal = (tableint*)(data + 1);
#if defined(USE_SSE) && HNSW_INSERT_SEARCH_PREFETCH_AHEAD > 0
            if (size > 0) {
                tableint prefetch_id = datal[0];
                _mm_prefetch((char*)(visited_array + prefetch_id),
                             HNSW_INSERT_SEARCH_PREFETCH_HINT_VALUE);
                _mm_prefetch((char*)(visited_array + prefetch_id + 64),
                             HNSW_INSERT_SEARCH_PREFETCH_HINT_VALUE);
                prefetchInsertDataByInternalId(prefetch_id);
            }
            if (size > 1) {
                prefetchInsertDataByInternalId(datal[1]);
            }
#endif

#if HNSW_INSERT_SEARCH_ADDR_BATCH
            struct PendingInsertCandidate {
                tableint id;
                char* data;
                dist_t dist;
            };
            std::vector<PendingInsertCandidate> pending;
            pending.reserve(size);
            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = datal[j];
#if defined(USE_SSE) && HNSW_INSERT_SEARCH_PREFETCH_AHEAD > 0
                if (j + HNSW_INSERT_SEARCH_PREFETCH_AHEAD < size) {
                    tableint prefetch_id = datal[j + HNSW_INSERT_SEARCH_PREFETCH_AHEAD];
                    _mm_prefetch((char*)(visited_array + prefetch_id),
                                 HNSW_INSERT_SEARCH_PREFETCH_HINT_VALUE);
                    prefetchInsertDataByInternalId(prefetch_id);
                }
#endif
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;
                visited_array[candidate_id] = visited_array_tag;
                pending.push_back(
                    {candidate_id, getInsertDataByInternalId(candidate_id), dist_t{}});
            }

            if (!pending.empty()) {
                std::vector<size_t> addr_order(pending.size());
                for (size_t i = 0; i < pending.size(); i++) {
                    addr_order[i] = i;
                }
                std::sort(addr_order.begin(), addr_order.end(),
                          [&pending](size_t a, size_t b) {
                              return pending[a].data < pending[b].data;
                          });
                for (size_t pos = 0; pos < addr_order.size(); pos++) {
#if defined(USE_SSE) && HNSW_INSERT_SEARCH_PREFETCH_AHEAD > 0
                    if (pos + HNSW_INSERT_SEARCH_PREFETCH_AHEAD < addr_order.size()) {
                        size_t prefetch_idx =
                            addr_order[pos + HNSW_INSERT_SEARCH_PREFETCH_AHEAD];
                        _mm_prefetch(pending[prefetch_idx].data,
                                     HNSW_INSERT_SEARCH_PREFETCH_HINT_VALUE);
                    }
#endif
                    size_t idx = addr_order[pos];
                    pending[idx].dist =
                        fstdistfunc_(data_point, pending[idx].data, dist_func_param_);
                }
                local_dist_comps += pending.size();
                metric_graph_base_search_addr_batch_candidates.fetch_add(
                    pending.size(), std::memory_order_relaxed);
            }

            for (const auto& cand : pending) {
                dist_t dist1 = cand.dist;
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, cand.id);
#if defined(USE_SSE) && HNSW_INSERT_SEARCH_PREFETCH_AHEAD > 0
                    prefetchInsertDataByInternalId(candidateSet.top().second);
#endif

                    if (!isMarkedDeleted(cand.id))
                        top_candidates.emplace(dist1, cand.id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
#else
            auto process_insert_search_candidate = [&](tableint candidate_id,
                                                       char* currObj1) {
#if HNSW_INSERT_SEARCH_NT_DISTANCE
                (void)currObj1;
                dist_t dist1 = insertConstructionDistance(data_point, candidate_id);
#else
                dist_t dist1 =
                    fstdistfunc_(data_point, currObj1, dist_func_param_);
#endif
                local_dist_comps++;
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#if defined(USE_SSE) && HNSW_INSERT_SEARCH_PREFETCH_AHEAD > 0
                    prefetchInsertDataByInternalId(candidateSet.top().second);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            };

#if HNSW_CACHE_AWARE_INSERT_SEARCH_ORDER && HNSW_SEARCH_ACTIVE_CACHE_BUCKET_TRACKING
            size_t deferred_insert_search_reads = 0;
            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = datal[j];
#if defined(USE_SSE) && HNSW_INSERT_SEARCH_PREFETCH_AHEAD > 0
                if (j + HNSW_INSERT_SEARCH_PREFETCH_AHEAD < size) {
                    tableint prefetch_id = datal[j + HNSW_INSERT_SEARCH_PREFETCH_AHEAD];
                    _mm_prefetch((char*)(visited_array + prefetch_id),
                                 HNSW_INSERT_SEARCH_PREFETCH_HINT_VALUE);
                    prefetchInsertDataByInternalId(prefetch_id);
                }
#endif
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;
                char* currObj1 = getInsertDataByInternalId(candidate_id);
                if (cacheAddressHasActiveBucket(currObj1)) {
                    deferred_insert_search_reads++;
                    continue;
                }
                visited_array[candidate_id] = visited_array_tag;
                process_insert_search_candidate(candidate_id, currObj1);
            }
            if (deferred_insert_search_reads > 0) {
                metric_insert_search_cache_defer_lists.fetch_add(
                    1, std::memory_order_relaxed);
                metric_insert_search_cache_defer_nodes.fetch_add(
                    deferred_insert_search_reads, std::memory_order_relaxed);
                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = datal[j];
                    if (visited_array[candidate_id] == visited_array_tag)
                        continue;
                    visited_array[candidate_id] = visited_array_tag;
                    process_insert_search_candidate(
                        candidate_id, getInsertDataByInternalId(candidate_id));
                }
            }
#else
            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = datal[j];
#if defined(USE_SSE) && HNSW_INSERT_SEARCH_PREFETCH_AHEAD > 0
                if (j + HNSW_INSERT_SEARCH_PREFETCH_AHEAD < size) {
                    tableint prefetch_id = datal[j + HNSW_INSERT_SEARCH_PREFETCH_AHEAD];
                    _mm_prefetch((char*)(visited_array + prefetch_id),
                                 HNSW_INSERT_SEARCH_PREFETCH_HINT_VALUE);
                    prefetchInsertDataByInternalId(prefetch_id);
                }
#endif
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;
                visited_array[candidate_id] = visited_array_tag;
                process_insert_search_candidate(candidate_id,
                                                getInsertDataByInternalId(candidate_id));
            }
#endif

#endif
#if HNSW_INSERT_SEARCH_GLOBAL_METRICS
            // critical section ends with `lock` going out of scope at the
            // end of this while body; record holding time before destructor.
            metric_graph_search_critical_ns.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - search_lock_acquired).count()),
                std::memory_order_relaxed);
#endif
        }
        releaseVisitedListScratch(vl);
#if HNSW_INSERT_SEARCH_GLOBAL_METRICS
        metric_graph_base_search_ns.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - base_search_start)
                    .count()),
            std::memory_order_relaxed);
        metric_graph_base_search_expansions.fetch_add(local_expansions,
                                                      std::memory_order_relaxed);
        metric_graph_base_search_edges_scanned.fetch_add(local_edges_scanned,
                                                         std::memory_order_relaxed);
        metric_graph_base_search_dist_comps.fetch_add(local_dist_comps,
                                                       std::memory_order_relaxed);
#endif

        return top_candidates;
    }

    // bare_bone_search means there is no check for deletions and stop condition is ignored in
    // return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
    searchBaseLayerST(tableint ep_id, const void* data_point, size_t ef,
                      BaseFilterFunctor* isIdAllowed = nullptr,
                      BaseSearchStopCondition<dist_t>* stop_condition = nullptr,
                      SearchTelemetry* telemetry = nullptr) const {
        VisitedList* vl = acquireVisitedListScratch();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || (!isMarkedDeleted(ep_id) &&
                                 ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getSearchDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            if (telemetry) {
                telemetry->distance_computations++;
                telemetry->level0_distance_computations++;
            }
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (telemetry) {
                telemetry->result_pushes++;
            }
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
            if (telemetry) {
                telemetry->candidate_pushes++;
            }
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
            if (telemetry) {
                telemetry->candidate_pushes++;
            }
        }

        visited_array[ep_id] = visited_array_tag;
        if (telemetry) {
            telemetry->visited_nodes++;
        }

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search =
                        stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();
            if (telemetry) {
                telemetry->candidate_pops++;
            }

            tableint current_node_id = current_node_pair.second;
#if HNSW_SEARCH_ACTIVE_NODE_TRACKING
            SearchActiveNodeGuard search_active_node_guard(this, current_node_id);
#endif
#if HNSW_SEARCH_ACTIVE_CACHE_BUCKET_TRACKING
            SearchActiveCacheNodeGuard search_active_cache_guard(this, current_node_id);
#endif
            std::shared_lock<std::shared_mutex> level_lock;
            if (use_node_lock_in_search_) {
                uint64_t lock_wait_start = telemetry ? s3_rdtsc_ns() : 0;
                level_lock = std::shared_lock<std::shared_mutex>(link_list_locks_[current_node_id]);
                if (telemetry) {
                    telemetry->level0_lock_wait_ns += s3_rdtsc_ns() - lock_wait_start;
                }
            }
            int* data = (int*)get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
            //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations += size;
            }
            if (telemetry) {
                telemetry->level0_expansions++;
                telemetry->level0_edges_scanned += size;
            }

#ifdef USE_SSE
            _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getSearchDataByInternalId(*(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char*)(data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#if defined(USE_SSE) && HNSW_SEARCH_PREFETCH_AHEAD > 0
                if (j + HNSW_SEARCH_PREFETCH_AHEAD <= size) {
                    tableint prefetch_id = *(data + j + HNSW_SEARCH_PREFETCH_AHEAD);
                    _mm_prefetch((char*)(visited_array + prefetch_id),
                                 HNSW_SEARCH_PREFETCH_HINT_VALUE);
                    _mm_prefetch(getSearchDataByInternalId(prefetch_id),
                                 HNSW_SEARCH_PREFETCH_HINT_VALUE);
                }
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;
                    if (telemetry) {
                        telemetry->visited_nodes++;
                    }

                    char* currObj1 = getSearchDataByInternalId(candidate_id);
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (telemetry) {
                        telemetry->distance_computations++;
                        telemetry->level0_distance_computations++;
                    }

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate =
                            stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
                        if (telemetry) {
                            telemetry->candidate_pushes++;
                        }
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ +
                                         candidate_set.top().second * size_data_per_element_ +
                                         offsetLevel0_,  ///////////
                                     _MM_HINT_T0);       ////////////////////////
#endif

                        if (bare_bone_search ||
                            (!isMarkedDeleted(candidate_id) &&
                            ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (telemetry) {
                                telemetry->result_pushes++;
                            }
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id),
                                                                    currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(
                                    getExternalLabel(id), getSearchDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        releaseVisitedListScratch(vl);
        return top_candidates;
    }

    // Search→Search sharing variant of searchBaseLayerST
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
    searchBaseLayerShared(tableint ep_id, const void* data_point, size_t ef,
                         BaseFilterFunctor* isIdAllowed = nullptr,
                         BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        
        if (!enable_search_sharing_ || !search_frontier_pool_) {
            // Fall back to regular search if sharing disabled
            return searchBaseLayerST<bare_bone_search, collect_metrics>(ep_id, data_point, ef, isIdAllowed, stop_condition);
        }

        // Register this search with the frontier pool
        uint64_t timestamp = s3_rdtsc_ns();
        int frontier_slot = search_frontier_pool_->registerSearch(data_point, timestamp);
        
        // Try to get useful candidates from other active searches
        auto shared_candidates = search_frontier_pool_->findUsefulFrontier(
            data_point, fstdistfunc_, dist_func_param_, static_cast<dist_t>(0.1), 1000000000ULL);
        
        VisitedList* vl = acquireVisitedListScratch();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            candidate_set;

        dist_t lowerBound;
        
        // Initialize with entry point
        if (bare_bone_search || (!isMarkedDeleted(ep_id) &&
                                 ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        // Add shared candidates to initial beam if we found any
        if (!shared_candidates.empty()) {
            search_sharing_hits_.fetch_add(1, std::memory_order_relaxed);
            for (const auto& cand : shared_candidates) {
                if (!isMarkedDeleted(cand.second) &&
                    ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(cand.second)))) {
                    // Re-compute distance to this query
                    char* cand_data = getDataByInternalId(cand.second);
                    dist_t dist = fstdistfunc_(data_point, cand_data, dist_func_param_);
                    
                    if (top_candidates.size() < ef || dist < lowerBound) {
                        top_candidates.emplace(dist, cand.second);
                        candidate_set.emplace(-dist, cand.second);
                        visited_array[cand.second] = visited_array_tag;
                        
                        if (!bare_bone_search && stop_condition) {
                            stop_condition->add_point_to_result(getExternalLabel(cand.second), cand_data, dist);
                        }
                        
                        // Update lower bound
                        if (top_candidates.size() > ef) {
                            top_candidates.pop();
                        }
                        if (!top_candidates.empty()) {
                            lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }
        } else {
            search_sharing_misses_.fetch_add(1, std::memory_order_relaxed);
        }

        visited_array[ep_id] = visited_array_tag;

        int iteration_count = 0;
        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search =
                        stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            // Periodically update frontier pool with current beam
            if (frontier_slot >= 0 && iteration_count % search_sharing_check_interval_ == 0) {
                // Convert top_candidates to vector for sharing
                std::vector<std::pair<dist_t, tableint>> current_beam;
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                                    CompareByFirst> temp_queue = top_candidates;
                while (!temp_queue.empty()) {
                    current_beam.push_back(temp_queue.top());
                    temp_queue.pop();
                }
                search_frontier_pool_->updateFrontier(frontier_slot, current_beam);
            }
            iteration_count++;

            tableint current_node_id = current_node_pair.second;
            std::shared_lock<std::shared_mutex> level_lock;
            if (use_node_lock_in_search_) {
                level_lock = std::shared_lock<std::shared_mutex>(link_list_locks_[current_node_id]);
            }
            int* data = (int*)get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
            
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations += size;
            }

#ifdef USE_SSE
            _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_,
                         _MM_HINT_T0);
            _mm_prefetch((char*)(data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
#ifdef USE_SSE
                _mm_prefetch((char*)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(
                    data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                    _MM_HINT_T0);
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char* currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate =
                            stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ +
                                         candidate_set.top().second * size_data_per_element_ +
                                         offsetLevel0_,
                                     _MM_HINT_T0);
#endif

                        if (bare_bone_search ||
                            (!isMarkedDeleted(candidate_id) &&
                             ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id),
                                                                    currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(
                                    getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        // Deregister from frontier pool when search completes
        if (frontier_slot >= 0) {
            search_frontier_pool_->deregister(frontier_slot);
        }

        releaseVisitedListScratch(vl);
        return top_candidates;
    }

    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>& top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                    fstdistfunc_(getInsertDataForDistance(second_pair.second),
                                 getInsertDataForDistance(curent_pair.second),
                                 dist_func_param_);
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

    linklistsizeint* get_linklist0(tableint internal_id) const {
#if HNSW_L0_ADJ_COW_PUBLISH
        if (level0_linklist_overrides_) {
            char* override_ptr =
                level0_linklist_overrides_[internal_id].load(std::memory_order_acquire);
            if (override_ptr) {
                return reinterpret_cast<linklistsizeint*>(override_ptr);
            }
        }
#endif
        return getBaseLinklist0(internal_id);
    }

    linklistsizeint* get_linklist0(tableint internal_id, char* data_level0_memory_) const {
        return (linklistsizeint*)(data_level0_memory_ + internal_id * size_data_per_element_ +
                                  offsetLevel0_);
    }

    linklistsizeint* get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint*)(linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }

    linklistsizeint* get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }

    tableint mutuallyConnectNewElement(
        const void* data_point, tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>& top_candidates,
        int level, bool isUpdate) {
        const auto graph_connect_start = std::chrono::steady_clock::now();
        uint64_t local_link_updates = 0;
        size_t Mcurmax = level ? maxM_ : maxM0_;
        const size_t select_input = top_candidates.size();
        const auto select_start = std::chrono::steady_clock::now();
        {
            InsertVectorReadAdmissionGuard vector_read_admission(this);
            getNeighborsByHeuristic2(top_candidates, M_);
        }
        const auto select_end = std::chrono::steady_clock::now();
        metric_graph_select_new_neighbors_ns.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    select_end - select_start)
                    .count()),
            std::memory_order_relaxed);
        metric_graph_select_new_neighbors_input.fetch_add(select_input,
                                                          std::memory_order_relaxed);
        metric_graph_select_new_neighbors_selected.fetch_add(top_candidates.size(),
                                                             std::memory_order_relaxed);
        if (top_candidates.size() > M_)
            throw std::runtime_error(
                "Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            const auto inserted_link_start = std::chrono::steady_clock::now();
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock<std::shared_mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint* ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint* data = (tableint*)(ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
                local_link_updates++;
                if (!recordEdgeUpdate()) {
                    setListCount(ll_cur, idx + 1);
                    return next_closest_entry_point;
                }
            }
            metric_graph_inserted_node_link_ns.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - inserted_link_start)
                        .count()),
                std::memory_order_relaxed);
            metric_graph_inserted_node_edges_written.fetch_add(
                selectedNeighbors.size(), std::memory_order_relaxed);
        }

        std::vector<tableint> repairNeighbors = selectedNeighbors;
#if HNSW_INSERT_REPAIR_ORDER_BY_ADDR
        std::sort(repairNeighbors.begin(), repairNeighbors.end(),
                  [this, level](tableint a, tableint b) {
                      const auto addr_a = reinterpret_cast<uintptr_t>(
                          level == 0 ? get_linklist0(a) : get_linklist(a, level));
                      const auto addr_b = reinterpret_cast<uintptr_t>(
                          level == 0 ? get_linklist0(b) : get_linklist(b, level));
                      return addr_a < addr_b;
                  });
#endif
        deferCacheHotRepairTargets(repairNeighbors, level);
        deferSearchActiveRepairTargets(repairNeighbors, level);

        const auto existing_loop_start = std::chrono::steady_clock::now();
        ExistingNeighborRepairAdmissionGuard repair_admission(this, level);
        InsertVectorReadAdmissionGuard vector_read_admission(this);
        for (size_t idx = 0; idx < repairNeighbors.size(); idx++) {
            tableint neighbor = repairNeighbors[idx];
            const auto lock_wait_start = std::chrono::steady_clock::now();
            std::unique_lock<std::shared_mutex> lock(link_list_locks_[neighbor]);
            const auto lock_acquired = std::chrono::steady_clock::now();
            metric_graph_unique_lock_wait_ns.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        lock_acquired - lock_wait_start)
                        .count()),
                std::memory_order_relaxed);
            metric_graph_existing_neighbor_visits.fetch_add(1, std::memory_order_relaxed);

            linklistsizeint* ll_other;
            if (level == 0)
                ll_other = get_linklist0(neighbor);
            else
                ll_other = get_linklist(neighbor, level);

            size_t sz_link_list_other = getListCount(ll_other);
            metric_graph_existing_neighbor_loaded_edges.fetch_add(
                sz_link_list_other, std::memory_order_relaxed);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (neighbor == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[neighbor])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint* data = (tableint*)(ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                const auto scan_start = std::chrono::steady_clock::now();
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
                metric_graph_existing_neighbor_load_scan_ns.fetch_add(
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - scan_start)
                            .count()),
                    std::memory_order_relaxed);
            }

            // If cur_c is already present in the neighboring connections of
            // `selectedNeighbors[idx]` then no need to modify any connections or run the
            // heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    const auto append_start = std::chrono::steady_clock::now();
#if HNSW_L0_ADJ_COW_PUBLISH
                    if (level == 0 && partial_state_.limit == 0) {
                        bool first_publish = false;
                        linklistsizeint* ll_cow =
                            copyLevel0AdjForCow(neighbor, &first_publish);
                        tableint* cow_data = (tableint*)(ll_cow + 1);
                        cow_data[sz_link_list_other] = cur_c;
                        setListCount(ll_cow, sz_link_list_other + 1);
                        local_link_updates++;
                        if (!recordEdgeUpdate())
                            return next_closest_entry_point;
                        publishLevel0AdjCow(neighbor, reinterpret_cast<char*>(ll_cow),
                                             first_publish);
                        metric_l0_adj_cow_append_publishes.fetch_add(
                            1, std::memory_order_relaxed);
                    } else
#endif
                    {
#if HNSW_L0_ADJ_COW_PUBLISH
                        if (level == 0 && partial_state_.limit > 0) {
                            metric_l0_adj_cow_partial_fallbacks.fetch_add(
                                1, std::memory_order_relaxed);
                        }
#endif
                        if (shouldCountL0RepairNtPartialFallback(level)) {
                            metric_l0_repair_nt_partial_fallbacks.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                        const bool use_l0_nt = shouldUseL0RepairNtStore(level);
                        storeL0RepairNeighbor(data + sz_link_list_other, cur_c,
                                              use_l0_nt);
                        if (use_l0_nt) {
                            fenceL0RepairNtStores(use_l0_nt);
                            metric_l0_repair_nt_edges.fetch_add(
                                1, std::memory_order_relaxed);
                            metric_l0_repair_nt_appends.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                        setListCount(ll_other, sz_link_list_other + 1);
                        local_link_updates++;
                        if (!recordEdgeUpdate())
                            return next_closest_entry_point;
                    }
                    metric_graph_existing_neighbor_append_ns.fetch_add(
                        static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - append_start)
                                .count()),
                        std::memory_order_relaxed);
                    metric_graph_existing_neighbor_appends.fetch_add(
                        1, std::memory_order_relaxed);
                    metric_graph_existing_neighbor_edges_written.fetch_add(
                        1, std::memory_order_relaxed);
                } else {
                    metric_graph_existing_neighbor_prunes.fetch_add(
                        1, std::memory_order_relaxed);
                    metric_graph_existing_neighbor_prune_candidates.fetch_add(
                        sz_link_list_other + 1, std::memory_order_relaxed);
                    const auto prune_start = std::chrono::steady_clock::now();
                    uint64_t prune_dist_comps = 0;
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max =
                        fstdistfunc_(getInsertDataForDistance(cur_c),
                                     getInsertDataForDistance(neighbor), dist_func_param_);
                    prune_dist_comps++;
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>,
                                        std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
                        candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(fstdistfunc_(getInsertDataForDistance(data[j]),
                                                        getInsertDataForDistance(neighbor),
                                                        dist_func_param_),
                                           data[j]);
                        prune_dist_comps++;
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);
                    metric_graph_existing_neighbor_prune_ns.fetch_add(
                        static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - prune_start)
                                .count()),
                        std::memory_order_relaxed);
                    metric_graph_existing_neighbor_prune_dist_comps.fetch_add(
                        prune_dist_comps, std::memory_order_relaxed);

                    const auto rewrite_start = std::chrono::steady_clock::now();
                    const size_t old_count = sz_link_list_other;
#if HNSW_L0_ADJ_COW_PUBLISH
                    if (level == 0 && partial_state_.limit == 0) {
                        std::vector<tableint> rewritten;
                        rewritten.reserve(Mcurmax);
                        while (candidates.size() > 0) {
                            rewritten.push_back(candidates.top().second);
                            candidates.pop();
                        }
                        bool first_publish = false;
                        linklistsizeint* ll_cow =
                            copyLevel0AdjForCow(neighbor, &first_publish);
                        tableint* cow_data = (tableint*)(ll_cow + 1);
                        for (size_t rewrite_idx = 0; rewrite_idx < rewritten.size();
                             rewrite_idx++) {
                            cow_data[rewrite_idx] = rewritten[rewrite_idx];
                            local_link_updates++;
                            if (!recordEdgeUpdate()) {
                                setListCount(ll_cow, rewrite_idx + 1);
                                return next_closest_entry_point;
                            }
                        }
                        setListCount(ll_cow, rewritten.size());
                        publishLevel0AdjCow(neighbor, reinterpret_cast<char*>(ll_cow),
                                             first_publish);
                        metric_l0_adj_cow_prune_publishes.fetch_add(
                            1, std::memory_order_relaxed);
                        metric_graph_existing_neighbor_rewrite_ns.fetch_add(
                            static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - rewrite_start)
                                    .count()),
                            std::memory_order_relaxed);
                        metric_graph_existing_neighbor_edges_written.fetch_add(
                            static_cast<uint64_t>(rewritten.size()), std::memory_order_relaxed);
                        if (old_count > rewritten.size()) {
                            metric_graph_existing_neighbor_edges_pruned.fetch_add(
                                old_count - rewritten.size(), std::memory_order_relaxed);
                        }
                    } else
#endif
#if HNSW_INSERT_REPAIR_BULK_REWRITE
                    if (partial_state_.limit == 0) {
                        std::vector<tableint> rewritten;
                        rewritten.reserve(Mcurmax);
                        while (candidates.size() > 0) {
                            rewritten.push_back(candidates.top().second);
                            candidates.pop();
                        }
                        for (size_t rewrite_idx = 0; rewrite_idx < rewritten.size();
                             rewrite_idx++) {
                            local_link_updates++;
                            if (!recordEdgeUpdate()) {
                                setListCount(ll_other, rewrite_idx);
                                return next_closest_entry_point;
                            }
                        }
                        if (!rewritten.empty()) {
                            memcpy(data, rewritten.data(), rewritten.size() * sizeof(tableint));
                        }
                        setListCount(ll_other, rewritten.size());
                        metric_graph_existing_neighbor_rewrite_ns.fetch_add(
                            static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - rewrite_start)
                                    .count()),
                            std::memory_order_relaxed);
                        metric_graph_existing_neighbor_edges_written.fetch_add(
                            static_cast<uint64_t>(rewritten.size()), std::memory_order_relaxed);
                        if (old_count > rewritten.size()) {
                            metric_graph_existing_neighbor_edges_pruned.fetch_add(
                                old_count - rewritten.size(), std::memory_order_relaxed);
                        }
                    } else
#endif
                    {
#if HNSW_L0_ADJ_COW_PUBLISH
                        if (level == 0 && partial_state_.limit > 0) {
                            metric_l0_adj_cow_partial_fallbacks.fetch_add(
                                1, std::memory_order_relaxed);
                        }
#endif
                        if (shouldCountL0RepairNtPartialFallback(level)) {
                            metric_l0_repair_nt_partial_fallbacks.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                        const bool use_l0_nt = shouldUseL0RepairNtStore(level);
                        int indx = 0;
                        while (candidates.size() > 0) {
                            storeL0RepairNeighbor(data + indx, candidates.top().second,
                                                  use_l0_nt);
                            local_link_updates++;
                            if (!recordEdgeUpdate()) {
                                setListCount(ll_other, indx + 1);
                                return next_closest_entry_point;
                            }
                            candidates.pop();
                            indx++;
                        }

                        if (use_l0_nt) {
                            fenceL0RepairNtStores(use_l0_nt);
                            metric_l0_repair_nt_edges.fetch_add(
                                static_cast<uint64_t>(indx), std::memory_order_relaxed);
                            metric_l0_repair_nt_rewrites.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                        setListCount(ll_other, indx);
                        metric_graph_existing_neighbor_rewrite_ns.fetch_add(
                            static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - rewrite_start)
                                    .count()),
                            std::memory_order_relaxed);
                        metric_graph_existing_neighbor_edges_written.fetch_add(
                            static_cast<uint64_t>(indx), std::memory_order_relaxed);
                        if (old_count > static_cast<size_t>(indx)) {
                            metric_graph_existing_neighbor_edges_pruned.fetch_add(
                                old_count - static_cast<size_t>(indx),
                                std::memory_order_relaxed);
                        }
                    }
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]),
                    getDataByInternalId(rez[idx]), dist_func_param_); if (d > d_max) { indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
            metric_graph_link_critical_ns.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - lock_acquired)
                        .count()),
                std::memory_order_relaxed);
        }
        metric_graph_existing_neighbor_update_loop_ns.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - existing_loop_start)
                    .count()),
            std::memory_order_relaxed);
        metric_graph_connect_calls.fetch_add(1, std::memory_order_relaxed);
        metric_graph_connect_ns.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - graph_connect_start)
                    .count()),
            std::memory_order_relaxed);
        metric_graph_link_updates.fetch_add(local_link_updates,
                                            std::memory_order_relaxed);

        return next_closest_entry_point;
    }

    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error(
                "Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<LinkListLock>(new_max_elements).swap(link_list_locks_);
        resizeSearchActiveNodeTable(new_max_elements);
        resizeLevel0CowOverrides(new_max_elements);
        freeInsertVectorReadTrace();
        freeSearchVectorReadTrace();

        // Reallocate base layer
        char* data_level0_memory_new =
            (char*)realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error(
                "Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

#if HNSW_SEARCH_VECTOR_SHADOW
        char* search_vector_shadow_memory_new =
            (char*)realloc(search_vector_shadow_memory_, new_max_elements * data_size_);
        if (search_vector_shadow_memory_new == nullptr)
            throw std::runtime_error(
                "Not enough memory: resizeIndex failed to allocate search vector shadow");
        search_vector_shadow_memory_ = search_vector_shadow_memory_new;
#endif
#if HNSW_INSERT_VECTOR_SHADOW || HNSW_INSERT_VECTOR_PARTIAL_SHADOW || \
	    HNSW_INSERT_VECTOR_HOT_SHADOW
	        allocateInsertVectorShadow(new_max_elements);
	        rebuildInsertVectorShadow(cur_element_count);
#endif
#if HNSW_INSERT_VECTOR_REGION_SHADOW
	        allocateInsertVectorRegionShadow(new_max_elements);
#endif
	        allocateInsertVectorReadTrace(new_max_elements);
	        allocateSearchVectorReadTrace(new_max_elements);

        // Reallocate all other layers
        char** linkLists_new = (char**)realloc(linkLists_, sizeof(void*) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error(
                "Not enough memory: resizeIndex failed to allocate other layers");
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
                element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(std::ostream& output) const {
        if (!output.good())
            throw std::runtime_error("Cannot write index: bad output stream");

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

        if (cur_element_count > 0 && data_level0_memory_) {
#if HNSW_L0_ADJ_COW_PUBLISH
            if (level0_linklist_overrides_) {
                std::vector<char> element(size_data_per_element_);
                for (size_t i = 0; i < cur_element_count; i++) {
                    const char* base =
                        data_level0_memory_ + i * size_data_per_element_;
                    memcpy(element.data(), base, size_data_per_element_);
                    memcpy(element.data() + offsetLevel0_, get_linklist0(i),
                           size_links_level0_);
                    output.write(element.data(), size_data_per_element_);
                }
            } else
#endif
            {
                output.write(data_level0_memory_,
                             cur_element_count * size_data_per_element_);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize =
                element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize && linkLists_[i]) {
                output.write(linkLists_[i], linkListSize);
            }
        }
    }

    void saveIndex(const std::string& location) {
        std::ofstream output(location, std::ios::binary);
        if (!output.is_open())
            throw std::runtime_error("Cannot open file");
        saveIndex(output);
        output.close();
    }

    void loadIndex(std::istream& input, SpaceInterface<dist_t>* s, size_t max_elements_i = 0) {
        if (!input.good())
            throw std::runtime_error("Cannot read index: bad input stream");

        clear();

        input.clear();
        input.seekg(0, std::ios::end);
        std::streampos total_filesize = input.tellg();
        if (total_filesize == std::streampos(-1))
            throw std::runtime_error("Index seems to be corrupted or unsupported");
        input.seekg(0, std::ios::beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
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
        input.seekg(cur_element_count * size_data_per_element_, std::ios::cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() > total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, std::ios::cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, std::ios::beg);

        data_level0_memory_ = (char*)malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        if (cur_element_count > 0) {
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);
        }

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = computeLevel0LinksBytes(maxM0_);
	        validateLevel0Layout("loadIndex");
	        allocateSearchVectorShadow(max_elements);
	        allocateInsertVectorShadow(max_elements);
	        allocateInsertVectorRegionShadow(max_elements);
	        allocateInsertVectorReadTrace(max_elements);
	        allocateSearchVectorReadTrace(max_elements);
        allocateSearchActiveNodeTable(max_elements);
        allocateSearchActiveCacheBuckets();
        allocateLevel0CowOverrides(max_elements);
        rebuildSearchVectorShadow(cur_element_count);
        rebuildInsertVectorShadow(cur_element_count);
        std::vector<LinkListLock>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char**)malloc(sizeof(void*) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char*)malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error(
                        "Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_)
                    deleted_elements.insert(i);
            }
        }
    }

    void loadIndex(const std::string& location, SpaceInterface<dist_t>* s,
                   size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        loadIndex(input, s, max_elements_i);
        input.close();
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

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t*)dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*)data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }

    /*
     * Marks an element with the given label deleted, does NOT really change the current graph.
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
     * Uses the last 16 bits of the memory for the linked list size to store the mark,
     * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost
     * all cases.
     */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }

    /*
     * Removes the deleted mark of the node, does NOT really change the current graph.
     *
     * Note: the method is not safe to use when replacement of deleted elements is enabled,
     *  because elements marked as deleted can be completely removed by addPoint
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
            unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }

    /*
     * Checks the first 16 bits of the memory to see if the element is marked deleted.
     */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }

    unsigned short int getListCount(linklistsizeint* ptr) const {
        return *((unsigned short int*)ptr);
    }

    void setListCount(linklistsizeint* ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr)) = *((unsigned short int*)&size);
    }

    /*
     * Adds point. Updates the point if it is already in the index.
     * If replacement of deleted elements is enabled: replaces previously deleted point if any,
     * updating it with new point
     */
    void addPoint(const void* data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
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
            // we assume that there are no concurrent operations on deleted element
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

    void updatePoint(const void* dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);
        copySearchVectorShadow(internalId, dataPoint);
        copyInsertVectorShadow(internalId, dataPoint);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just
        // return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
                    candidates;
                size_t size = sCand.find(neigh) == sCand.end()
                                  ? sCand.size()
                                  : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getInsertDataForDistance(neigh),
                                                   getInsertDataForDistance(cand),
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
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock<std::shared_mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint* ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint* data = (tableint*)(ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }

    void repairConnectionsForUpdate(const void* dataPoint, tableint entryPointInternalId,
                                    tableint dataPointInternalId, int dataPointLevel,
                                    int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist =
                fstdistfunc_(dataPoint, getInsertDataForDistance(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int* data;
                    std::unique_lock<std::shared_mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint* datal = (tableint*)(data + 1);
#ifdef USE_SSE
                    prefetchInsertDataByInternalId(*datal);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        prefetchInsertDataByInternalId(*(datal + i + 1));
#endif
                        tableint cand = datal[i];
                        dist_t d =
                            fstdistfunc_(dataPoint, getInsertDataForDistance(cand),
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
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
                topCandidates = searchBaseLayer(currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
                filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases
            // where `topCandidates` could just contains entry point itself. To prevent self loops,
            // the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(
                        fstdistfunc_(dataPoint, getInsertDataForDistance(entryPointInternalId),
                                     dist_func_param_),
                        entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId,
                                                    filteredTopCandidates, level, true);
            }
        }
    }

    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock<std::shared_mutex> lock(link_list_locks_[internalId]);
        unsigned int* data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint* ll = (tableint*)(data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }

    tableint addPoint(const void* data_point, labeltype label, int level) {
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
                        throw std::runtime_error("Can't use addPoint to update deleted elements if "
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
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        TelemetryGuard telemetry_guard(this, true);
        PartialGuard partial_guard;
        std::unique_lock<std::shared_mutex> lock_el(link_list_locks_[cur_c]);

        int curlevel = 0;
        if (!isNSW_)
            curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock<std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0,
#if HNSW_INSERT_ZERO_LINKS_ONLY
               size_links_level0_);
#else
               size_data_per_element_);
#endif

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);
        copySearchVectorShadow(cur_c, data_point);
        copyInsertVectorShadow(cur_c, data_point);
#if HNSW_INSERT_VECTOR_CLDEMOTE
        demoteInsertedVectorData(cur_c);
#endif

        if (curlevel) {
            linkLists_[cur_c] = (char*)malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            // Track per-layer entry points for Path Skip
            bool need_byproduct =
                (enable_s3_ || enable_path_skip_ || enable_candidate_injection_ ||
                 enable_warm_start_) && s3_store_;
            std::vector<tableint> layer_entry_points;
            if (need_byproduct) {
                layer_entry_points.resize(maxlevelcopy + 1, currObj);
            }

            if (curlevel < maxlevelcopy) {
                InsertVectorReadAdmissionGuard vector_read_admission(this);
                const auto upper_search_start = std::chrono::steady_clock::now();
                uint64_t upper_dist_comps = 0;
                uint64_t upper_edges_scanned = 0;
                dist_t curdist = insertConstructionDistance(data_point, currObj);
                upper_dist_comps++;
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int* data;
                        std::unique_lock<std::shared_mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);
                        upper_edges_scanned += static_cast<uint64_t>(size);

                        tableint* datal = (tableint*)(data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = insertConstructionDistance(data_point, cand);
                            upper_dist_comps++;
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                    // Record entry point at this layer after greedy descent
                    if (need_byproduct && level > 0 && level <= (int)layer_entry_points.size()) {
                        layer_entry_points[level - 1] = currObj;
                    }
                }
                metric_graph_upper_search_ns.fetch_add(
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - upper_search_start)
                            .count()),
                    std::memory_order_relaxed);
                metric_graph_upper_search_dist_comps.fetch_add(
                    upper_dist_comps, std::memory_order_relaxed);
                metric_graph_upper_search_edges_scanned.fetch_add(
                    upper_edges_scanned, std::memory_order_relaxed);
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                // Record entry point for this layer
                if (need_byproduct && level < (int)layer_entry_points.size()) {
                    layer_entry_points[level] = currObj;
                }

                std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
                    top_candidates = searchBaseLayer(currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(
                        insertConstructionDistance(data_point, enterpoint_copy),
                        enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                // Store byproduct for level 0 (S3 + Path Skip + Candidate Injection)
                if (need_byproduct && level == 0) {
                    std::vector<std::pair<dist_t, tableint>> byproduct_candidates;
                    std::vector<std::pair<dist_t, tableint>> l0_candidates;
                    auto temp_candidates = top_candidates;
                    while (!temp_candidates.empty()) {
                        byproduct_candidates.push_back(temp_candidates.top());
                        l0_candidates.push_back(temp_candidates.top());
                        temp_candidates.pop();
                    }
                    s3_store_->add(cur_c, std::move(byproduct_candidates),
                                   data_point, s3_rdtsc_ns(),
                                   std::move(layer_entry_points),
                                   std::move(l0_candidates));
                }

                currObj =
                    mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
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
#if HNSW_INSERT_VECTOR_CLDEMOTE_AFTER_REPAIR
	        demoteInsertedVectorData(cur_c);
#endif
	        return cur_c;
	    }

    std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
        const void* query_data, size_t k,
        BaseFilterFunctor* isIdAllowed = nullptr) const override {
        return searchKnnWithTelemetry(query_data, k, isIdAllowed, nullptr);
    }

    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnWithTelemetry(
        const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr,
        SearchTelemetry* telemetry = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0)
            return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist =
            fstdistfunc_(query_data, getSearchDataByInternalId(enterpoint_node_), dist_func_param_);
        if (telemetry) {
            telemetry->distance_computations++;
            telemetry->upper_distance_computations++;
        }

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                std::shared_lock<std::shared_mutex> level_lock;
                if (use_node_lock_in_search_) {
                    uint64_t lock_wait_start = telemetry ? s3_rdtsc_ns() : 0;
                    level_lock = std::shared_lock<std::shared_mutex>(link_list_locks_[currObj]);
                    if (telemetry) {
                        telemetry->upper_lock_wait_ns += s3_rdtsc_ns() - lock_wait_start;
                    }
                }

                unsigned int* data = (unsigned int*)get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;
                if (telemetry) {
                    telemetry->upper_hops++;
                    telemetry->upper_edges_scanned += size;
                }

                tableint* datal = (tableint*)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d =
                        fstdistfunc_(query_data, getSearchDataByInternalId(cand), dist_func_param_);
                    if (telemetry) {
                        telemetry->distance_computations++;
                        telemetry->upper_distance_computations++;
                    }

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates =
                searchBaseLayerST<true>(currObj, query_data, std::max(ef_, k),
                                        isIdAllowed, nullptr, telemetry);
        } else {
            top_candidates =
                searchBaseLayerST<false>(currObj, query_data, std::max(ef_, k),
                                         isIdAllowed, nullptr, telemetry);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    // Search that also returns all visited node labels at layer-0 (for V0 validation experiments)
    std::pair<std::priority_queue<std::pair<dist_t, labeltype>>, std::vector<labeltype>>
    searchKnnWithVisited(const void* query_data, size_t k,
                         BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        std::vector<labeltype> visited_labels;

        if (cur_element_count == 0)
            return std::make_pair(result, visited_labels);

        tableint currObj = enterpoint_node_;
        dist_t curdist =
            fstdistfunc_(query_data, getSearchDataByInternalId(enterpoint_node_), dist_func_param_);

        // Greedy descent through upper layers (same as searchKnn)
        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                std::shared_lock<std::shared_mutex> level_lock;
                if (use_node_lock_in_search_) {
                    level_lock = std::shared_lock<std::shared_mutex>(link_list_locks_[currObj]);
                }
                unsigned int* data = (unsigned int*)get_linklist(currObj, level);
                int size = getListCount(data);
                tableint* datal = (tableint*)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d =
                        fstdistfunc_(query_data, getSearchDataByInternalId(cand), dist_func_param_);
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        // Base layer beam search with visited node collection
        VisitedList* vl = acquireVisitedListScratch();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        size_t ef = std::max(ef_, k);

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            candidate_set;

        dist_t dist = fstdistfunc_(query_data, getSearchDataByInternalId(currObj), dist_func_param_);
        dist_t lowerBound = dist;
        top_candidates.emplace(dist, currObj);
        candidate_set.emplace(-dist, currObj);
        visited_array[currObj] = visited_array_tag;
        visited_labels.push_back(getExternalLabel(currObj));

        while (!candidate_set.empty()) {
            auto current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            if (candidate_dist > lowerBound && top_candidates.size() == ef)
                break;
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            std::shared_lock<std::shared_mutex> level_lock;
            if (use_node_lock_in_search_) {
                level_lock = std::shared_lock<std::shared_mutex>(link_list_locks_[current_node_id]);
            }
            int* data = (int*)get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;
                    visited_labels.push_back(getExternalLabel(candidate_id));

                    char* currObj1 = getSearchDataByInternalId(candidate_id);
                    dist_t d = fstdistfunc_(query_data, currObj1, dist_func_param_);

                    if (top_candidates.size() < ef || lowerBound > d) {
                        candidate_set.emplace(-d, candidate_id);
                        top_candidates.emplace(d, candidate_id);

                        if (top_candidates.size() > ef) {
                            top_candidates.pop();
                        }
                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        releaseVisitedListScratch(vl);

        // Build result
        while (top_candidates.size() > k) top_candidates.pop();
        while (top_candidates.size() > 0) {
            auto rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }

        return std::make_pair(result, visited_labels);
    }

    // Also expose layer-0 neighbors for a given label
    std::vector<labeltype> getNeighborsByLabel(labeltype label) const {
        std::vector<labeltype> neighbors;

        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internal_id = search->second;

        int* data = (int*)get_linklist0(internal_id);
        size_t size = getListCount((linklistsizeint*)data);

        for (size_t j = 1; j <= size; j++) {
            tableint neighbor_id = *(data + j);
            neighbors.push_back(getExternalLabel(neighbor_id));
        }

        return neighbors;
    }

    // S3-enabled search that tries to use byproducts from recent inserts
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnS3(
        const void* query_data, size_t k,
        BaseFilterFunctor* isIdAllowed = nullptr) const {

        if (!enable_s3_ || !s3_store_) {
            return searchKnn(query_data, k, isIdAllowed);
        }

        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;

        const void* ep_data = getDataByInternalId(enterpoint_node_);
        const auto* byproduct = s3_store_->findByproductAdaptive(query_data, fstdistfunc_, dist_func_param_, ep_data);

        if (byproduct != nullptr) {
            s3_hits_.fetch_add(1, std::memory_order_relaxed);

            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst> top_candidates;

            size_t dist_saved = byproduct->candidates.size();
            for (const auto& cand : byproduct->candidates) {
                if (!isMarkedDeleted(cand.second) &&
                    (!isIdAllowed || (*isIdAllowed)(getExternalLabel(cand.second)))) {
                    dist_t dist = fstdistfunc_(query_data, getDataByInternalId(cand.second),
                                               dist_func_param_);
                    top_candidates.emplace(dist, cand.second);
                }
            }

            byproduct->consumed.store(true, std::memory_order_release);
            s3_dist_saved_.fetch_add(dist_saved, std::memory_order_relaxed);

            // If not enough candidates, do additional beam search
            if (top_candidates.size() < k) {
                tableint entry_point = top_candidates.empty() ?
                    enterpoint_node_ : top_candidates.top().second;

                bool bare_bone_search = !num_deleted_ && !isIdAllowed;
                std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>,
                                    CompareByFirst> additional;
                if (bare_bone_search) {
                    additional = searchBaseLayerST<true>(entry_point, query_data, std::max(ef_, k), isIdAllowed);
                } else {
                    additional = searchBaseLayerST<false>(entry_point, query_data, std::max(ef_, k), isIdAllowed);
                }
                while (!additional.empty()) {
                    top_candidates.push(additional.top());
                    additional.pop();
                }
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (!top_candidates.empty()) {
                auto rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
        } else {
            s3_misses_.fetch_add(1, std::memory_order_relaxed);
            result = searchKnn(query_data, k, isIdAllowed);
        }

        return result;
    }

    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnWarmStart(
        const void* query_data, size_t k,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        
        if (!enable_warm_start_) {
            // Fall back to regular search if warm start disabled
            return searchKnn(query_data, k, isIdAllowed);
        }
        
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;

        // Determine entry point for beam search
        tableint entry = enterpoint_node_;
        dist_t entry_dist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_),
                                        dist_func_param_);
        bool used_warm_start = false;
        
        // Try to find a better entry point from S3 store
        if (s3_store_) {
            size_t current_head = s3_store_->head.load(std::memory_order_acquire);
            size_t current_tail = s3_store_->tail.load(std::memory_order_acquire);
            
            // Linear scan through ring buffer to find best candidate
            for (size_t i = current_tail; i != current_head; i = (i == 0 ? s3_store_->capacity - 1 : i - 1)) {
                size_t idx = (i == 0 ? s3_store_->capacity - 1 : i - 1);
                const auto& bp = s3_store_->buffer[idx];
                
                if (bp.insert_vector == nullptr) continue;
                if (bp.consumed.load(std::memory_order_acquire)) continue;
                if (bp.candidates.empty()) continue;
                
                // Use the best candidate from this byproduct (closest to insert vector)
                tableint best_candidate = bp.candidates[0].second;
                
                // Skip if marked as deleted or filtered out
                if (isMarkedDeleted(best_candidate)) continue;
                if (isIdAllowed && !(*isIdAllowed)(getExternalLabel(best_candidate))) continue;
                
                // Check if this candidate is closer to query than current entry
                dist_t dist_to_query = fstdistfunc_(query_data, getDataByInternalId(best_candidate),
                                                   dist_func_param_);
                
                if (dist_to_query < entry_dist) {
                    entry = best_candidate;
                    entry_dist = dist_to_query;
                    used_warm_start = true;
                }
            }
        }
        
        if (used_warm_start) {
            warm_start_hits_.fetch_add(1, std::memory_order_relaxed);
            // TODO: Implement hops saved calculation for improvement metric
        }
        
        // Perform upper layer navigation starting from entry point
        tableint currObj = entry;
        dist_t curdist = entry_dist;

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                std::shared_lock<std::shared_mutex> lock;
                if (use_node_lock_in_search_) {
                    lock = std::shared_lock<std::shared_mutex>(
                        link_list_locks_[currObj]);
                }

                unsigned int* data;
                data = get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;
                tableint* datal = (tableint*)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand),
                                           dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        // Perform beam search at layer 0
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates =
                searchBaseLayerST<true>(currObj, query_data, std::max(ef_, k),
                                        isIdAllowed);
        } else {
            top_candidates =
                searchBaseLayerST<false>(currObj, query_data, std::max(ef_, k),
                                         isIdAllowed);
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

    // Search→Search sharing enabled search
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnSearchSharing(
        const void* query_data, size_t k,
        BaseFilterFunctor* isIdAllowed = nullptr) const {

        if (!enable_search_sharing_ || !search_frontier_pool_) {
            return searchKnn(query_data, k, isIdAllowed);
        }

        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
        
        // Search from level maxlevel_ to 1 using regular search
        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                std::shared_lock<std::shared_mutex> level_lock;
                if (use_node_lock_in_search_) {
                    level_lock = std::shared_lock<std::shared_mutex>(link_list_locks_[currObj]);
                }
                unsigned int* data = (unsigned int*)get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;
                tableint* datal = (tableint*)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint candidate = datal[i];
                    if (candidate < 0 || candidate >= max_elements_)
                        throw std::runtime_error("candidate error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(candidate), dist_func_param_);
                    if (d < curdist) {
                        curdist = d;
                        currObj = candidate;
                        changed = true;
                    }
                }
            }
        }
        
        // Use shared search for level 0
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> top_candidates;
        
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerShared<true>(currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerShared<false>(currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (!top_candidates.empty()) {
            auto rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }

        return result;
    }

    // Path Skip: skip upper layers using insert's per-layer entry points
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnPathSkip(
        const void* query_data, size_t k,
        BaseFilterFunctor* isIdAllowed = nullptr) const {

        if (!enable_path_skip_ || !s3_store_) {
            return searchKnn(query_data, k, isIdAllowed);
        }

        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;

        const void* ep_data = getDataByInternalId(enterpoint_node_);
        const auto* byproduct = s3_store_->findByproductAdaptive(query_data, fstdistfunc_, dist_func_param_, ep_data);

        if (byproduct != nullptr && !byproduct->layer_entry_points.empty()) {
            s3_hits_.fetch_add(1, std::memory_order_relaxed);

            // Use the layer-0 entry point from the insert's traversal path
            tableint l0_entry = byproduct->layer_entry_points[0];

            // Do beam search at layer 0 starting from the insert's layer-0 entry point
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst> top_candidates;
            bool bare_bone_search = !num_deleted_ && !isIdAllowed;
            if (bare_bone_search) {
                top_candidates = searchBaseLayerST<true>(l0_entry, query_data, std::max(ef_, k), isIdAllowed);
            } else {
                top_candidates = searchBaseLayerST<false>(l0_entry, query_data, std::max(ef_, k), isIdAllowed);
            }

            while (top_candidates.size() > k) top_candidates.pop();
            while (!top_candidates.empty()) {
                auto rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
        } else {
            s3_misses_.fetch_add(1, std::memory_order_relaxed);
            result = searchKnn(query_data, k, isIdAllowed);
        }
        return result;
    }

    // Candidate Injection: inject insert's level-0 candidates as extra seeds
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnCandidateInjection(
        const void* query_data, size_t k,
        BaseFilterFunctor* isIdAllowed = nullptr) const {

        if (!enable_candidate_injection_ || !s3_store_) {
            return searchKnn(query_data, k, isIdAllowed);
        }

        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;

        // Standard upper-layer traversal
        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                std::shared_lock<std::shared_mutex> level_lock;
                if (use_node_lock_in_search_) {
                    level_lock = std::shared_lock<std::shared_mutex>(link_list_locks_[currObj]);
                }
                unsigned int* data = (unsigned int*)get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;
                tableint* datal = (tableint*)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
                    if (d < curdist) { curdist = d; currObj = cand; changed = true; }
                }
            }
        }

        // Layer-0 beam search with injected candidates
        VisitedList* vl = acquireVisitedListScratch();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst> candidate_set;

        // Seed with normal entry point
        dist_t dist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
        dist_t lowerBound = dist;
        top_candidates.emplace(dist, currObj);
        candidate_set.emplace(-dist, currObj);
        visited_array[currObj] = visited_array_tag;

        // Inject byproduct candidates as additional seeds
        const void* ep_data_ci = getDataByInternalId(enterpoint_node_);
        const auto* byproduct = s3_store_->findByproductAdaptive(query_data, fstdistfunc_, dist_func_param_, ep_data_ci);
        if (byproduct != nullptr && !byproduct->level0_candidates.empty()) {
            for (const auto& cand : byproduct->level0_candidates) {
                tableint cand_id = cand.second;
                if (cand_id >= cur_element_count) continue;
                if (visited_array[cand_id] == visited_array_tag) continue;
                visited_array[cand_id] = visited_array_tag;
                dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand_id), dist_func_param_);
                candidate_set.emplace(-d, cand_id);
                if (!isMarkedDeleted(cand_id) && (!isIdAllowed || (*isIdAllowed)(getExternalLabel(cand_id)))) {
                    top_candidates.emplace(d, cand_id);
                    if (d < lowerBound) lowerBound = d;
                }
            }
        }

        size_t ef = std::max(ef_, k);
        // Standard beam search loop
        while (!candidate_set.empty()) {
            auto current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;
            if (candidate_dist > lowerBound && top_candidates.size() == ef) break;
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            std::shared_lock<std::shared_mutex> level_lock;
            if (use_node_lock_in_search_) {
                level_lock = std::shared_lock<std::shared_mutex>(link_list_locks_[current_node_id]);
            }
            int* data = (int*)get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
            metric_hops++;
            metric_distance_computations += size;

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char* currObj1 = getDataByInternalId(candidate_id);
                dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
                if (top_candidates.size() < ef || lowerBound > dist1) {
                    candidate_set.emplace(-dist1, candidate_id);
                    if (!isMarkedDeleted(candidate_id) &&
                        (!isIdAllowed || (*isIdAllowed)(getExternalLabel(candidate_id)))) {
                        top_candidates.emplace(dist1, candidate_id);
                    }
                    while (top_candidates.size() > ef) top_candidates.pop();
                    if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
                }
            }
        }
        releaseVisitedListScratch(vl);

        while (top_candidates.size() > k) top_candidates.pop();
        while (!top_candidates.empty()) {
            auto rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    // Combined: Path Skip + Candidate Injection
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnCombined(
        const void* query_data, size_t k,
        BaseFilterFunctor* isIdAllowed = nullptr) const {

        if (!s3_store_) {
            return searchKnn(query_data, k, isIdAllowed);
        }

        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;

        const void* ep_data_combined = getDataByInternalId(enterpoint_node_);
        const auto* byproduct = s3_store_->findByproductAdaptive(query_data, fstdistfunc_, dist_func_param_, ep_data_combined);

        if (byproduct != nullptr && !byproduct->layer_entry_points.empty()) {
            s3_hits_.fetch_add(1, std::memory_order_relaxed);

            // Path Skip: use layer-0 entry point
            tableint l0_entry = byproduct->layer_entry_points[0];

            // Beam search at layer 0 with injected candidates
            VisitedList* vl = acquireVisitedListScratch();
            vl_type* visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst> candidate_set;

            dist_t dist = fstdistfunc_(query_data, getDataByInternalId(l0_entry), dist_func_param_);
            dist_t lowerBound = dist;
            top_candidates.emplace(dist, l0_entry);
            candidate_set.emplace(-dist, l0_entry);
            visited_array[l0_entry] = visited_array_tag;

            // Candidate Injection
            for (const auto& cand : byproduct->level0_candidates) {
                tableint cand_id = cand.second;
                if (cand_id >= cur_element_count) continue;
                if (visited_array[cand_id] == visited_array_tag) continue;
                visited_array[cand_id] = visited_array_tag;
                dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand_id), dist_func_param_);
                candidate_set.emplace(-d, cand_id);
                if (!isMarkedDeleted(cand_id) && (!isIdAllowed || (*isIdAllowed)(getExternalLabel(cand_id)))) {
                    top_candidates.emplace(d, cand_id);
                    if (d < lowerBound) lowerBound = d;
                }
            }

            size_t ef = std::max(ef_, k);
            while (!candidate_set.empty()) {
                auto current_node_pair = candidate_set.top();
                dist_t candidate_dist = -current_node_pair.first;
                if (candidate_dist > lowerBound && top_candidates.size() == ef) break;
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                std::shared_lock<std::shared_mutex> level_lock;
                if (use_node_lock_in_search_) {
                    level_lock = std::shared_lock<std::shared_mutex>(link_list_locks_[current_node_id]);
                }
                int* data = (int*)get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
                metric_hops++;
                metric_distance_computations += size;

                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char* currObj1 = getDataByInternalId(candidate_id);
                    dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef || lowerBound > dist1) {
                        candidate_set.emplace(-dist1, candidate_id);
                        if (!isMarkedDeleted(candidate_id) &&
                            (!isIdAllowed || (*isIdAllowed)(getExternalLabel(candidate_id)))) {
                            top_candidates.emplace(dist1, candidate_id);
                        }
                        while (top_candidates.size() > ef) top_candidates.pop();
                        if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
                    }
                }
            }
            releaseVisitedListScratch(vl);

            while (top_candidates.size() > k) top_candidates.pop();
            while (!top_candidates.empty()) {
                auto rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
        } else {
            s3_misses_.fetch_add(1, std::memory_order_relaxed);
            result = searchKnn(query_data, k, isIdAllowed);
        }
        return result;
    }

    std::vector<std::pair<dist_t, labeltype>> searchStopConditionClosest(
        const void* query_data, BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0)
            return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist =
            fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                std::shared_lock<std::shared_mutex> level_lock;
                if (use_node_lock_in_search_) {
                    level_lock = std::shared_lock<std::shared_mutex>(link_list_locks_[currObj]);
                }
                unsigned int* data = (unsigned int*)get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;

                tableint* datal = (tableint*)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d =
                        fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            top_candidates;
        top_candidates =
            searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

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
                linklistsizeint* ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint* data = (tableint*)(ll_cur + 1);
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
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i = 0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }
};
}  // namespace hnswlib
