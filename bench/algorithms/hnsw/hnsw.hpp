#pragma once

#include <omp.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <sched.h>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>
#include <limits>

#include "../index.hpp"
#include "hnswlib/hnswlib/hnswlib.h"

enum class HNSWWorkScheduler : size_t {
    kBatchParallel = 0,
    kThreadPool = 1,
};

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class HNSW : public IndexBase<T, TagT, LabelT> {
   public:
    HNSW(size_t max_elements, size_t dim, size_t num_threads, size_t M,
         size_t ef_construction, bool use_node_lock_in_search = true,
         size_t work_scheduler = 0)
        : dim_(dim),
          num_threads_(num_threads),
          max_elements_(max_elements),
          space_(dim),
          arena_(num_threads),
          measured_search_arena_enabled_(read_env_bool("HNSW_MEASURED_SEARCH_ARENA", false)),
          measured_search_arena_(
              read_env_size("HNSW_MEASURED_SEARCH_ARENA_THREADS", num_threads)),
          M_(M),
          ef_c_(ef_construction),
          scheduler_(work_scheduler == static_cast<size_t>(HNSWWorkScheduler::kThreadPool)
                         ? HNSWWorkScheduler::kThreadPool
                         : HNSWWorkScheduler::kBatchParallel) {
        index_ = new hnswlib::HierarchicalNSW<T>(&space_, max_elements, M,
                                                 ef_construction, 0, 100, false,
                                                 use_node_lock_in_search);
    }

    ~HNSW() override {
        delete index_;
        index_ = nullptr;
    }

    void build(const T* data, const TagT* tags, size_t num_points) override {
        const bool reset_insert_trace_after_build =
            read_env_bool("HNSW_INSERT_VECTOR_READ_TRACE_RESET_AFTER_BUILD", false);
        if (reset_insert_trace_after_build) {
            index_->setInsertVectorReadTraceEnabled(false);
        }
	#pragma omp parallel for num_threads(num_threads_)
	        for (size_t i = 0; i < num_points; i++) {
	            index_->addPoint((void*)(data + i * dim_), tags[i]);
	        }
	        index_->setInsertVectorRegionShadowMaxId(num_points);
	        index_->buildInsertVectorGraphHotReplica();
#if HNSW_HOT_VEC_LEVEL_RENUMBER
	        // Run the renumber pass once, in this single-threaded post-build
	        // window, before any concurrent search/insert begins.
	        index_->doHotVecLevelRenumber();
#endif
        if (reset_insert_trace_after_build) {
            index_->resetInsertVectorReadTrace();
            index_->setInsertVectorReadTraceEnabled(true);
        }
        if (read_env_bool("HNSW_SEARCH_VECTOR_READ_TRACE_RESET_AFTER_BUILD", false)) {
            index_->resetSearchVectorReadTrace();
        }
    }

    int insert(const T* data, const TagT tag) override {
        index_->enableInsertVectorReadAdmission(true);
        index_->addPoint(data, tag);
        index_->enableInsertVectorReadAdmission(false);
        return 0;
    }

    int batch_insert(const T* batch_data, const TagT* batch_tags,
                     size_t num_points) override {
        if (scheduler_ == HNSWWorkScheduler::kThreadPool) {
            return batch_insert_thread_pool(batch_data, batch_tags, num_points);
        }
        int success_count = 0;
        index_->enableInsertVectorReadAdmission(true);
        arena_.execute([&] {
            tbb::parallel_for(size_t(0), num_points, [&](size_t i) {
                pin_current_thread_to_env_cpu("HNSW_INSERT_CPU_LIST",
                                              "ANNCHOR_INSERT_CPU_LIST", i);
                index_->addPoint(batch_data + i * dim_, batch_tags[i]);
                __sync_fetch_and_add(&success_count, 1);
            });
        });
        index_->enableInsertVectorReadAdmission(false);
        return success_count == num_points ? 0 : -1;
    }

    int batch_partial_insert(const T* batch_data,
                                  const TagT* batch_tags,
                                  size_t num_points,
                                  const size_t* partial_limits,
                                  size_t* update_counts) {
        if (num_points == 0) {
            return 0;
        }

        bool enableTelemetry = update_counts != nullptr;
        index_->enableInsertTelemetry(enableTelemetry);
        index_->enableInsertVectorReadAdmission(true);

        arena_.execute([&] {
            tbb::parallel_for(size_t(0), num_points, [&](size_t i) {
                pin_current_thread_to_env_cpu("HNSW_INSERT_CPU_LIST",
                                              "ANNCHOR_INSERT_CPU_LIST", i);
                if (enableTelemetry) {
                    index_->setTelemetryOutputTarget(update_counts + i);
                } else {
                    index_->setTelemetryOutputTarget(nullptr);
                }

                bool hasLimit = partial_limits != nullptr;
                if (hasLimit) {
                    index_->configurePartialInsert(partial_limits[i]);
                }

                index_->addPoint(batch_data + i * dim_, batch_tags[i]);

                if (hasLimit) {
                    index_->disablePartialInsert();
                }
            });
        });

        if (enableTelemetry) {
            index_->enableInsertTelemetry(false);
        }
        index_->enableInsertVectorReadAdmission(false);

        return 0;
    }

    void set_query_params(const QParams& params) override {
        ef_s_ = params.ef_search;
        index_->setEf(ef_s_);
    }

    int search(const T* query, size_t k,
               std::vector<TagT>& result_tags) override {
        auto result = index_->searchKnn(query, k);
        while (!result.empty()) {
            result_tags.push_back(result.top().second);
            result.pop();
        }
        return 0;
    }

    int batch_search(const T* batch_queries, size_t k, size_t num_queries,
                     TagT** batch_results, size_t* watermark_out = nullptr,
                     size_t visible_ts = std::numeric_limits<size_t>::max()) override {
        if (scheduler_ == HNSWWorkScheduler::kThreadPool) {
            return batch_search_thread_pool(batch_queries, k, num_queries, batch_results);
        }
        bool suspended = index_->isSharingSuspended();
        bool use_s3 = !suspended && index_->isS3Enabled();
        bool use_search_sharing = !suspended && index_->isSearchSharingEnabled();
        bool use_warm_start = !suspended && index_->isWarmStartEnabled();
        bool use_ps = !suspended && index_->isPathSkipEnabled();
        bool use_ci = !suspended && index_->isCandidateInjectionEnabled();
        arena_.execute([&] {
            tbb::parallel_for(size_t(0), num_queries, [&](size_t i) {
                pin_current_thread_to_env_cpu("HNSW_SEARCH_CPU_LIST",
                                              "ANNCHOR_SEARCH_CPU_LIST", i);
                const T* q = batch_queries + i * dim_;
                auto result = [&]() {
                    if (use_ps && use_ci) return index_->searchKnnCombined(q, k);
                    else if (use_ps) return index_->searchKnnPathSkip(q, k);
                    else if (use_ci) return index_->searchKnnCandidateInjection(q, k);
                    else if (use_search_sharing) return index_->searchKnnSearchSharing(q, k);
                    else if (use_warm_start) return index_->searchKnnWarmStart(q, k);
                    else if (use_s3) return index_->searchKnnS3(q, k);
                    else return index_->searchKnn(q, k);
                }();
                size_t j = 0;
                std::vector<TagT> results;
                while (!result.empty()) {
                    results.push_back(result.top().second);
                    result.pop();
                }
                std::reverse(results.begin(), results.end());
                for (j = 0; j < results.size(); ++j) {
                    batch_results[i][j] = results[j];
                }
            });
        });
        return 0;
    }

    int batch_search_measured_work(const T* batch_queries, size_t k,
                                   size_t num_queries, TagT** batch_results,
                                   SearchWorkStats* per_query_stats,
                                   size_t* watermark_out = nullptr,
                                   size_t visible_ts = std::numeric_limits<size_t>::max()) override {
        (void)visible_ts;
        if (watermark_out) {
            *watermark_out = visible_ts;
        }
        if (scheduler_ == HNSWWorkScheduler::kThreadPool) {
            return batch_search(batch_queries, k, num_queries, batch_results,
                                watermark_out, visible_ts);
        }

        bool suspended = index_->isSharingSuspended();
        bool use_s3 = !suspended && index_->isS3Enabled();
        bool use_search_sharing = !suspended && index_->isSearchSharingEnabled();
        bool use_warm_start = !suspended && index_->isWarmStartEnabled();
        bool use_ps = !suspended && index_->isPathSkipEnabled();
        bool use_ci = !suspended && index_->isCandidateInjectionEnabled();
        if (use_s3 || use_search_sharing || use_warm_start || use_ps || use_ci) {
            return batch_search(batch_queries, k, num_queries, batch_results,
                                watermark_out, visible_ts);
        }

        measured_search_task_arena().execute([&] {
            tbb::parallel_for(size_t(0), num_queries, [&](size_t i) {
                pin_current_thread_to_env_cpu("HNSW_MEASURED_SEARCH_CPU_LIST",
                                              "ANNCHOR_MEASURED_SEARCH_CPU_LIST", i);
                const T* q = batch_queries + i * dim_;
                typename hnswlib::HierarchicalNSW<T>::SearchTelemetry telemetry{};

                const auto search_start = std::chrono::steady_clock::now();
                struct timespec cpu_start {};
                if (per_query_stats) {
                    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_start);
                }
                const int start_cpu = per_query_stats ? sched_getcpu() : -1;
                auto result = index_->searchKnnWithTelemetry(q, k, nullptr,
                                                             per_query_stats ? &telemetry : nullptr);
                const auto search_end = std::chrono::steady_clock::now();
                const int end_cpu = per_query_stats ? sched_getcpu() : -1;

                uint64_t thread_cpu_ns = 0;
                if (per_query_stats) {
                    struct timespec cpu_end {};
                    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_end);
                    const int64_t cpu_ns =
                        static_cast<int64_t>(cpu_end.tv_sec - cpu_start.tv_sec) *
                            1000000000LL +
                        static_cast<int64_t>(cpu_end.tv_nsec - cpu_start.tv_nsec);
                    thread_cpu_ns = cpu_ns > 0 ? static_cast<uint64_t>(cpu_ns) : 0;
                }
                const uint64_t search_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        search_end - search_start)
                        .count());

                const auto copy_start = std::chrono::steady_clock::now();
                std::vector<TagT> results;
                while (!result.empty()) {
                    results.push_back(result.top().second);
                    result.pop();
                }
                std::reverse(results.begin(), results.end());
                size_t limit = std::min(results.size(), static_cast<size_t>(k));
                for (size_t j = 0; j < limit; ++j) {
                    batch_results[i][j] = results[j];
                }
                const auto copy_end = std::chrono::steady_clock::now();
                const uint64_t copy_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        copy_end - copy_start)
                        .count());

                if (per_query_stats) {
                    per_query_stats[i].searchknn_ns = search_ns;
                    per_query_stats[i].searchknn_thread_cpu_ns = thread_cpu_ns;
                    per_query_stats[i].work_start_cpu = start_cpu;
                    per_query_stats[i].work_end_cpu = end_cpu;
                    per_query_stats[i].result_copy_ns = copy_ns;
                    per_query_stats[i].upper_lock_wait_ns = telemetry.upper_lock_wait_ns;
                    per_query_stats[i].level0_lock_wait_ns = telemetry.level0_lock_wait_ns;
                    per_query_stats[i].distance_computations = telemetry.distance_computations;
                    per_query_stats[i].upper_distance_computations =
                        telemetry.upper_distance_computations;
                    per_query_stats[i].level0_distance_computations =
                        telemetry.level0_distance_computations;
                    per_query_stats[i].upper_hops = telemetry.upper_hops;
                    per_query_stats[i].upper_edges_scanned = telemetry.upper_edges_scanned;
                    per_query_stats[i].level0_expansions = telemetry.level0_expansions;
                    per_query_stats[i].level0_edges_scanned = telemetry.level0_edges_scanned;
                    per_query_stats[i].candidate_pops = telemetry.candidate_pops;
                    per_query_stats[i].candidate_pushes = telemetry.candidate_pushes;
                    per_query_stats[i].visited_nodes = telemetry.visited_nodes;
                    per_query_stats[i].result_pushes = telemetry.result_pushes;
                }
            });
        });
        return 0;
    }

    void dump_stats(std::string& str) {
        std::stringstream ss;
        ss << "index_memory_mb:" << index_->getIndexMemory() / (1024 * 1024)
           << ", dist_computations:"
           << index_->metric_distance_computations.load()
           << ", hops:" << index_->metric_hops.load();
        const auto level0_begin =
            reinterpret_cast<uintptr_t>(index_->data_level0_memory_);
        const auto level0_end =
            level0_begin + index_->max_elements_ * index_->size_data_per_element_;
        const auto linklists_ptr_begin =
            reinterpret_cast<uintptr_t>(index_->linkLists_);
        const auto linklists_ptr_end =
            linklists_ptr_begin + index_->max_elements_ * sizeof(char*);
        const auto locks_begin =
            index_->link_list_locks_.empty()
                ? uintptr_t{0}
                : reinterpret_cast<uintptr_t>(index_->link_list_locks_.data());
        const auto locks_end =
            locks_begin + index_->link_list_locks_.size() * sizeof(std::shared_mutex);
        const auto element_levels_begin =
            index_->element_levels_.empty()
                ? uintptr_t{0}
                : reinterpret_cast<uintptr_t>(index_->element_levels_.data());
        const auto element_levels_end =
            element_levels_begin + index_->element_levels_.size() * sizeof(int);
        const auto hnsw_object_begin = reinterpret_cast<uintptr_t>(index_);
        const auto hnsw_object_end =
            hnsw_object_begin + sizeof(*index_);
        ss << std::hex
           << ", hnsw_level0_begin:0x" << level0_begin
           << ", hnsw_level0_end:0x" << level0_end
           << ", hnsw_linklists_ptr_begin:0x" << linklists_ptr_begin
           << ", hnsw_linklists_ptr_end:0x" << linklists_ptr_end
           << ", hnsw_locks_begin:0x" << locks_begin
           << ", hnsw_locks_end:0x" << locks_end
           << ", hnsw_element_levels_begin:0x" << element_levels_begin
           << ", hnsw_element_levels_end:0x" << element_levels_end
           << ", hnsw_object_begin:0x" << hnsw_object_begin
           << ", hnsw_object_end:0x" << hnsw_object_end
           << ", hnsw_metric_insert_search_wait_addr:0x"
           << reinterpret_cast<uintptr_t>(&index_->metric_graph_search_unique_lock_wait_ns)
           << ", hnsw_metric_insert_search_acqs_addr:0x"
           << reinterpret_cast<uintptr_t>(&index_->metric_graph_search_unique_lock_acqs)
           << ", hnsw_metric_insert_base_search_addr:0x"
           << reinterpret_cast<uintptr_t>(&index_->metric_graph_base_search_ns)
           << std::dec
           << ", hnsw_size_data_per_element:" << index_->size_data_per_element_
           << ", hnsw_offset_level0:" << index_->offsetLevel0_
           << ", hnsw_maxM0:" << index_->maxM0_
           << ", hnsw_offset_data:" << index_->offsetData_
           << ", hnsw_offset_data_mod64:" << index_->offsetData_ % 64
           << ", hnsw_label_offset:" << index_->label_offset_
           << ", hnsw_size_links_level0:" << index_->size_links_level0_
           << ", hnsw_level0_layout_overhead_bytes:"
           << index_->level0LayoutOverheadBytes()
           << ", hnsw_align_data_to_cacheline:" << HNSW_ALIGN_DATA_TO_CACHELINE
           << ", hnsw_search_vector_shadow:" << HNSW_SEARCH_VECTOR_SHADOW
           << ", hnsw_search_vector_shadow_bytes:" << index_->searchVectorShadowBytes()
           << ", hnsw_insert_vector_shadow:" << HNSW_INSERT_VECTOR_SHADOW
           << ", hnsw_insert_vector_partial_shadow:" << HNSW_INSERT_VECTOR_PARTIAL_SHADOW
           << ", hnsw_insert_vector_partial_shadow_num:"
           << HNSW_INSERT_VECTOR_PARTIAL_SHADOW_NUM
           << ", hnsw_insert_vector_partial_shadow_den:"
           << HNSW_INSERT_VECTOR_PARTIAL_SHADOW_DEN
           << ", hnsw_insert_vector_partial_shadow_mode:"
           << HNSW_INSERT_VECTOR_PARTIAL_SHADOW_MODE
           << ", hnsw_insert_vector_shadow_offset_bytes:"
           << HNSW_INSERT_VECTOR_SHADOW_OFFSET_BYTES
           << ", hnsw_insert_vector_cold_prefetch:" << HNSW_INSERT_VECTOR_COLD_PREFETCH
           << ", hnsw_insert_vector_hot_shadow:" << HNSW_INSERT_VECTOR_HOT_SHADOW
           << ", hnsw_insert_vector_hot_online:" << HNSW_INSERT_VECTOR_HOT_ONLINE
           << ", hnsw_insert_vector_hot_online_budget:"
           << HNSW_INSERT_VECTOR_HOT_ONLINE_BUDGET
           << ", hnsw_insert_vector_hot_online_threshold:"
           << HNSW_INSERT_VECTOR_HOT_ONLINE_THRESHOLD
           << ", hnsw_insert_vector_hot_online_sample_mask:"
           << HNSW_INSERT_VECTOR_HOT_ONLINE_SAMPLE_MASK
           << ", hnsw_insert_vector_graph_hot:" << HNSW_INSERT_VECTOR_GRAPH_HOT
           << ", hnsw_insert_vector_graph_hot_budget:"
           << HNSW_INSERT_VECTOR_GRAPH_HOT_BUDGET
	           << ", hnsw_insert_vector_graph_hot_mode:"
	           << HNSW_INSERT_VECTOR_GRAPH_HOT_MODE
	           << ", hnsw_insert_vector_region_shadow:"
	           << HNSW_INSERT_VECTOR_REGION_SHADOW
	           << ", hnsw_insert_vector_region_shadow_bytes_budget:"
	           << HNSW_INSERT_VECTOR_REGION_SHADOW_BYTES
	           << ", hnsw_insert_vector_region_shift:"
	           << HNSW_INSERT_VECTOR_REGION_SHIFT
	           << ", hnsw_insert_vector_region_threshold:"
	           << HNSW_INSERT_VECTOR_REGION_THRESHOLD
	           << ", hnsw_insert_vector_region_sample_mask:"
	           << HNSW_INSERT_VECTOR_REGION_SAMPLE_MASK
	           << ", hnsw_insert_vector_region_stats:"
	           << HNSW_INSERT_VECTOR_REGION_STATS
	           << ", hnsw_insert_vector_region_stats_sample_mask:"
	           << HNSW_INSERT_VECTOR_REGION_STATS_SAMPLE_MASK
	           << ", hnsw_insert_vector_read_trace:" << HNSW_INSERT_VECTOR_READ_TRACE
	           << ", hnsw_insert_vector_read_trace_reads:"
	           << index_->insertVectorReadTraceReads()
	           << ", hnsw_insert_vector_shadow_slots:" << index_->insertVectorShadowSlots()
	           << ", hnsw_insert_vector_shadow_bytes:" << index_->insertVectorShadowBytes()
	           << ", hnsw_insert_vector_region_shadow_slots:"
	           << index_->insertVectorRegionShadowSlots()
	           << ", hnsw_insert_vector_region_shadow_bytes:"
	           << index_->insertVectorRegionShadowBytes()
	           << ", hnsw_insert_vector_region_shadow_region_bytes:"
	           << index_->insertVectorRegionShadowRegionBytesStat()
	           << ", hnsw_insert_vector_region_count:"
	           << index_->insertVectorRegionCount()
	           << ", hnsw_insert_vector_region_shadow_max_id:"
	           << index_->insertVectorRegionShadowMaxId()
	           << ", hnsw_insert_vector_region_shadow_samples:"
	           << index_->insertVectorRegionShadowSamples()
	           << ", hnsw_insert_vector_region_shadow_promotions:"
	           << index_->insertVectorRegionShadowPromotions()
	           << ", hnsw_insert_vector_region_shadow_hits:"
	           << index_->insertVectorRegionShadowHits()
	           << ", hnsw_insert_vector_region_shadow_fallbacks:"
	           << index_->insertVectorRegionShadowFallbacks()
	           << ", hnsw_insert_vector_region_shadow_budget_full:"
	           << index_->insertVectorRegionShadowBudgetFull()
	           << ", hnsw_insert_vector_region_shadow_copy_ms:"
	           << index_->insertVectorRegionShadowCopyMs()
	           << ", hnsw_insert_vector_region_shadow_cross_region:"
	           << index_->insertVectorRegionShadowCrossRegion()
	           << ", hnsw_insert_vector_thread_cache:"
	           << HNSW_INSERT_VECTOR_THREAD_CACHE
	           << ", hnsw_insert_vector_thread_cache_slots:"
	           << HNSW_INSERT_VECTOR_THREAD_CACHE_SLOTS
	           << ", hnsw_insert_vector_thread_filter_slots:"
	           << HNSW_INSERT_VECTOR_THREAD_FILTER_SLOTS
	           << ", hnsw_insert_vector_thread_cache_threshold:"
	           << HNSW_INSERT_VECTOR_THREAD_CACHE_THRESHOLD
	           << ", hnsw_insert_vector_thread_cache_stats:"
	           << HNSW_INSERT_VECTOR_THREAD_CACHE_STATS
	           << ", hnsw_insert_vector_thread_cache_stats_sample_mask:"
	           << HNSW_INSERT_VECTOR_THREAD_CACHE_STATS_SAMPLE_MASK
	           << ", hnsw_insert_vector_thread_cache_stream_fill:"
	           << HNSW_INSERT_VECTOR_THREAD_CACHE_STREAM_FILL
	           << ", hnsw_insert_vector_thread_cache_hits:"
	           << index_->insertVectorThreadCacheHits()
	           << ", hnsw_insert_vector_thread_cache_fallbacks:"
	           << index_->insertVectorThreadCacheFallbacks()
	           << ", hnsw_insert_vector_thread_cache_filter_hits:"
	           << index_->insertVectorThreadCacheFilterHits()
	           << ", hnsw_insert_vector_thread_cache_admissions:"
	           << index_->insertVectorThreadCacheAdmissions()
	           << ", hnsw_insert_vector_thread_cache_copies:"
	           << index_->insertVectorThreadCacheCopies()
	           << ", hnsw_insert_vector_hot_online_samples:"
	           << index_->insertVectorHotOnlineSamples()
           << ", hnsw_insert_vector_hot_online_promotions:"
           << index_->insertVectorHotOnlinePromotions()
           << ", hnsw_insert_vector_hot_online_budget_full:"
           << index_->insertVectorHotOnlineBudgetFull()
           << ", hnsw_insert_vector_graph_hot_select_ms:"
           << index_->insertVectorGraphHotSelectMs()
           << ", hnsw_insert_vector_graph_hot_copy_ms:"
           << index_->insertVectorGraphHotCopyMs()
           << ", hnsw_insert_vector_graph_hot_scanned_edges:"
           << index_->insertVectorGraphHotScannedEdges()
           << ", hnsw_insert_vector_graph_hot_selected:"
           << index_->insertVectorGraphHotSelected()
           << ", hnsw_insert_vector_graph_hot_candidates:"
           << index_->insertVectorGraphHotCandidates()
           << ", hnsw_l0_adj_cow_publish:" << HNSW_L0_ADJ_COW_PUBLISH
           << ", hnsw_l0_adj_cow_nodes:" << index_->level0AdjCowNodes()
           << ", hnsw_l0_adj_cow_publishes:" << index_->level0AdjCowPublishes()
           << ", hnsw_l0_adj_cow_append_publishes:"
           << index_->level0AdjCowAppendPublishes()
           << ", hnsw_l0_adj_cow_prune_publishes:"
           << index_->level0AdjCowPrunePublishes()
           << ", hnsw_l0_adj_cow_bytes:" << index_->level0AdjCowBytes()
           << ", hnsw_l0_adj_cow_partial_fallbacks:"
           << index_->level0AdjCowPartialFallbacks()
           << ", hnsw_l0_repair_nt_store:" << HNSW_L0_REPAIR_NT_STORE
           << ", hnsw_l0_repair_nt_edges:" << index_->l0RepairNtEdges()
           << ", hnsw_l0_repair_nt_rewrites:" << index_->l0RepairNtRewrites()
           << ", hnsw_l0_repair_nt_appends:" << index_->l0RepairNtAppends()
           << ", hnsw_l0_repair_nt_partial_fallbacks:"
           << index_->l0RepairNtPartialFallbacks()
           << ", hnsw_insert_search_prefetch_ahead:"
           << HNSW_INSERT_SEARCH_PREFETCH_AHEAD
           << ", hnsw_insert_search_prefetch_hint:"
           << HNSW_INSERT_SEARCH_PREFETCH_HINT
           << ", hnsw_insert_search_addr_batch:"
           << HNSW_INSERT_SEARCH_ADDR_BATCH
           << ", hnsw_insert_search_addr_batch_candidates:"
           << index_->metric_graph_base_search_addr_batch_candidates.load()
           << ", hnsw_insert_vector_read_max_active:"
           << HNSW_INSERT_VECTOR_READ_MAX_ACTIVE
           << ", hnsw_insert_vector_read_wait_ns:"
           << HNSW_INSERT_VECTOR_READ_WAIT_NS
           << ", hnsw_insert_vector_read_admissions:"
           << index_->insertVectorReadAdmissions()
           << ", hnsw_insert_vector_read_waits:" << index_->insertVectorReadWaits()
           << ", hnsw_insert_vector_read_wait_ms:"
           << static_cast<double>(index_->insertVectorReadWaitNs()) / 1e6
           << ", hnsw_active_insert_vector_readers:"
           << index_->activeInsertVectorReaders()
           << ", hnsw_search_active_node_tracking:"
           << HNSW_SEARCH_ACTIVE_NODE_TRACKING
           << ", hnsw_search_aware_repair_reorder:"
           << HNSW_SEARCH_AWARE_REPAIR_REORDER
           << ", hnsw_search_active_node_table_bytes:"
           << index_->searchActiveNodeTableBytes()
           << ", hnsw_search_active_node_marks:"
           << index_->searchActiveNodeMarks()
           << ", hnsw_insert_search_hot_reorder_lists:"
           << index_->insertSearchHotReorderLists()
           << ", hnsw_insert_search_hot_reorder_nodes:"
           << index_->insertSearchHotReorderNodes()
           << ", hnsw_search_active_cache_bucket_tracking:"
           << HNSW_SEARCH_ACTIVE_CACHE_BUCKET_TRACKING
           << ", hnsw_search_active_cache_bucket_store_only:"
           << HNSW_SEARCH_ACTIVE_CACHE_BUCKET_STORE_ONLY
           << ", hnsw_cache_aware_repair_reorder:"
           << HNSW_CACHE_AWARE_REPAIR_REORDER
           << ", hnsw_cache_aware_insert_search_order:"
           << HNSW_CACHE_AWARE_INSERT_SEARCH_ORDER
           << ", hnsw_search_active_cache_buckets:"
           << HNSW_SEARCH_ACTIVE_CACHE_BUCKETS
           << ", hnsw_search_active_cache_guard_buckets:"
           << HNSW_SEARCH_ACTIVE_CACHE_GUARD_MAX_BUCKETS
           << ", hnsw_search_active_cache_bucket_bytes:"
           << index_->searchActiveCacheBucketBytes()
           << ", hnsw_search_active_cache_bucket_marks:"
           << index_->searchActiveCacheBucketMarks()
           << ", hnsw_insert_cache_hot_reorder_lists:"
           << index_->insertCacheHotReorderLists()
           << ", hnsw_insert_cache_hot_reorder_nodes:"
           << index_->insertCacheHotReorderNodes()
           << ", hnsw_insert_search_cache_defer_lists:"
           << index_->insertSearchCacheDeferLists()
           << ", hnsw_insert_search_cache_defer_nodes:"
           << index_->insertSearchCacheDeferNodes()
           << ", hnsw_current_count:" << index_->getCurrentElementCount();
        if (index_->isS3Enabled() || index_->isPathSkipEnabled() ||
            index_->isCandidateInjectionEnabled() || index_->isSearchSharingEnabled() ||
            index_->isWarmStartEnabled()) {
            ss << ", s3_hits:" << index_->getS3Hits()
               << ", s3_misses:" << index_->getS3Misses()
               << ", s3_dist_saved:" << index_->getS3DistSaved()
               << ", s3_adaptive_accepts:" << index_->getS3AdaptiveAccepts()
               << ", s3_adaptive_rejects:" << index_->getS3AdaptiveRejects()
               << ", search_sharing_hits:" << index_->getSearchSharingHits()
               << ", search_sharing_misses:" << index_->getSearchSharingMisses()
               << ", warm_start_hits:" << index_->getWarmStartHits()
               << ", warm_start_improvement:" << index_->getWarmStartImprovement();
        }
        const char* trace_path = std::getenv("HNSW_INSERT_VECTOR_READ_TRACE_PATH");
        if (trace_path && trace_path[0] != '\0' &&
            !insert_vector_trace_dumped_.exchange(true, std::memory_order_acq_rel)) {
            size_t top_k = 2000000;
            if (const char* raw_top_k = std::getenv("HNSW_INSERT_VECTOR_READ_TRACE_TOPK")) {
                if (raw_top_k[0] != '\0') {
                    top_k = static_cast<size_t>(std::strtoull(raw_top_k, nullptr, 10));
                }
            }
            index_->dumpInsertVectorReadTrace(trace_path, top_k);
        }
        const char* search_trace_path = std::getenv("HNSW_SEARCH_VECTOR_READ_TRACE_PATH");
        if (search_trace_path && search_trace_path[0] != '\0' &&
            !search_vector_trace_dumped_.exchange(true, std::memory_order_acq_rel)) {
            size_t top_k = 2000000;
            if (const char* raw_top_k = std::getenv("HNSW_SEARCH_VECTOR_READ_TRACE_TOPK")) {
                if (raw_top_k[0] != '\0') {
                    top_k = static_cast<size_t>(std::strtoull(raw_top_k, nullptr, 10));
                }
            }
            index_->dumpSearchVectorReadTrace(search_trace_path, top_k);
        }
        maybe_dump_address_ranges();
        str = ss.str();
    }

    int batch_base_neighbors(const TagT* labels, size_t num_labels,
                             size_t max_neighbors, TagT* out,
                             size_t* counts) const override {
        if (!labels || !out || !counts || max_neighbors == 0) {
            return -1;
        }
        for (size_t i = 0; i < num_labels; ++i) {
            counts[i] = 0;
            hnswlib::tableint internal_id = 0;
            {
                std::unique_lock<std::mutex> lock_table(index_->label_lookup_lock);
                auto it = index_->label_lookup_.find(
                    static_cast<hnswlib::labeltype>(labels[i]));
                if (it == index_->label_lookup_.end()) continue;
                internal_id = it->second;
            }
            if (internal_id >= index_->getCurrentElementCount()) continue;
            std::shared_lock<std::shared_mutex> lock(index_->link_list_locks_[internal_id]);
            auto* data = index_->get_linklist0(internal_id);
            const size_t sz = index_->getListCount(data);
            auto* datal = reinterpret_cast<hnswlib::tableint*>(data + 1);
            const size_t take = std::min(max_neighbors, sz);
            for (size_t j = 0; j < take; ++j) {
                const auto nb = datal[j];
                if (nb >= index_->getCurrentElementCount()) continue;
                out[i * max_neighbors + counts[i]] =
                    static_cast<TagT>(index_->getExternalLabel(nb));
                counts[i]++;
            }
        }
        return 0;
    }

    uint64_t graph_link_write_probe(const TagT* labels, size_t num_labels,
                                    size_t loops, size_t max_edges) override {
        if (!labels || num_labels == 0 || loops == 0) return 0;
        uint64_t writes = 0;
        for (size_t loop = 0; loop < loops; ++loop) {
            for (size_t i = 0; i < num_labels; ++i) {
                hnswlib::tableint internal_id = 0;
                {
                    std::unique_lock<std::mutex> lock_table(index_->label_lookup_lock);
                    auto it = index_->label_lookup_.find(
                        static_cast<hnswlib::labeltype>(labels[i]));
                    if (it == index_->label_lookup_.end()) continue;
                    internal_id = it->second;
                }
                if (internal_id >= index_->getCurrentElementCount()) continue;

                std::unique_lock<std::shared_mutex> lock(index_->link_list_locks_[internal_id]);
                auto* linklist = index_->get_linklist0(internal_id);
                const size_t sz = index_->getListCount(linklist);
                const size_t take = max_edges > 0 ? std::min(max_edges, sz) : sz;

                auto* count_ptr =
                    reinterpret_cast<volatile hnswlib::linklistsizeint*>(linklist);
                const auto count_value = *count_ptr;
                *count_ptr = count_value;
                writes++;

                auto* data = reinterpret_cast<hnswlib::tableint*>(linklist + 1);
                for (size_t j = 0; j < take; ++j) {
                    auto* ptr = reinterpret_cast<volatile hnswlib::tableint*>(data + j);
                    const auto value = *ptr;
                    *ptr = value;
                    writes++;
                }
            }
        }
        return writes;
    }

    bool graph_mutation_stats(GraphMutationStats* out) const override {
        if (out == nullptr) {
            return false;
        }
        out->connect_calls = index_->metric_graph_connect_calls.load();
        out->connect_ns = index_->metric_graph_connect_ns.load();
        out->link_critical_ns = index_->metric_graph_link_critical_ns.load();
        out->unique_lock_wait_ns = index_->metric_graph_unique_lock_wait_ns.load();
        out->search_unique_lock_wait_ns =
            index_->metric_graph_search_unique_lock_wait_ns.load();
        out->search_unique_lock_acqs =
            index_->metric_graph_search_unique_lock_acqs.load();
        out->search_critical_ns =
            index_->metric_graph_search_critical_ns.load();
        out->link_updates = index_->metric_graph_link_updates.load();
        out->upper_search_ns = index_->metric_graph_upper_search_ns.load();
        out->upper_search_dist_comps =
            index_->metric_graph_upper_search_dist_comps.load();
        out->upper_search_edges_scanned =
            index_->metric_graph_upper_search_edges_scanned.load();
        out->base_search_ns = index_->metric_graph_base_search_ns.load();
        out->base_search_expansions =
            index_->metric_graph_base_search_expansions.load();
        out->base_search_edges_scanned =
            index_->metric_graph_base_search_edges_scanned.load();
        out->base_search_dist_comps =
            index_->metric_graph_base_search_dist_comps.load();
        out->select_new_neighbors_ns =
            index_->metric_graph_select_new_neighbors_ns.load();
        out->select_new_neighbors_input =
            index_->metric_graph_select_new_neighbors_input.load();
        out->select_new_neighbors_selected =
            index_->metric_graph_select_new_neighbors_selected.load();
        out->select_new_neighbors_heuristic_dist_comps =
            index_->metric_graph_select_new_neighbors_heuristic_dist_comps.load();
        out->inserted_node_link_ns =
            index_->metric_graph_inserted_node_link_ns.load();
        out->inserted_node_edges_written =
            index_->metric_graph_inserted_node_edges_written.load();
        out->existing_neighbor_update_loop_ns =
            index_->metric_graph_existing_neighbor_update_loop_ns.load();
        out->existing_neighbor_load_scan_ns =
            index_->metric_graph_existing_neighbor_load_scan_ns.load();
        out->existing_neighbor_loaded_edges =
            index_->metric_graph_existing_neighbor_loaded_edges.load();
        out->existing_neighbor_append_ns =
            index_->metric_graph_existing_neighbor_append_ns.load();
        out->existing_neighbor_prune_ns =
            index_->metric_graph_existing_neighbor_prune_ns.load();
        out->existing_neighbor_prune_candidates =
            index_->metric_graph_existing_neighbor_prune_candidates.load();
        out->existing_neighbor_prune_dist_comps =
            index_->metric_graph_existing_neighbor_prune_dist_comps.load();
        out->existing_neighbor_undo_record_ns =
            index_->metric_graph_existing_neighbor_undo_record_ns.load();
        out->existing_neighbor_rewrite_ns =
            index_->metric_graph_existing_neighbor_rewrite_ns.load();
        out->existing_neighbor_edges_written =
            index_->metric_graph_existing_neighbor_edges_written.load();
        out->existing_neighbor_edges_pruned =
            index_->metric_graph_existing_neighbor_edges_pruned.load();
        out->existing_neighbor_visits =
            index_->metric_graph_existing_neighbor_visits.load();
        out->existing_neighbor_appends =
            index_->metric_graph_existing_neighbor_appends.load();
        out->existing_neighbor_prunes =
            index_->metric_graph_existing_neighbor_prunes.load();
        out->existing_neighbor_pruned_edges_recorded =
            index_->metric_graph_existing_neighbor_pruned_edges_recorded.load();
        return true;
    }

    bool supports_snapshot() const override { return true; }

    int snapshot(std::vector<uint8_t>& out) override {
        try {
            std::ostringstream oss(std::ios::binary);
            index_->saveIndex(oss);
            std::string blob = oss.str();
            out.assign(blob.begin(), blob.end());
        } catch (const std::exception&) {
            return -1;
        }
        return 0;
    }

    int restore(const uint8_t* data, size_t size) override {
        if (size > 0 && data == nullptr) {
            return -1;
        }

        try {
            std::string buffer;
            if (size > 0) {
                buffer.assign(reinterpret_cast<const char*>(data), size);
            }
            std::istringstream iss(buffer, std::ios::binary);
            index_->loadIndex(iss, &space_, max_elements_);
            if (ef_s_ > 0) {
                index_->setEf(ef_s_);
            }
        } catch (const std::exception&) {
            return -1;
        }
        return 0;
    }

    // S3 methods
    void set_enable_s3(bool enable) { index_->setEnableS3(enable); }
    bool is_s3_enabled() const { return index_->isS3Enabled(); }
    void set_s3_proximity_threshold(T threshold) { index_->setS3ProximityThreshold(threshold); }
    uint64_t get_s3_hits() const { return index_->getS3Hits(); }
    uint64_t get_s3_misses() const { return index_->getS3Misses(); }
    uint64_t get_s3_dist_saved() const { return index_->getS3DistSaved(); }

    // Search→Search sharing methods
    void set_enable_search_sharing(bool enable) { index_->setEnableSearchSharing(enable); }
    bool is_search_sharing_enabled() const { return index_->isSearchSharingEnabled(); }
    void set_search_sharing_check_interval(int interval) { index_->setSearchSharingCheckInterval(interval); }
    int get_search_sharing_check_interval() const { return index_->getSearchSharingCheckInterval(); }
    uint64_t get_search_sharing_hits() const { return index_->getSearchSharingHits(); }
    uint64_t get_search_sharing_misses() const { return index_->getSearchSharingMisses(); }

    // Path Skip & Candidate Injection methods
    void set_enable_path_skip(bool enable) { index_->setEnablePathSkip(enable); }
    void set_enable_candidate_injection(bool enable) { index_->setEnableCandidateInjection(enable); }
    bool is_path_skip_enabled() const { return index_->isPathSkipEnabled(); }
    bool is_candidate_injection_enabled() const { return index_->isCandidateInjectionEnabled(); }

    // Adaptive locality detection
    void set_s3_adaptive_enabled(bool enabled) { index_->setS3AdaptiveEnabled(enabled); }
    bool is_s3_adaptive_enabled() const { return index_->isS3AdaptiveEnabled(); }
    void set_s3_adaptive_threshold(float threshold) { index_->setS3AdaptiveThreshold(threshold); }
    
    void set_enable_warm_start(bool enable) { index_->setEnableWarmStart(enable); }
    bool is_warm_start_enabled() const { return index_->isWarmStartEnabled(); }

    void suspend_sharing() { index_->suspendSharing(); }
    void resume_sharing() { index_->resumeSharing(); }
    float get_s3_adaptive_threshold() const { return index_->getS3AdaptiveThreshold(); }
    uint64_t get_s3_adaptive_accepts() const { return index_->getS3AdaptiveAccepts(); }
    uint64_t get_s3_adaptive_rejects() const { return index_->getS3AdaptiveRejects(); }

    int search_s3(const T* query, size_t k, std::vector<TagT>& result_tags) {
        auto result = index_->searchKnnS3(query, k);
        while (!result.empty()) {
            result_tags.push_back(result.top().second);
            result.pop();
        }
        return 0;
    }

    int search_search_sharing(const T* query, size_t k, std::vector<TagT>& result_tags) {
        auto result = index_->searchKnnSearchSharing(query, k);
        while (!result.empty()) {
            result_tags.push_back(result.top().second);
            result.pop();
        }
        return 0;
    }

    int search_warm_start(const T* query, size_t k, std::vector<TagT>& result_tags) {
        auto result = index_->searchKnnWarmStart(query, k);
        while (!result.empty()) {
            result_tags.push_back(result.top().second);
            result.pop();
        }
        return 0;
    }

    void enable_insert_telemetry(bool enable) { index_->enableInsertTelemetry(enable); }
    size_t last_insert_update_count() const { return index_->getLastInsertUpdateCount(); }
    void configure_partial_insert(size_t limit) { index_->configurePartialInsert(limit); }
    void disable_partial_insert() { index_->disablePartialInsert(); }

    size_t dim_;
    size_t num_threads_;
    size_t max_elements_;
    size_t M_;
    size_t ef_c_;
    size_t ef_s_;
    hnswlib::L2Space space_;
    hnswlib::HierarchicalNSW<T>* index_;

   private:
    void maybe_dump_address_ranges() const {
        const char* path = std::getenv("HNSW_ADDR_RANGE_DUMP");
        if (path == nullptr || path[0] == '\0') {
            path = std::getenv("ANNCHOR_M3_ADDR_RANGE_DUMP");
        }
        if (path == nullptr || path[0] == '\0' || index_ == nullptr) return;

        std::ofstream out(path);
        if (!out) return;
        out << "class,id,level,begin,end,size\n";
        auto write_range = [&](const char* cls, size_t id, int level,
                               uintptr_t begin, uintptr_t end) {
            if (begin == 0 || end <= begin) return;
            out << cls << ',' << id << ',' << level << ",0x" << std::hex
                << begin << ",0x" << end << std::dec << ','
                << (end - begin) << '\n';
        };

        const uintptr_t level0_begin =
            reinterpret_cast<uintptr_t>(index_->data_level0_memory_);
        write_range("level0_records", 0, 0, level0_begin,
                    level0_begin +
                        index_->max_elements_ * index_->size_data_per_element_);

        const uintptr_t ptr_begin =
            reinterpret_cast<uintptr_t>(index_->linkLists_);
        write_range("upper_link_pointer_array", 0, 0, ptr_begin,
                    ptr_begin + index_->max_elements_ * sizeof(char*));

        const uintptr_t locks_begin =
            index_->link_list_locks_.empty()
                ? uintptr_t{0}
                : reinterpret_cast<uintptr_t>(index_->link_list_locks_.data());
        write_range("node_lock_array", 0, 0, locks_begin,
                    locks_begin + index_->link_list_locks_.size() *
                                      sizeof(index_->link_list_locks_[0]));

        const uintptr_t levels_begin =
            index_->element_levels_.empty()
                ? uintptr_t{0}
                : reinterpret_cast<uintptr_t>(index_->element_levels_.data());
        write_range("element_levels_array", 0, 0, levels_begin,
                    levels_begin + index_->element_levels_.size() *
                                       sizeof(index_->element_levels_[0]));

        const uintptr_t object_begin = reinterpret_cast<uintptr_t>(index_);
        write_range("hnsw_object_metadata", 0, 0, object_begin,
                    object_begin + sizeof(*index_));

        if (index_->visited_list_pool_) {
            index_->visited_list_pool_->dumpAddressRanges(out, "shared");
        }
    }

    tbb::task_arena arena_;
    bool measured_search_arena_enabled_;
    tbb::task_arena measured_search_arena_;
    HNSWWorkScheduler scheduler_;
    std::atomic<bool> insert_vector_trace_dumped_{false};
    std::atomic<bool> search_vector_trace_dumped_{false};

    static std::vector<int> read_env_cpu_list(const char* primary,
                                              const char* fallback = nullptr) {
        const char* raw = nullptr;
        if (primary != nullptr) {
            raw = std::getenv(primary);
        }
        if ((raw == nullptr || raw[0] == '\0') && fallback != nullptr) {
            raw = std::getenv(fallback);
        }
        std::vector<int> cpus;
        if (raw == nullptr || raw[0] == '\0') {
            return cpus;
        }

        std::stringstream ss(raw);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (token.empty()) {
                continue;
            }
            auto dash = token.find('-');
            if (dash == std::string::npos) {
                char* end = nullptr;
                long cpu = std::strtol(token.c_str(), &end, 10);
                if (end != token.c_str() && cpu >= 0) {
                    cpus.push_back(static_cast<int>(cpu));
                }
                continue;
            }
            const std::string begin_text = token.substr(0, dash);
            const std::string end_text = token.substr(dash + 1);
            char* begin_end = nullptr;
            char* finish_end = nullptr;
            long begin = std::strtol(begin_text.c_str(), &begin_end, 10);
            long finish = std::strtol(end_text.c_str(), &finish_end, 10);
            if (begin_end == begin_text.c_str() || finish_end == end_text.c_str()) {
                continue;
            }
            if (begin < 0 || finish < 0) {
                continue;
            }
            if (begin <= finish) {
                for (long cpu = begin; cpu <= finish; ++cpu) {
                    cpus.push_back(static_cast<int>(cpu));
                }
            } else {
                for (long cpu = begin; cpu >= finish; --cpu) {
                    cpus.push_back(static_cast<int>(cpu));
                }
            }
        }
        return cpus;
    }

    static void pin_current_thread_to_env_cpu(const char* primary,
                                              const char* fallback,
                                              size_t ordinal) {
        const std::vector<int> cpus = read_env_cpu_list(primary, fallback);
        if (cpus.empty()) {
            return;
        }
        const int cpu = cpus[ordinal % cpus.size()];
        thread_local int current_cpu = -1;
        if (current_cpu == cpu) {
            return;
        }

        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) == 0) {
            current_cpu = cpu;
        }
    }

    static bool read_env_bool(const char* name, bool default_value) {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return default_value;
        }
        return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
               value[0] == 'y' || value[0] == 'Y';
    }

    static size_t read_env_size(const char* name, size_t default_value) {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return default_value;
        }
        char* end = nullptr;
        unsigned long parsed = std::strtoul(value, &end, 10);
        if (end == value || parsed == 0) {
            return default_value;
        }
        return static_cast<size_t>(parsed);
    }

    tbb::task_arena& measured_search_task_arena() {
        return measured_search_arena_enabled_ ? measured_search_arena_ : arena_;
    }

    int batch_insert_thread_pool(const T* batch_data, const TagT* batch_tags,
                                 size_t num_points) {
        if (num_points == 0) {
            return 0;
        }
        index_->enableInsertVectorReadAdmission(true);
        auto remaining = std::make_shared<std::atomic<size_t>>(num_points);
        auto completion = std::make_shared<std::promise<void>>();
        auto future = completion->get_future();
        for (size_t i = 0; i < num_points; ++i) {
            const T* point = batch_data + i * dim_;
            TagT tag = batch_tags[i];
            arena_.enqueue([this, point, tag, remaining, completion]() {
                index_->addPoint(point, tag);
                if (remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    completion->set_value();
                }
            });
        }
        future.wait();
        index_->enableInsertVectorReadAdmission(false);
        return 0;
    }

    int batch_search_thread_pool(const T* batch_queries, size_t k,
                                 size_t num_queries, TagT** batch_results) {
        if (num_queries == 0) {
            return 0;
        }
        auto remaining = std::make_shared<std::atomic<size_t>>(num_queries);
        auto completion = std::make_shared<std::promise<void>>();
        auto future = completion->get_future();
        for (size_t i = 0; i < num_queries; ++i) {
            const T* query = batch_queries + i * dim_;
            arena_.enqueue([this, query, k, i, batch_results, remaining, completion]() {
                auto result = index_->searchKnn(query, k);
                std::vector<TagT> results;
                while (!result.empty()) {
                    results.push_back(result.top().second);
                    result.pop();
                }
                std::reverse(results.begin(), results.end());
                size_t limit = std::min(results.size(), static_cast<size_t>(k));
                for (size_t j = 0; j < limit; ++j) {
                    batch_results[i][j] = results[j];
                }
                if (remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    completion->set_value();
                }
            });
        }
        future.wait();
        return 0;
    }
};
