#pragma once

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sched.h>
#include <sstream>
#include <time.h>
#include <vector>

#include "../index.hpp"
#include "../index_cgo.hpp"
#define annchor annchor_m1
#include "ANNchor/src/hnswlib/hnswlib.h"
#undef annchor
#include <memory>
#include <cmath>

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class ANNchorM1 : public IndexBase<T, TagT, LabelT> {
   public:
    ANNchorM1(size_t max_elements, size_t dim, size_t num_threads, size_t M,
           size_t ef_construction, bool use_node_lock_in_search = true, MetricType metric = METRIC_L2)
        : dim_(dim),
          num_threads_(num_threads),
          M_(M),
          ef_c_(ef_construction),
          ef_s_(0),
          max_elements_(max_elements),
          use_node_lock_in_search_(use_node_lock_in_search),
          metric_(metric),
          arena_(read_env_size("ANNCHOR_SHARED_ARENA_THREADS", num_threads)),
          separate_arenas_(read_env_bool("ANNCHOR_SEPARATE_SEARCH_ARENA", false)),
          search_arena_(read_env_size("ANNCHOR_SEARCH_ARENA_THREADS", num_threads)),
          insert_arena_(read_env_size("ANNCHOR_INSERT_ARENA_THREADS", num_threads)),
          measured_search_arena_enabled_(read_env_bool("ANNCHOR_MEASURED_SEARCH_ARENA", false)),
          measured_search_arena_(read_env_size("ANNCHOR_MEASURED_SEARCH_ARENA_THREADS", num_threads)) {
        if (metric_ == METRIC_IP || metric_ == METRIC_COSINE) {
            space_.reset(new annchor_m1::InnerProductSpace(dim));
        } else {
            space_.reset(new annchor_m1::L2Space(dim));
        }
        index_ = new annchor_m1::HierarchicalNSW<T>(
            space_.get(), max_elements, M, ef_construction, 100, false, use_node_lock_in_search, num_threads_);
        index_->setEnableDirectionCapsule(true);
    }

    ~ANNchorM1() override {
        delete index_;
    }

    void build(const T* data, const TagT* tags, size_t num_points) override {
        const bool direction_capsule_enabled = index_->isDirectionCapsuleEnabled();
        index_->setEnableDirectionCapsule(false);
        (void)batch_insert(data, tags, num_points);
        index_->setEnableDirectionCapsule(direction_capsule_enabled);
    }

    int insert(const T* data, const TagT tag) override {
        if (metric_ == METRIC_COSINE) {
            std::vector<T> temp_vec(data, data + dim_);
            normalize_vector(temp_vec.data(), 1);
            index_->addPoint(temp_vec.data(), tag);
        } else {
            index_->addPoint(data, tag);
        }
        return 0;
    }

    int batch_insert(const T* batch_data, const TagT* batch_tags,
                     size_t num_points) override {
        if (num_points == 0) return 0;
        bool mvcc = index_->isMvccEnabled();

        if (metric_ == METRIC_COSINE) {
	             std::vector<T> temp_batch(batch_data, batch_data + num_points * dim_);
	             normalize_vector(temp_batch.data(), num_points);
	             insert_task_arena().execute([&] {
	                 tbb::parallel_for(size_t(0), num_points, [&](size_t i) {
	                     pin_current_thread_to_env_cpu("ANNCHOR_INSERT_CPU_LIST", i);
	                     const T* point = temp_batch.data() + i * dim_;
	                     auto id = index_->addPoint(
	                         point, static_cast<annchor_m1::labeltype>(batch_tags[i]), -1, mvcc);
                     if (mvcc) index_->markReady(id);
                 });
             });
	        } else {
	             insert_task_arena().execute([&] {
	                 tbb::parallel_for(size_t(0), num_points, [&](size_t i) {
	                     pin_current_thread_to_env_cpu("ANNCHOR_INSERT_CPU_LIST", i);
	                     const T* point = batch_data + i * dim_;
	                     auto id = index_->addPoint(
	                         point, static_cast<annchor_m1::labeltype>(batch_tags[i]), -1, mvcc);
                     if (mvcc) index_->markReady(id);
                  });
              });
        }
        if (mvcc) index_->advanceWatermark();

        return 0;
    }

    void set_query_params(const QParams& params) override {
        ef_s_ = params.ef_search;
        index_->setEf(ef_s_);
    }

    int search(const T* query, size_t k,
               std::vector<TagT>& result_tags) override {
        std::priority_queue<std::pair<T, annchor_m1::labeltype>> result;
        if (metric_ == METRIC_COSINE) {
            std::vector<T> temp_query(query, query + dim_);
            normalize_vector(temp_query.data(), 1);
            result = index_->searchKnn(temp_query.data(), k);
        } else {
            result = index_->searchKnn(query, k);
        }

        while (!result.empty()) {
            result_tags.push_back(result.top().second);
            result.pop();
        }
        return 0;
    }

	    int batch_search(const T* batch_queries, size_t k, size_t num_queries,
	                     TagT** batch_results, size_t* watermark_out = nullptr,
	                     size_t visible_ts = std::numeric_limits<size_t>::max()) override {
	        return batch_search_with_arena(
	            search_task_arena(), batch_queries, k, num_queries, batch_results,
	            watermark_out, visible_ts, nullptr, "ANNCHOR_SEARCH_CPU_LIST");
	    }

    int batch_search_measured(const T* batch_queries, size_t k, size_t num_queries,
                              TagT** batch_results, size_t* watermark_out = nullptr,
                              size_t visible_ts = std::numeric_limits<size_t>::max()) override {
	        return batch_search_with_arena(
	            measured_search_task_arena(), batch_queries, k, num_queries,
	            batch_results, watermark_out, visible_ts, nullptr,
	            "ANNCHOR_MEASURED_SEARCH_CPU_LIST");
	    }

    int batch_search_measured_work(const T* batch_queries, size_t k,
                                   size_t num_queries, TagT** batch_results,
                                   SearchWorkStats* per_query_stats,
                                   size_t* watermark_out = nullptr,
                                   size_t visible_ts = std::numeric_limits<size_t>::max()) override {
	        return batch_search_with_arena(
	            measured_search_task_arena(), batch_queries, k, num_queries,
	            batch_results, watermark_out, visible_ts, per_query_stats,
	            "ANNCHOR_MEASURED_SEARCH_CPU_LIST");
	    }

    int batch_search_with_arena(tbb::task_arena& task_arena,
                                const T* batch_queries, size_t k,
	                                size_t num_queries, TagT** batch_results,
	                                size_t* watermark_out,
	                                size_t visible_ts,
	                                SearchWorkStats* per_query_stats = nullptr,
	                                const char* cpu_list_env = nullptr) {
        size_t wm = visible_ts;
        if (watermark_out) *watermark_out = wm;

        const auto arena_start = std::chrono::steady_clock::now();
        std::atomic<uint64_t> local_searchknn_ns{0};
        std::atomic<uint64_t> local_result_copy_ns{0};
	        task_arena.execute([&] {
	            tbb::parallel_for(size_t(0), num_queries, [&](size_t i) {
	                pin_current_thread_to_env_cpu(cpu_list_env, i);
	                const T* q = batch_queries + i * dim_;
                std::priority_queue<std::pair<T, annchor_m1::labeltype>> result;

                const auto searchknn_start = std::chrono::steady_clock::now();
	                struct timespec cpu_start {};
	                if (per_query_stats) {
	                    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_start);
	                }
	                const int work_start_cpu =
	                    per_query_stats ? sched_getcpu() : -1;
	                annchor_m1::SearchWorkStats query_stats{};
	                auto* stats_ptr = per_query_stats ? &query_stats : nullptr;
                if (metric_ == METRIC_COSINE) {
                    std::vector<T> temp_query(q, q + dim_);
                    normalize_vector(temp_query.data(), 1);
                    result = index_->searchKnnWithStats(
                        temp_query.data(), k, nullptr, wm, stats_ptr);
                } else {
                    result = index_->searchKnnWithStats(q, k, nullptr, wm,
                                                        stats_ptr);
                }
	                const auto searchknn_end = std::chrono::steady_clock::now();
	                const int work_end_cpu =
	                    per_query_stats ? sched_getcpu() : -1;
	                uint64_t searchknn_thread_cpu_ns = 0;
                if (per_query_stats) {
                    struct timespec cpu_end {};
                    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_end);
                    const int64_t cpu_ns =
                        static_cast<int64_t>(cpu_end.tv_sec - cpu_start.tv_sec) *
                            1000000000LL +
                        static_cast<int64_t>(cpu_end.tv_nsec - cpu_start.tv_nsec);
                    searchknn_thread_cpu_ns =
                        cpu_ns > 0 ? static_cast<uint64_t>(cpu_ns) : 0;
                }
                const auto searchknn_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        searchknn_end - searchknn_start)
                        .count());
                local_searchknn_ns.fetch_add(
                    searchknn_ns,
                    std::memory_order_relaxed);

                const auto copy_start = std::chrono::steady_clock::now();
                size_t sz = std::min(result.size(), k);
                for (size_t j = 0; j < k; ++j) {
                    batch_results[i][j] = std::numeric_limits<TagT>::max();
                }
                for (size_t j = 0; j < sz; ++j) {
                    batch_results[i][sz - j - 1] = result.top().second;
                    result.pop();
                }
                const auto copy_end = std::chrono::steady_clock::now();
                const auto copy_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        copy_end - copy_start)
                        .count());
                local_result_copy_ns.fetch_add(
                    copy_ns,
                    std::memory_order_relaxed);
                if (per_query_stats) {
                    per_query_stats[i].searchknn_ns = searchknn_ns;
	                    per_query_stats[i].searchknn_thread_cpu_ns =
	                        searchknn_thread_cpu_ns;
	                    per_query_stats[i].work_start_cpu = work_start_cpu;
	                    per_query_stats[i].work_end_cpu = work_end_cpu;
	                    per_query_stats[i].result_copy_ns = copy_ns;
                    per_query_stats[i].entry_ns = query_stats.entry_ns;
                    per_query_stats[i].upper_search_ns =
                        query_stats.upper_search_ns;
                    per_query_stats[i].base_search_ns =
                        query_stats.base_search_ns;
                    per_query_stats[i].result_materialize_ns =
                        query_stats.result_materialize_ns;
                    per_query_stats[i].snapshot_guard_ns =
                        query_stats.snapshot_guard_ns;
                    per_query_stats[i].visited_list_get_ns =
                        query_stats.visited_list_get_ns;
                    per_query_stats[i].visited_list_release_ns =
                        query_stats.visited_list_release_ns;
                    per_query_stats[i].upper_lock_wait_ns =
                        query_stats.upper_lock_wait_ns;
                    per_query_stats[i].level0_lock_wait_ns =
                        query_stats.level0_lock_wait_ns;
                    per_query_stats[i].distance_computations =
                        query_stats.distance_computations;
                    per_query_stats[i].upper_distance_computations =
                        query_stats.upper_distance_computations;
                    per_query_stats[i].level0_distance_computations =
                        query_stats.level0_distance_computations;
                    per_query_stats[i].upper_hops = query_stats.upper_hops;
                    per_query_stats[i].upper_edges_scanned =
                        query_stats.upper_edges_scanned;
                    per_query_stats[i].level0_expansions =
                        query_stats.level0_expansions;
                    per_query_stats[i].level0_edges_scanned =
                        query_stats.level0_edges_scanned;
                    per_query_stats[i].candidate_pops =
                        query_stats.candidate_pops;
                    per_query_stats[i].candidate_pushes =
                        query_stats.candidate_pushes;
                    per_query_stats[i].visited_nodes =
                        query_stats.visited_nodes;
                    per_query_stats[i].result_pushes =
                        query_stats.result_pushes;
                    per_query_stats[i].invisible_expansions =
                        query_stats.invisible_expansions;
                    per_query_stats[i].invisible_edges =
                        query_stats.invisible_edges;
                    per_query_stats[i].invisible_candidate_dist_comps =
                        query_stats.invisible_candidate_dist_comps;
                    per_query_stats[i].invisible_candidate_enqueues =
                        query_stats.invisible_candidate_enqueues;
                    per_query_stats[i].future_skip_hops =
                        query_stats.future_skip_hops;
                }
            });
        });
        const auto arena_end = std::chrono::steady_clock::now();
        metric_batch_search_calls_.fetch_add(1, std::memory_order_relaxed);
        metric_batch_search_queries_.fetch_add(num_queries, std::memory_order_relaxed);
        metric_batch_search_arena_ns_.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    arena_end - arena_start)
                    .count()),
            std::memory_order_relaxed);
        metric_batch_search_searchknn_ns_.fetch_add(
            local_searchknn_ns.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        metric_batch_search_result_copy_ns_.fetch_add(
            local_result_copy_ns.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        return 0;
    }

    long get_inflight_points() const override {
        return index_->getInflightPoints();
    }

    int batch_base_neighbors(const TagT* labels, size_t num_labels,
                             size_t max_neighbors, TagT* out,
                             size_t* counts) const override {
        if (!labels || !out || !counts || max_neighbors == 0) return -1;

        for (size_t i = 0; i < num_labels; ++i) {
            counts[i] = 0;
            annchor_m1::tableint internal_id = 0;
            {
                std::unique_lock<std::mutex> lock_table(index_->label_lookup_lock);
                auto it = index_->label_lookup_.find(
                    static_cast<annchor_m1::labeltype>(labels[i]));
                if (it == index_->label_lookup_.end()) continue;
                internal_id = it->second;
            }
            if (internal_id >= index_->getCurrentElementCount()) continue;
            auto* data = index_->get_linklist0(internal_id);
            const size_t sz = index_->getListCount(data);
            auto* datal = reinterpret_cast<annchor_m1::tableint*>(data + 1);
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
                annchor_m1::tableint internal_id = 0;
                {
                    std::unique_lock<std::mutex> lock_table(index_->label_lookup_lock);
                    auto it = index_->label_lookup_.find(
                        static_cast<annchor_m1::labeltype>(labels[i]));
                    if (it == index_->label_lookup_.end()) continue;
                    internal_id = it->second;
                }
                if (internal_id >= index_->getCurrentElementCount()) continue;

                auto* linklist = index_->get_linklist0(internal_id);
                const size_t sz = index_->getListCount(linklist);
                const size_t take = max_edges > 0 ? std::min(max_edges, sz) : sz;

                auto* count_ptr =
                    reinterpret_cast<volatile annchor_m1::linklistsizeint*>(
                        linklist);
                const auto count_value = *count_ptr;
                *count_ptr = count_value;
                writes++;

                auto* data =
                    reinterpret_cast<annchor_m1::tableint*>(linklist + 1);
                for (size_t j = 0; j < take; ++j) {
                    auto* ptr =
                        reinterpret_cast<volatile annchor_m1::tableint*>(
                            data + j);
                    const auto value = *ptr;
                    *ptr = value;
                    writes++;
                }
            }
        }
        return writes;
    }

    void dump_stats(std::string& str) override {
        std::stringstream ss;
        ss << "index_memory_mb:" << index_->indexFileSize() / (1024 * 1024)
           << ", dist_computations:"
           << index_->metric_distance_computations.load()
           << ", hops:" << index_->metric_hops.load()
           << ", future_skip_queries:" << index_->metric_future_skip_queries.load()
           << ", future_skip_hops_total:" << index_->metric_future_skip_hops_total.load()
           << ", future_skip_hop_max:" << index_->metric_future_skip_hop_max.load()
           << ", graph_connect_calls:" << index_->metric_graph_connect_calls.load()
           << ", graph_connect_ns:" << index_->metric_graph_connect_ns.load()
           << ", graph_link_critical_ns:" << index_->metric_graph_link_critical_ns.load()
		           << ", graph_unique_lock_wait_ns:" << index_->metric_graph_unique_lock_wait_ns.load()
		           << ", graph_link_updates:" << index_->metric_graph_link_updates.load()
	           << ", graph_upper_search_ns:" << index_->metric_graph_upper_search_ns.load()
	           << ", graph_upper_search_dist_comps:" << index_->metric_graph_upper_search_dist_comps.load()
	           << ", graph_upper_search_edges_scanned:" << index_->metric_graph_upper_search_edges_scanned.load()
	           << ", graph_base_search_ns:" << index_->metric_graph_base_search_ns.load()
	           << ", graph_base_search_expansions:" << index_->metric_graph_base_search_expansions.load()
	           << ", graph_base_search_edges_scanned:" << index_->metric_graph_base_search_edges_scanned.load()
	           << ", graph_base_search_dist_comps:" << index_->metric_graph_base_search_dist_comps.load()
	           << ", graph_select_new_neighbors_ns:" << index_->metric_graph_select_new_neighbors_ns.load()
	           << ", graph_select_new_neighbors_input:" << index_->metric_graph_select_new_neighbors_input.load()
	           << ", graph_select_new_neighbors_selected:" << index_->metric_graph_select_new_neighbors_selected.load()
	           << ", graph_select_new_neighbors_heuristic_dist_comps:" << index_->metric_graph_select_new_neighbors_heuristic_dist_comps.load()
		           << ", graph_inserted_node_link_ns:" << index_->metric_graph_inserted_node_link_ns.load()
	           << ", graph_inserted_node_edges_written:" << index_->metric_graph_inserted_node_edges_written.load()
		           << ", graph_existing_neighbor_update_loop_ns:" << index_->metric_graph_existing_neighbor_update_loop_ns.load()
	           << ", graph_existing_neighbor_load_scan_ns:" << index_->metric_graph_existing_neighbor_load_scan_ns.load()
	           << ", graph_existing_neighbor_loaded_edges:" << index_->metric_graph_existing_neighbor_loaded_edges.load()
	           << ", graph_existing_neighbor_append_ns:" << index_->metric_graph_existing_neighbor_append_ns.load()
	           << ", graph_existing_neighbor_prune_ns:" << index_->metric_graph_existing_neighbor_prune_ns.load()
	           << ", graph_existing_neighbor_prune_candidates:" << index_->metric_graph_existing_neighbor_prune_candidates.load()
	           << ", graph_existing_neighbor_prune_dist_comps:" << index_->metric_graph_existing_neighbor_prune_dist_comps.load()
		           << ", graph_existing_neighbor_undo_record_ns:" << index_->metric_graph_existing_neighbor_undo_record_ns.load()
	           << ", graph_existing_neighbor_rewrite_ns:" << index_->metric_graph_existing_neighbor_rewrite_ns.load()
	           << ", graph_existing_neighbor_edges_written:" << index_->metric_graph_existing_neighbor_edges_written.load()
	           << ", graph_existing_neighbor_edges_pruned:" << index_->metric_graph_existing_neighbor_edges_pruned.load()
	           << ", graph_existing_neighbor_visits:" << index_->metric_graph_existing_neighbor_visits.load()
	           << ", graph_existing_neighbor_appends:" << index_->metric_graph_existing_neighbor_appends.load()
	           << ", graph_existing_neighbor_prunes:" << index_->metric_graph_existing_neighbor_prunes.load()
	           << ", graph_existing_neighbor_pruned_edges_recorded:" << index_->metric_graph_existing_neighbor_pruned_edges_recorded.load()
	           << ", rewrite_active_expansions:" << index_->metric_search_rewrite_active_expansions.load()
	           << ", rewrite_active_queries:" << index_->metric_search_rewrite_active_queries.load()
	           << ", rewrite_recent_expansions:" << index_->metric_search_rewrite_recent_expansions.load()
	           << ", rewrite_recent_queries:" << index_->metric_search_rewrite_recent_queries.load()
	           << ", rewrite_period_expansions:" << index_->metric_search_rewrite_period_expansions.load()
	           << ", rewrite_period_queries:" << index_->metric_search_rewrite_period_queries.load()
	           << ", rewrite_period_active_sum:" << index_->metric_search_rewrite_period_active_sum.load()
	           << ", rewrite_period_active_max:" << index_->metric_search_rewrite_period_active_max.load()
	           << ", batch_search_calls:" << metric_batch_search_calls_.load()
	           << ", batch_search_queries:" << metric_batch_search_queries_.load()
	           << ", batch_search_arena_ns:" << metric_batch_search_arena_ns_.load()
	           << ", batch_search_searchknn_ns:" << metric_batch_search_searchknn_ns_.load()
	           << ", batch_search_result_copy_ns:" << metric_batch_search_result_copy_ns_.load()
	           << ", invisible_expansions:" << index_->metric_search_invisible_expansions.load()
	           << ", invisible_expansion_edges:" << index_->metric_search_invisible_expansion_edges.load()
	           << ", invisible_candidate_dist_comps:" << index_->metric_search_invisible_candidate_dist_comps.load()
	           << ", invisible_candidate_enqueues:" << index_->metric_search_invisible_candidate_enqueues.load()
	           << ", phase_overlap_samples:" << index_->metric_search_phase_overlap_samples.load()
	           << ", phase_existing_update_samples:" << index_->metric_search_phase_existing_update_samples.load()
	           << ", phase_existing_update_queries:" << index_->metric_search_phase_existing_update_queries.load()
	           << ", phase_link_critical_samples:" << index_->metric_search_phase_link_critical_samples.load()
	           << ", phase_link_critical_queries:" << index_->metric_search_phase_link_critical_queries.load()
	           << ", phase_load_scan_samples:" << index_->metric_search_phase_load_scan_samples.load()
	           << ", phase_load_scan_queries:" << index_->metric_search_phase_load_scan_queries.load()
	           << ", phase_append_samples:" << index_->metric_search_phase_append_samples.load()
	           << ", phase_append_queries:" << index_->metric_search_phase_append_queries.load()
	           << ", phase_prune_samples:" << index_->metric_search_phase_prune_samples.load()
	           << ", phase_prune_queries:" << index_->metric_search_phase_prune_queries.load()
	           << ", phase_undo_record_samples:" << index_->metric_search_phase_undo_record_samples.load()
	           << ", phase_undo_record_queries:" << index_->metric_search_phase_undo_record_queries.load()
	           << ", phase_rewrite_samples:" << index_->metric_search_phase_rewrite_samples.load()
	           << ", phase_rewrite_queries:" << index_->metric_search_phase_rewrite_queries.load()
	           << ", hot_region_search_marks:" << index_->metric_hot_region_search_marks.load()
	           << ", hot_region_repair_checks:" << index_->metric_hot_region_repair_checks.load()
	           << ", hot_region_repair_conflicts:" << index_->metric_hot_region_repair_conflicts.load()
	           << ", hot_region_repair_hot_neighbors:" << index_->metric_hot_region_repair_hot_neighbors.load()
	           << ", hot_region_repair_reordered_calls:" << index_->metric_hot_region_repair_reordered_calls.load()
	           << ", hot_region_candidate_filter_calls:" << index_->metric_hot_region_candidate_filter_calls.load()
	           << ", hot_region_candidate_filter_cold_kept:" << index_->metric_hot_region_candidate_filter_cold_kept.load()
	           << ", hot_region_candidate_filter_hot_dropped:" << index_->metric_hot_region_candidate_filter_hot_dropped.load()
           << ", undo_recovery_enabled:" << (index_->isUndoRecoveryEnabled() ? 1 : 0)
           << ", search_visibility_mode:" << index_->getSearchVisibilityMode()
           << ", search_visibility_by_label:" << (index_->getSearchVisibilityByLabel() ? 1 : 0)
           << ", undo_recovery_mode:" << index_->getM2Mode()
           << ", recovery_triggers:" << index_->m2_triggered_.load()
           << ", recovered_edges_total:" << index_->m2_recovered_edges_.load()
           << ", recovered_edges_useful:" << index_->m2_useful_edges_.load()
           << ", recovery_modified_expansions:" << index_->m2_modified_expansions_.load()
           << ", recovery_attempts:" << index_->m2_recovery_attempts_.load()
           << ", recovery_candidate_edges:" << index_->m2_recovery_candidate_edges_.load()
           << ", recovery_get_ns:" << index_->m2_recovery_get_ns_.load()
           << ", recovery_loop_ns:" << index_->m2_recovery_loop_ns_.load()
           << ", direction_capsule_enabled:" << (index_->isDirectionCapsuleEnabled() ? 1 : 0)
           << ", direction_capsule_bits:" << index_->getDirectionCapsuleBits()
           << ", direction_capsule_route_r:" << index_->getDirectionCapsuleRouteR()
           << ", direction_capsule_points:" << index_->getDirectionCapsulePoints()
           << ", direction_capsule_entries:" << index_->getDirectionCapsuleEntries()
           << ", direction_capsule_edge_tests:" << index_->getDirectionCapsuleEdgeTests()
           << ", direction_capsule_bucket_count:" << index_->getDirectionCapsuleBucketCount()
           << ", separate_search_arena:" << (separate_arenas_ ? 1 : 0)
           << ", search_arena_threads:" << search_arena_.max_concurrency()
           << ", insert_arena_threads:" << insert_arena_.max_concurrency();

        str = ss.str();
    }

    bool graph_mutation_stats(GraphMutationStats* out) const override {
        if (out == nullptr) return false;
        out->connect_calls = index_->metric_graph_connect_calls.load();
        out->connect_ns = index_->metric_graph_connect_ns.load();
        out->link_critical_ns = index_->metric_graph_link_critical_ns.load();
        out->unique_lock_wait_ns = index_->metric_graph_unique_lock_wait_ns.load();
        out->link_updates = index_->metric_graph_link_updates.load();
        out->upper_search_ns = index_->metric_graph_upper_search_ns.load();
        out->upper_search_dist_comps = index_->metric_graph_upper_search_dist_comps.load();
        out->upper_search_edges_scanned = index_->metric_graph_upper_search_edges_scanned.load();
        out->base_search_ns = index_->metric_graph_base_search_ns.load();
        out->base_search_expansions = index_->metric_graph_base_search_expansions.load();
        out->base_search_edges_scanned = index_->metric_graph_base_search_edges_scanned.load();
        out->base_search_dist_comps = index_->metric_graph_base_search_dist_comps.load();
        out->select_new_neighbors_ns = index_->metric_graph_select_new_neighbors_ns.load();
        out->select_new_neighbors_input = index_->metric_graph_select_new_neighbors_input.load();
        out->select_new_neighbors_selected = index_->metric_graph_select_new_neighbors_selected.load();
        out->select_new_neighbors_heuristic_dist_comps =
            index_->metric_graph_select_new_neighbors_heuristic_dist_comps.load();
        out->inserted_node_link_ns = index_->metric_graph_inserted_node_link_ns.load();
        out->inserted_node_edges_written = index_->metric_graph_inserted_node_edges_written.load();
        out->existing_neighbor_update_loop_ns = index_->metric_graph_existing_neighbor_update_loop_ns.load();
        out->existing_neighbor_load_scan_ns = index_->metric_graph_existing_neighbor_load_scan_ns.load();
        out->existing_neighbor_loaded_edges = index_->metric_graph_existing_neighbor_loaded_edges.load();
        out->existing_neighbor_append_ns = index_->metric_graph_existing_neighbor_append_ns.load();
        out->existing_neighbor_prune_ns = index_->metric_graph_existing_neighbor_prune_ns.load();
        out->existing_neighbor_prune_candidates = index_->metric_graph_existing_neighbor_prune_candidates.load();
        out->existing_neighbor_prune_dist_comps = index_->metric_graph_existing_neighbor_prune_dist_comps.load();
        out->existing_neighbor_undo_record_ns = index_->metric_graph_existing_neighbor_undo_record_ns.load();
        out->existing_neighbor_rewrite_ns = index_->metric_graph_existing_neighbor_rewrite_ns.load();
        out->existing_neighbor_edges_written = index_->metric_graph_existing_neighbor_edges_written.load();
        out->existing_neighbor_edges_pruned = index_->metric_graph_existing_neighbor_edges_pruned.load();
        out->existing_neighbor_visits = index_->metric_graph_existing_neighbor_visits.load();
        out->existing_neighbor_appends = index_->metric_graph_existing_neighbor_appends.load();
        out->existing_neighbor_prunes = index_->metric_graph_existing_neighbor_prunes.load();
        out->existing_neighbor_pruned_edges_recorded = index_->metric_graph_existing_neighbor_pruned_edges_recorded.load();
        return true;
    }

    // MVCC Mechanism 1 methods (only these are exposed)
    void set_enable_mvcc(bool enable) {
        index_->setEnableMvcc(enable);
    }
    bool is_mvcc_enabled() const { return index_->isMvccEnabled(); }
    void set_enable_undo_recovery(bool enable) {
        index_->setM2Mode(read_env_int("ANNCHOR_M2_RECOVERY_MODE", 1));
        index_->resetM2Stats();
        index_->setEnableUndoRecovery(enable);
    }
    bool is_undo_recovery_enabled() const {
        return index_->isUndoRecoveryEnabled();
    }

    void set_visibility_mode(int mode) {
        index_->setSearchVisibilityMode(mode);
    }
    int visibility_mode() const { return index_->getSearchVisibilityMode(); }
    void set_visibility_by_label(bool enabled) {
        index_->setSearchVisibilityByLabel(enabled);
    }

    void set_direction_capsule(bool enable, size_t bits = 4, size_t route_r = 4) {
        index_->setDirectionCapsuleParams(bits, route_r);
        index_->setEnableDirectionCapsule(enable);
    }
    
    size_t compact(long safe_ts = -1) { return index_->compact(safe_ts); }

    bool supports_snapshot() const override { return true; }

    int snapshot(std::vector<uint8_t>& out) override {
        std::string tmp_file = "/tmp/annchor_m1_snapshot_" + std::to_string((uintptr_t)this) + ".bin";
        try {
            index_->saveIndex(tmp_file);
        } catch (...) {
            return -1;
        }

        std::ifstream file(tmp_file, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return -1;
        
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        out.resize(size);
        if (!file.read((char*)out.data(), size)) {
            return -1;
        }
        file.close();
        std::remove(tmp_file.c_str());
        return 0;
    }

    int restore(const uint8_t* data, size_t size) override {
        std::string tmp_file = "/tmp/annchor_m1_restore_" + std::to_string((uintptr_t)this) + ".bin";
        std::ofstream file(tmp_file, std::ios::binary);
        if (!file.write((const char*)data, size)) {
            return -1;
        }
        file.close();

        try {
            delete index_;
            index_ = new annchor_m1::HierarchicalNSW<T>(
                space_.get(), tmp_file, false, max_elements_, false); 
            index_->setEnableDirectionCapsule(true);
        } catch (...) {
            return -1;
        }
        std::remove(tmp_file.c_str());
        return 0;
    }

   private:
    size_t num_threads_;
    size_t dim_;
    size_t M_;
    size_t ef_c_;
    size_t ef_s_;
    size_t max_elements_;
    bool use_node_lock_in_search_;
    MetricType metric_;
    std::unique_ptr<annchor_m1::SpaceInterface<T>> space_;
    annchor_m1::HierarchicalNSW<T>* index_;
    std::atomic<uint64_t> metric_batch_search_calls_{0};
    std::atomic<uint64_t> metric_batch_search_queries_{0};
    std::atomic<uint64_t> metric_batch_search_arena_ns_{0};
    std::atomic<uint64_t> metric_batch_search_searchknn_ns_{0};
    std::atomic<uint64_t> metric_batch_search_result_copy_ns_{0};

    static std::vector<int> read_env_cpu_list(const char* name) {
        std::vector<int> cpus;
        if (name == nullptr) return cpus;
        const char* raw = std::getenv(name);
        if (raw == nullptr || raw[0] == '\0') return cpus;

        std::stringstream ss(raw);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (token.empty()) continue;
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
            if (begin < 0 || finish < 0) continue;
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

    static void pin_current_thread_to_env_cpu(const char* name, size_t ordinal) {
        if (name == nullptr) return;
        const std::vector<int> cpus = read_env_cpu_list(name);
        if (cpus.empty()) return;
        const int cpu = cpus[ordinal % cpus.size()];
        thread_local int current_cpu = -1;
        if (current_cpu == cpu) return;

        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) == 0) {
            current_cpu = cpu;
        }
    }

    static bool read_env_bool(const char* name, bool default_value) {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') return default_value;
        return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
               value[0] == 'y' || value[0] == 'Y';
    }

    static size_t read_env_size(const char* name, size_t default_value) {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') return default_value;
        char* end = nullptr;
        unsigned long parsed = std::strtoul(value, &end, 10);
        if (end == value || parsed == 0) return default_value;
        return static_cast<size_t>(parsed);
    }

    static int read_env_int(const char* name, int default_value) {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') return default_value;
        char* end = nullptr;
        long parsed = std::strtol(value, &end, 10);
        if (end == value) return default_value;
        return static_cast<int>(parsed);
    }

    tbb::task_arena& search_task_arena() {
        return separate_arenas_ ? search_arena_ : arena_;
    }

    tbb::task_arena& insert_task_arena() {
        return separate_arenas_ ? insert_arena_ : arena_;
    }

    tbb::task_arena& measured_search_task_arena() {
        return measured_search_arena_enabled_ ? measured_search_arena_ : search_task_arena();
    }

    void normalize_vector(T* data, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            T* vec = data + i * dim_;
            float norm = 0;
            for (size_t j = 0; j < dim_; ++j) norm += vec[j] * vec[j];
            norm = 1.0f / (sqrt(norm) + 1e-30f);
            for (size_t j = 0; j < dim_; ++j) vec[j] *= norm;
        }
    }
    tbb::task_arena arena_;
    bool separate_arenas_;
    tbb::task_arena search_arena_;
    tbb::task_arena insert_arena_;
    bool measured_search_arena_enabled_;
    tbb::task_arena measured_search_arena_;
};
