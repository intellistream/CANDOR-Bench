#pragma once

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>
#include <mutex>
#include <queue>
#include <sched.h>
#include <shared_mutex>
#include <sstream>
#include <time.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "../index.hpp"
#include "../index_cgo.hpp"
#define annchor annchor_m2
#include "ANNchor/src/hnswlib/hnswlib.h"
#undef annchor
#include <memory>
#include <cmath>

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class ANNchorM2 : public IndexBase<T, TagT, LabelT> {
   public:
    ANNchorM2(size_t max_elements, size_t dim, size_t num_threads, size_t M,
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
          measured_search_arena_(read_env_size("ANNCHOR_MEASURED_SEARCH_ARENA_THREADS", num_threads)),
          label_to_id_cache_(max_elements) {
        fresh_early_abort_l2_ =
            read_env_bool("ANNCHOR_M2_EARLY_ABORT_L2", false);
        fresh_early_abort_interval_ =
            read_env_size("ANNCHOR_M2_EARLY_ABORT_INTERVAL", 32);
        if (metric_ == METRIC_IP || metric_ == METRIC_COSINE) {
            space_.reset(new annchor_m2::InnerProductSpace(dim));
        } else {
            space_.reset(new annchor_m2::L2Space(dim));
        }
        index_ = new annchor_m2::HierarchicalNSW<T>(
            space_.get(), max_elements, M, ef_construction, 100, false, use_node_lock_in_search, num_threads_);
        reset_label_cache();
    }

    ~ANNchorM2() override {
        delete index_;
    }

    void build(const T* data, const TagT* tags, size_t num_points) override {
        reset_graph_regions();
        reset_label_cache();
        (void)batch_insert(data, tags, num_points);
        prebuild_graph_regions(num_points);
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
        struct ActiveBatchGuard {
            ANNchorM2* owner;
            size_t id{0};
            ~ActiveBatchGuard() {
                if (owner && id != 0) owner->unregister_active_insert_batch(id);
            }
        } active_batch_guard{this, 0};

        if (metric_ == METRIC_COSINE) {
	             std::vector<T> temp_batch(batch_data, batch_data + num_points * dim_);
	             normalize_vector(temp_batch.data(), num_points);
	             if (mvcc) {
	                 active_batch_guard.id = register_active_insert_batch(
	                     temp_batch.data(), batch_tags, num_points);
	             }
	             insert_task_arena().execute([&] {
	                 tbb::parallel_for(size_t(0), num_points, [&](size_t i) {
	                     pin_current_thread_to_env_cpu("ANNCHOR_INSERT_CPU_LIST", i);
	                     const T* point = temp_batch.data() + i * dim_;
	                     auto id = index_->addPoint(
	                         point, static_cast<annchor_m2::labeltype>(batch_tags[i]), -1, mvcc);
                     cache_label_id(batch_tags[i], id);
                     if (mvcc) index_->markReady(id);
                 });
             });
	        } else {
	             if (mvcc) {
	                 active_batch_guard.id = register_active_insert_batch(
	                     batch_data, batch_tags, num_points);
	             }
	             insert_task_arena().execute([&] {
	                 tbb::parallel_for(size_t(0), num_points, [&](size_t i) {
	                     pin_current_thread_to_env_cpu("ANNCHOR_INSERT_CPU_LIST", i);
	                     const T* point = batch_data + i * dim_;
	                     auto id = index_->addPoint(
	                         point, static_cast<annchor_m2::labeltype>(batch_tags[i]), -1, mvcc);
                     cache_label_id(batch_tags[i], id);
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
        std::priority_queue<std::pair<T, annchor_m2::labeltype>> result;
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
	            "ANNCHOR_MEASURED_SEARCH_CPU_LIST", false);
	    }

    int batch_search_path_work(const T* batch_queries, size_t k,
                               size_t num_queries, TagT** batch_results,
                               SearchWorkStats* per_query_stats,
                               size_t* watermark_out = nullptr,
                               size_t visible_ts = std::numeric_limits<size_t>::max()) override {
	        return batch_search_with_arena(
	            measured_search_task_arena(), batch_queries, k, num_queries,
	            batch_results, watermark_out, visible_ts, per_query_stats,
	            "ANNCHOR_MEASURED_SEARCH_CPU_LIST", true);
	    }

    int batch_search_with_arena(tbb::task_arena& task_arena,
                                const T* batch_queries, size_t k,
	                                size_t num_queries, TagT** batch_results,
	                                size_t* watermark_out,
	                                size_t visible_ts,
	                                SearchWorkStats* per_query_stats = nullptr,
	                                const char* cpu_list_env = nullptr,
	                                bool capture_path = false) {
        size_t wm = visible_ts;
        if (watermark_out) *watermark_out = wm;

        const auto arena_start = std::chrono::steady_clock::now();
        std::atomic<uint64_t> local_searchknn_ns{0};
        std::atomic<uint64_t> local_result_copy_ns{0};
        auto run_one_query = [&](size_t i) {
	                pin_current_thread_to_env_cpu(cpu_list_env, i);
	                const T* q = batch_queries + i * dim_;
                std::priority_queue<std::pair<T, annchor_m2::labeltype>> result;

                const auto searchknn_start = std::chrono::steady_clock::now();
	                struct timespec cpu_start {};
	                if (per_query_stats) {
	                    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_start);
	                }
	                const int work_start_cpu =
	                    per_query_stats ? sched_getcpu() : -1;
	                annchor_m2::SearchWorkStats query_stats{};
	                query_stats.capture_path = capture_path;
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
                    per_query_stats[i].path_count = std::min<uint32_t>(
                        query_stats.path_count,
                        static_cast<uint32_t>(SEARCH_PATH_CAPTURE_CAP));
                    for (uint32_t pj = 0; pj < per_query_stats[i].path_count;
                         ++pj) {
                        per_query_stats[i].path_labels[pj] =
                            query_stats.path_labels[pj];
                        per_query_stats[i].path_dists[pj] =
                            query_stats.path_dists[pj];
                    }
                }
        };
        if (read_env_bool("ANNCHOR_M2_SEARCH_BATCH_SERIAL", false)) {
            for (size_t i = 0; i < num_queries; ++i) {
                run_one_query(i);
            }
        } else {
            task_arena.execute([&] {
                tbb::parallel_for(size_t(0), num_queries, [&](size_t i) {
                    run_one_query(i);
                });
            });
        }
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

    size_t get_inflight_labels(TagT* out, size_t max_labels) const override {
        if (!out || max_labels == 0) return 0;
        return snapshot_active_insert_batch_labels(
            out, max_labels, std::numeric_limits<uint64_t>::max());
    }

    size_t get_inflight_labels_before(TagT* out, size_t max_labels,
                                      uint64_t snapshot_raw_ns) const override {
        if (!out || max_labels == 0) return 0;
        return snapshot_active_insert_batch_labels(out, max_labels,
                                                  snapshot_raw_ns);
    }

    int batch_base_neighbors(const TagT* labels, size_t num_labels,
                             size_t max_neighbors, TagT* out,
                             size_t* counts) const override {
        if (!labels || !out || !counts || max_neighbors == 0) return -1;

        for (size_t i = 0; i < num_labels; ++i) {
            counts[i] = 0;
            annchor_m2::tableint internal_id = 0;
            {
                std::unique_lock<std::mutex> lock_table(index_->label_lookup_lock);
                auto it = index_->label_lookup_.find(
                    static_cast<annchor_m2::labeltype>(labels[i]));
                if (it == index_->label_lookup_.end()) continue;
                internal_id = it->second;
            }
            if (internal_id >= index_->getCurrentElementCount()) continue;
            auto* data = index_->get_linklist0(internal_id);
            const size_t sz = index_->getListCount(data);
            auto* datal = reinterpret_cast<annchor_m2::tableint*>(data + 1);
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
                annchor_m2::tableint internal_id = 0;
                {
                    std::unique_lock<std::mutex> lock_table(index_->label_lookup_lock);
                    auto it = index_->label_lookup_.find(
                        static_cast<annchor_m2::labeltype>(labels[i]));
                    if (it == index_->label_lookup_.end()) continue;
                    internal_id = it->second;
                }
                if (internal_id >= index_->getCurrentElementCount()) continue;

                auto* linklist = index_->get_linklist0(internal_id);
                const size_t sz = index_->getListCount(linklist);
                const size_t take = max_edges > 0 ? std::min(max_edges, sz) : sz;

                auto* count_ptr =
                    reinterpret_cast<volatile annchor_m2::linklistsizeint*>(
                        linklist);
                const auto count_value = *count_ptr;
                *count_ptr = count_value;
                writes++;

                auto* data =
                    reinterpret_cast<annchor_m2::tableint*>(linklist + 1);
                for (size_t j = 0; j < take; ++j) {
                    auto* ptr =
                        reinterpret_cast<volatile annchor_m2::tableint*>(
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
        size_t graph_region_count = 0;
        size_t graph_region_built_limit = 0;
        size_t graph_region_max_size = 0;
        size_t graph_region_nonempty = 0;
        size_t graph_region_total_size = 0;
        size_t graph_region_edge_anchors = 0;
        {
            std::shared_lock<std::shared_mutex> guard(region_mu_);
            graph_region_count = region_sizes_.size();
            graph_region_built_limit = region_built_limit_;
            for (size_t sz : region_sizes_) {
                graph_region_total_size += sz;
                if (sz > 0) graph_region_nonempty++;
                if (sz > graph_region_max_size) graph_region_max_size = sz;
            }
            for (const auto& anchors : region_edge_anchors_) {
                graph_region_edge_anchors += anchors.size();
            }
        }
        std::stringstream ss;
        ss << "index_memory_mb:" << index_->indexFileSize() / (1024 * 1024)
           << ", graph_region_count:" << graph_region_count
           << ", graph_region_nonempty:" << graph_region_nonempty
           << ", graph_region_built_limit:" << graph_region_built_limit
           << ", graph_region_total_size:" << graph_region_total_size
           << ", graph_region_max_size:" << graph_region_max_size
           << ", graph_region_cap:" << region_cap_config_
           << ", graph_region_overlap:" << region_overlap_config_
           << ", graph_region_edge_anchor_k:" << region_edge_anchor_config_
           << ", graph_region_edge_anchors:" << graph_region_edge_anchors
           << ", graph_region_max:" << region_max_limit(region_cap_config_)
           << ", graph_region_assign_edge_evidence:" << metric_m2_region_assign_evidence_.load()
           << ", graph_region_assign_local_fallback:" << metric_m2_region_assign_local_fallback_.load()
           << ", graph_region_open_no_candidate:" << metric_m2_region_open_no_candidate_.load()
           << ", graph_region_open_full:" << metric_m2_region_open_full_.load()
           << ", graph_region_assign_forced:" << metric_m2_region_assign_forced_.load()
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
	           << ", m2_micro_regions_built:" << metric_m2_micro_regions_built_.load()
	           << ", m2_micro_region_entries:" << metric_m2_micro_region_entries_.load()
	           << ", m2_micro_regions_opened:" << metric_m2_micro_regions_opened_.load()
	           << ", m2_micro_regions_pruned:" << metric_m2_micro_regions_pruned_.load()
	           << ", m2_member_bound_checks:" << metric_m2_member_bound_checks_.load()
	           << ", m2_member_bound_skips:" << metric_m2_member_bound_skips_.load()
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

    // M2 keeps the M1 visibility boundary and adds active-batch fresh
    // validation. Published fresh points are filtered by graph regions; points
    // from insert batches that were already active when the query entered are
    // read directly from the saved batch vectors below.
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

    void set_enable_prune_only_node_lock(bool enable) {
        (void)enable;
    }

    void set_enable_prune_snapshot_delegation(bool enable) {
        (void)enable;
    }

    size_t dim() const { return dim_; }

    int fresh_merge_graph_region(const T* batch_queries, size_t k,
                                 size_t num_queries, const T* fresh_data,
                                 const TagT* fresh_tags,
                                 size_t fresh_count,
                                 const TagT* committed_results,
                                 TagT** batch_results,
                                 size_t committed_view_limit,
                                 const C_FreshJoinParams& params,
                                 C_FreshJoinStats* stats) {
        if (fresh_count > 0 && !fresh_data) return -1;
        auto fresh_ptr_at = [this, fresh_data](size_t fi) -> const T* {
            return fresh_data + fi * dim_;
        };
        return fresh_merge_graph_region_impl(
            batch_queries, k, num_queries, fresh_tags, fresh_count,
            fresh_ptr_at, committed_results, batch_results,
            committed_view_limit, params, stats);
    }

    template <typename FreshPtrAt>
    int fresh_merge_graph_region_impl(const T* batch_queries, size_t k,
                                 size_t num_queries,
                                 const TagT* fresh_tags,
                                 size_t fresh_count,
                                 FreshPtrAt fresh_ptr_at,
                                 const TagT* committed_results,
                                 TagT** batch_results,
                                 size_t committed_view_limit,
                                 const C_FreshJoinParams& params,
                                 C_FreshJoinStats* stats) {
        if (!batch_queries || !committed_results || !batch_results) return -1;
        if (fresh_count > 0 && !fresh_tags) return -1;
        if (k == 0 || num_queries == 0) return 0;

        if (stats) {
            stats->queries = num_queries;
            stats->fresh_vectors = fresh_count;
            stats->active_placement_centers = 0;
            stats->active_placement_regions = 0;
            stats->selected_candidates = 0;
            stats->exacted_candidates = 0;
            stats->exact_fallback_queries = 0;
            stats->floor_fallback_queries = 0;
        }

        const size_t current_count = index_->getCurrentElementCount();
        size_t committed_limit =
            committed_view_limit == 0
                ? current_count
                : std::min(committed_view_limit, current_count);
        if (fresh_count == 0) {
            copy_committed_results(k, num_queries, committed_results,
                                   batch_results);
            return 0;
        }
        if (committed_limit == 0) {
            return exact_fresh_merge_by_accessor(
                batch_queries, k, num_queries, fresh_tags, fresh_count,
                fresh_ptr_at, committed_results, batch_results, stats, true);
        }

        const size_t default_overlap = dim_ <= 512 ? 1 : 2;
        const size_t overlap_budget =
            std::min<size_t>(read_env_size_allow_zero(
                                 "ANNCHOR_M2_REGION_OVERLAP",
                                 default_overlap),
                             8);
        const size_t edge_anchor_k =
            std::min<size_t>(read_env_size_allow_zero(
                                 "ANNCHOR_M2_EDGE_ANCHOR_K", 2),
                             32);
        const size_t region_halo =
            std::min<size_t>(read_env_size_allow_zero(
                                 "ANNCHOR_M2_REGION_HALO", 0),
                             16);
        const size_t region_cap =
            read_env_size("ANNCHOR_M2_REGION_CAP", 512);
        const size_t micro_region_cap =
            std::max<size_t>(
                1, std::min<size_t>(
                       region_cap,
                       read_env_size("ANNCHOR_M2_MICRO_REGION_CAP",
                                     dim_ <= 512 ? 64 : 96)));
        const size_t small_threshold =
            params.small_fresh_threshold == 0 ? 200 : params.small_fresh_threshold;
        const bool use_micro_postings =
            read_env_bool("ANNCHOR_M2_USE_MICRO_POSTINGS", dim_ <= 1024);
        // Cover-extension hysteresis: the committed region cover is extended
        // under a unique lock on the query path. Committed count grows with
        // every insert, so without slack almost every merge would take that
        // exclusive lock to add a few hundred nodes, serializing merges and
        // inflating search p99. Allow the cover to lag committed_limit by up to
        // this slack (lagging nodes fall back to insert-evidence regions), so
        // the exclusive rebuild fires only once per chunk.
        const size_t cover_lag_slack =
            read_env_size("ANNCHOR_M2_COVER_LAG_SLACK", 8192);

        std::vector<RegionKeySet> fresh_regions;
        std::vector<RegionKeySet> query_regions;
        FreshRegionPostings fresh_postings;
        std::vector<uint32_t> candidate_marks;
        uint32_t candidate_mark = 1;
        auto fill_region_stats = [&]() {
            if (stats) {
                stats->active_placement_centers = region_sizes_.size();
                stats->active_placement_regions = region_sizes_.size();
            }
        };
        auto compute_region_keys = [&]() {
            fresh_regions.resize(fresh_count);
            for (size_t fi = 0; fi < fresh_count; ++fi) {
                fresh_regions[fi] = fresh_region_keys_locked(
                    fresh_ptr_at(fi), fresh_tags[fi], committed_limit,
                    overlap_budget);
            }
            query_regions.resize(num_queries);
            for (size_t qi = 0; qi < num_queries; ++qi) {
                query_regions[qi] = query_region_keys_locked(
                    batch_queries + qi * dim_, committed_results + qi * k, k,
                    params, committed_limit, overlap_budget);
            }
        };
        if (fresh_count > small_threshold) {
            bool region_ready = false;
            bool empty_regions = false;
            {
                std::shared_lock<std::shared_mutex> guard(region_mu_);
                region_ready =
                    region_cap_config_ == region_cap &&
                    region_overlap_config_ == overlap_budget &&
                    region_edge_anchor_config_ == edge_anchor_k &&
                    region_halo_config_ == region_halo &&
                    region_built_limit_ > 0 &&
                    committed_limit <= region_built_limit_ + cover_lag_slack;
                if (region_ready) {
                    fill_region_stats();
                    empty_regions = region_sizes_.empty();
                    if (!empty_regions) compute_region_keys();
                }
            }

            if (!region_ready) {
                std::unique_lock<std::shared_mutex> guard(region_mu_);
                ensure_graph_regions_locked(committed_limit, region_cap,
                                            overlap_budget, edge_anchor_k,
                                            region_halo);
                fill_region_stats();
                empty_regions = region_sizes_.empty();
                guard.unlock();

                if (!empty_regions) {
                    std::shared_lock<std::shared_mutex> read_guard(region_mu_);
                    compute_region_keys();
                }
            }

            if (empty_regions) {
                return exact_fresh_merge_by_accessor(
                    batch_queries, k, num_queries, fresh_tags, fresh_count,
                    fresh_ptr_at, committed_results, batch_results, stats,
                    true);
            }
        }
        if (fresh_count > small_threshold && use_micro_postings) {
            fresh_postings = build_fresh_region_postings(
                fresh_count, fresh_regions, fresh_ptr_at, micro_region_cap);
            if (stats) {
                stats->active_placement_regions =
                    fresh_postings.micro_regions.size();
            }
            metric_m2_micro_regions_built_.fetch_add(
                fresh_postings.micro_regions.size(), std::memory_order_relaxed);
            metric_m2_micro_region_entries_.fetch_add(
                fresh_postings.entries, std::memory_order_relaxed);
            if (fresh_postings.enabled) {
                candidate_marks.assign(fresh_count, 0);
            }
        } else if (fresh_count > small_threshold) {
            if (stats) {
                stats->active_placement_regions = region_sizes_.size();
            }
        }

        for (size_t qi = 0; qi < num_queries; ++qi) {
            const T* query = batch_queries + qi * dim_;
            std::vector<ScoredTag> top =
                committed_topk(query, committed_results + qi * k, k);
            bool exact_all = fresh_count <= small_threshold;
            if (!exact_all) {
                exact_all = query_regions[qi].regions.empty();
            }

            size_t selected = 0;
            if (exact_all) {
                if (stats) stats->exact_fallback_queries++;
                for (size_t fi = 0; fi < fresh_count; ++fi) {
                    exact_admit(query, fresh_ptr_at(fi), fresh_tags[fi], k,
                                top);
                }
                selected = fresh_count;
            } else {
                const size_t floor_candidates =
                    params.candidate_floor_pct > 0.0
                        ? static_cast<size_t>(
                              std::ceil(params.candidate_floor_pct *
                                        static_cast<double>(fresh_count)))
                        : 0;
                std::vector<size_t> candidates;
                candidates.reserve(std::min(fresh_count, size_t{64}));
                const bool use_marks = !candidate_marks.empty();
                if (use_marks) {
                    reset_candidate_mark(candidate_marks, candidate_mark);
                }
                if (fresh_postings.enabled && use_marks) {
                    collect_posting_candidates(
                        query, query_regions[qi], fresh_postings, fresh_ptr_at,
                        top, k, candidates, candidate_marks, candidate_mark);
                } else {
                    for (size_t fi = 0; fi < fresh_count; ++fi) {
                        if (fresh_regions[fi].regions.empty() ||
                            regions_intersect(query_regions[qi], fresh_regions[fi])) {
                            if (use_marks) {
                                add_marked_candidate(candidates, candidate_marks,
                                                     candidate_mark, fi);
                            } else {
                                candidates.push_back(fi);
                            }
                        }
                    }
                }
                if (floor_candidates > 0 && candidates.size() < floor_candidates) {
                    if (stats) {
                        stats->floor_fallback_queries++;
                        stats->exact_fallback_queries++;
                    }
                    for (size_t fi = 0; fi < fresh_count; ++fi) {
                        exact_admit(query, fresh_ptr_at(fi), fresh_tags[fi], k,
                                    top);
                    }
                    selected = fresh_count;
                } else {
                    selected = candidates.size();
                    for (size_t fi : candidates) {
                        exact_admit(query, fresh_ptr_at(fi), fresh_tags[fi], k,
                                    top);
                    }
                }
            }

            if (stats) {
                stats->selected_candidates += selected;
                stats->exacted_candidates += selected;
            }
            std::sort(top.begin(), top.end(), [](const ScoredTag& a,
                                                 const ScoredTag& b) {
                if (a.dist != b.dist) return a.dist < b.dist;
                return a.tag < b.tag;
            });
            for (size_t j = 0; j < k; ++j) {
                batch_results[qi][j] =
                    j < top.size() ? top[j].tag
                                   : std::numeric_limits<TagT>::max();
            }
        }
        return 0;
    }

    int fresh_merge_graph_region_labels(const T* batch_queries, size_t k,
                                        size_t num_queries,
                                        const TagT* fresh_tags,
                                        size_t fresh_count,
                                        const TagT* committed_results,
                                        TagT** batch_results,
                                        const C_FreshJoinParams& params,
                                        C_FreshJoinStats* stats) {
        if (!batch_queries || !committed_results || !batch_results) return -1;
        if (fresh_count > 0 && !fresh_tags) return -1;
        if (k == 0 || num_queries == 0) return 0;

        struct ActiveVectorRef {
            std::shared_ptr<const std::vector<T>> vectors;
            size_t offset{0};
        };

        std::unordered_set<TagT> requested_tags;
        requested_tags.reserve(fresh_count * 2 + 1);
        for (size_t i = 0; i < fresh_count; ++i) {
            requested_tags.insert(fresh_tags[i]);
        }

        std::unordered_map<TagT, ActiveVectorRef> active_vectors;
        active_vectors.reserve(requested_tags.size());
        {
            std::lock_guard<std::mutex> guard(active_insert_batch_mu_);
            auto add_batch = [&](const ActiveInsertBatch& batch) {
                if (!batch.vectors) return;
                for (size_t i = 0; i < batch.labels.size(); ++i) {
                    const TagT label = batch.labels[i];
                    if (requested_tags.find(label) == requested_tags.end()) {
                        continue;
                    }
                    if (active_vectors.find(label) != active_vectors.end()) {
                        continue;
                    }
                    const size_t offset = i * dim_;
                    if (offset + dim_ > batch.vectors->size()) continue;
                    active_vectors.emplace(
                        label, ActiveVectorRef{batch.vectors, offset});
                }
            };
            for (const auto& batch : active_insert_batches_) add_batch(batch);
            for (const auto& batch : retired_insert_batches_) add_batch(batch);
        }

        const size_t current_count = index_->getCurrentElementCount();
        std::vector<TagT> filtered_tags;
        std::vector<const T*> fresh_ptrs;
        filtered_tags.reserve(fresh_count);
        fresh_ptrs.reserve(fresh_count);
        std::unordered_set<TagT> seen;
        seen.reserve(fresh_count * 2 + 1);
        size_t committed_view_limit = current_count;

        for (size_t i = 0; i < fresh_count; ++i) {
            const TagT tag = fresh_tags[i];
            if (!seen.insert(tag).second) continue;

            auto active_it = active_vectors.find(tag);
            if (active_it != active_vectors.end()) {
                filtered_tags.push_back(tag);
                fresh_ptrs.push_back(
                    active_it->second.vectors->data() +
                    active_it->second.offset);
                committed_view_limit = std::min(
                    committed_view_limit, static_cast<size_t>(tag));
                continue;
            }

            annchor_m2::tableint id = 0;
            if (!label_to_internal(tag, id) || id >= current_count) continue;
            filtered_tags.push_back(tag);
            fresh_ptrs.push_back(data_by_id(id));
            committed_view_limit =
                std::min(committed_view_limit, static_cast<size_t>(id));
        }

        if (filtered_tags.empty()) {
            copy_committed_results(k, num_queries, committed_results,
                                   batch_results);
            if (stats) {
                stats->queries = num_queries;
                stats->fresh_vectors = 0;
                stats->active_placement_centers = 0;
                stats->active_placement_regions = 0;
                stats->selected_candidates = 0;
                stats->exacted_candidates = 0;
                stats->exact_fallback_queries = 0;
                stats->floor_fallback_queries = 0;
            }
            return 0;
        }

        auto fresh_ptr_at = [&fresh_ptrs](size_t fi) -> const T* {
            return fresh_ptrs[fi];
        };
        return fresh_merge_graph_region_impl(
            batch_queries, k, num_queries, filtered_tags.data(),
            filtered_tags.size(), fresh_ptr_at, committed_results,
            batch_results, committed_view_limit, params, stats);
    }
    
    size_t compact(long safe_ts = -1) { return index_->compact(safe_ts); }

    bool supports_snapshot() const override { return true; }

    int snapshot(std::vector<uint8_t>& out) override {
        std::string tmp_file = "/tmp/annchor_m2_snapshot_" + std::to_string((uintptr_t)this) + ".bin";
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
        std::string tmp_file = "/tmp/annchor_m2_restore_" + std::to_string((uintptr_t)this) + ".bin";
        std::ofstream file(tmp_file, std::ios::binary);
        if (!file.write((const char*)data, size)) {
            return -1;
        }
        file.close();

        try {
            delete index_;
            index_ = new annchor_m2::HierarchicalNSW<T>(
                space_.get(), tmp_file, false, max_elements_, false); 
            reset_label_cache();
            rebuild_label_cache_from_lookup();
            reset_graph_regions();
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
    std::unique_ptr<annchor_m2::SpaceInterface<T>> space_;
    annchor_m2::HierarchicalNSW<T>* index_;
    std::atomic<uint64_t> metric_batch_search_calls_{0};
    std::atomic<uint64_t> metric_batch_search_queries_{0};
    std::atomic<uint64_t> metric_batch_search_arena_ns_{0};
    std::atomic<uint64_t> metric_batch_search_searchknn_ns_{0};
    std::atomic<uint64_t> metric_batch_search_result_copy_ns_{0};
    mutable std::atomic<uint64_t> metric_m2_micro_regions_built_{0};
    mutable std::atomic<uint64_t> metric_m2_micro_region_entries_{0};
    mutable std::atomic<uint64_t> metric_m2_micro_regions_opened_{0};
    mutable std::atomic<uint64_t> metric_m2_micro_regions_pruned_{0};
    mutable std::atomic<uint64_t> metric_m2_member_bound_checks_{0};
    mutable std::atomic<uint64_t> metric_m2_member_bound_skips_{0};
    mutable std::atomic<uint64_t> metric_m2_region_assign_evidence_{0};
    mutable std::atomic<uint64_t> metric_m2_region_assign_local_fallback_{0};
    mutable std::atomic<uint64_t> metric_m2_region_open_no_candidate_{0};
    mutable std::atomic<uint64_t> metric_m2_region_open_full_{0};
    mutable std::atomic<uint64_t> metric_m2_region_assign_forced_{0};
    std::vector<std::atomic<uint32_t>> label_to_id_cache_;
    static constexpr uint32_t kInvalidCachedId =
        std::numeric_limits<uint32_t>::max();

    size_t region_max_limit(size_t region_cap) const {
        const size_t safe_cap = std::max<size_t>(region_cap, 1);
        const size_t region_span = std::max<size_t>(safe_cap / 2, 1);
        const size_t current = std::max<size_t>(index_->getCurrentElementCount(), 1);
        // Region ids are a lightweight graph cover, not a fixed partition.  The
        // budget grows with the committed graph so dense areas can split when
        // local candidate regions saturate, while sparse areas keep reusing
        // under-filled regions.
        const size_t adaptive =
            std::max<size_t>(4096, (current + region_span - 1) / region_span);
        const size_t bounded = std::min<size_t>(adaptive, 65536);
        return read_env_size("ANNCHOR_M2_REGION_MAX", bounded);
    }

    static std::vector<int> read_env_cpu_list(const char* name) {
        std::vector<int> cpus;
        if (name == nullptr) return cpus;
        const char* raw = std::getenv(name);
        if (raw == nullptr || raw[0] == '\0') return cpus;

        std::stringstream ss(raw);
        std::string cpu_range;
        while (std::getline(ss, cpu_range, ',')) {
            if (cpu_range.empty()) continue;
            auto dash = cpu_range.find('-');
            if (dash == std::string::npos) {
                char* end = nullptr;
                long cpu = std::strtol(cpu_range.c_str(), &end, 10);
                if (end != cpu_range.c_str() && cpu >= 0) {
                    cpus.push_back(static_cast<int>(cpu));
                }
                continue;
            }
            const std::string begin_text = cpu_range.substr(0, dash);
            const std::string end_text = cpu_range.substr(dash + 1);
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

    static size_t read_env_size_allow_zero(const char* name,
                                           size_t default_value) {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') return default_value;
        char* end = nullptr;
        unsigned long parsed = std::strtoul(value, &end, 10);
        if (end == value) return default_value;
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

    void reset_graph_regions() {
        std::unique_lock<std::shared_mutex> guard(region_mu_);
        region_built_limit_ = 0;
        region_cap_config_ = 0;
        region_overlap_config_ = 0;
        region_edge_anchor_config_ = 0;
        region_halo_config_ = 0;
        node_regions_.clear();
        region_sizes_.clear();
        region_edge_anchors_.clear();
    }

    void prebuild_graph_regions(size_t committed_limit) const {
        if (committed_limit == 0) return;
        const size_t overlap_budget =
            std::min<size_t>(read_env_size_allow_zero(
                                 "ANNCHOR_M2_REGION_OVERLAP",
                                 dim_ <= 512 ? 1 : 2),
                             8);
        const size_t edge_anchor_k =
            std::min<size_t>(read_env_size_allow_zero(
                                 "ANNCHOR_M2_EDGE_ANCHOR_K", 2),
                             32);
        const size_t region_halo =
            std::min<size_t>(read_env_size_allow_zero(
                                 "ANNCHOR_M2_REGION_HALO", 0),
                             16);
        const size_t region_cap =
            read_env_size("ANNCHOR_M2_REGION_CAP", 512);
        std::unique_lock<std::shared_mutex> guard(region_mu_);
        ensure_graph_regions_locked(
            std::min(committed_limit, index_->getCurrentElementCount()),
            region_cap, overlap_budget, edge_anchor_k, region_halo);
    }

    struct RegionMeta {
        uint32_t primary{std::numeric_limits<uint32_t>::max()};
        std::vector<uint32_t> overlap;
        std::vector<uint32_t> halo;
    };

    struct EdgeAnchor {
        annchor_m2::tableint outside{0};
        annchor_m2::tableint inside{0};
        T len{0};
    };

    struct RegionCandidate {
        uint32_t region{std::numeric_limits<uint32_t>::max()};
        bool evidence_valid{false};
        bool edge_valid{false};
        bool primary_evidence{false};
        T best_neighbor_dist{std::numeric_limits<T>::max()};
        T edge_inside_dist{std::numeric_limits<T>::max()};
    };

    struct RegionKeySet {
        static constexpr uint32_t kBitsetBits = 4096;
        static constexpr size_t kBitsetWords = kBitsetBits / 64;
        std::vector<uint32_t> regions;
        std::array<uint64_t, kBitsetWords> bits{};
        uint16_t words{0};
        bool overflow{false};
    };

    struct FreshMicroRegion {
        size_t anchor{0};
        std::vector<size_t> posting;
        std::vector<T> radial;
        T radius{0};
    };

    struct FreshRegionPostings {
        bool enabled{false};
        std::vector<std::vector<size_t>> by_region;
        std::vector<FreshMicroRegion> micro_regions;
        std::vector<size_t> unpublished;
        size_t entries{0};
    };

    struct ScoredTag {
        TagT tag{0};
        T dist{0};
    };

    T dist_ptr(const T* lhs, const T* rhs) const {
        return index_->fstdistfunc_(lhs, rhs, index_->dist_func_param_);
    }

    T fresh_dist_for_admit(const T* query, const T* fresh, T cutoff,
                           bool has_cutoff) const {
        if (metric_ != METRIC_L2 || !fresh_early_abort_l2_ || !has_cutoff ||
            cutoff == std::numeric_limits<T>::max()) {
            return dist_ptr(query, fresh);
        }
        T acc = 0;
        const size_t interval = std::max<size_t>(fresh_early_abort_interval_, 1);
        size_t j = 0;
        for (; j + interval <= dim_; j += interval) {
            T block = 0;
            for (size_t e = 0; e < interval; ++e) {
                const T diff = query[j + e] - fresh[j + e];
                block += diff * diff;
            }
            acc += block;
            if (acc > cutoff) return acc;
        }
        for (; j < dim_; ++j) {
            const T diff = query[j] - fresh[j];
            acc += diff * diff;
        }
        return acc;
    }

    const T* data_by_id(annchor_m2::tableint id) const {
        return reinterpret_cast<const T*>(index_->getDataByInternalId(id));
    }

    void reset_label_cache() {
        for (auto& cached : label_to_id_cache_) {
            cached.store(kInvalidCachedId, std::memory_order_relaxed);
        }
    }

    void cache_label_id(TagT label, annchor_m2::tableint id) {
        const uint64_t idx = static_cast<uint64_t>(label);
        if (idx >= label_to_id_cache_.size()) return;
        label_to_id_cache_[idx].store(static_cast<uint32_t>(id),
                                      std::memory_order_release);
    }

    void rebuild_label_cache_from_lookup() {
        std::unique_lock<std::mutex> lock_table(index_->label_lookup_lock);
        for (const auto& entry : index_->label_lookup_) {
            const uint64_t label = static_cast<uint64_t>(entry.first);
            if (label >= label_to_id_cache_.size()) continue;
            label_to_id_cache_[label].store(static_cast<uint32_t>(entry.second),
                                            std::memory_order_release);
        }
    }

    bool label_to_internal(TagT label, annchor_m2::tableint& out) const {
        const uint64_t idx = static_cast<uint64_t>(label);
        if (idx < label_to_id_cache_.size()) {
            const uint32_t cached =
                label_to_id_cache_[idx].load(std::memory_order_acquire);
            if (cached != kInvalidCachedId &&
                cached < index_->getCurrentElementCount()) {
                out = static_cast<annchor_m2::tableint>(cached);
                return true;
            }
        }
        std::unique_lock<std::mutex> lock_table(index_->label_lookup_lock);
        auto it = index_->label_lookup_.find(
            static_cast<annchor_m2::labeltype>(label));
        if (it == index_->label_lookup_.end()) return false;
        out = it->second;
        return out < index_->getCurrentElementCount();
    }

    void add_region_candidate(std::vector<RegionCandidate>& candidates,
                              const T* x, uint32_t region,
                              bool primary_evidence, T neighbor_dist) const {
        if (region == std::numeric_limits<uint32_t>::max()) return;
        if (region >= region_sizes_.size()) return;
        bool edge_valid = false;
        T edge_inside_dist = std::numeric_limits<T>::max();
        if (region < region_edge_anchors_.size()) {
            for (const EdgeAnchor& edge : region_edge_anchors_[region]) {
                if (edge.inside >= index_->getCurrentElementCount() ||
                    edge.outside >= index_->getCurrentElementCount()) {
                    continue;
                }
                const T inside_dist = dist_ptr(x, data_by_id(edge.inside));
                const T outside_dist = dist_ptr(x, data_by_id(edge.outside));
                if (inside_dist < edge_inside_dist) {
                    edge_inside_dist = inside_dist;
                }
                if (inside_dist < outside_dist) {
                    edge_valid = true;
                }
            }
        }
        const bool evidence_valid = edge_valid;
        for (auto& entry : candidates) {
            if (entry.region == region) {
                entry.evidence_valid = entry.evidence_valid || evidence_valid;
                entry.edge_valid = entry.edge_valid || edge_valid;
                entry.primary_evidence =
                    entry.primary_evidence || primary_evidence;
                if (neighbor_dist < entry.best_neighbor_dist) {
                    entry.best_neighbor_dist = neighbor_dist;
                }
                if (edge_inside_dist < entry.edge_inside_dist) {
                    entry.edge_inside_dist = edge_inside_dist;
                }
                return;
            }
        }
        RegionCandidate entry;
        entry.region = region;
        entry.evidence_valid = evidence_valid;
        entry.edge_valid = edge_valid;
        entry.primary_evidence = primary_evidence;
        entry.best_neighbor_dist = neighbor_dist;
        entry.edge_inside_dist = edge_inside_dist;
        candidates.push_back(entry);
    }

    std::vector<RegionCandidate> neighbor_region_candidates_locked(
        annchor_m2::tableint id, size_t committed_limit) const {
        std::vector<RegionCandidate> candidates;
        if (id >= index_->getCurrentElementCount()) return candidates;
        const T* x = data_by_id(id);

        auto* ll = index_->get_linklist0(id);
        const size_t sz = index_->getListCount(ll);
        auto* neighbors = reinterpret_cast<annchor_m2::tableint*>(ll + 1);
        for (size_t j = 0; j < sz; ++j) {
            const annchor_m2::tableint nb = neighbors[j];
            if (nb >= committed_limit || nb >= node_regions_.size()) continue;
            const RegionMeta& meta = node_regions_[nb];
            if (meta.primary == std::numeric_limits<uint32_t>::max()) continue;
            const T neighbor_dist = dist_ptr(x, data_by_id(nb));
            auto add = [&](uint32_t region, bool primary_evidence) {
                if (region == std::numeric_limits<uint32_t>::max() ||
                    region >= region_sizes_.size()) {
                    return;
                }
                add_region_candidate(candidates, x, region, primary_evidence,
                                     neighbor_dist);
            };
            add(meta.primary, true);
            for (uint32_t r : meta.overlap) {
                add(r, false);
            }
        }
        return candidates;
    }

    static bool add_region_key_unique(RegionKeySet& out, uint32_t region) {
        if (region == std::numeric_limits<uint32_t>::max()) return false;
        for (uint32_t existing : out.regions) {
            if (existing == region) return false;
        }
        out.regions.push_back(region);
        if (region < RegionKeySet::kBitsetBits) {
            const size_t word = region >> 6;
            out.bits[word] |= uint64_t{1} << (region & 63);
            if (out.words < word + 1) {
                out.words = static_cast<uint16_t>(word + 1);
            }
        } else {
            out.overflow = true;
        }
        return true;
    }

    uint32_t open_region_locked() const {
        const uint32_t region = static_cast<uint32_t>(region_sizes_.size());
        region_sizes_.push_back(0);
        region_edge_anchors_.emplace_back();
        return region;
    }

    static void sort_region_candidates(
        std::vector<RegionCandidate>& candidates,
        const std::vector<size_t>& region_sizes) {
        std::sort(candidates.begin(), candidates.end(),
                  [&](const auto& a, const auto& b) {
                      if (a.evidence_valid != b.evidence_valid) {
                          return a.evidence_valid > b.evidence_valid;
                      }
                      if (a.edge_valid != b.edge_valid) {
                          return a.edge_valid > b.edge_valid;
                      }
                      if (a.primary_evidence != b.primary_evidence) {
                          return a.primary_evidence > b.primary_evidence;
                      }
                      if (a.edge_inside_dist != b.edge_inside_dist) {
                          return a.edge_inside_dist < b.edge_inside_dist;
                      }
                      if (a.best_neighbor_dist != b.best_neighbor_dist) {
                          return a.best_neighbor_dist < b.best_neighbor_dist;
                      }
                      const size_t as = a.region < region_sizes.size()
                                            ? region_sizes[a.region]
                                            : std::numeric_limits<size_t>::max();
                      const size_t bs = b.region < region_sizes.size()
                                            ? region_sizes[b.region]
                                            : std::numeric_limits<size_t>::max();
                      if (as != bs) return as < bs;
                      return a.region < b.region;
                  });
    }

    uint32_t choose_primary_region_locked(
        const std::vector<RegionCandidate>& candidates,
        annchor_m2::tableint id, size_t region_cap) const {
        (void)id;
        const size_t region_max = region_max_limit(region_cap);
        if (candidates.empty()) {
            metric_m2_region_open_no_candidate_.fetch_add(
                1, std::memory_order_relaxed);
            if (region_sizes_.empty() || region_sizes_.size() < region_max) {
                return open_region_locked();
            }
            metric_m2_region_assign_forced_.fetch_add(
                1, std::memory_order_relaxed);
            return static_cast<uint32_t>(
                std::min_element(region_sizes_.begin(), region_sizes_.end()) -
                region_sizes_.begin());
        }

        const bool has_evidence = std::any_of(
            candidates.begin(), candidates.end(),
            [](const RegionCandidate& entry) { return entry.evidence_valid; });
        std::vector<RegionCandidate> sorted = candidates;
        if (has_evidence) {
            sort_region_candidates(sorted, region_sizes_);
            for (const auto& entry : sorted) {
                if (!entry.evidence_valid) continue;
                if (entry.region < region_sizes_.size() &&
                    region_sizes_[entry.region] < region_cap) {
                    metric_m2_region_assign_evidence_.fetch_add(
                        1, std::memory_order_relaxed);
                    return entry.region;
                }
            }
            // All graph-supported local regions are full, so open the next
            // region at this dense graph neighborhood instead of widening the
            // existing region globally.
            if (region_sizes_.size() < region_max) {
                metric_m2_region_open_full_.fetch_add(
                    1, std::memory_order_relaxed);
                return open_region_locked();
            }
        } else {
            sort_region_candidates_local_fallback(sorted, region_sizes_);
        }
        for (const auto& entry : sorted) {
            if (entry.region < region_sizes_.size() &&
                region_sizes_[entry.region] < region_cap) {
                metric_m2_region_assign_local_fallback_.fetch_add(
                    1, std::memory_order_relaxed);
                return entry.region;
            }
        }
        if (region_sizes_.size() < region_max) {
            metric_m2_region_open_full_.fetch_add(
                1, std::memory_order_relaxed);
            return open_region_locked();
        }
        metric_m2_region_assign_forced_.fetch_add(
            1, std::memory_order_relaxed);
        return sorted.front().region;
    }

    static void sort_region_candidates_local_fallback(
        std::vector<RegionCandidate>& candidates,
        const std::vector<size_t>& region_sizes) {
        std::sort(candidates.begin(), candidates.end(),
                  [&](const auto& a, const auto& b) {
                      if (a.primary_evidence != b.primary_evidence) {
                          return a.primary_evidence > b.primary_evidence;
                      }
                      if (a.best_neighbor_dist != b.best_neighbor_dist) {
                          return a.best_neighbor_dist < b.best_neighbor_dist;
                      }
                      const size_t as = a.region < region_sizes.size()
                                            ? region_sizes[a.region]
                                            : std::numeric_limits<size_t>::max();
                      const size_t bs = b.region < region_sizes.size()
                                            ? region_sizes[b.region]
                                            : std::numeric_limits<size_t>::max();
                      if (as != bs) return as < bs;
                      return a.region < b.region;
                  });
    }

    RegionKeySet keys_from_candidates_locked(
        const std::vector<RegionCandidate>& candidates,
        size_t overlap_budget) const {
        RegionKeySet out;
        if (candidates.empty()) return out;
        const bool has_evidence = std::any_of(
            candidates.begin(), candidates.end(),
            [](const RegionCandidate& entry) { return entry.evidence_valid; });
        if (!has_evidence) return out;
        std::vector<RegionCandidate> sorted = candidates;
        sort_region_candidates(sorted, region_sizes_);
        out.regions.reserve(1 + overlap_budget);
        for (const auto& entry : sorted) {
            if (!entry.evidence_valid) continue;
            add_region_key_unique(out, entry.region);
            if (out.regions.size() >= 1 + overlap_budget) break;
        }
        return out;
    }

    RegionKeySet keys_from_meta_locked(const RegionMeta& meta,
                                       size_t overlap_budget,
                                       bool include_halo = false,
                                       size_t halo_budget = 0) const {
        RegionKeySet out;
        if (meta.primary == std::numeric_limits<uint32_t>::max()) return out;
        out.regions.reserve(1 + overlap_budget +
                            (include_halo ? halo_budget : 0));
        add_region_key_unique(out, meta.primary);
        for (uint32_t r : meta.overlap) {
            if (out.regions.size() >= 1 + overlap_budget) break;
            add_region_key_unique(out, r);
        }
        if (include_halo) {
            const size_t base_budget = 1 + overlap_budget;
            const size_t total_budget = base_budget + halo_budget;
            for (uint32_t r : meta.halo) {
                if (out.regions.size() >= total_budget) break;
                add_region_key_unique(out, r);
            }
        }
        return out;
    }

    void add_edge_anchor_locked(uint32_t region, annchor_m2::tableint outside,
                                annchor_m2::tableint inside) const {
        if (region_edge_anchor_config_ == 0) return;
        if (region >= region_edge_anchors_.size()) return;
        if (outside == inside) return;
        if (outside >= index_->getCurrentElementCount() ||
            inside >= index_->getCurrentElementCount()) {
            return;
        }
        auto& anchors = region_edge_anchors_[region];
        for (const EdgeAnchor& anchor : anchors) {
            if (anchor.outside == outside && anchor.inside == inside) return;
        }
        EdgeAnchor candidate;
        candidate.outside = outside;
        candidate.inside = inside;
        candidate.len = dist_ptr(data_by_id(outside), data_by_id(inside));
        if (anchors.size() < region_edge_anchor_config_) {
            anchors.push_back(candidate);
        } else {
            auto worst = std::max_element(
                anchors.begin(), anchors.end(),
                [](const EdgeAnchor& a, const EdgeAnchor& b) {
                    return a.len < b.len;
                });
            if (worst == anchors.end() || candidate.len >= worst->len) {
                return;
            }
            *worst = candidate;
        }
        std::sort(anchors.begin(), anchors.end(),
                  [](const EdgeAnchor& a, const EdgeAnchor& b) {
                      if (a.len != b.len) return a.len < b.len;
                      if (a.inside != b.inside) return a.inside < b.inside;
                      return a.outside < b.outside;
                  });
    }

    void refresh_edge_anchors_from_graph_locked(
        annchor_m2::tableint id) const {
        if (region_edge_anchor_config_ == 0) return;
        if (id >= node_regions_.size()) return;
        const RegionMeta& meta = node_regions_[id];
        if (meta.primary == std::numeric_limits<uint32_t>::max()) return;

        std::vector<uint32_t> owned_regions;
        owned_regions.reserve(1 + meta.overlap.size());
        owned_regions.push_back(meta.primary);
        for (uint32_t region : meta.overlap) {
            bool seen = false;
            for (uint32_t existing : owned_regions) {
                if (existing == region) {
                    seen = true;
                    break;
                }
            }
            if (!seen) owned_regions.push_back(region);
        }

        auto* ll = index_->get_linklist0(id);
        const size_t sz = index_->getListCount(ll);
        auto* neighbors = reinterpret_cast<annchor_m2::tableint*>(ll + 1);
        for (uint32_t region : owned_regions) {
            if (region == std::numeric_limits<uint32_t>::max()) continue;
            for (size_t j = 0; j < sz; ++j) {
                const annchor_m2::tableint nb = neighbors[j];
                if (nb >= node_regions_.size()) continue;
                const RegionMeta& nb_meta = node_regions_[nb];
                if (nb_meta.primary == std::numeric_limits<uint32_t>::max()) {
                    continue;
                }
                if (nb_meta.primary == region) continue;
                add_edge_anchor_locked(region, nb, id);
            }
        }
    }

    void refresh_halo_regions_from_graph_locked(
        annchor_m2::tableint id) const {
        if (region_halo_config_ == 0) return;
        if (id >= node_regions_.size() ||
            id >= index_->getCurrentElementCount()) {
            return;
        }
        RegionMeta& meta = node_regions_[id];
        if (meta.primary == std::numeric_limits<uint32_t>::max()) return;

        std::vector<uint32_t> owned_regions;
        owned_regions.reserve(1 + meta.overlap.size());
        owned_regions.push_back(meta.primary);
        for (uint32_t region : meta.overlap) {
            bool seen = false;
            for (uint32_t existing : owned_regions) {
                if (existing == region) {
                    seen = true;
                    break;
                }
            }
            if (!seen) owned_regions.push_back(region);
        }

        std::vector<std::pair<uint32_t, T>> candidates;
        auto region_owned = [&](uint32_t region) {
            for (uint32_t existing : owned_regions) {
                if (existing == region) return true;
            }
            return false;
        };
        auto add_candidate = [&](uint32_t region, T len) {
            if (region == std::numeric_limits<uint32_t>::max()) return;
            if (region >= region_sizes_.size() || region_owned(region)) return;
            for (auto& entry : candidates) {
                if (entry.first == region) {
                    if (len < entry.second) entry.second = len;
                    return;
                }
            }
            candidates.emplace_back(region, len);
        };

        auto* ll = index_->get_linklist0(id);
        const size_t sz = index_->getListCount(ll);
        auto* neighbors = reinterpret_cast<annchor_m2::tableint*>(ll + 1);
        const T* x = data_by_id(id);
        for (size_t j = 0; j < sz; ++j) {
            const annchor_m2::tableint nb = neighbors[j];
            if (nb >= node_regions_.size() ||
                nb >= index_->getCurrentElementCount()) {
                continue;
            }
            const RegionMeta& nb_meta = node_regions_[nb];
            if (nb_meta.primary == std::numeric_limits<uint32_t>::max()) {
                continue;
            }
            add_candidate(nb_meta.primary, dist_ptr(x, data_by_id(nb)));
        }

        std::sort(candidates.begin(), candidates.end(),
                  [&](const auto& a, const auto& b) {
                      if (a.second != b.second) return a.second < b.second;
                      const size_t as = a.first < region_sizes_.size()
                                            ? region_sizes_[a.first]
                                            : std::numeric_limits<size_t>::max();
                      const size_t bs = b.first < region_sizes_.size()
                                            ? region_sizes_[b.first]
                                            : std::numeric_limits<size_t>::max();
                      if (as != bs) return as < bs;
                      return a.first < b.first;
                  });

        meta.halo.clear();
        meta.halo.reserve(region_halo_config_);
        for (const auto& entry : candidates) {
            if (meta.halo.size() >= region_halo_config_) break;
            meta.halo.push_back(entry.first);
        }
    }

    void ensure_graph_regions_locked(size_t committed_limit, size_t region_cap,
                                     size_t overlap_budget,
                                     size_t edge_anchor_k,
                                     size_t region_halo) const {
        if (region_cap_config_ != region_cap ||
            region_overlap_config_ != overlap_budget ||
            region_edge_anchor_config_ != edge_anchor_k ||
            region_halo_config_ != region_halo) {
            region_built_limit_ = 0;
            region_cap_config_ = region_cap;
            region_overlap_config_ = overlap_budget;
            region_edge_anchor_config_ = edge_anchor_k;
            region_halo_config_ = region_halo;
            node_regions_.clear();
            region_sizes_.clear();
            region_edge_anchors_.clear();
        }
        if (committed_limit <= region_built_limit_) return;
        node_regions_.resize(committed_limit);

        for (size_t raw = region_built_limit_; raw < committed_limit; ++raw) {
            const annchor_m2::tableint id =
                static_cast<annchor_m2::tableint>(raw);
            auto candidates = neighbor_region_candidates_locked(id, raw);
            const uint32_t primary =
                choose_primary_region_locked(candidates, id, region_cap);
            if (primary >= region_sizes_.size()) continue;

            RegionMeta meta;
            meta.primary = primary;
            const bool has_evidence = std::any_of(
                candidates.begin(), candidates.end(),
                [](const RegionCandidate& entry) { return entry.evidence_valid; });
            if (has_evidence) {
                std::vector<RegionCandidate> sorted = candidates;
                sort_region_candidates(sorted, region_sizes_);
                for (const auto& entry : sorted) {
                    if (meta.overlap.size() >= overlap_budget) break;
                    if (entry.region == primary || !entry.evidence_valid) {
                        continue;
                    }
                    meta.overlap.push_back(entry.region);
                }
            }
            node_regions_[raw] = std::move(meta);
            region_sizes_[primary]++;
            refresh_edge_anchors_from_graph_locked(id);
            refresh_halo_regions_from_graph_locked(id);
        }
        region_built_limit_ = committed_limit;
    }

    RegionKeySet insert_evidence_region_keys_locked(
        annchor_m2::tableint id, size_t committed_limit,
        size_t overlap_budget) const {
        RegionKeySet out;
        const std::vector<annchor_m2::tableint> insert_neighbors =
            index_->getM2BaseInsertNeighbors(id);
        if (insert_neighbors.empty()) return out;

        const size_t region_cap =
            std::min<size_t>(
                RegionKeySet::kBitsetBits,
                read_env_size("ANNCHOR_M2_INSERT_EVIDENCE_REGION_MAX",
                              std::min<size_t>(
                                  std::max<size_t>(M_, 8), size_t{32})));
        out.regions.reserve(std::min(region_cap, insert_neighbors.size()));
        auto add_meta_regions = [&](annchor_m2::tableint nb) {
            if (out.regions.size() >= region_cap) return false;
            if (nb >= committed_limit || nb >= node_regions_.size()) return true;
            RegionKeySet keys =
                keys_from_meta_locked(node_regions_[nb], overlap_budget);
            for (uint32_t region : keys.regions) {
                add_region_key_unique(out, region);
                if (out.regions.size() >= region_cap) return false;
            }
            return true;
        };
        for (annchor_m2::tableint nb : insert_neighbors) {
            if (!add_meta_regions(nb)) break;
        }
        return out;
    }

    RegionKeySet fresh_region_keys_locked(const T* vec, TagT tag,
                                          size_t committed_limit,
                                          size_t overlap_budget) const {
        annchor_m2::tableint id = 0;
        if (label_to_internal(tag, id) && id < index_->getCurrentElementCount()) {
            if (id < committed_limit && id < node_regions_.size()) {
                RegionKeySet keys =
                    keys_from_meta_locked(node_regions_[id], overlap_budget);
                if (!keys.regions.empty()) return keys;
            }
            RegionKeySet insert_keys =
                insert_evidence_region_keys_locked(id, committed_limit,
                                                   overlap_budget);
            if (!insert_keys.regions.empty()) return insert_keys;
        }
        (void)vec;
        return RegionKeySet{};
    }

    RegionKeySet query_region_keys_locked(const T* query,
                                          const TagT* committed_row, size_t k,
                                          const C_FreshJoinParams& params,
                                          size_t committed_limit,
                                          size_t overlap_budget) const {
        RegionKeySet out;
        const size_t max_query_regions =
            std::min<size_t>(read_env_size("ANNCHOR_M2_QUERY_REGION_MAX", 64),
                             RegionKeySet::kBitsetBits);
        const size_t min_support =
            read_env_size("ANNCHOR_M2_QUERY_REGION_MIN_SUPPORT",
                          dim_ > 512 ? 2 : 1);
        std::vector<std::pair<uint32_t, size_t>> region_support;
        // In high dimensions the metric radius bound is weak because distances
        // concentrate. Require repeated evidence from the committed top-k
        // neighborhood before opening a region by default. For very high
        // dimensions, the caller also skips micro-region postings by default and
        // uses this region set for direct fresh-region intersection.
        auto add_support = [&](uint32_t region) {
            if (region == std::numeric_limits<uint32_t>::max()) return;
            for (auto& entry : region_support) {
                if (entry.first == region) {
                    entry.second++;
                    return;
                }
            }
            region_support.emplace_back(region, 1);
        };
        auto add_key = [&](uint32_t region) {
            if (out.regions.size() >= max_query_regions) return false;
            add_region_key_unique(out, region);
            return out.regions.size() < max_query_regions;
        };
        out.regions.reserve((params.query_result_prefix == 0 ? k : params.query_result_prefix) *
                            (1 + overlap_budget + region_halo_config_));

        const size_t prefix = std::min<size_t>(
            k, params.query_result_prefix == 0 ? k : params.query_result_prefix);
        const bool add_neighbor_regions =
            read_env_bool("ANNCHOR_M2_QUERY_NEIGHBOR_REGIONS", dim_ > 512);
        const size_t neighbor_seed_prefix =
            std::min<size_t>(
                prefix,
                read_env_size("ANNCHOR_M2_QUERY_NEIGHBOR_PREFIX",
                              dim_ > 512 ? std::min<size_t>(prefix, 6)
                                         : std::min<size_t>(prefix, 4)));
        const size_t neighbor_limit =
            std::min<size_t>(
                read_env_size("ANNCHOR_M2_QUERY_NEIGHBOR_LIMIT",
                              dim_ > 512 ? 32 : 16),
                256);
        auto add_meta_regions = [&](annchor_m2::tableint id) {
            if (id >= node_regions_.size()) return;
            RegionKeySet keys =
                keys_from_meta_locked(node_regions_[id], overlap_budget,
                                      true, region_halo_config_);
            for (uint32_t region : keys.regions) {
                if (min_support <= 1) {
                    if (!add_key(region)) return;
                } else {
                    add_support(region);
                }
            }
        };
        for (size_t j = 0; j < prefix; ++j) {
            if (out.regions.size() >= max_query_regions) break;
            if (committed_row[j] == std::numeric_limits<TagT>::max()) continue;
            annchor_m2::tableint id = 0;
            if (!label_to_internal(committed_row[j], id)) continue;
            if (id >= committed_limit || id >= node_regions_.size()) continue;
            add_meta_regions(id);
            if (!add_neighbor_regions || j >= neighbor_seed_prefix ||
                out.regions.size() >= max_query_regions) {
                continue;
            }
            auto* ll = index_->get_linklist0(id);
            const size_t sz = index_->getListCount(ll);
            auto* neighbors = reinterpret_cast<annchor_m2::tableint*>(ll + 1);
            const size_t take = std::min(sz, neighbor_limit);
            for (size_t ni = 0; ni < take; ++ni) {
                const annchor_m2::tableint nb = neighbors[ni];
                if (nb >= committed_limit || nb >= node_regions_.size()) continue;
                add_meta_regions(nb);
                if (out.regions.size() >= max_query_regions) break;
            }
        }
        if (min_support > 1) {
            std::sort(region_support.begin(), region_support.end(),
                      [&](const auto& a, const auto& b) {
                          if (a.second != b.second) return a.second > b.second;
                          const size_t as = a.first < region_sizes_.size()
                                                ? region_sizes_[a.first]
                                                : std::numeric_limits<size_t>::max();
                          const size_t bs = b.first < region_sizes_.size()
                                                ? region_sizes_[b.first]
                                                : std::numeric_limits<size_t>::max();
                          if (as != bs) return as < bs;
                          return a.first < b.first;
                      });
            for (const auto& entry : region_support) {
                if (entry.second < min_support) continue;
                if (!add_key(entry.first)) break;
            }
            if (out.regions.empty() && !region_support.empty()) {
                add_key(region_support.front().first);
            }
        }
        (void)query;
        return out;
    }

    T metric_distance_value(T raw_dist) const {
        if (metric_ == METRIC_L2) {
            return static_cast<T>(
                std::sqrt(std::max<double>(0.0, static_cast<double>(raw_dist))));
        }
        return raw_dist;
    }

    T top_worst_dist_or_inf(const std::vector<ScoredTag>& top,
                            size_t k) const {
        if (top.size() < k) return std::numeric_limits<T>::max();
        T worst = top.empty() ? std::numeric_limits<T>::max() : top[0].dist;
        for (const auto& entry : top) {
            if (entry.dist > worst) worst = entry.dist;
        }
        return worst;
    }

    T region_local_radius_limit(uint32_t region) const {
        if (metric_ != METRIC_L2) return std::numeric_limits<T>::max();
        if (region >= region_edge_anchors_.size()) {
            return std::numeric_limits<T>::max();
        }
        T limit = 0;
        for (const EdgeAnchor& edge : region_edge_anchors_[region]) {
            limit = std::max(limit, metric_distance_value(edge.len));
        }
        return limit > 0 ? limit : std::numeric_limits<T>::max();
    }


    template <typename FreshPtrAt>
    FreshRegionPostings build_fresh_region_postings(
        size_t fresh_count,
        const std::vector<RegionKeySet>& fresh_regions,
        FreshPtrAt fresh_ptr_at,
        size_t micro_region_cap) const {
        FreshRegionPostings postings;
        const size_t posting_region_limit =
            read_env_size("ANNCHOR_M2_POSTING_REGION_LIMIT", 65536);
        // Cap micros per committed region so the sub-clustering can't degenerate
        // into occupancy-1 singletons (e.g. Deep1M: the edge-anchor radius_limit
        // is smaller than typical fresh-to-fresh distances, so the radius gate
        // rejects every merge -> one micro per fresh point -> O(fresh) scan with
        // poor locality -> the mechanism loses to brute).
        //
        // The cap is per-region ceil(sqrt(fresh_in_region)) by default. Basis:
        // for n points split into k clusters, per-query work ~ (scan k anchors)
        // + (scan n/k members) = k + n/k, minimized at k = sqrt(n) (the standard
        // IVF nlist~sqrt(N) balance). This is self-adaptive to how many fresh
        // land in each region and does not depend on the fragile edge-length
        // threshold. ANNCHOR_M2_MICRO_MAX_PER_REGION>0 forces a fixed cap instead.
        const size_t fixed_micros_per_region =
            read_env_size("ANNCHOR_M2_MICRO_MAX_PER_REGION", 0);
        uint32_t max_region = 0;
        bool has_region = false;
        for (size_t fi = 0; fi < fresh_regions.size(); ++fi) {
            if (fresh_regions[fi].regions.empty()) {
                postings.unpublished.push_back(fi);
                continue;
            }
            for (uint32_t region : fresh_regions[fi].regions) {
                if (region >= posting_region_limit) return FreshRegionPostings{};
                max_region = std::max(max_region, region);
                has_region = true;
            }
        }
        if (!has_region) {
            postings.enabled = true;
            return postings;
        }

        postings.by_region.resize(static_cast<size_t>(max_region) + 1);
        postings.micro_regions.reserve(fresh_count);

        // Per-region micro cap = ceil(sqrt(fresh_in_region)) (IVF sqrt(n) rule),
        // or a fixed override when the env is set.
        std::vector<uint32_t> region_micro_cap(
            static_cast<size_t>(max_region) + 1, 0);
        {
            std::vector<uint32_t> region_fresh_count(
                static_cast<size_t>(max_region) + 1, 0);
            for (size_t fi = 0; fi < fresh_regions.size(); ++fi) {
                for (uint32_t region : fresh_regions[fi].regions) {
                    if (region < region_fresh_count.size()) {
                        region_fresh_count[region]++;
                    }
                }
            }
            for (size_t r = 0; r < region_micro_cap.size(); ++r) {
                if (fixed_micros_per_region > 0) {
                    region_micro_cap[r] =
                        static_cast<uint32_t>(fixed_micros_per_region);
                } else {
                    const double n = static_cast<double>(region_fresh_count[r]);
                    region_micro_cap[r] = static_cast<uint32_t>(
                        std::max<size_t>(1, static_cast<size_t>(
                                                std::ceil(std::sqrt(n)))));
                }
            }
        }

        auto insert_into_label = [&](uint32_t region, size_t fi) {
            if (region >= postings.by_region.size()) return;
            const T* x = fresh_ptr_at(fi);
            const T radius_limit = region_local_radius_limit(region);
            size_t best_micro = postings.micro_regions.size();
            T best_dist = std::numeric_limits<T>::max();
            for (size_t micro_id : postings.by_region[region]) {
                FreshMicroRegion& micro = postings.micro_regions[micro_id];
                if (micro.posting.size() >= micro_region_cap) continue;
                const T raw = dist_ptr(x, fresh_ptr_at(micro.anchor));
                const T dist = metric_distance_value(raw);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_micro = micro_id;
                }
            }

            const bool region_at_micro_cap =
                region < region_micro_cap.size() &&
                postings.by_region[region].size() >= region_micro_cap[region];
            if (best_micro != postings.micro_regions.size()) {
                FreshMicroRegion& micro = postings.micro_regions[best_micro];
                const T new_radius = std::max(micro.radius, best_dist);
                // Respect the metric radius for tight micros, but once the
                // region is at its micro cap, force-merge into the nearest one
                // so the structure can't blow up into singletons.
                if (new_radius <= radius_limit || region_at_micro_cap) {
                    micro.posting.push_back(fi);
                    micro.radial.push_back(best_dist);
                    micro.radius = new_radius;
                    postings.entries++;
                    return;
                }
            }

            FreshMicroRegion micro;
            micro.anchor = fi;
            micro.posting.reserve(micro_region_cap);
            micro.radial.reserve(micro_region_cap);
            micro.posting.push_back(fi);
            micro.radial.push_back(0);
            const size_t micro_id = postings.micro_regions.size();
            postings.micro_regions.push_back(std::move(micro));
            postings.by_region[region].push_back(micro_id);
            postings.entries++;
        };

        for (size_t fi = 0; fi < fresh_regions.size(); ++fi) {
            for (uint32_t region : fresh_regions[fi].regions) {
                insert_into_label(region, fi);
            }
        }
        postings.enabled = true;
        return postings;
    }

    static void add_marked_candidate(std::vector<size_t>& candidates,
                                     std::vector<uint32_t>& marks,
                                     uint32_t mark, size_t id) {
        if (id >= marks.size() || marks[id] == mark) return;
        marks[id] = mark;
        candidates.push_back(id);
    }

    static void reset_candidate_mark(std::vector<uint32_t>& marks,
                                     uint32_t& mark) {
        mark++;
        if (mark == 0) {
            std::fill(marks.begin(), marks.end(), 0);
            mark = 1;
        }
    }

    template <typename FreshPtrAt>
    size_t collect_posting_candidates(
        const T* query, const RegionKeySet& query_regions,
        const FreshRegionPostings& postings,
        FreshPtrAt fresh_ptr_at, const std::vector<ScoredTag>& top, size_t k,
        std::vector<size_t>& candidates,
        std::vector<uint32_t>& marks,
        uint32_t mark) const {
        size_t touched = 0;
        for (size_t id : postings.unpublished) {
            add_marked_candidate(candidates, marks, mark, id);
        }
        touched += postings.unpublished.size();

        const bool use_bounds = metric_ == METRIC_L2;
        const T tau2 = top_worst_dist_or_inf(top, k);
        for (uint32_t region : query_regions.regions) {
            if (region >= postings.by_region.size()) continue;
            const auto& bucket = postings.by_region[region];
            touched += bucket.size();
            for (size_t micro_id : bucket) {
                if (micro_id >= postings.micro_regions.size()) continue;
                const FreshMicroRegion& micro =
                    postings.micro_regions[micro_id];
                if (micro.anchor >= marks.size()) continue;
                bool open_micro = true;
                T dq_anchor = 0;
                if (use_bounds && tau2 != std::numeric_limits<T>::max()) {
                    const T raw = dist_ptr(query, fresh_ptr_at(micro.anchor));
                    dq_anchor = metric_distance_value(raw);
                    const T lb =
                        std::max<T>(0, dq_anchor - micro.radius);
                    if (lb * lb > tau2) {
                        open_micro = false;
                    }
                }
                if (!open_micro) {
                    metric_m2_micro_regions_pruned_.fetch_add(
                        1, std::memory_order_relaxed);
                    continue;
                }
                metric_m2_micro_regions_opened_.fetch_add(
                    1, std::memory_order_relaxed);
                touched += micro.posting.size();
                for (size_t i = 0; i < micro.posting.size(); ++i) {
                    const size_t id = micro.posting[i];
                    if (use_bounds && tau2 != std::numeric_limits<T>::max() &&
                        i < micro.radial.size()) {
                        metric_m2_member_bound_checks_.fetch_add(
                            1, std::memory_order_relaxed);
                        const T lb =
                            static_cast<T>(std::fabs(
                                static_cast<double>(dq_anchor - micro.radial[i])));
                        if (lb * lb > tau2) {
                            metric_m2_member_bound_skips_.fetch_add(
                                1, std::memory_order_relaxed);
                            continue;
                        }
                    }
                    add_marked_candidate(candidates, marks, mark, id);
                }
            }
        }
        return touched;
    }


    static bool dense_bitset_intersects(const RegionKeySet& lhs,
                                        const RegionKeySet& rhs) {
        const size_t words = std::min<size_t>(lhs.words, rhs.words);
#if defined(__AVX2__)
        size_t i = 0;
        for (; i + 4 <= words; i += 4) {
            const __m256i a = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(lhs.bits.data() + i));
            const __m256i b = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(rhs.bits.data() + i));
            const __m256i both = _mm256_and_si256(a, b);
            if (!_mm256_testz_si256(both, both)) return true;
        }
        for (; i < words; ++i) {
            if ((lhs.bits[i] & rhs.bits[i]) != 0) return true;
        }
#else
        for (size_t i = 0; i < words; ++i) {
            if ((lhs.bits[i] & rhs.bits[i]) != 0) return true;
        }
#endif
        return false;
    }

    static bool regions_intersect(const RegionKeySet& lhs,
                                  const RegionKeySet& rhs) {
        if (!lhs.overflow && !rhs.overflow) {
            const RegionKeySet& sparse =
                lhs.regions.size() <= rhs.regions.size() ? lhs : rhs;
            const RegionKeySet& dense =
                lhs.regions.size() <= rhs.regions.size() ? rhs : lhs;
            if (sparse.regions.size() <= 8 ||
                sparse.regions.size() * 2 <= std::min<size_t>(lhs.words, rhs.words)) {
                for (uint32_t region : sparse.regions) {
                    const size_t word = region >> 6;
                    if (word >= dense.words) continue;
                    if ((dense.bits[word] & (uint64_t{1} << (region & 63))) != 0) {
                        return true;
                    }
                }
                return false;
            }
            return dense_bitset_intersects(lhs, rhs);
        }
        for (uint32_t a : lhs.regions) {
            for (uint32_t b : rhs.regions) {
                if (a == b) return true;
            }
        }
        return false;
    }


    std::vector<ScoredTag> committed_topk(const T* query,
                                          const TagT* committed_row,
                                          size_t k) const {
        std::vector<ScoredTag> top;
        top.reserve(k);
        for (size_t j = 0; j < k; ++j) {
            const TagT tag = committed_row[j];
            if (tag == std::numeric_limits<TagT>::max()) continue;
            annchor_m2::tableint id = 0;
            if (!label_to_internal(tag, id)) continue;
            top.push_back(ScoredTag{tag, dist_ptr(query, data_by_id(id))});
        }
        return top;
    }

    void exact_admit(const T* query, const T* fresh, TagT tag, size_t k,
                     std::vector<ScoredTag>& top) const {
        if (top.size() < k) {
            const T dist = dist_ptr(query, fresh);
            top.push_back(ScoredTag{tag, dist});
            return;
        }
        auto worst = std::max_element(top.begin(), top.end(),
                                      [](const ScoredTag& a,
                                         const ScoredTag& b) {
                                          if (a.dist != b.dist) return a.dist < b.dist;
                                          return a.tag < b.tag;
                                      });
        const T cutoff =
            worst == top.end() ? std::numeric_limits<T>::max() : worst->dist;
        const T dist = fresh_dist_for_admit(query, fresh, cutoff,
                                            worst != top.end());
        if (worst != top.end() &&
            (dist < worst->dist || (dist == worst->dist && tag < worst->tag))) {
            *worst = ScoredTag{tag, dist};
        }
    }

    void copy_committed_results(size_t k, size_t num_queries,
                                const TagT* committed_results,
                                TagT** batch_results) const {
        for (size_t qi = 0; qi < num_queries; ++qi) {
            for (size_t j = 0; j < k; ++j) {
                batch_results[qi][j] = committed_results[qi * k + j];
            }
        }
    }

    template <typename FreshPtrAt>
    int exact_fresh_merge_by_accessor(
                          const T* batch_queries, size_t k,
                          size_t num_queries, const TagT* fresh_tags,
                          size_t fresh_count, FreshPtrAt fresh_ptr_at,
                          const TagT* committed_results,
                          TagT** batch_results, C_FreshJoinStats* stats,
                          bool count_fallback) const {
        for (size_t qi = 0; qi < num_queries; ++qi) {
            const T* query = batch_queries + qi * dim_;
            std::vector<ScoredTag> top =
                committed_topk(query, committed_results + qi * k, k);
            for (size_t fi = 0; fi < fresh_count; ++fi) {
                exact_admit(query, fresh_ptr_at(fi), fresh_tags[fi], k, top);
            }
            std::sort(top.begin(), top.end(), [](const ScoredTag& a,
                                                 const ScoredTag& b) {
                if (a.dist != b.dist) return a.dist < b.dist;
                return a.tag < b.tag;
            });
            for (size_t j = 0; j < k; ++j) {
                batch_results[qi][j] =
                    j < top.size() ? top[j].tag
                                   : std::numeric_limits<TagT>::max();
            }
        }
        if (stats) {
            stats->selected_candidates += num_queries * fresh_count;
            stats->exacted_candidates += num_queries * fresh_count;
            if (count_fallback) stats->exact_fallback_queries += num_queries;
        }
        return 0;
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

    size_t register_active_insert_batch(const T* data, const TagT* tags,
                                        size_t count) const {
        if (!data || !tags || count == 0) return 0;
        ActiveInsertBatch batch;
        batch.id = active_insert_batch_next_id_.fetch_add(
            1, std::memory_order_relaxed);
        batch.start_raw_ns = monotonic_raw_ns();
        batch.labels.assign(tags, tags + count);
        // Keep a C++ copy for queries that enter while this insert batch is
        // still running. Some labels may not yet resolve through the index, but
        // OCC validation still needs the vector itself.
        batch.vectors = std::make_shared<std::vector<T>>(
            data, data + count * dim_);
        std::lock_guard<std::mutex> guard(active_insert_batch_mu_);
        active_insert_batches_.push_back(std::move(batch));
        return active_insert_batches_.back().id;
    }

    void unregister_active_insert_batch(size_t id) const {
        if (id == 0) return;
        const uint64_t finish_raw_ns = monotonic_raw_ns();
        std::lock_guard<std::mutex> guard(active_insert_batch_mu_);
        auto it = std::find_if(active_insert_batches_.begin(),
                               active_insert_batches_.end(),
                               [id](const ActiveInsertBatch& batch) {
                                   return batch.id == id;
                               });
        if (it == active_insert_batches_.end()) return;
        it->finish_raw_ns = finish_raw_ns;
        retired_insert_batches_.push_back(std::move(*it));
        active_insert_batches_.erase(it);
        constexpr size_t kMaxRetiredInsertBatches = 32;
        if (retired_insert_batches_.size() > kMaxRetiredInsertBatches) {
            const size_t extra =
                retired_insert_batches_.size() - kMaxRetiredInsertBatches;
            retired_insert_batches_.erase(retired_insert_batches_.begin(),
                                          retired_insert_batches_.begin() +
                                              static_cast<std::ptrdiff_t>(extra));
        }
    }

    size_t snapshot_active_insert_batch_labels(TagT* out,
                                               size_t max_labels,
                                               uint64_t snapshot_raw_ns) const {
        if (!out || max_labels == 0) return 0;
        std::lock_guard<std::mutex> guard(active_insert_batch_mu_);
        size_t written = 0;
        auto include_batch = [snapshot_raw_ns](const ActiveInsertBatch& batch) {
            return batch.start_raw_ns <= snapshot_raw_ns &&
                   (batch.finish_raw_ns == 0 ||
                    batch.finish_raw_ns > snapshot_raw_ns);
        };
        auto write_batch = [&](const ActiveInsertBatch& batch) {
            if (!include_batch(batch)) return false;
            for (TagT label : batch.labels) {
                if (written >= max_labels) return true;
                out[written++] = label;
            }
            return false;
        };
        for (const auto& batch : active_insert_batches_) {
            if (write_batch(batch)) return written;
        }
        for (const auto& batch : retired_insert_batches_) {
            if (write_batch(batch)) return written;
        }
        return written;
    }

    static uint64_t monotonic_raw_ns() {
        struct timespec ts;
#if defined(CLOCK_MONOTONIC_RAW)
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
        clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
        return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
               static_cast<uint64_t>(ts.tv_nsec);
    }

    struct ActiveInsertBatch {
        size_t id{0};
        uint64_t start_raw_ns{0};
        uint64_t finish_raw_ns{0};
        std::vector<TagT> labels;
        std::shared_ptr<const std::vector<T>> vectors;
    };

    mutable std::mutex active_insert_batch_mu_;
    mutable std::vector<ActiveInsertBatch> active_insert_batches_;
    mutable std::vector<ActiveInsertBatch> retired_insert_batches_;
    mutable std::atomic<size_t> active_insert_batch_next_id_{1};

    mutable std::shared_mutex region_mu_;
    mutable size_t region_built_limit_{0};
    mutable size_t region_cap_config_{0};
    mutable size_t region_overlap_config_{0};
    mutable size_t region_edge_anchor_config_{0};
    mutable size_t region_halo_config_{0};
    mutable std::vector<RegionMeta> node_regions_;
    mutable std::vector<size_t> region_sizes_;
    mutable std::vector<std::vector<EdgeAnchor>> region_edge_anchors_;
    tbb::task_arena arena_;
    bool separate_arenas_;
    tbb::task_arena search_arena_;
    tbb::task_arena insert_arena_;
    bool measured_search_arena_enabled_;
    tbb::task_arena measured_search_arena_;
    bool fresh_early_abort_l2_{false};
    size_t fresh_early_abort_interval_{32};
};
