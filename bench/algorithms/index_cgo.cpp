#include "index_cgo.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <time.h>
#include <vector>

#include "annchor-m1/annchor_m1.hpp"
#include "annchor-trim/annchor_trim.hpp"
#include "annchor-preempt/annchor_preempt.hpp"
#include "hnsw_visible/hnsw_visible.hpp"
#include "hnsw/hnsw.hpp"
#include "index.hpp"
#include "parlayann/parlay_hnsw.hpp"
#include "parlayann/parlay_vamana.hpp"
#include "segment/segment.hpp"
#include "vamana/vamana.hpp"

#define BUFFER_LEN 2048

IndexBase<float>* create_annchor_m2_float(const IndexParams& params);
IndexBase<float>* create_annchor_m3_float(const IndexParams& params);
bool annchor_m2_set_enable_mvcc_float(IndexBase<float>* base,
                                             bool enable);
bool annchor_m2_set_enable_undo_recovery_float(IndexBase<float>* base,
                                                      bool enable);
bool annchor_m3_set_enable_mvcc_float(IndexBase<float>* base, bool enable);
bool annchor_m3_set_enable_undo_recovery_float(IndexBase<float>* base,
                                               bool enable);
int annchor_m2_fresh_merge_float(IndexBase<float>* base,
                                        const float* batch_queries,
                                        size_t k,
                                        size_t num_queries,
                                        const float* fresh_data,
                                        const uint32_t* fresh_tags,
                                        size_t fresh_count,
                                        const uint32_t* committed_results,
                                        uint32_t** batch_results,
                                        size_t committed_view_limit,
                                        const C_FreshJoinParams& params,
                                        C_FreshJoinStats* stats);
int annchor_m2_fresh_merge_labels_float(IndexBase<float>* base,
                                        const float* batch_queries,
                                        size_t k,
                                        size_t num_queries,
                                        const uint32_t* fresh_tags,
                                        size_t fresh_count,
                                        const uint32_t* committed_results,
                                        uint32_t** batch_results,
                                        const C_FreshJoinParams& params,
                                        C_FreshJoinStats* stats);
int annchor_m3_fresh_merge_float(IndexBase<float>* base,
                                 const float* batch_queries,
                                 size_t k,
                                 size_t num_queries,
                                 const float* fresh_data,
                                 const uint32_t* fresh_tags,
                                 size_t fresh_count,
                                 const uint32_t* committed_results,
                                 uint32_t** batch_results,
                                 size_t committed_view_limit,
                                 const C_FreshJoinParams& params,
                                 C_FreshJoinStats* stats);
int annchor_m3_fresh_merge_labels_float(IndexBase<float>* base,
                                        const float* batch_queries,
                                        size_t k,
                                        size_t num_queries,
                                        const uint32_t* fresh_tags,
                                        size_t fresh_count,
                                        const uint32_t* committed_results,
                                        uint32_t** batch_results,
                                        const C_FreshJoinParams& params,
                                        C_FreshJoinStats* stats);

extern "C" {

uint64_t annbench_monotonic_raw_ns(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
           static_cast<uint64_t>(ts.tv_nsec);
}

void* create_index(IndexType type, IndexParams params) {
    IndexBase<float>* index = nullptr;
    switch (type) {
        case INDEX_TYPE_HNSW:
            if (params.data_type == DATA_TYPE_FLOAT) {
                index = new HNSW<float>(params.max_elements, params.dim,
                                        params.num_threads, params.M,
                                        params.ef_construction,
                                        params.use_node_lock,
                                        params.worker_scheduler);
            }
            break;
        case INDEX_TYPE_PARLAYHNSW:
            if (params.data_type == DATA_TYPE_FLOAT) {
                index = new ParlayHNSW<float>(
                    params.max_elements, params.dim, params.num_threads,
                    params.M, params.ef_construction, params.level_m,
                    params.alpha, params.visit_limit);
            }
            break;
        case INDEX_TYPE_PARLAYVAMANA:
            if (params.data_type == DATA_TYPE_FLOAT) {
                index = new ParlayVamana<float>(
                    params.max_elements, params.dim, params.num_threads,
                    params.M, params.ef_construction, params.alpha);
            }
            break;
        case INDEX_TYPE_VAMANA:
            if (params.data_type == DATA_TYPE_FLOAT) {
                index = new Vamana<float>(params.max_elements, params.dim,
                                          params.num_threads, params.M,
                                          params.ef_construction, params.alpha);
            }
            break;
        case INDEX_TYPE_ANNCHOR:
            break;
        case INDEX_TYPE_SEGMENT:
            if (params.data_type == DATA_TYPE_FLOAT) {
                index = new SegmentIndex(params);
            }
            break;
        case INDEX_TYPE_ANNCHOR_DEV:
            break;
        case INDEX_TYPE_ANNCHOR_M1:
            if (params.data_type == DATA_TYPE_FLOAT) {
                index = new ANNchorM1<float>(params.max_elements, params.dim,
                                                 params.num_threads, params.M,
                                                 params.ef_construction, params.use_node_lock,
                                                 static_cast<MetricType>(params.metric));
            }
            break;
        case INDEX_TYPE_ANNCHOR_PREEMPT:
            if (params.data_type == DATA_TYPE_FLOAT) {
                index = new ANNchorPreempt<float>(params.max_elements, params.dim,
                                                  params.num_threads, params.M,
                                                  params.ef_construction, params.use_node_lock,
                                                  static_cast<MetricType>(params.metric));
            }
            break;
        case INDEX_TYPE_ANNCHOR_M3:
            index = create_annchor_m3_float(params);
            break;
        case INDEX_TYPE_ANNCHOR_TRIM:
            if (params.data_type == DATA_TYPE_FLOAT) {
                index = new ANNchorTrim<float>(params.max_elements, params.dim,
                                               params.num_threads, params.M,
                                               params.ef_construction, params.use_node_lock,
                                               static_cast<MetricType>(params.metric));
            }
            break;
        case INDEX_TYPE_ANNCHOR_M2:
            index = create_annchor_m2_float(params);
            break;
        case INDEX_TYPE_HNSW_VISIBLE:
            if (params.data_type == DATA_TYPE_FLOAT) {
                index = new HNSWVisible<float>(
                    params.max_elements, params.dim, params.num_threads,
                    params.M, params.ef_construction, params.use_node_lock,
                    static_cast<MetricType>(params.metric));
            }
            break;
        default:
            return nullptr;
    }
    return static_cast<void*>(index);
}

void destroy_index(void* index_ptr) {
    if (index_ptr) {
        delete static_cast<IndexBase<float>*>(index_ptr);
    }
}

int build(void* index_ptr, float* data, uint32_t* tags, size_t num_points) {
    if (!index_ptr || !data || !tags) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    index->build(data, tags, num_points);
    return 0;
}

void set_query_params(void* index_ptr, C_QueryParams params) {
    if (!index_ptr) return;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    QParams qparams(params.ef_search, params.beam_width, params.alpha,
                    params.visit_limit);
    index->set_query_params(qparams);
}

int search(void* index_ptr, float* query, size_t k, uint32_t* res_tags) {
    if (!index_ptr || !query || !res_tags) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    std::vector<uint32_t> results;
    index->search(query, k, results);
    for (size_t i = 0; i < results.size(); ++i) {
        res_tags[i] = results[i];
    }
    return 0;
}

int batch_insert(void* index_ptr, float* batch_data, uint32_t* batch_tags,
                 size_t batch_size) {
    if (!index_ptr || !batch_data || !batch_tags) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    return index->batch_insert(batch_data, batch_tags, batch_size);
}

int batch_partial_insert(void* index_ptr, float* batch_data,
                              uint32_t* batch_tags, size_t batch_size,
                              size_t* partial_limits, size_t* update_counts) {
    if (!index_ptr || !batch_data || !batch_tags) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    if (auto hnsw = dynamic_cast<HNSW<float>*>(index)) {
        return hnsw->batch_partial_insert(batch_data, batch_tags,
                                               batch_size, partial_limits,
                                               update_counts);
    }
    return index->batch_insert(batch_data, batch_tags, batch_size);
}

int batch_search(void* index_ptr, float* batch_queries, size_t k,
                 size_t num_queries, uint32_t** batch_results, size_t* watermark_out,
                 size_t visible_ts) {
    if (!index_ptr || !batch_queries) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    return index->batch_search(batch_queries, k, num_queries, batch_results, watermark_out, visible_ts);
}

int batch_search_measured(void* index_ptr, float* batch_queries, size_t k,
                          size_t num_queries, uint32_t** batch_results,
                          size_t* watermark_out, size_t visible_ts) {
    if (!index_ptr || !batch_queries) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    return index->batch_search_measured(
        batch_queries, k, num_queries, batch_results, watermark_out, visible_ts);
}

int batch_search_measured_work(void* index_ptr, float* batch_queries, size_t k,
                               size_t num_queries, uint32_t** batch_results,
                               C_SearchWorkStats* per_query_stats,
                               size_t* watermark_out, size_t visible_ts) {
    if (!index_ptr || !batch_queries || !per_query_stats) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    std::vector<SearchWorkStats> stats(num_queries);
    int rc = index->batch_search_measured_work(
        batch_queries, k, num_queries, batch_results, stats.data(),
        watermark_out, visible_ts);
    if (rc != 0) return rc;
    for (size_t i = 0; i < num_queries; ++i) {
        per_query_stats[i].searchknn_ns = stats[i].searchknn_ns;
        per_query_stats[i].searchknn_thread_cpu_ns =
            stats[i].searchknn_thread_cpu_ns;
        per_query_stats[i].work_start_cpu = stats[i].work_start_cpu;
        per_query_stats[i].work_end_cpu = stats[i].work_end_cpu;
        per_query_stats[i].result_copy_ns = stats[i].result_copy_ns;
        per_query_stats[i].entry_ns = stats[i].entry_ns;
        per_query_stats[i].upper_search_ns = stats[i].upper_search_ns;
        per_query_stats[i].base_search_ns = stats[i].base_search_ns;
        per_query_stats[i].result_materialize_ns = stats[i].result_materialize_ns;
        per_query_stats[i].snapshot_guard_ns = stats[i].snapshot_guard_ns;
        per_query_stats[i].visited_list_get_ns = stats[i].visited_list_get_ns;
        per_query_stats[i].visited_list_release_ns = stats[i].visited_list_release_ns;
        per_query_stats[i].upper_lock_wait_ns = stats[i].upper_lock_wait_ns;
        per_query_stats[i].level0_lock_wait_ns = stats[i].level0_lock_wait_ns;
        per_query_stats[i].distance_computations = stats[i].distance_computations;
        per_query_stats[i].upper_distance_computations =
            stats[i].upper_distance_computations;
        per_query_stats[i].level0_distance_computations =
            stats[i].level0_distance_computations;
        per_query_stats[i].distance_compute_ns = stats[i].distance_compute_ns;
        per_query_stats[i].upper_distance_compute_ns =
            stats[i].upper_distance_compute_ns;
        per_query_stats[i].level0_distance_compute_ns =
            stats[i].level0_distance_compute_ns;
        per_query_stats[i].level0_queue_pop_ns = stats[i].level0_queue_pop_ns;
        per_query_stats[i].level0_adj_fetch_ns = stats[i].level0_adj_fetch_ns;
        per_query_stats[i].level0_locality_capture_ns =
            stats[i].level0_locality_capture_ns;
        per_query_stats[i].level0_candidate_loop_ns =
            stats[i].level0_candidate_loop_ns;
        per_query_stats[i].level0_visited_check_ns =
            stats[i].level0_visited_check_ns;
        per_query_stats[i].level0_visibility_check_ns =
            stats[i].level0_visibility_check_ns;
        per_query_stats[i].level0_candidate_accept_ns =
            stats[i].level0_candidate_accept_ns;
        per_query_stats[i].upper_hops = stats[i].upper_hops;
        per_query_stats[i].upper_edges_scanned = stats[i].upper_edges_scanned;
        per_query_stats[i].level0_expansions = stats[i].level0_expansions;
        per_query_stats[i].level0_edges_scanned = stats[i].level0_edges_scanned;
        per_query_stats[i].candidate_pops = stats[i].candidate_pops;
        per_query_stats[i].candidate_pushes = stats[i].candidate_pushes;
        per_query_stats[i].visited_nodes = stats[i].visited_nodes;
        per_query_stats[i].result_pushes = stats[i].result_pushes;
        per_query_stats[i].invisible_expansions = stats[i].invisible_expansions;
        per_query_stats[i].invisible_edges = stats[i].invisible_edges;
        per_query_stats[i].invisible_candidate_dist_comps =
            stats[i].invisible_candidate_dist_comps;
        per_query_stats[i].invisible_candidate_enqueues =
            stats[i].invisible_candidate_enqueues;
        per_query_stats[i].future_skip_hops = stats[i].future_skip_hops;
        per_query_stats[i].rewrite_active_expansions =
            stats[i].rewrite_active_expansions;
        per_query_stats[i].rewrite_recent_expansions =
            stats[i].rewrite_recent_expansions;
        per_query_stats[i].rewrite_period_expansions =
            stats[i].rewrite_period_expansions;
        per_query_stats[i].rewrite_period_active_sum =
            stats[i].rewrite_period_active_sum;
        per_query_stats[i].rewrite_period_active_max =
            stats[i].rewrite_period_active_max;
        per_query_stats[i].expand_visible_count =
            stats[i].expand_visible_count;
        per_query_stats[i].expand_recent_1k_hits =
            stats[i].expand_recent_1k_hits;
        per_query_stats[i].expand_recent_4k_hits =
            stats[i].expand_recent_4k_hits;
        per_query_stats[i].expand_recent_16k_hits =
            stats[i].expand_recent_16k_hits;
        per_query_stats[i].expand_label_gap_sum =
            stats[i].expand_label_gap_sum;
        per_query_stats[i].expand_label_span = stats[i].expand_label_span;
        per_query_stats[i].expand_unique_label_4k_buckets =
            stats[i].expand_unique_label_4k_buckets;
        per_query_stats[i].expand_unique_data_4k_pages =
            stats[i].expand_unique_data_4k_pages;
        per_query_stats[i].expand_unique_data_2m_pages =
            stats[i].expand_unique_data_2m_pages;
        per_query_stats[i].expand_unique_adj_4k_pages =
            stats[i].expand_unique_adj_4k_pages;
        per_query_stats[i].expand_unique_adj_2m_pages =
            stats[i].expand_unique_adj_2m_pages;
        per_query_stats[i].expand_unique_overflow =
            stats[i].expand_unique_overflow;
        per_query_stats[i].path_count = stats[i].path_count;
        const size_t path_count =
            stats[i].path_count < C_SEARCH_PATH_CAPTURE_CAP
                ? stats[i].path_count
                : C_SEARCH_PATH_CAPTURE_CAP;
        for (size_t j = 0; j < path_count; ++j) {
            per_query_stats[i].path_labels[j] = stats[i].path_labels[j];
            per_query_stats[i].path_dists[j] = stats[i].path_dists[j];
        }
    }
    return 0;
}

int batch_search_path_work(void* index_ptr, float* batch_queries, size_t k,
                           size_t num_queries, uint32_t** batch_results,
                           C_SearchWorkStats* per_query_stats,
                           size_t* watermark_out, size_t visible_ts) {
    if (!index_ptr || !batch_queries || !per_query_stats) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    std::vector<SearchWorkStats> stats(num_queries);
    int rc = index->batch_search_path_work(
        batch_queries, k, num_queries, batch_results, stats.data(),
        watermark_out, visible_ts);
    if (rc != 0) return rc;
    for (size_t i = 0; i < num_queries; ++i) {
        per_query_stats[i].rewrite_active_expansions =
            stats[i].rewrite_active_expansions;
        per_query_stats[i].rewrite_recent_expansions =
            stats[i].rewrite_recent_expansions;
        per_query_stats[i].rewrite_period_expansions =
            stats[i].rewrite_period_expansions;
        per_query_stats[i].rewrite_period_active_sum =
            stats[i].rewrite_period_active_sum;
        per_query_stats[i].rewrite_period_active_max =
            stats[i].rewrite_period_active_max;
        per_query_stats[i].expand_visible_count =
            stats[i].expand_visible_count;
        per_query_stats[i].expand_recent_1k_hits =
            stats[i].expand_recent_1k_hits;
        per_query_stats[i].expand_recent_4k_hits =
            stats[i].expand_recent_4k_hits;
        per_query_stats[i].expand_recent_16k_hits =
            stats[i].expand_recent_16k_hits;
        per_query_stats[i].expand_label_gap_sum =
            stats[i].expand_label_gap_sum;
        per_query_stats[i].expand_label_span = stats[i].expand_label_span;
        per_query_stats[i].expand_unique_label_4k_buckets =
            stats[i].expand_unique_label_4k_buckets;
        per_query_stats[i].expand_unique_data_4k_pages =
            stats[i].expand_unique_data_4k_pages;
        per_query_stats[i].expand_unique_data_2m_pages =
            stats[i].expand_unique_data_2m_pages;
        per_query_stats[i].expand_unique_adj_4k_pages =
            stats[i].expand_unique_adj_4k_pages;
        per_query_stats[i].expand_unique_adj_2m_pages =
            stats[i].expand_unique_adj_2m_pages;
        per_query_stats[i].expand_unique_overflow =
            stats[i].expand_unique_overflow;
        per_query_stats[i].path_count = stats[i].path_count;
        const size_t path_count =
            stats[i].path_count < C_SEARCH_PATH_CAPTURE_CAP
                ? stats[i].path_count
                : C_SEARCH_PATH_CAPTURE_CAP;
        for (size_t j = 0; j < path_count; ++j) {
            per_query_stats[i].path_labels[j] = stats[i].path_labels[j];
            per_query_stats[i].path_dists[j] = stats[i].path_dists[j];
        }
    }
    return 0;
}

int annchor_fresh_join_merge(void* index_ptr, float* batch_queries, size_t k,
                             size_t num_queries, float* fresh_data,
                             uint32_t* fresh_tags, size_t fresh_count,
                             uint32_t* committed_results,
                             uint32_t** batch_results,
                             size_t committed_view_limit,
                             C_FreshJoinParams params,
                             C_FreshJoinStats* stats) {
    if (!index_ptr || !batch_queries || !committed_results || !batch_results) {
        return -1;
    }
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    int rc = annchor_m2_fresh_merge_float(
        base, batch_queries, k, num_queries, fresh_data, fresh_tags,
        fresh_count, committed_results, batch_results, committed_view_limit,
        params, stats);
    if (rc != -2) return rc;
    return annchor_m3_fresh_merge_float(
        base, batch_queries, k, num_queries, fresh_data, fresh_tags,
        fresh_count, committed_results, batch_results, committed_view_limit,
        params, stats);
}

int annchor_fresh_join_merge_labels(void* index_ptr, float* batch_queries,
                                    size_t k, size_t num_queries,
                                    uint32_t* fresh_tags, size_t fresh_count,
                                    uint32_t* committed_results,
                                    uint32_t** batch_results,
                                    C_FreshJoinParams params,
                                    C_FreshJoinStats* stats) {
    if (!index_ptr || !batch_queries || !committed_results || !batch_results) {
        return -1;
    }
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    int rc = annchor_m2_fresh_merge_labels_float(
        base, batch_queries, k, num_queries, fresh_tags, fresh_count,
        committed_results, batch_results, params, stats);
    if (rc != -2) return rc;
    return annchor_m3_fresh_merge_labels_float(
        base, batch_queries, k, num_queries, fresh_tags, fresh_count,
        committed_results, batch_results, params, stats);
}

void dump_stats(void* index_ptr, char* str) {
    if (!index_ptr) return;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    std::string stats_str;
    index->dump_stats(stats_str);
    strncpy(str, stats_str.c_str(), BUFFER_LEN);
    str[BUFFER_LEN - 1] = '\0';
}

size_t dump_stats_len(void* index_ptr) {
    if (!index_ptr) return 0;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    std::string stats_str;
    index->dump_stats(stats_str);
    return stats_str.size() + 1;
}

void dump_stats_copy(void* index_ptr, char* str, size_t len) {
    if (!index_ptr || !str || len == 0) return;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    std::string stats_str;
    index->dump_stats(stats_str);
    strncpy(str, stats_str.c_str(), len);
    str[len - 1] = '\0';
}

bool graph_mutation_stats(void* index_ptr, C_GraphMutationStats* out) {
    if (!index_ptr || !out) return false;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    GraphMutationStats stats{};
    if (!index->graph_mutation_stats(&stats)) return false;
    out->connect_calls = stats.connect_calls;
    out->connect_ns = stats.connect_ns;
    out->link_critical_ns = stats.link_critical_ns;
    out->unique_lock_wait_ns = stats.unique_lock_wait_ns;
    out->search_unique_lock_wait_ns = stats.search_unique_lock_wait_ns;
    out->search_unique_lock_acqs = stats.search_unique_lock_acqs;
    out->search_critical_ns = stats.search_critical_ns;
    out->link_updates = stats.link_updates;
    out->upper_search_ns = stats.upper_search_ns;
    out->upper_search_dist_comps = stats.upper_search_dist_comps;
    out->upper_search_edges_scanned = stats.upper_search_edges_scanned;
    out->base_search_ns = stats.base_search_ns;
    out->base_search_expansions = stats.base_search_expansions;
    out->base_search_edges_scanned = stats.base_search_edges_scanned;
    out->base_search_dist_comps = stats.base_search_dist_comps;
    out->select_new_neighbors_ns = stats.select_new_neighbors_ns;
    out->select_new_neighbors_input = stats.select_new_neighbors_input;
    out->select_new_neighbors_selected = stats.select_new_neighbors_selected;
    out->select_new_neighbors_heuristic_dist_comps =
        stats.select_new_neighbors_heuristic_dist_comps;
    out->inserted_node_link_ns = stats.inserted_node_link_ns;
    out->inserted_node_edges_written = stats.inserted_node_edges_written;
    out->existing_neighbor_update_loop_ns = stats.existing_neighbor_update_loop_ns;
    out->existing_neighbor_load_scan_ns = stats.existing_neighbor_load_scan_ns;
    out->existing_neighbor_loaded_edges = stats.existing_neighbor_loaded_edges;
    out->existing_neighbor_append_ns = stats.existing_neighbor_append_ns;
    out->existing_neighbor_prune_ns = stats.existing_neighbor_prune_ns;
    out->existing_neighbor_prune_candidates = stats.existing_neighbor_prune_candidates;
    out->existing_neighbor_prune_dist_comps = stats.existing_neighbor_prune_dist_comps;
    out->existing_neighbor_undo_record_ns = stats.existing_neighbor_undo_record_ns;
    out->existing_neighbor_rewrite_ns = stats.existing_neighbor_rewrite_ns;
    out->existing_neighbor_edges_written = stats.existing_neighbor_edges_written;
    out->existing_neighbor_edges_pruned = stats.existing_neighbor_edges_pruned;
    out->existing_neighbor_visits = stats.existing_neighbor_visits;
    out->existing_neighbor_appends = stats.existing_neighbor_appends;
    out->existing_neighbor_prunes = stats.existing_neighbor_prunes;
    out->existing_neighbor_pruned_edges_recorded = stats.existing_neighbor_pruned_edges_recorded;
    return true;
}

int snapshot_index(void* index_ptr, uint8_t** buffer, size_t* size) {
    if (!index_ptr || !buffer || !size) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    if (!index->supports_snapshot()) return -2;

    std::vector<uint8_t> tmp;
    if (index->snapshot(tmp) != 0) return -3;

    size_t bytes = tmp.size();
    uint8_t* data = nullptr;
    if (bytes > 0) {
        data = reinterpret_cast<uint8_t*>(malloc(bytes));
        if (!data) {
            return -4;
        }
        memcpy(data, tmp.data(), bytes);
    }
    *buffer = data;
    *size = bytes;
    return 0;
}

int restore_index(void* index_ptr, const uint8_t* buffer, size_t size) {
    if (!index_ptr || !buffer) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    if (!index->supports_snapshot()) return -2;
    return index->restore(buffer, size);
}

void free_snapshot_buffer(uint8_t* buffer) {
    if (buffer) free(buffer);
}

void hnsw_enable_insert_telemetry(void* index_ptr, bool enabled) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto h = dynamic_cast<HNSW<float>*>(base)) h->enable_insert_telemetry(enabled);
}

size_t hnsw_last_insert_updates(void* index_ptr) {
    if (!index_ptr) return 0;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto h = dynamic_cast<HNSW<float>*>(base)) return h->last_insert_update_count();
    return 0;
}

void hnsw_configure_partial_insert(void* index_ptr, size_t limit) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto h = dynamic_cast<HNSW<float>*>(base)) h->configure_partial_insert(limit);
}

void hnsw_disable_partial_insert(void* index_ptr) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto h = dynamic_cast<HNSW<float>*>(base)) h->disable_partial_insert();
}

void hnsw_set_enable_s3(void* index_ptr, bool enable) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto h = dynamic_cast<HNSW<float>*>(base)) h->set_enable_s3(enable);
}

void hnsw_set_s3_proximity_threshold(void* index_ptr, float threshold) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto h = dynamic_cast<HNSW<float>*>(base)) h->set_s3_proximity_threshold(threshold);
}

void hnsw_set_enable_path_skip(void* index_ptr, bool enable) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto h = dynamic_cast<HNSW<float>*>(base)) h->set_enable_path_skip(enable);
}

void hnsw_set_enable_candidate_injection(void* index_ptr, bool enable) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto h = dynamic_cast<HNSW<float>*>(base)) h->set_enable_candidate_injection(enable);
}

void hnsw_set_enable_search_sharing(void* index_ptr, bool enable) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto h = dynamic_cast<HNSW<float>*>(base)) h->set_enable_search_sharing(enable);
}

void hnsw_set_search_sharing_check_interval(void* index_ptr, int interval) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto h = dynamic_cast<HNSW<float>*>(base)) h->set_search_sharing_check_interval(interval);
}

void hnsw_set_enable_warm_start(void* index_ptr, bool enable) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto h = dynamic_cast<HNSW<float>*>(base)) h->set_enable_warm_start(enable);
}

void annchor_set_enable_mvcc(void* index_ptr, bool enable) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto a = dynamic_cast<ANNchorTrim<float>*>(base)) a->set_enable_mvcc(enable);
    if (auto a = dynamic_cast<ANNchorPreempt<float>*>(base)) a->set_enable_mvcc(enable);
    if (auto a = dynamic_cast<ANNchorM1<float>*>(base)) a->set_enable_mvcc(enable);
    annchor_m2_set_enable_mvcc_float(base, enable);
    annchor_m3_set_enable_mvcc_float(base, enable);
}

void annchor_set_enable_undo_recovery(void* index_ptr, bool enable) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto a = dynamic_cast<ANNchorTrim<float>*>(base)) a->set_enable_undo_recovery(enable);
    if (auto a = dynamic_cast<ANNchorPreempt<float>*>(base)) a->set_enable_undo_recovery(enable);
    if (auto a = dynamic_cast<ANNchorM1<float>*>(base)) a->set_enable_undo_recovery(enable);
    annchor_m2_set_enable_undo_recovery_float(base, enable);
    annchor_m3_set_enable_undo_recovery_float(base, enable);
}

void hnsw_visible_set_visibility_mode(void* index_ptr, int mode) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto v = dynamic_cast<HNSWVisible<float>*>(base)) {
        v->set_visibility_mode(mode);
        return;
    }
    if (auto a = dynamic_cast<ANNchorM1<float>*>(base)) {
        a->set_visibility_mode(mode);
    }
}

void annchor_trim_set_enable_recovery_filter(void* index_ptr, bool enable) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto a = dynamic_cast<ANNchorTrim<float>*>(base)) a->set_enable_trim_recovery_filter(enable);
}

void annchor_trim_set_recovery_relax_factor(void* index_ptr, float factor) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto a = dynamic_cast<ANNchorTrim<float>*>(base)) a->set_trim_recovery_relax_factor(factor);
}

void annchor_trim_set_recovery_margin_ratio(void* index_ptr, float ratio) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto a = dynamic_cast<ANNchorTrim<float>*>(base)) a->set_trim_recovery_margin_ratio(ratio);
}

void annchor_set_enable_s3(void* index_ptr, bool enable) {
    (void)index_ptr;
    (void)enable;
}

void annchor_set_s3_proximity_threshold(void* index_ptr, float threshold) {
    (void)index_ptr;
    (void)threshold;
}

void annchor_set_enable_slipstream(void* index_ptr, bool enable) {
    (void)index_ptr;
    (void)enable;
}

void annchor_m3_set_prune_only_node_lock(void* index_ptr, bool enable) {
    (void)index_ptr;
    (void)enable;
}

void annchor_set_slipstream_ttl(void* index_ptr, uint64_t ttl_ns) {
    (void)index_ptr;
    (void)ttl_ns;
}

void annchor_set_slipstream_quality_gate(void* index_ptr, bool enable) {
    (void)index_ptr;
    (void)enable;
}

void annchor_set_slipstream_skip_ratio(void* index_ptr, float ratio) {
    (void)index_ptr;
    (void)ratio;
}

void annchor_dev_set_enable_cnr(void* index_ptr, bool enable) {
    (void)index_ptr;
    (void)enable;
}

void annchor_dev_set_cnr_degree_threshold(void* index_ptr, float threshold) {
    (void)index_ptr;
    (void)threshold;
}

void annchor_dev_set_cnr_max_recover(void* index_ptr, int max_recover) {
    (void)index_ptr;
    (void)max_recover;
}

void annchor_dev_set_cnr_stagnation_hops(void* index_ptr, int hops) {
    (void)index_ptr;
    (void)hops;
}

void annchor_dev_set_enable_m2_dual_path(void* index_ptr, bool enable) {
    (void)index_ptr;
    (void)enable;
}

void annchor_dev_set_m2_risk_hops(void* index_ptr, int hops) {
    (void)index_ptr;
    (void)hops;
}

void annchor_dev_set_m2_assist_budget(void* index_ptr, int budget) {
    (void)index_ptr;
    (void)budget;
}

void annchor_preempt_set_enable_m2(void* index_ptr, bool enable) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto a = dynamic_cast<ANNchorPreempt<float>*>(base)) a->set_enable_preempt_m2(enable);
}

void annchor_preempt_set_quantum_points(void* index_ptr, int quantum_points) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto a = dynamic_cast<ANNchorPreempt<float>*>(base)) a->set_preempt_quantum_points(quantum_points);
}

void annchor_preempt_set_search_backlog_threshold(void* index_ptr, int threshold) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto a = dynamic_cast<ANNchorPreempt<float>*>(base)) a->set_preempt_search_backlog_threshold(threshold);
}

void annchor_preempt_set_max_yields_per_batch(void* index_ptr, int max_yields_per_batch) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto a = dynamic_cast<ANNchorPreempt<float>*>(base)) a->set_preempt_max_yields_per_batch(max_yields_per_batch);
}

void annchor_preempt_set_budget_window_us(void* index_ptr, int window_us) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto a = dynamic_cast<ANNchorPreempt<float>*>(base)) a->set_preempt_budget_window_us(window_us);
}

void annchor_preempt_set_budget_pct(void* index_ptr, float budget_pct) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto a = dynamic_cast<ANNchorPreempt<float>*>(base)) a->set_preempt_budget_pct(static_cast<double>(budget_pct));
}

void annchor_preempt_set_priority_cap(void* index_ptr, int priority_cap) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto a = dynamic_cast<ANNchorPreempt<float>*>(base)) a->set_preempt_priority_cap(priority_cap);
}

void annchor_preempt_set_runtime_search_backlog(int backlog) {
    ANNchorPreempt<float>::set_runtime_search_backlog(backlog);
}

void annchor_preempt_set_runtime_priority_searches(int priority_searches) {
    ANNchorPreempt<float>::set_runtime_priority_searches(priority_searches);
}


void annchor_m3_set_slipstream_ttl(void* index_ptr, uint64_t ttl_ns) {
    annchor_set_slipstream_ttl(index_ptr, ttl_ns);
}

void annchor_m3_set_slipstream_mode(void* index_ptr, int mode) {
    (void)index_ptr;
    (void)mode;
}

long annchor_get_inflight_points(void* index_ptr) {
    if (!index_ptr) return -1;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    return base->get_inflight_points();
}

size_t annchor_get_inflight_labels(void* index_ptr, uint32_t* out,
                                   size_t max_labels) {
    if (!index_ptr || !out || max_labels == 0) return 0;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    return base->get_inflight_labels(out, max_labels);
}

size_t annchor_get_inflight_labels_before(void* index_ptr, uint32_t* out,
                                          size_t max_labels,
                                          uint64_t snapshot_raw_ns) {
    if (!index_ptr || !out || max_labels == 0) return 0;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    return base->get_inflight_labels_before(out, max_labels, snapshot_raw_ns);
}

int annchor_batch_base_neighbors(void* index_ptr, const uint32_t* labels,
                                 size_t num_labels, size_t max_neighbors,
                                 uint32_t* out, size_t* counts) {
    if (!index_ptr || !labels || !out || !counts) return -1;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    return base->batch_base_neighbors(labels, num_labels, max_neighbors, out, counts);
}

uint64_t graph_link_write_probe(void* index_ptr, const uint32_t* labels,
                                size_t num_labels, size_t loops,
                                size_t max_edges) {
    if (!index_ptr || !labels || num_labels == 0 || loops == 0) return 0;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    return base->graph_link_write_probe(labels, num_labels, loops, max_edges);
}

#if defined(__linux__)
#include <malloc.h>
#endif

}  // extern "C"
