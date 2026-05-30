#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define C_SEARCH_PATH_CAPTURE_CAP 128

typedef enum {
    INDEX_TYPE_HNSW = 0,
    INDEX_TYPE_PARLAYHNSW = 1,
    INDEX_TYPE_PARLAYVAMANA = 2,
    INDEX_TYPE_VAMANA = 3,
    INDEX_TYPE_ANNCHOR = 4,
    INDEX_TYPE_SEGMENT = 5,
    INDEX_TYPE_ANNCHOR_DEV = 6,
    INDEX_TYPE_ANNCHOR_M1 = 7,
    INDEX_TYPE_ANNCHOR_PREEMPT = 8,
    INDEX_TYPE_ANNCHOR_M3 = 9,
    INDEX_TYPE_ANNCHOR_TRIM = 10,
    INDEX_TYPE_ANNCHOR_M2 = 11,
    INDEX_TYPE_HNSW_VISIBLE = 12,
} IndexType;

void annchor_m3_set_slipstream_ttl(void* index_ptr, uint64_t ttl_ns);
void annchor_m3_set_slipstream_mode(void* index_ptr, int mode);
void annchor_m3_set_prune_only_node_lock(void* index_ptr, bool enable);

typedef enum {
    DATA_TYPE_FLOAT = 0,
    DATA_TYPE_INT8 = 1,
    DATA_TYPE_UINT8 = 2,
} DataType;

typedef enum {
    METRIC_L2 = 0,
    METRIC_IP = 1,
    METRIC_COSINE = 2,
} MetricType;

typedef struct {
    size_t dim;
    size_t max_elements;
    size_t M;
    size_t ef_construction;
    float level_m;
    float alpha;
    size_t visit_limit;
    size_t seal_threshold;
    size_t num_threads;
    DataType data_type;
    bool use_node_lock;
    int metric;
    size_t worker_scheduler;
    IndexType sealed_index_type;
} IndexParams;

typedef struct {
    size_t ef_search;
    size_t beam_width;
    float alpha;
    size_t visit_limit;
} C_QueryParams;

typedef struct {
    uint64_t connect_calls;
    uint64_t connect_ns;
    uint64_t link_critical_ns;
    uint64_t unique_lock_wait_ns;
    uint64_t search_unique_lock_wait_ns;
    uint64_t search_unique_lock_acqs;
    uint64_t search_critical_ns;
    uint64_t link_updates;
    uint64_t upper_search_ns;
    uint64_t upper_search_dist_comps;
    uint64_t upper_search_edges_scanned;
    uint64_t base_search_ns;
    uint64_t base_search_expansions;
    uint64_t base_search_edges_scanned;
    uint64_t base_search_dist_comps;
    uint64_t select_new_neighbors_ns;
    uint64_t select_new_neighbors_input;
    uint64_t select_new_neighbors_selected;
    uint64_t select_new_neighbors_heuristic_dist_comps;
    uint64_t inserted_node_link_ns;
    uint64_t inserted_node_edges_written;
    uint64_t existing_neighbor_update_loop_ns;
    uint64_t existing_neighbor_load_scan_ns;
    uint64_t existing_neighbor_loaded_edges;
    uint64_t existing_neighbor_append_ns;
    uint64_t existing_neighbor_prune_ns;
    uint64_t existing_neighbor_prune_candidates;
    uint64_t existing_neighbor_prune_dist_comps;
    uint64_t existing_neighbor_undo_record_ns;
    uint64_t existing_neighbor_rewrite_ns;
    uint64_t existing_neighbor_edges_written;
    uint64_t existing_neighbor_edges_pruned;
    uint64_t existing_neighbor_visits;
    uint64_t existing_neighbor_appends;
    uint64_t existing_neighbor_prunes;
    uint64_t existing_neighbor_pruned_edges_recorded;
} C_GraphMutationStats;

typedef struct {
    uint64_t searchknn_ns;
    uint64_t searchknn_thread_cpu_ns;
    int32_t work_start_cpu;
    int32_t work_end_cpu;
    uint64_t result_copy_ns;
    uint64_t entry_ns;
    uint64_t upper_search_ns;
    uint64_t base_search_ns;
    uint64_t result_materialize_ns;
    uint64_t snapshot_guard_ns;
    uint64_t visited_list_get_ns;
    uint64_t visited_list_release_ns;
    uint64_t upper_lock_wait_ns;
    uint64_t level0_lock_wait_ns;
    uint64_t distance_computations;
    uint64_t upper_distance_computations;
    uint64_t level0_distance_computations;
    uint64_t distance_compute_ns;
    uint64_t upper_distance_compute_ns;
    uint64_t level0_distance_compute_ns;
    uint64_t level0_queue_pop_ns;
    uint64_t level0_adj_fetch_ns;
    uint64_t level0_locality_capture_ns;
    uint64_t level0_candidate_loop_ns;
    uint64_t level0_visited_check_ns;
    uint64_t level0_visibility_check_ns;
    uint64_t level0_candidate_accept_ns;
    uint64_t upper_hops;
    uint64_t upper_edges_scanned;
    uint64_t level0_expansions;
    uint64_t level0_edges_scanned;
    uint64_t candidate_pops;
    uint64_t candidate_pushes;
    uint64_t visited_nodes;
    uint64_t result_pushes;
    uint64_t invisible_expansions;
    uint64_t invisible_edges;
    uint64_t invisible_candidate_dist_comps;
    uint64_t invisible_candidate_enqueues;
    uint64_t future_skip_hops;
    uint64_t rewrite_active_expansions;
    uint64_t rewrite_recent_expansions;
    uint64_t rewrite_period_expansions;
    uint64_t rewrite_period_active_sum;
    uint64_t rewrite_period_active_max;
    uint64_t expand_visible_count;
    uint64_t expand_recent_1k_hits;
    uint64_t expand_recent_4k_hits;
    uint64_t expand_recent_16k_hits;
    uint64_t expand_label_gap_sum;
    uint64_t expand_label_span;
    uint64_t expand_unique_label_4k_buckets;
    uint64_t expand_unique_data_4k_pages;
    uint64_t expand_unique_data_2m_pages;
    uint64_t expand_unique_adj_4k_pages;
    uint64_t expand_unique_adj_2m_pages;
    uint64_t expand_unique_overflow;
    uint32_t path_count;
    uint32_t path_labels[C_SEARCH_PATH_CAPTURE_CAP];
    float path_dists[C_SEARCH_PATH_CAPTURE_CAP];
} C_SearchWorkStats;

typedef struct {
    size_t query_result_prefix;
    size_t small_fresh_threshold;
    double candidate_floor_pct;
} C_FreshJoinParams;

typedef struct {
    size_t queries;
    size_t fresh_vectors;
    size_t active_placement_centers;
    size_t active_placement_regions;
    size_t selected_candidates;
    size_t exacted_candidates;
    size_t exact_fallback_queries;
    size_t floor_fallback_queries;
} C_FreshJoinStats;

void* create_index(IndexType type, IndexParams params);
void destroy_index(void* index_ptr);

int build(void* index_ptr, float* data, uint32_t* tags, size_t num_points);
void set_query_params(void* index_ptr, C_QueryParams params);
int search(void* index_ptr, float* query, size_t k, uint32_t* res_tags);
int batch_insert(void* index_ptr, float* batch_data, uint32_t* batch_tags,
                 size_t batch_size);
int batch_partial_insert(void* index_ptr, float* batch_data,
                              uint32_t* batch_tags, size_t batch_size,
                              size_t* partial_limits, size_t* update_counts);
int batch_search(void* index_ptr, float* batch_queries, size_t k,
                 size_t num_queries, uint32_t** batch_results, size_t* watermark_out,
                 size_t visible_ts);
int batch_search_measured(void* index_ptr, float* batch_queries, size_t k,
                          size_t num_queries, uint32_t** batch_results,
                          size_t* watermark_out, size_t visible_ts);
int batch_search_measured_work(void* index_ptr, float* batch_queries, size_t k,
                               size_t num_queries, uint32_t** batch_results,
                               C_SearchWorkStats* per_query_stats,
                               size_t* watermark_out, size_t visible_ts);
int batch_search_path_work(void* index_ptr, float* batch_queries, size_t k,
                           size_t num_queries, uint32_t** batch_results,
                           C_SearchWorkStats* per_query_stats,
                           size_t* watermark_out, size_t visible_ts);
int annchor_fresh_join_merge(void* index_ptr, float* batch_queries, size_t k,
                             size_t num_queries, float* fresh_data,
                             uint32_t* fresh_tags, size_t fresh_count,
                             uint32_t* committed_results,
                             uint32_t** batch_results,
                             size_t committed_view_limit,
                             C_FreshJoinParams params,
                             C_FreshJoinStats* stats);
int annchor_fresh_join_merge_labels(void* index_ptr, float* batch_queries,
                                    size_t k, size_t num_queries,
                                    uint32_t* fresh_tags, size_t fresh_count,
                                    uint32_t* committed_results,
                                    uint32_t** batch_results,
                                    C_FreshJoinParams params,
                                    C_FreshJoinStats* stats);

void save_stat(void* index_ptr, const char* filename);

void dump_stats(void* index_ptr, char* str);
size_t dump_stats_len(void* index_ptr);
void dump_stats_copy(void* index_ptr, char* str, size_t len);
bool graph_mutation_stats(void* index_ptr, C_GraphMutationStats* out);
uint64_t annbench_monotonic_raw_ns(void);

int snapshot_index(void* index_ptr, uint8_t** buffer, size_t* size);
int restore_index(void* index_ptr, const uint8_t* buffer, size_t size);
void free_snapshot_buffer(uint8_t* buffer);

void hnsw_enable_insert_telemetry(void* index_ptr, bool enabled);
size_t hnsw_last_insert_updates(void* index_ptr);
void hnsw_configure_partial_insert(void* index_ptr, size_t limit);
void hnsw_disable_partial_insert(void* index_ptr);
void hnsw_set_enable_s3(void* index_ptr, bool enable);
void hnsw_set_s3_proximity_threshold(void* index_ptr, float threshold);
void hnsw_set_enable_path_skip(void* index_ptr, bool enable);
void hnsw_set_enable_candidate_injection(void* index_ptr, bool enable);
void hnsw_set_enable_search_sharing(void* index_ptr, bool enable);
void hnsw_set_search_sharing_check_interval(void* index_ptr, int interval);
void hnsw_set_enable_warm_start(void* index_ptr, bool enable);

void annchor_set_enable_mvcc(void* index_ptr, bool enable);
void annchor_set_enable_undo_recovery(void* index_ptr, bool enable);
void hnsw_visible_set_visibility_mode(void* index_ptr, int mode);
void annchor_trim_set_enable_recovery_filter(void* index_ptr, bool enable);
void annchor_trim_set_recovery_relax_factor(void* index_ptr, float factor);
void annchor_trim_set_recovery_margin_ratio(void* index_ptr, float ratio);
void annchor_set_enable_s3(void* index_ptr, bool enable);
void annchor_dev_set_enable_cnr(void* index_ptr, bool enable);
void annchor_dev_set_cnr_degree_threshold(void* index_ptr, float threshold);
void annchor_dev_set_cnr_max_recover(void* index_ptr, int max_recover);
void annchor_dev_set_cnr_stagnation_hops(void* index_ptr, int hops);
void annchor_dev_set_enable_m2_dual_path(void* index_ptr, bool enable);
void annchor_dev_set_m2_risk_hops(void* index_ptr, int hops);
void annchor_dev_set_m2_assist_budget(void* index_ptr, int budget);
void annchor_set_s3_proximity_threshold(void* index_ptr, float threshold);
void annchor_set_enable_slipstream(void* index_ptr, bool enable);
void annchor_set_slipstream_ttl(void* index_ptr, uint64_t ttl_ns);
void annchor_set_slipstream_quality_gate(void* index_ptr, bool enable);
void annchor_set_slipstream_skip_ratio(void* index_ptr, float ratio);

void annchor_preempt_set_enable_m2(void* index_ptr, bool enable);
void annchor_preempt_set_quantum_points(void* index_ptr, int quantum_points);
void annchor_preempt_set_search_backlog_threshold(void* index_ptr, int threshold);
void annchor_preempt_set_max_yields_per_batch(void* index_ptr, int max_yields_per_batch);
void annchor_preempt_set_budget_window_us(void* index_ptr, int window_us);
void annchor_preempt_set_budget_pct(void* index_ptr, float budget_pct);
void annchor_preempt_set_priority_cap(void* index_ptr, int priority_cap);
void annchor_preempt_set_runtime_search_backlog(int backlog);
void annchor_preempt_set_runtime_priority_searches(int priority_searches);

// OCC inflight metric: returns (cur_element_count - watermark - 1), i.e. points
// that have entered addPoint but are not yet committed to the query-visible
// watermark. Returns -1 if the index is not ANNchor or MVCC is disabled.
long annchor_get_inflight_points(void* index_ptr);

// Export labels of real ANNchor-side in-flight points. These are points that
// have entered addPoint inside the C++ index but have not completed insertion.
// Returns the number of labels written into out.
size_t annchor_get_inflight_labels(void* index_ptr, uint32_t* out,
                                   size_t max_labels);
size_t annchor_get_inflight_labels_before(void* index_ptr, uint32_t* out,
                                          size_t max_labels,
                                          uint64_t snapshot_raw_ns);

// Export base-layer graph neighbors for labels. Returns 0 on success.
int annchor_batch_base_neighbors(void* index_ptr, const uint32_t* labels,
                                 size_t num_labels, size_t max_neighbors,
                                 uint32_t* out, size_t* counts);

uint64_t graph_link_write_probe(void* index_ptr, const uint32_t* labels,
                                size_t num_labels, size_t loops,
                                size_t max_edges);
#ifdef __cplusplus
}
#endif
