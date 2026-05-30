#pragma once

#include <omp.h>
#include <stdint.h>
#include <limits>

#include <string>
#include <vector>

static constexpr size_t SEARCH_PATH_CAPTURE_CAP = 128;
static constexpr size_t SEARCH_LOCALITY_BUCKET_CAP = 128;

struct GraphMutationStats {
    uint64_t connect_calls;
    uint64_t connect_ns;
    uint64_t link_critical_ns;
    uint64_t unique_lock_wait_ns;
    uint64_t search_unique_lock_wait_ns; // searchBaseLayer per-visit unique_lock
    uint64_t search_unique_lock_acqs;
    uint64_t search_critical_ns;          // time inside that critical section
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
};

struct SearchWorkStats {
    uint64_t searchknn_ns = 0;
    uint64_t searchknn_thread_cpu_ns = 0;
    int32_t work_start_cpu = -1;
    int32_t work_end_cpu = -1;
    uint64_t result_copy_ns = 0;
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
    uint64_t distance_compute_ns = 0;
    uint64_t upper_distance_compute_ns = 0;
    uint64_t level0_distance_compute_ns = 0;
    uint64_t level0_queue_pop_ns = 0;
    uint64_t level0_adj_fetch_ns = 0;
    uint64_t level0_locality_capture_ns = 0;
    uint64_t level0_candidate_loop_ns = 0;
    uint64_t level0_visited_check_ns = 0;
    uint64_t level0_visibility_check_ns = 0;
    uint64_t level0_candidate_accept_ns = 0;
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
    uint64_t rewrite_active_expansions = 0;
    uint64_t rewrite_recent_expansions = 0;
    uint64_t rewrite_period_expansions = 0;
    uint64_t rewrite_period_active_sum = 0;
    uint64_t rewrite_period_active_max = 0;
    uint64_t expand_visible_count = 0;
    uint64_t expand_recent_1k_hits = 0;
    uint64_t expand_recent_4k_hits = 0;
    uint64_t expand_recent_16k_hits = 0;
    uint64_t expand_label_gap_sum = 0;
    uint64_t expand_label_span = 0;
    uint64_t expand_unique_label_4k_buckets = 0;
    uint64_t expand_unique_data_4k_pages = 0;
    uint64_t expand_unique_data_2m_pages = 0;
    uint64_t expand_unique_adj_4k_pages = 0;
    uint64_t expand_unique_adj_2m_pages = 0;
    uint64_t expand_unique_overflow = 0;
    uint32_t expand_label_min = std::numeric_limits<uint32_t>::max();
    uint32_t expand_label_max = 0;
    uint32_t expand_last_label = 0;
    bool expand_label_seen = false;
    uint64_t expand_label_4k_buckets[SEARCH_LOCALITY_BUCKET_CAP] = {};
    uint64_t expand_data_4k_pages[SEARCH_LOCALITY_BUCKET_CAP] = {};
    uint64_t expand_data_2m_pages[SEARCH_LOCALITY_BUCKET_CAP] = {};
    uint64_t expand_adj_4k_pages[SEARCH_LOCALITY_BUCKET_CAP] = {};
    uint64_t expand_adj_2m_pages[SEARCH_LOCALITY_BUCKET_CAP] = {};
    uint32_t expand_label_4k_bucket_count = 0;
    uint32_t expand_data_4k_page_count = 0;
    uint32_t expand_data_2m_page_count = 0;
    uint32_t expand_adj_4k_page_count = 0;
    uint32_t expand_adj_2m_page_count = 0;
    uint32_t path_count = 0;
    uint32_t path_labels[SEARCH_PATH_CAPTURE_CAP] = {};
    float path_dists[SEARCH_PATH_CAPTURE_CAP] = {};
};

struct QParams {
    size_t ef_search;
    size_t beam_width;
    float alpha;
    size_t visit_limit;

    QParams() = default;

    QParams(size_t ef_search) : ef_search(ef_search) {}

    QParams(size_t ef_search, size_t beam_width, float alpha,
            size_t visit_limit)
        : ef_search(ef_search),
          beam_width(beam_width),
          alpha(alpha),
          visit_limit(visit_limit) {}
};

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class IndexBase {
   public:
    virtual ~IndexBase() = default;
    virtual void build(const T* data, const TagT* tags, size_t num_points) = 0;
    virtual int insert(const T* point, const TagT tag) = 0;
    virtual int batch_insert(const T* batch_data, const TagT* batch_tags,
                             size_t num_points) = 0;
    virtual void set_query_params(const QParams& params) = 0;
    virtual int search(const T* query, size_t k,
                       std::vector<TagT>& res_tags) = 0;
    virtual int batch_search(const T* batch_queries, size_t k,
                             size_t num_queries, TagT** batch_results,
                             size_t* watermark_out = nullptr,
                             size_t visible_ts = std::numeric_limits<size_t>::max()) = 0;
    virtual int batch_search_measured(const T* batch_queries, size_t k,
                                      size_t num_queries, TagT** batch_results,
                                      size_t* watermark_out = nullptr,
                                      size_t visible_ts = std::numeric_limits<size_t>::max()) {
        return batch_search(batch_queries, k, num_queries, batch_results,
                            watermark_out, visible_ts);
    }
    virtual int batch_search_measured_work(
        const T* batch_queries, size_t k, size_t num_queries,
        TagT** batch_results, SearchWorkStats* per_query_stats,
        size_t* watermark_out = nullptr,
        size_t visible_ts = std::numeric_limits<size_t>::max()) {
        (void)per_query_stats;
        return batch_search_measured(batch_queries, k, num_queries,
                                     batch_results, watermark_out, visible_ts);
    }
    virtual int batch_search_path_work(
        const T* batch_queries, size_t k, size_t num_queries,
        TagT** batch_results, SearchWorkStats* per_query_stats,
        size_t* watermark_out = nullptr,
        size_t visible_ts = std::numeric_limits<size_t>::max()) {
        (void)per_query_stats;
        return batch_search(batch_queries, k, num_queries, batch_results,
                            watermark_out, visible_ts);
    }
    virtual void dump_stats(std::string& str) {}

    virtual bool supports_snapshot() const { return false; }
    virtual int snapshot(std::vector<uint8_t>& out) { return -1; }
    virtual int restore(const uint8_t* data, size_t size) { return -1; }

    // OCC inflight metric. Returns -1 if not supported by this index type.
    virtual long get_inflight_points() const { return -1; }

    // Export labels from insert batches currently active inside the index.
    // Returns the number written into out.
    virtual size_t get_inflight_labels(TagT* out, size_t max_labels) const {
        (void)out;
        (void)max_labels;
        return 0;
    }

    // Export labels for insert batches that had already started by
    // snapshot_raw_ns and had not completed at that time.
    virtual size_t get_inflight_labels_before(TagT* out, size_t max_labels,
                                              uint64_t snapshot_raw_ns) const {
        (void)snapshot_raw_ns;
        return get_inflight_labels(out, max_labels);
    }

    // Best-effort base-layer graph neighbor export for graph-native inflight
    // OCC filters. Implementations return one count per input label and write
    // at most max_neighbors labels into out[i * max_neighbors:].
    virtual int batch_base_neighbors(const TagT* labels, size_t num_labels,
                                     size_t max_neighbors, TagT* out,
                                     size_t* counts) const {
        (void)labels;
        (void)num_labels;
        (void)max_neighbors;
        (void)out;
        (void)counts;
        return -1;
    }

    // Experiment-only graph memory write probe. Implementations may write the
    // same values back into base-layer link-list storage for the supplied
    // labels, returning the number of cache-line-visible scalar writes. The
    // default is unsupported and leaves graph semantics unchanged.
    virtual uint64_t graph_link_write_probe(const TagT* labels,
                                            size_t num_labels, size_t loops,
                                            size_t max_edges) {
        (void)labels;
        (void)num_labels;
        (void)loops;
        (void)max_edges;
        return 0;
    }

    virtual bool graph_mutation_stats(GraphMutationStats* out) const {
        (void)out;
        return false;
    }
};
