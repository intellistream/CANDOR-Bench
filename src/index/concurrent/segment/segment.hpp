#pragma once

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <limits>

#include "index.hpp"
#include "../index_types.hpp"
#include <tbb/task_arena.h>

// SIMD-accelerated L2 for float buffers; implemented in segment.cpp.
float l2_distance_simd(const float* a, const float* b, size_t dim);

// Segmented index: sealed ANN segments + a growing brute-force buffer.
class SegmentIndex : public IndexBase<float> {
   public:
    explicit SegmentIndex(const IndexParams& params);
    ~SegmentIndex() override;

    void build(const float* data, const uint32_t* tags,
               size_t num_points) override;
    int insert(const float* point, const uint32_t tag) override;
    int batch_insert(const float* batch_data, const uint32_t* batch_tags,
                     size_t num_points) override;
    void set_query_params(const QParams& params) override;
    int search(const float* query, size_t k,
               std::vector<uint32_t>& result_tags) override;
    int batch_search(const float* batch_queries, size_t k, size_t num_queries,
                     uint32_t** batch_results, size_t* watermark_out = nullptr,
                     size_t visible_ts = std::numeric_limits<size_t>::max()) override;
    int batch_search_measured_work(
        const float* batch_queries, size_t k, size_t num_queries,
        uint32_t** batch_results, SearchWorkStats* per_query_stats,
        size_t* watermark_out = nullptr,
        size_t visible_ts = std::numeric_limits<size_t>::max()) override;
    void dump_stats(std::string& str) override;

   private:
    struct Snapshot {
        std::vector<uint32_t> buffer_tags;
        std::vector<std::shared_ptr<std::vector<uint32_t>>> raw_segments;
        std::vector<std::shared_ptr<IndexBase<float>>> sealed_segments;
        std::shared_ptr<IndexBase<float>> active_delta;
        size_t active_delta_points = 0;
    };

    struct BuildJob {
        std::shared_ptr<std::vector<uint32_t>> tags;
    };

    struct SegmentSearchStats {
        uint64_t sealed_segments = 0;
        uint64_t raw_segments = 0;
        uint64_t raw_points = 0;
        uint64_t buffer_points = 0;
        uint64_t active_delta_points = 0;
        uint64_t sealed_result_rechecks = 0;
        uint64_t candidates = 0;
        uint64_t sealed_search_ns = 0;
        uint64_t active_delta_search_ns = 0;
        uint64_t exact_distance_ns = 0;
        uint64_t selection_ns = 0;
    };

    IndexParams params_;
    size_t dim_;
    size_t max_elements_;
    IndexType sealed_type_;
    int seal_threshold_;
    size_t builder_threads_;
    bool active_delta_hnsw_{false};
    std::vector<float> store_;
    tbb::task_arena search_arena_;
    bool measured_search_arena_enabled_{false};
    tbb::task_arena measured_search_arena_;

    mutable std::mutex mu_;
    std::vector<uint32_t> buffer_tags_;
    std::vector<std::shared_ptr<std::vector<uint32_t>>> raw_segments_;
    std::vector<std::shared_ptr<IndexBase<float>>> sealed_segments_;
    std::shared_ptr<IndexBase<float>> active_delta_;
    size_t active_delta_points_{0};
    size_t active_delta_rotations_{0};
    QParams current_qparams_{};
    bool has_qparams_{false};

    std::mutex active_delta_insert_mu_;
    std::mutex builder_mu_;
    std::condition_variable builder_cv_;
    std::deque<BuildJob> build_queue_;
    bool stop_builder_{false};
    std::thread builder_thread_;
    size_t seal_jobs_enqueued_{0};
    size_t seal_jobs_completed_{0};
    mutable std::atomic<uint64_t> metric_search_queries_{0};
    mutable std::atomic<uint64_t> metric_search_raw_points_{0};
    mutable std::atomic<uint64_t> metric_search_buffer_points_{0};
    mutable std::atomic<uint64_t> metric_search_active_delta_points_{0};
    mutable std::atomic<uint64_t> metric_search_sealed_result_rechecks_{0};
    mutable std::atomic<uint64_t> metric_search_candidates_{0};
    mutable std::atomic<uint64_t> metric_search_sealed_segments_{0};
    mutable std::atomic<uint64_t> metric_search_raw_segments_{0};
    mutable std::atomic<uint64_t> metric_search_sealed_search_ns_{0};
    mutable std::atomic<uint64_t> metric_search_active_delta_search_ns_{0};
    mutable std::atomic<uint64_t> metric_search_exact_distance_ns_{0};
    mutable std::atomic<uint64_t> metric_search_selection_ns_{0};

    IndexType resolve_sealed_type(IndexType requested) const;
    const float* vector_for_tag(uint32_t tag) const;
    void cache_vectors(const float* data, const uint32_t* tags,
                       size_t num_points);
    std::shared_ptr<IndexBase<float>> build_segment(const float* data,
                                                    const uint32_t* tags,
                                                    size_t num_points,
                                                    size_t build_threads);
    std::shared_ptr<IndexBase<float>> build_segment_from_tags(
        const std::shared_ptr<std::vector<uint32_t>>& tags);
    std::unique_ptr<IndexBase<float>> create_base_index(size_t capacity,
                                                        size_t threads,
                                                        bool use_node_lock) const;
    std::shared_ptr<IndexBase<float>> create_active_delta_index() const;
    int batch_insert_active_delta(const float* batch_data,
                                  const uint32_t* batch_tags,
                                  size_t num_points);
    void rotate_active_delta_locked();
    void enqueue_seal(std::shared_ptr<std::vector<uint32_t>> tags);
    void builder_loop();
    Snapshot take_snapshot() const;
    void search_with_snapshot(const float* query, size_t k,
                              const Snapshot& snap,
                              std::vector<uint32_t>& result_tags,
                              SegmentSearchStats* stats = nullptr) const;
    int batch_search_impl(const float* batch_queries, size_t k,
                          size_t num_queries, uint32_t** batch_results,
                          SearchWorkStats* per_query_stats,
                          bool measured);
    tbb::task_arena& search_task_arena(bool measured);
};
