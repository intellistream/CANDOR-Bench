#pragma once

#include <omp.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>

#include "index.hpp"
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
#pragma omp parallel for num_threads(num_threads_)
        for (size_t i = 0; i < num_points; i++) {
            index_->addPoint((void*)(data + i * dim_), tags[i]);
        }
    }

    int insert(const T* data, const TagT tag) override {
        index_->addPoint(data, tag);
        return 0;
    }

    int batch_insert(const T* batch_data, const TagT* batch_tags,
                     size_t num_points) override {
        if (scheduler_ == HNSWWorkScheduler::kThreadPool) {
            return batch_insert_thread_pool(batch_data, batch_tags, num_points);
        }
        int success_count = 0;
        arena_.execute([&] {
            tbb::parallel_for(size_t(0), num_points, [&](size_t i) {
                index_->addPoint(batch_data + i * dim_, batch_tags[i]);
                __sync_fetch_and_add(&success_count, 1);
            });
        });
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

        arena_.execute([&] {
            tbb::parallel_for(size_t(0), num_points, [&](size_t i) {
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
        bool use_ps = !suspended && index_->isPathSkipEnabled();
        bool use_ci = !suspended && index_->isCandidateInjectionEnabled();
        arena_.execute([&] {
            tbb::parallel_for(size_t(0), num_queries, [&](size_t i) {
                const T* q = batch_queries + i * dim_;
                auto result = [&]() {
                    if (use_ps && use_ci) return index_->searchKnnCombined(q, k);
                    else if (use_ps) return index_->searchKnnPathSkip(q, k);
                    else if (use_ci) return index_->searchKnnCandidateInjection(q, k);
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

    void dump_stats(std::string& str) {
        std::stringstream ss;
        ss << "index_memory_mb:" << index_->getIndexMemory() / (1024 * 1024)
           << ", dist_computations:"
           << index_->metric_distance_computations.load()
           << ", hops:" << index_->metric_hops.load();
        if (index_->isS3Enabled() || index_->isPathSkipEnabled() || index_->isCandidateInjectionEnabled()) {
            ss << ", s3_hits:" << index_->getS3Hits()
               << ", s3_misses:" << index_->getS3Misses()
               << ", s3_dist_saved:" << index_->getS3DistSaved()
               << ", s3_adaptive_accepts:" << index_->getS3AdaptiveAccepts()
               << ", s3_adaptive_rejects:" << index_->getS3AdaptiveRejects();
        }
        str = ss.str();
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
    
    // WarmStart not implemented in underlying HNSW library
    void set_enable_warm_start(bool enable) { /* Not implemented */ }
    bool is_warm_start_enabled() const { return false; }

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
    tbb::task_arena arena_;
    HNSWWorkScheduler scheduler_;

    int batch_insert_thread_pool(const T* batch_data, const TagT* batch_tags,
                                 size_t num_points) {
        if (num_points == 0) {
            return 0;
        }
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
