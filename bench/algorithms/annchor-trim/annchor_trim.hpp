#pragma once

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include "../index.hpp"
#include "../index_cgo.hpp"
#include "../annchor/ANNchor/src/hnswlib/hnswlib.h"
#include <memory>

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class ANNchorTrim : public IndexBase<T, TagT, LabelT> {
   public:
    ANNchorTrim(size_t max_elements, size_t dim, size_t num_threads, size_t M,
                size_t ef_construction, bool use_node_lock_in_search = true,
                MetricType metric = METRIC_L2)
        : dim_(dim),
          num_threads_(num_threads),
          M_(M),
          ef_c_(ef_construction),
          ef_s_(0),
          max_elements_(max_elements),
          use_node_lock_in_search_(use_node_lock_in_search),
          metric_(metric),
          trim_filter_enabled_(metric == METRIC_L2),
          trim_recovery_relax_factor_(kDefaultTrimRelaxFactor),
          trim_recovery_margin_ratio_(kDefaultTrimMarginRatio),
          arena_(num_threads) {
        if (metric_ == METRIC_IP || metric_ == METRIC_COSINE) {
            space_.reset(new annchor::InnerProductSpace(dim));
        } else {
            space_.reset(new annchor::L2Space(dim));
        }
        index_ = new annchor::HierarchicalNSW<T>(
            space_.get(), max_elements, M, ef_construction, 100, false,
            use_node_lock_in_search, num_threads_);
        configure_trim_filter();
    }

    ~ANNchorTrim() override {
        delete index_;
    }

    void build(const T* data, const TagT* tags, size_t num_points) override {
        (void)batch_insert(data, tags, num_points);
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
            arena_.execute([&] {
                tbb::parallel_for(size_t(0), num_points, [&](size_t i) {
                    const T* point = temp_batch.data() + i * dim_;
                    auto id = index_->addPoint(
                        point, static_cast<annchor::labeltype>(batch_tags[i]), -1, mvcc);
                    if (mvcc) index_->markReady(id);
                });
            });
        } else {
            arena_.execute([&] {
                tbb::parallel_for(size_t(0), num_points, [&](size_t i) {
                    const T* point = batch_data + i * dim_;
                    auto id = index_->addPoint(
                        point, static_cast<annchor::labeltype>(batch_tags[i]), -1, mvcc);
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
        std::priority_queue<std::pair<T, annchor::labeltype>> result;
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
        size_t wm = visible_ts;
        if (watermark_out) *watermark_out = wm;

        arena_.execute([&] {
            tbb::parallel_for(size_t(0), num_queries, [&](size_t i) {
                const T* q = batch_queries + i * dim_;
                std::priority_queue<std::pair<T, annchor::labeltype>> result;

                if (metric_ == METRIC_COSINE) {
                    std::vector<T> temp_query(q, q + dim_);
                    normalize_vector(temp_query.data(), 1);
                    result = index_->searchKnn(temp_query.data(), k, nullptr, wm);
                } else {
                    result = index_->searchKnn(q, k, nullptr, wm);
                }

                size_t sz = std::min(result.size(), k);
                for (size_t j = 0; j < sz; ++j) {
                    batch_results[i][sz - j - 1] = result.top().second;
                    result.pop();
                }
            });
        });
        return 0;
    }

    void dump_stats(std::string& str) override {
        std::stringstream ss;
        ss << "index_memory_mb:" << index_->indexFileSize() / (1024 * 1024)
           << ", dist_computations:" << index_->metric_distance_computations.load()
           << ", hops:" << index_->metric_hops.load()
           << ", undo_recovery_enabled:" << (index_->isUndoRecoveryEnabled() ? 1 : 0)
           << ", trim_filter_enabled:" << (index_->isTrimRecoveryFilterEnabled() ? 1 : 0)
           << ", trim_relax_factor:" << index_->getTrimRecoveryRelaxFactor()
           << ", undo_lookups:" << index_->metric_undo_log_lookups.load()
           << ", recovered_edges_total:" << index_->metric_recovered_edges_total.load()
           << ", recovered_edges_useful:" << index_->metric_recovered_edges_useful.load()
           << ", recovered_single_node_max:" << index_->metric_recovered_edges_single_node_max.load()
           << ", trim_candidates_estimated:" << index_->metric_trim_candidates_estimated.load()
           << ", trim_candidates_filtered:" << index_->metric_trim_candidates_filtered.load()
           << ", trim_candidates_exact:" << index_->metric_trim_candidates_exact.load()
           << ", trim_groups_total:" << index_->metric_trim_groups_total.load()
           << ", trim_groups_pruned:" << index_->metric_trim_groups_pruned.load()
           << ", trim_margin_ratio:" << index_->getTrimRecoveryMarginRatio()
           << ", trim_margin_filtered:" << index_->metric_trim_margin_filtered.load();

        str = ss.str();
    }

    void set_enable_mvcc(bool enable) {
        index_->setEnableMvcc(enable);
    }
    bool is_mvcc_enabled() const { return index_->isMvccEnabled(); }
    void set_enable_undo_recovery(bool enable) {
        index_->setEnableUndoRecovery(enable);
    }
    void set_enable_trim_recovery_filter(bool enable) {
        trim_filter_enabled_ = enable;
        index_->setEnableTrimRecoveryFilter(enable);
    }
    bool is_trim_recovery_filter_enabled() const {
        return trim_filter_enabled_;
    }
    void set_trim_recovery_relax_factor(float factor) {
        trim_recovery_relax_factor_ = (factor < 1.0f) ? 1.0f : factor;
        index_->setTrimRecoveryRelaxFactor(trim_recovery_relax_factor_);
    }
    float get_trim_recovery_relax_factor() const {
        return trim_recovery_relax_factor_;
    }
    void set_trim_recovery_margin_ratio(float ratio) {
        trim_recovery_margin_ratio_ = ratio;
        index_->setTrimRecoveryMarginRatio(ratio);
    }
    float get_trim_recovery_margin_ratio() const {
        return trim_recovery_margin_ratio_;
    }
    bool is_undo_recovery_enabled() const {
        return index_->isUndoRecoveryEnabled();
    }

    size_t compact(long safe_ts = -1) { return index_->compact(safe_ts); }

    bool supports_snapshot() const override { return true; }

    int snapshot(std::vector<uint8_t>& out) override {
        std::string tmp_file = "/tmp/annchor_trim_snapshot_" + std::to_string((uintptr_t)this) + ".bin";
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
        std::string tmp_file = "/tmp/annchor_trim_restore_" + std::to_string((uintptr_t)this) + ".bin";
        std::ofstream file(tmp_file, std::ios::binary);
        if (!file.write((const char*)data, size)) {
            return -1;
        }
        file.close();

        try {
            delete index_;
            index_ = new annchor::HierarchicalNSW<T>(
                space_.get(), tmp_file, false, max_elements_, false);
            configure_trim_filter();
        } catch (...) {
            return -1;
        }
        std::remove(tmp_file.c_str());
        return 0;
    }

   private:
    static constexpr float kDefaultTrimRelaxFactor = 1.5f;
    static constexpr float kDefaultTrimMarginRatio = -1.0f;

    size_t num_threads_;
    size_t dim_;
    size_t M_;
    size_t ef_c_;
    size_t ef_s_;
    size_t max_elements_;
    bool use_node_lock_in_search_;
    MetricType metric_;
    bool trim_filter_enabled_;
    float trim_recovery_relax_factor_;
    float trim_recovery_margin_ratio_;
    std::unique_ptr<annchor::SpaceInterface<T>> space_;
    annchor::HierarchicalNSW<T>* index_;
    tbb::task_arena arena_;

    void configure_trim_filter() {
        index_->setEnableTrimRecoveryFilter(trim_filter_enabled_);
        index_->setTrimRecoveryRelaxFactor(trim_recovery_relax_factor_);
        index_->setTrimRecoveryMarginRatio(trim_recovery_margin_ratio_);
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
};
