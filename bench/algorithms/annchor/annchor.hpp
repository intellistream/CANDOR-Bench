#pragma once

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

#include "../index.hpp"
#include "../index_cgo.hpp"
#include "ANNchor/src/hnswlib/hnswlib.h"
#include <memory>
#include <cmath>

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class ANNchor : public IndexBase<T, TagT, LabelT> {
   public:
    ANNchor(size_t max_elements, size_t dim, size_t num_threads, size_t M,
           size_t ef_construction, bool use_node_lock_in_search = true, MetricType metric = METRIC_L2)
        : dim_(dim),
          num_threads_(num_threads),
          M_(M),
          ef_c_(ef_construction),
          ef_s_(0),
          max_elements_(max_elements),
          use_node_lock_in_search_(use_node_lock_in_search),
          metric_(metric),
          arena_(num_threads) {
        if (metric_ == METRIC_IP || metric_ == METRIC_COSINE) {
            space_.reset(new annchor::InnerProductSpace(dim));
        } else {
            space_.reset(new annchor::L2Space(dim));
        }
        index_ = new annchor::HierarchicalNSW<T>(
            space_.get(), max_elements, M, ef_construction, 100, false, use_node_lock_in_search, num_threads_);
    }

    ~ANNchor() override {
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

    // EQR (Epoch-aware Query Reordering): store centroid of last insert batch.
    // Search queries are reordered by proximity to this centroid so that
    // spatially-close searches execute together, benefiting from cache lines
    // warmed by the preceding insert batch. MVCC-specific: only MVCC knows
    // the epoch boundary (which inserts just happened).
    std::vector<T> last_insert_centroid_;
    bool eqr_enabled_{false};

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

        // EQR: compute centroid of this insert batch for query reordering
        if (eqr_enabled_ && num_points > 0) {
            const T* src = batch_data;
            last_insert_centroid_.resize(dim_);
            std::fill(last_insert_centroid_.begin(), last_insert_centroid_.end(), T(0));
            for (size_t i = 0; i < num_points; ++i) {
                for (size_t d = 0; d < dim_; ++d) {
                    last_insert_centroid_[d] += src[i * dim_ + d];
                }
            }
            T inv = T(1) / T(num_points);
            for (size_t d = 0; d < dim_; ++d) {
                last_insert_centroid_[d] *= inv;
            }
        }

        return 0;
    }

    // S3-enabled search
    int search_s3(const T* query, size_t k, std::vector<TagT>& result_tags) {
        std::priority_queue<std::pair<T, annchor::labeltype>> result;
        if (metric_ == METRIC_COSINE) {
            std::vector<T> temp_query(query, query + dim_);
            normalize_vector(temp_query.data(), 1);
            result = index_->searchKnnS3(temp_query.data(), k);
        } else {
            result = index_->searchKnnS3(query, k);
        }

        while (!result.empty()) {
            result_tags.push_back(result.top().second);
            result.pop();
        }
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

        bool use_s3 = index_->isS3Enabled() || index_->isSlipstreamEnabled();

        // EQR: Reorder queries by proximity to last insert centroid.
        // Spatially-close queries execute together within the same TBB grain,
        // maximizing cache reuse from both the preceding insert batch and
        // from each other. Only under MVCC: epoch boundary tells us where
        // inserts just landed.
        std::vector<size_t> order(num_queries);
        std::iota(order.begin(), order.end(), 0);

        if (eqr_enabled_ && !last_insert_centroid_.empty() && num_queries > 1) {
            // Compute L2 distance from each query to insert centroid
            std::vector<T> dists(num_queries);
            const T* c = last_insert_centroid_.data();
            for (size_t i = 0; i < num_queries; ++i) {
                const T* q = batch_queries + i * dim_;
                T d = 0;
                for (size_t dd = 0; dd < dim_; ++dd) {
                    T diff = q[dd] - c[dd];
                    d += diff * diff;
                }
                dists[i] = d;
            }
            // Sort indices by distance to centroid
            std::sort(order.begin(), order.end(),
                      [&dists](size_t a, size_t b) { return dists[a] < dists[b]; });
        }

        arena_.execute([&] {
            tbb::parallel_for(size_t(0), num_queries, [&](size_t idx) {
                size_t i = order[idx];  // EQR: execute in reordered sequence
                const T* q = batch_queries + i * dim_;
                std::priority_queue<std::pair<T, annchor::labeltype>> result;

                if (metric_ == METRIC_COSINE) {
                    std::vector<T> temp_query(q, q + dim_);
                    normalize_vector(temp_query.data(), 1);
                    if (use_s3) {
                        result = index_->searchKnnS3(temp_query.data(), k);
                    } else {
                        result = index_->searchKnn(temp_query.data(), k, nullptr, wm);
                    }
                } else {
                    if (use_s3) {
                        result = index_->searchKnnS3(q, k);
                    } else {
                        result = index_->searchKnn(q, k, nullptr, wm);
                    }
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
           << ", dist_computations:"
           << index_->metric_distance_computations.load()
           << ", hops:" << index_->metric_hops.load();

        size_t skip_queries = index_->metric_future_skip_queries.load();
        size_t skip_max = index_->metric_future_skip_hop_max.load();
        size_t skip_avg = 0;
        if (skip_queries > 0) {
            size_t skip_total = index_->metric_future_skip_hops_total.load();
            skip_avg = skip_total / skip_queries;
        }

        ss << ", future_skip_queries:" << skip_queries
           << ", future_skip_hop_max:" << skip_max
           << ", future_skip_hop_avg:" << skip_avg
           << ", undo_recovery_enabled:" << (index_->isUndoRecoveryEnabled() ? 1 : 0)
           << ", undo_lookups:" << index_->metric_undo_log_lookups.load()
           << ", recovered_edges_total:" << index_->metric_recovered_edges_total.load()
           << ", recovered_edges_useful:" << index_->metric_recovered_edges_useful.load()
           << ", recovered_single_node_max:" << index_->metric_recovered_edges_single_node_max.load();
        
        // Add S3 metrics if enabled
        if (index_->isS3Enabled()) {
            ss << ", s3_hits:" << index_->getS3Hits()
               << ", s3_misses:" << index_->getS3Misses()
               << ", s3_dist_saved:" << index_->getS3DistSaved();
        }

        // Add Slipstream metrics if enabled
        if (index_->isSlipstreamEnabled()) {
            ss << ", slipstream_publish:" << index_->getSlipstreamPublishCount()
               << ", slipstream_insert_publish:" << index_->getSlipstreamInsertPublishCount()
               << ", slipstream_search_publish:" << index_->getSlipstreamSearchPublishCount()
               << ", slipstream_hit:" << index_->getSlipstreamHitCount()
               << ", slipstream_miss:" << index_->getSlipstreamMissCount()
               << ", slipstream_seeds_injected:" << index_->getSlipstreamSeedsInjected()
               << ", slipstream_quality_reject:" << index_->getSlipstreamQualityRejectCount()
               << ", slipstream_cooldown_reject:" << index_->getSlipstreamCooldownRejectCount()
               << ", slipstream_regret:" << index_->getSlipstreamRegretCount()
               << ", slipstream_helpful:" << index_->getSlipstreamHelpfulCount()
               << ", slipstream_gate:" << (index_->isSlipstreamQualityGateEnabled() ? 1 : 0)
               << ", slipstream_skip_ratio:" << index_->getSlipstreamSkipRatio();
        }
        
        ss << ", " << index_->getProfileStats();
        str = ss.str();
    }

    void set_enable_mvcc(bool enable) {
        index_->setEnableMvcc(enable);
        eqr_enabled_ = false;  // EQR disabled (sort overhead outweighs benefit)
    }
    bool is_mvcc_enabled() const { return index_->isMvccEnabled(); }
    void set_enable_undo_recovery(bool enable) {
        index_->setEnableUndoRecovery(enable);
    }
    bool is_undo_recovery_enabled() const {
        return index_->isUndoRecoveryEnabled();
    }
    
    // S3 methods
    void set_enable_bcs(bool enable) { (void)enable; }
    void set_enable_bgr(bool enable) { (void)enable; }
    void bgr_repair() {}
    size_t compact(long safe_ts = -1) {
        (void)safe_ts;
        return 0;
    }
    void set_enable_s3(bool enable) { index_->setEnableS3(enable); }
    bool is_s3_enabled() const { return index_->isS3Enabled(); }
    void set_s3_proximity_threshold(T threshold) { index_->setS3ProximityThreshold(threshold); }

    void set_enable_eul(bool enable) { (void)enable; }
    void set_enable_mep(bool enable) { (void)enable; }

    // M3 Slipstream
    void set_enable_slipstream(bool enable) { index_->setEnableSlipstream(enable); }
    bool is_slipstream_enabled() const { return index_->isSlipstreamEnabled(); }
    void set_slipstream_ttl(uint64_t ttl_ns) { index_->setSlipstreamTTL(ttl_ns); }
    void set_slipstream_mode(int mode) { index_->setSlipstreamMode(mode); }
    void set_slipstream_quality_gate(bool enable) {
        index_->setSlipstreamQualityGate(enable);
    }
    void set_slipstream_skip_ratio(T ratio) { index_->setSlipstreamSkipRatio(ratio); }

    bool supports_snapshot() const override { return true; }

    int snapshot(std::vector<uint8_t>& out) override {
        std::string tmp_file = "/tmp/annchor_snapshot_" + std::to_string((uintptr_t)this) + ".bin";
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
        std::string tmp_file = "/tmp/annchor_restore_" + std::to_string((uintptr_t)this) + ".bin";
        std::ofstream file(tmp_file, std::ios::binary);
        if (!file.write((const char*)data, size)) {
            return -1;
        }
        file.close();

        try {
            delete index_;
            index_ = new annchor::HierarchicalNSW<T>(
                space_.get(), tmp_file, false, max_elements_, false); 
        } catch (...) {
            return -1;
        }
        std::remove(tmp_file.c_str());
        return 0;
    }

    size_t num_threads_;
    size_t dim_;
    size_t M_;
    size_t ef_c_;
    size_t ef_s_;
    size_t max_elements_;
    bool use_node_lock_in_search_;
    MetricType metric_;
    std::unique_ptr<annchor::SpaceInterface<T>> space_;
    annchor::HierarchicalNSW<T>* index_;

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
};
