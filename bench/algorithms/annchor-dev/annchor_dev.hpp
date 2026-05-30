#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include "../index.hpp"
#include "../index_cgo.hpp"
namespace annchor_dev {
    #include "ANNchor/src/hnswlib/hnswlib.h"
}
#include <memory>
#include <cmath>

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class ANNchorDev : public IndexBase<T, TagT, LabelT> {
   public:
    ANNchorDev(size_t max_elements, size_t dim, size_t num_threads, size_t M,
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
            space_.reset(new annchor_dev::annchor::InnerProductSpace(dim));
        } else {
            space_.reset(new annchor_dev::annchor::L2Space(dim));
        }
        index_ = new annchor_dev::annchor::HierarchicalNSW<T>(
            space_.get(), max_elements, M, ef_construction, 100, false, use_node_lock_in_search, num_threads_);
    }

    ~ANNchorDev() override {
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
                         point, static_cast<annchor_dev::annchor::labeltype>(batch_tags[i]), -1, mvcc);
                     if (mvcc) index_->markReady(id);
                 });
             });
        } else {
             arena_.execute([&] {
                 tbb::parallel_for(size_t(0), num_points, [&](size_t i) {
                     const T* point = batch_data + i * dim_;
                     auto id = index_->addPoint(
                         point, static_cast<annchor_dev::annchor::labeltype>(batch_tags[i]), -1, mvcc);
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
        std::priority_queue<std::pair<T, annchor_dev::annchor::labeltype>> result;
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
                std::priority_queue<std::pair<T, annchor_dev::annchor::labeltype>> result;

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
           << ", dist_computations:"
           << index_->metric_distance_computations.load()
           << ", hops:" << index_->metric_hops.load()
           << ", cnr_triggered:" << index_->cnr_triggered_.load()
           << ", cnr_recovered_edges:" << index_->cnr_recovered_edges_.load()
           << ", cnr_useful_edges:" << index_->cnr_useful_edges_.load()
           << ", cnr_degree_deficit_triggers:" << index_->cnr_degree_deficit_triggers_.load()
           << ", cnr_stagnation_triggers:" << index_->cnr_stagnation_triggers_.load()
           << ", m2_triggered:" << index_->m2_triggered_.load()
           << ", m2_backup_injected:" << index_->m2_backup_injected_.load()
           << ", m2_backup_useful:" << index_->m2_backup_useful_.load()
           << ", m2_budget_exhausted:" << index_->m2_budget_exhausted_.load();

        // Undo log stats
        if (index_->undo_log_) {
            size_t log_nodes = 0, total_edges = 0, max_edges = 0;
            // Manual scan since logStats may not exist in stable
            for (size_t i = 0; i < index_->cur_element_count.load(); i++) {
                if (index_->undo_log_->hasLog(i)) {
                    const auto* log = index_->undo_log_->getLog(i);
                    if (log) {
                        size_t ne = log->edges.size();
                        log_nodes++;
                        total_edges += ne;
                        if (ne > max_edges) max_edges = ne;
                    }
                }
            }
            ss << ", undo_max_edges:" << max_edges
               << ", undo_log_nodes:" << log_nodes
               << ", undo_total_edges:" << total_edges;
            if (log_nodes > 0) ss << ", undo_avg_edges:" << (total_edges / log_nodes);
        }

        str = ss.str();
    }

    // MVCC methods
    void set_enable_mvcc(bool enable) { 
        index_->setEnableMvcc(enable);
    }
    bool is_mvcc_enabled() const { return index_->isMvccEnabled(); }
    
    // CNR methods
    void set_enable_cnr(bool enable) {
        index_->setEnableCnr(enable);
    }
    void set_cnr_degree_threshold(float t) {
        index_->setCnrDegreeThreshold(t);
    }
    void set_cnr_max_recover(int m) {
        index_->setCnrMaxRecover(m);
    }
    void set_cnr_stagnation_hops(int h) {
        index_->setCnrStagnationHops(h);
    }

    // M2 methods
    void set_enable_m2_dual_path(bool enable) {
        index_->setEnableM2DualPath(enable);
    }
    void set_m2_risk_hops(int h) {
        index_->setM2RiskHops(h);
    }
    void set_m2_assist_budget(int b) {
        index_->setM2AssistBudget(b);
    }

    size_t compact(long safe_ts = -1) { return index_->compact(safe_ts); }

    bool supports_snapshot() const override { return true; }

    int snapshot(std::vector<uint8_t>& out) override {
        std::string tmp_file = "/tmp/annchor_dev_snapshot_" + std::to_string((uintptr_t)this) + ".bin";
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
        std::string tmp_file = "/tmp/annchor_dev_restore_" + std::to_string((uintptr_t)this) + ".bin";
        std::ofstream file(tmp_file, std::ios::binary);
        if (!file.write((const char*)data, size)) {
            return -1;
        }
        file.close();

        try {
            delete index_;
            index_ = new annchor_dev::annchor::HierarchicalNSW<T>(
                space_.get(), tmp_file, false, max_elements_, false); 
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
    std::unique_ptr<annchor_dev::annchor::SpaceInterface<T>> space_;
    annchor_dev::annchor::HierarchicalNSW<T>* index_;

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
