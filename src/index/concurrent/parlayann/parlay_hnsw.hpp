#pragma once

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>
#include <limits>

#include "index.hpp"
#include "parlayann/algorithms/HNSW/HNSW.hpp"
#include "parlayann/algorithms/utils/euclidian_point.h"
#include "parlayann/algorithms/utils/point_range.h"
#include "parlayann/algorithms/utils/types.h"

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class ParlayHNSW : public IndexBase<T, TagT, LabelT> {
   public:
    using Point = parlayANN::Euclidian_Point<float>;
    using Range = parlayANN::PointRange<Point>;
    using desc = parlayANN::Desc_HNSW<T, Point>;

    ParlayHNSW(size_t max_elements, size_t dim, size_t num_threads, size_t M,
               size_t ef_construction, float m_l, float alpha,
               size_t visit_limit)
        : dim_(dim),
          graph_degree_(M),
          ef_construction_(ef_construction),
          m_l_(m_l),
          alpha_(alpha),
          num_threads_(num_threads),
          max_elements_(max_elements),
          total_points_(0) {
        setenv("PARLAY_NUM_THREADS", std::to_string(num_threads).c_str(), 1);
    }

    void build(const T* data, const TagT* tags, size_t num_points) override {
        data_range_ = Range(reinterpret_cast<const float*>(data), num_points,
                            dim_, max_elements_);
        total_points_ = num_points;

        auto ps = parlay::delayed_seq<Point>(
            total_points_, [this](size_t i) { return data_range_[i]; });
        std::cerr << "m_l: " << m_l_ << ", graph_degree: " << graph_degree_
                  << ", ef_construction: " << ef_construction_
                  << ", alpha: " << alpha_ << std::endl;

        index_ = std::make_unique<ANN::HNSW<desc>>(ps.begin(), ps.end(), dim_,
                                                   m_l_, graph_degree_,
                                                   ef_construction_, alpha_);
    }

    int batch_insert(const T* batch_data, const TagT* batch_tags,
                     size_t num_points) override {
        std::lock_guard<std::mutex> lock(index_mutex);

        assert((total_points_ + num_points) <= max_elements_);

        data_range_.extend(reinterpret_cast<const float*>(batch_data),
                           num_points);
        total_points_ += num_points;

        auto ps = parlay::delayed_seq<Point>(
            num_points, [this, num_points](size_t i) {
                return data_range_[total_points_ - num_points + i];
            });

        index_->batch_insert(ps.begin(), ps.end(), batch_tags[0]);
        return 0;
    }

    int insert(const T* point, const TagT tag) override {
        std::cerr << "ParlayHNSW does not support dynamic single insertion"
                  << std::endl;
        return -1;
    }

    void set_query_params(const QParams& params) override {
        visit_limit_ = params.visit_limit;
        beam_width_ = params.beam_width;
    }

    int search(const T* query, size_t k,
               std::vector<TagT>& result_tags) override {
        std::cerr << "ParlayHNSW does not support dynamic single search"
                  << std::endl;
        return -1;
    }

    int batch_search(const T* batch_queries, size_t k, size_t num_queries,
                     TagT** batch_results, size_t* watermark_out = nullptr,
                     size_t visible_ts = std::numeric_limits<size_t>::max()) override {
        parlayANN::QueryParams QP(
            k, beam_width_, 1.35, visit_limit_,
            std::min<int>(index_->get_threshold_m(0), 3 * visit_limit_));
        Range qpoints(batch_queries, num_queries, dim_);
        parlay::sequence<TagT> starts(1, 0);

        auto start = std::chrono::high_resolution_clock::now();
        auto graph = typename ANN::HNSW<desc>::graph(*index_, 0);
        parlay::parallel_for(0, num_queries, [&](size_t i) {
            auto q = qpoints[i];
            auto results = parlayANN::beam_search_impl<uint32_t>(
                q, graph, data_range_, starts, QP);
            for (size_t j = 0; j < k && j < results.first.first.size(); ++j) {
                batch_results[i][j] = results.first.first[j].first;
            }
        });
        return 0;
    }

    void dump_stats(std::string& str) {}

   private:
    std::mutex index_mutex;

    size_t dim_;
    uint32_t graph_degree_;  // M
    uint32_t ef_construction_;
    float m_l_;
    float alpha_;
    size_t visit_limit_;
    size_t beam_width_;
    size_t num_threads_;
    size_t max_elements_;
    size_t total_points_;

    std::unique_ptr<ANN::HNSW<desc>> index_;
    Range data_range_;
    QParams query_params_;
};
