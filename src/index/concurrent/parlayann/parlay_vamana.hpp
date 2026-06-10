#pragma once

#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>
#include <limits>

#include "index.hpp"
#include "parlayann/algorithms/utils/euclidian_point.h"
#include "parlayann/algorithms/utils/point_range.h"
#include "parlayann/algorithms/utils/types.h"
#include "parlayann/algorithms/vamana/index.h"
#include "parlayann/data_tools/utils/beamSearch.h"

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class ParlayVamana : public IndexBase<T, TagT, LabelT> {
   public:
    using Point = parlayANN::Euclidian_Point<float>;
    using Range = parlayANN::PointRange<Point>;
    using desc = parlayANN::Desc_HNSW<T, Point>;
    using BuildParams = parlayANN::BuildParams;
    using Graph = parlayANN::Graph<TagT>;
    using KnnIndex = parlayANN::knn_index<Range, Range, TagT>;
    using QueryParams = parlayANN::QueryParams;

    ParlayVamana(size_t max_elements, size_t dim, size_t num_threads, size_t M,
                 size_t ef_construction, float alpha)
        : max_elements_(max_elements),
          dim_(dim),
          num_threads_(num_threads),
          graph_degree_(M),
          ef_construction_(ef_construction),
          alpha_(alpha),
          total_points_(0) {
        setenv("PARLAY_NUM_THREADS", std::to_string(num_threads).c_str(), 1);
    }

    void build(const T* data, const TagT* tags, size_t num_points) override {
        data_range_ = Range(reinterpret_cast<const float*>(data), num_points,
                            dim_, max_elements_);
        total_points_ = num_points;
        actual_points_ = num_points;
        parlayANN::stats<TagT> build_stats(total_points_);
        G_ = std::make_unique<Graph>(graph_degree_, max_elements_);

        std::cout << "data_range_.size(): " << data_range_.size() << std::endl;
        std::cout << "graph_degree_: " << graph_degree_
                  << ", ef_construction_: " << ef_construction_
                  << ", alpha_: " << alpha_ << std::endl;

        BuildParams BP(graph_degree_, ef_construction_, alpha_, 1);
        index_ = std::make_unique<KnnIndex>(BP);
        index_->build_index(*G_, data_range_, data_range_, build_stats);
    }

    int batch_insert(const T* batch_data, const TagT* batch_tags,
                     size_t num_points) override {
        std::lock_guard<std::mutex> lock(index_mutex);
        size_t start_idx = actual_points_;
        data_range_.extend(reinterpret_cast<const float*>(batch_data),
                           num_points);
        total_points_ += num_points;
        actual_points_ += num_points;
        Range new_points(reinterpret_cast<const float*>(batch_data), num_points,
                         dim_);
        parlay::sequence<TagT> points = parlay::tabulate(
            num_points,
            [&](size_t i) { return static_cast<TagT>(start_idx + i); });
        BuildParams BP(graph_degree_, ef_construction_, alpha_, 1);
        parlayANN::stats<TagT> build_stats(total_points_);
        return index_->incr_batch_insert(points, *G_, data_range_, data_range_,
                                         build_stats, BP.alpha);
    }

    int insert(const T* point, const TagT tag) override {
        std::cerr << "ParlayVamana does not support dynamic single insertion"
                  << std::endl;
        return -1;
    }

    void set_query_params(const QParams& params) override {
        visit_limit_ = params.visit_limit;
        beam_width_ = params.beam_width;
    }

    int search(const T* query, size_t k,
               std::vector<TagT>& result_tags) override {
        std::cerr << "ParlayVamana does not support dynamic single search"
                  << std::endl;
        return -1;
    }

    int batch_search(const T* batch_queries, size_t k, size_t num_queries,
                     TagT** batch_results, size_t* watermark_out = nullptr,
                     size_t visible_ts = std::numeric_limits<size_t>::max()) override {
        std::cout << "beam_width_: " << beam_width_ << ", alpha_: " << alpha_
                  << ", visit_limit_: " << visit_limit_ << std::endl;
        QueryParams QP(k, beam_width_, alpha_, visit_limit_,
                       std::min<int>(G_->max_degree(), 3 * visit_limit_));
        Range query_points(reinterpret_cast<const float*>(batch_queries),
                           num_queries, dim_);

        parlay::sequence<TagT> starting_points = {0};

        parlay::parallel_for(0, num_queries, [&](size_t i) {
            auto p = query_points[i];

            auto search_results = parlayANN::beam_search(p, *G_, data_range_,
                                                         starting_points, QP);
            auto& beam_results = search_results.first.first;

            for (uint32_t j = 0; j < k && j < beam_results.size(); j++) {
                batch_results[i][j] = static_cast<TagT>(beam_results[j].first);
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
    float alpha_;
    size_t visit_limit_;
    size_t beam_width_;
    size_t num_threads_;
    size_t max_elements_;
    size_t total_points_;
    size_t actual_points_ = 0;

    std::unique_ptr<KnnIndex> index_;
    std::unique_ptr<Graph> G_;

    Range data_range_;
    QParams query_params_;
};
