#pragma once

// Stage-query generation: round_robin / chasing / peeking / zipfian.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace candor {
namespace driver {

enum class QueryMode {
    kRoundRobin,
    kChasing,
    kPeeking,
    kZipfian,
};

inline QueryMode parse_query_mode(const std::string& mode) {
    if (mode == "round_robin") return QueryMode::kRoundRobin;
    if (mode == "chasing") return QueryMode::kChasing;
    if (mode == "peeking") return QueryMode::kPeeking;
    if (mode == "zipfian") return QueryMode::kZipfian;
    throw std::invalid_argument("unsupported query_mode " + mode);
}

// Builds one batch of queries per call. Queries are copied into a flat
// buffer because a batch may gather non-contiguous rows of the workload
// (wrap-around, zipfian samples).
class QuerySource {
   public:
    QuerySource(QueryMode mode, const float* workload, size_t total_queries,
                size_t dim, size_t batch_size, size_t stage_window,
                double zipfian_skew, uint64_t seed)
        : mode_(mode),
          workload_(workload),
          total_queries_(total_queries),
          dim_(dim),
          batch_size_(batch_size),
          stage_window_(stage_window),
          rng_(seed ? seed : std::random_device{}()) {
        if (total_queries_ == 0) {
            throw std::invalid_argument(
                "query mode requires a workload query dataset");
        }
        if (mode_ == QueryMode::kZipfian) {
            build_zipf_cdf(zipfian_skew > 0 ? zipfian_skew : 0.99);
        }
    }

    // Fills out (num*dim floats) and tags; returns the number of queries
    // produced (0 = nothing to ask at this offset). Throws when the
    // requested window exceeds the workload.
    size_t next_batch(size_t end_insert_offset, std::vector<float>& out,
                      std::vector<uint32_t>& tags) {
        out.clear();
        tags.clear();
        if (batch_size_ == 0 || dim_ == 0) return 0;

        switch (mode_) {
            case QueryMode::kRoundRobin: {
                for (size_t i = 0; i < batch_size_; ++i) {
                    size_t idx = cursor_ % total_queries_;
                    append_query(idx, out, tags);
                    ++cursor_;
                }
                return batch_size_;
            }
            case QueryMode::kChasing: {
                if (end_insert_offset > total_queries_) {
                    throw std::runtime_error(
                        "query mode chasing window exceeds workload queries");
                }
                size_t window_end = end_insert_offset;
                size_t window_start =
                    window_end > stage_window_ ? window_end - stage_window_ : 0;
                size_t window_size = window_end - window_start;
                if (window_size == 0) return 0;
                for (size_t i = 0; i < batch_size_; ++i) {
                    append_query(window_start + (i % window_size), out, tags);
                }
                return batch_size_;
            }
            case QueryMode::kPeeking: {
                if (end_insert_offset > total_queries_) {
                    throw std::runtime_error(
                        "query mode peeking window exceeds workload queries");
                }
                size_t window_start = end_insert_offset;
                if (window_start >= total_queries_) return 0;
                size_t window_end = window_start + stage_window_;
                if (window_end > total_queries_) window_end = total_queries_;
                size_t window_size = window_end - window_start;
                if (window_size == 0) return 0;
                for (size_t i = 0; i < batch_size_; ++i) {
                    append_query(window_start + (i % window_size), out, tags);
                }
                return batch_size_;
            }
            case QueryMode::kZipfian: {
                for (size_t i = 0; i < batch_size_; ++i) {
                    append_query(sample_zipf(), out, tags);
                }
                return batch_size_;
            }
        }
        return 0;
    }

   private:
    void append_query(size_t idx, std::vector<float>& out,
                      std::vector<uint32_t>& tags) {
        const float* src = workload_ + idx * dim_;
        out.insert(out.end(), src, src + dim_);
        tags.push_back(static_cast<uint32_t>(idx));
    }

    void build_zipf_cdf(double skew) {
        cdf_.resize(total_queries_);
        double sum = 0.0;
        for (size_t k = 0; k < total_queries_; ++k) {
            cdf_[k] = 1.0 / std::pow(static_cast<double>(k + 1), skew);
            sum += cdf_[k];
        }
        double cumul = 0.0;
        for (size_t k = 0; k < total_queries_; ++k) {
            cumul += cdf_[k] / sum;
            cdf_[k] = cumul;
        }
        cdf_.back() = 1.0;
    }

    size_t sample_zipf() {
        double u;
        {
            std::lock_guard<std::mutex> lk(rng_mu_);
            u = uniform_(rng_);
        }
        size_t lo = 0, hi = cdf_.size() - 1;
        while (lo < hi) {
            size_t mid = (lo + hi) / 2;
            if (cdf_[mid] < u) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    QueryMode mode_;
    const float* workload_;
    size_t total_queries_;
    size_t dim_;
    size_t batch_size_;
    size_t stage_window_;
    size_t cursor_ = 0;
    std::vector<double> cdf_;
    std::mutex rng_mu_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};
};

}  // namespace driver
}  // namespace candor
