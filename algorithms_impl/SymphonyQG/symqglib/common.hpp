#pragma once

#include <cstdint>
#include <functional>

#include "./third/Eigen/Dense"
#include "./third/Eigen/src/Core/util/Constants.h"

namespace symqg {
#define RANDOM_QUERY_QUANTIZATION
#define QG_BQUERY 6
#define FORCE_INLINE inline __attribute__((always_inline))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

using PID = uint32_t;
constexpr uint32_t kPidMax = 0xFFFFFFFF;

template <typename T>
using RowMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using ColMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename T>
using DistFunc = std::function<T(const T*, const T*, size_t)>;

template <typename T>
struct Candidate {
    PID id;
    T distance;

    Candidate() = default;
    explicit Candidate(PID vec_id, T dis) : id(vec_id), distance(dis) {}

    auto operator<(const Candidate& other) const { return distance < other.distance; }

    auto operator>(const Candidate& other) const { return !(*this < other); }
};
}  // namespace symqg