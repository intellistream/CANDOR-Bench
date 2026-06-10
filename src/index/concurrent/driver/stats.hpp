#pragma once

// Result statistics for the benchmark driver.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "sync.hpp"

namespace candor {
namespace driver {

struct DriverStats {
    double elapsed_sec = 0;
    double insert_qps = 0, search_qps = 0;
    double mean_insert_op = 0, p95_insert_op = 0, p99_insert_op = 0;
    double mean_search_op = 0, p95_search_op = 0, p99_search_op = 0;
    double mean_insert_e2e = 0, p95_insert_e2e = 0, p99_insert_e2e = 0;
    double mean_search_e2e = 0, p95_search_e2e = 0, p99_search_e2e = 0;
    uint64_t insert_points = 0, search_points = 0;
    double search_lag_avg = 0;
    int64_t search_lag_max = 0;
    double peak_memory_mb = 0, avg_memory_mb = 0;
};

namespace detail {

inline double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

inline double mean(const std::vector<double>& v) {
    if (v.empty()) return 0;
    double s = 0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

// Nearest-rank estimator, sorted[int((n-1)*p)] — kept stable so
// numbers stay comparable across harness generations.
inline double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(static_cast<double>(v.size() - 1) * p);
    return v[idx];
}

inline uint64_t current_rss_bytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            uint64_t kb = 0;
            iss >> kb;
            return kb * 1024;
        }
    }
    return 0;
}

}  // namespace detail
}  // namespace driver
}  // namespace candor
