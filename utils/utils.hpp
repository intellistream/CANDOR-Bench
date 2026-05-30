#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

std::string to_string_with_precision(float value, int precision = 2);

struct SplitFileInfo {
    std::string filename;
    int start_offset;
    int end_offset;
};

struct Stat {
    std::string index_name;
    uint32_t num_points;
    uint32_t R;
    uint32_t Ls;
    uint32_t Lb;
    float alpha = 1.2;
    uint32_t num_threads;
    std::string dataset_name;
    uint32_t batch_size;

    float write_ratio;
    double insert_qps;
    double mean_insert_latency;
    double p95_insert_latency;
    double p99_insert_latency;

    double search_qps;
    double mean_search_latency;
    double p95_search_latency;
    double p99_search_latency;

    float overall_recall_at_10;
    std::string stagewise_result_path;

    Stat(std::string idx_name, std::string ds_name, uint32_t r, uint32_t lb,
         uint32_t ls, float wr, uint32_t threads, uint32_t batch_size,
         std::string res_path)
        : index_name(idx_name),
          dataset_name(ds_name),
          R(r),
          Lb(lb),
          Ls(ls),
          num_threads(threads),
          write_ratio(wr),
          alpha(1.2f),
          batch_size(100),
          stagewise_result_path(
              res_path + "/" + index_name + "_" + dataset_name + "_R" +
              std::to_string(r) + "_Lb" + std::to_string(lb) + "_Ls" +
              std::to_string(ls) + "_w" + to_string_with_precision(wr) + "_t" +
              std::to_string(threads) + ".res") {}
};

template <typename TagT>
struct SearchResult {
    uint64_t insert_offset;
    uint64_t query_tag;
    std::vector<TagT> tags;
    std::vector<float> distances;

    SearchResult(uint64_t offset, uint64_t tag, const std::vector<TagT> &t,
                 const std::vector<float> &d)
        : insert_offset(offset), query_tag(tag), tags(t), distances(d) {}
};

inline uint64_t le_to_host_uint64(const char *data) {
    uint64_t result = 0;
    for (int i = 0; i < 8; i++) {
        result |= (static_cast<uint64_t>(static_cast<unsigned char>(data[i]))
                   << (i * 8));
    }
    return result;
}

inline uint32_t le_to_host_uint32(const char *data) {
    uint32_t result = 0;
    for (int i = 0; i < 4; i++) {
        result |= (static_cast<uint32_t>(static_cast<unsigned char>(data[i]))
                   << (i * 8));
    }
    return result;
}

inline float le_to_host_float(const char *data) {
    uint32_t bits = le_to_host_uint32(data);
    return *reinterpret_cast<float *>(&bits);
}

std::string to_string_with_precision(float value, int precision) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    std::string str = ss.str();

    if (str.empty()) {
        return std::to_string(value);
    }

    size_t pos = str.find_last_not_of('0');
    if (pos != std::string::npos) {
        str.erase(pos + 1, std::string::npos);
    }

    if (!str.empty() && str.back() == '.') {
        str.pop_back();
    }
    return str;
}

void read_results_binary(std::vector<SearchResult<uint32_t>> &results,
                         const std::string &filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open result file: " + filepath);
    }

    uint64_t num_queries;
    file.read(reinterpret_cast<char *>(&num_queries), sizeof(uint64_t));
    if (file.fail()) {
        throw std::runtime_error(
            "Failed to read num_queries header from result file.");
    }

    results.clear();
    results.reserve(num_queries);
    for (uint64_t i = 0; i < num_queries; ++i) {
        SearchResult<uint32_t> sr{{}, {}, {}, {}};
        uint64_t num_tags, num_dists;

        file.read(reinterpret_cast<char *>(&sr.insert_offset),
                  sizeof(uint64_t));
        file.read(reinterpret_cast<char *>(&sr.query_tag), sizeof(uint64_t));
        file.read(reinterpret_cast<char *>(&num_tags), sizeof(uint64_t));
        sr.tags.resize(num_tags);
        if (num_tags > 0) {
            file.read(reinterpret_cast<char *>(sr.tags.data()),
                      num_tags * sizeof(uint32_t));
        }
        file.read(reinterpret_cast<char *>(&num_dists), sizeof(uint64_t));
        sr.distances.resize(num_dists);
        if (num_dists > 0) {
            file.read(reinterpret_cast<char *>(sr.distances.data()),
                      num_dists * sizeof(float));
        }

        if (file.fail()) {
            throw std::runtime_error("Error reading record " +
                                     std::to_string(i) + " from result file.");
        }
        results.push_back(sr);
    }
}

void parse_index_file_with_ranges(const std::string &index_path,
                                  std::vector<SplitFileInfo> &split_files) {
    std::ifstream index_file(index_path);
    if (!index_file.is_open()) {
        return;
    }

    split_files.clear();
    std::string line;
    int line_num = 0;
    while (std::getline(index_file, line)) {
        line_num++;
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string filename;
        int start_offset, end_offset;

        if (iss >> filename >> start_offset >> end_offset) {
            split_files.push_back({filename, start_offset, end_offset});
        } else {
        }
    }
    index_file.close();
}

inline std::vector<std::vector<float>> read_bin(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    int npts, dim;
    in.read(reinterpret_cast<char *>(&npts), sizeof(int));
    in.read(reinterpret_cast<char *>(&dim), sizeof(int));

    std::vector<std::vector<float>> data;
    data.reserve(npts);

    for (int i = 0; i < npts; ++i) {
        std::vector<float> vec(dim);
        in.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
        data.push_back(vec);
    }

    in.close();
    return data;
}

inline float euclidean_distance_simd(const std::vector<float> &a,
                                     const std::vector<float> &b) {
    if (a.size() != b.size())
        throw std::runtime_error("Vector dimensions mismatch");

    size_t n = a.size();
    float sum = 0.0f;
    size_t i = 0;

#ifdef __AVX2__
    if (n >= 8) {
        __m256 sum_vec = _mm256_setzero_ps();
        for (; i <= n - 8; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
        }
        float temp[8];
        _mm256_storeu_ps(temp, sum_vec);
        for (int j = 0; j < 8; ++j) {
            sum += temp[j];
        }
    }
#endif

    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}
