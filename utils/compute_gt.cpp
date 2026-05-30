#include <getopt.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "utils.hpp"

const int PARTSIZE = 10000000;
const int ALIGNMENT = 512;

template <class T>
T div_round_up(const T numerator, const T denominator) {
    return (numerator % denominator == 0) ? (numerator / denominator)
                                          : 1 + (numerator / denominator);
}

using pairIF = std::pair<size_t, float>;
struct cmpmaxstruct {
    bool operator()(const pairIF &l, const pairIF &r) const {
        return l.second < r.second;
    }
};

using maxPQIFCS =
    std::priority_queue<pairIF, std::vector<pairIF>, cmpmaxstruct>;

inline bool custom_dist(const std::pair<uint32_t, float> &a,
                        const std::pair<uint32_t, float> &b) {
    return a.second < b.second;
}

float manual_sdot(int N, const float *X, int incX, const float *Y, int incY) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += X[i * incX] * Y[i * incY];
    }
    return sum;
}

void manual_sgemm_dot_product_rows(size_t M, size_t N, size_t K, float alpha,
                                   const float *A, size_t ldA, const float *B,
                                   size_t ldB, float beta, float *C,
                                   size_t ldC) {
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            float dot_prod = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                dot_prod += A[i * ldA + k] * B[j * ldB + k];
            }
            C[i + j * ldC] = alpha * dot_prod + beta * C[i + j * ldC];
        }
    }
}

void manual_sgemm_add_outer_product(size_t M, size_t N, float alpha,
                                    const float *vec1, const float *vec2,
                                    float *C, size_t ldC) {
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            C[i + j * ldC] += alpha * vec1[i] * vec2[j];
        }
    }
}

void compute_l2sq(float *const points_l2sq, const float *const matrix,
                  const int64_t num_points, const uint64_t dim) {
    assert(points_l2sq != NULL);
#pragma omp parallel for schedule(static, 65536)
    for (int64_t d = 0; d < num_points; ++d) {
        points_l2sq[d] =
            manual_sdot((int64_t)dim, matrix + (ptrdiff_t)d * (ptrdiff_t)dim, 1,
                        matrix + (ptrdiff_t)d * (ptrdiff_t)dim, 1);
    }
}

void distsq_to_points(const size_t dim, float *dist_matrix, size_t npoints,
                      const float *const points, const float *const points_l2sq,
                      size_t nqueries, const float *const queries,
                      const float *const queries_l2sq, float *ones_vec = NULL) {
    bool ones_vec_alloc = false;
    if (ones_vec == NULL) {
        ones_vec = new float[nqueries > npoints ? nqueries : npoints];
        std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints,
                    (float)1.0);
        ones_vec_alloc = true;
    }

    manual_sgemm_dot_product_rows(npoints, nqueries, dim, (float)-2.0, points,
                                  dim, queries, dim, (float)0.0, dist_matrix,
                                  npoints);
    manual_sgemm_add_outer_product(npoints, nqueries, (float)1.0, points_l2sq,
                                   ones_vec, dist_matrix, npoints);
    manual_sgemm_add_outer_product(npoints, nqueries, (float)1.0, ones_vec,
                                   queries_l2sq, dist_matrix, npoints);

    if (ones_vec_alloc) delete[] ones_vec;
}

void exact_knn(
    const size_t dim, const size_t k,
    size_t *const closest_points,  // k * num_queries preallocated, col major,
                                   // queries columns
    float *const dist_closest_points,  // k * num_queries preallocated, Dist to
                                       // corresponding closes_points
    size_t npoints,
    float *points_in,  // points in Col major (actually row-major flat array)
    size_t nqueries, float *queries_in, float *points_l2sq = NULL,
    float *queries_l2sq =
        NULL)  // queries in Col major (actually row-major flat array)
{
    bool points_l2sq_alloc = false;
    if (points_l2sq == NULL) {
        points_l2sq = new float[npoints];
        compute_l2sq(points_l2sq, points_in, npoints, dim);
        points_l2sq_alloc = true;
    }

    bool queries_l2sq_alloc = false;
    if (queries_l2sq == NULL) {
        queries_l2sq = new float[nqueries];
        compute_l2sq(queries_l2sq, queries_in, nqueries, dim);
        queries_l2sq_alloc = true;
    }

    float *points = points_in;
    float *queries = queries_in;

    std::cout << "Going to compute " << k << " NNs for " << nqueries
              << " queries over " << npoints << " points in " << dim
              << " dimensions using L2 distance fn. " << std::endl;

    size_t q_batch_size = (1 << 9);
    float *dist_matrix = new float[(size_t)q_batch_size * (size_t)npoints];

    for (size_t b = 0; b < div_round_up(nqueries, q_batch_size); ++b) {
        int64_t q_b = b * q_batch_size;
        int64_t q_e = ((b + 1) * q_batch_size > nqueries)
                          ? nqueries
                          : (b + 1) * q_batch_size;

        distsq_to_points(
            dim, dist_matrix, npoints, points, points_l2sq, (size_t)(q_e - q_b),
            queries + (ptrdiff_t)q_b * (ptrdiff_t)dim, queries_l2sq + q_b);

        std::cout << "Computed distances for queries: [" << q_b << "," << q_e
                  << ")" << std::endl;

#pragma omp parallel for schedule(dynamic, 16)
        for (long long q = q_b; q < q_e; q++) {
            maxPQIFCS point_dist;
            for (size_t p = 0; p < k; p++) {
                point_dist.emplace(
                    p, dist_matrix[(ptrdiff_t)p +
                                   (ptrdiff_t)(q - q_b) * (ptrdiff_t)npoints]);
            }
            for (size_t p = k; p < npoints; p++) {
                if (point_dist.top().second >
                    dist_matrix[(ptrdiff_t)p +
                                (ptrdiff_t)(q - q_b) * (ptrdiff_t)npoints]) {
                    point_dist.pop();  // Remove largest
                    point_dist.emplace(
                        p, dist_matrix[(ptrdiff_t)p +
                                       (ptrdiff_t)(q - q_b) *
                                           (ptrdiff_t)npoints]);  // Add new
                                                                  // smaller
                }
            }
            // Extract results from priority queue (they are in reverse sorted
            // order)
            for (ptrdiff_t l = 0; l < (ptrdiff_t)k; ++l) {
                closest_points[(ptrdiff_t)(k - 1 - l) +
                               (ptrdiff_t)q * (ptrdiff_t)k] =
                    point_dist.top().first;
                dist_closest_points[(ptrdiff_t)(k - 1 - l) +
                                    (ptrdiff_t)q * (ptrdiff_t)k] =
                    point_dist.top().second;
                point_dist.pop();
            }
            assert(std::is_sorted(
                dist_closest_points + (ptrdiff_t)q * (ptrdiff_t)k,
                dist_closest_points + (ptrdiff_t)(q + 1) * (ptrdiff_t)k));
        }
        std::cout << "Computed exact k-NN for queries: [" << q_b << "," << q_e
                  << ")" << std::endl;
    }

    delete[] dist_matrix;

    if (points_l2sq_alloc) {
        delete[] points_l2sq;
    }
    if (queries_l2sq_alloc) {
        delete[] queries_l2sq;
    }
}

template <typename T>
inline int get_num_parts(const char *filename) {
    auto data = read_bin(filename);
    int npts = data.size();
    std::cout << "#pts = " << npts << std::endl;
    uint32_t num_parts = (npts % PARTSIZE) == 0
                             ? npts / PARTSIZE
                             : (uint32_t)std::floor(npts / PARTSIZE) + 1;
    std::cout << "Number of parts: " << num_parts << std::endl;
    return num_parts;
}

template <typename T>
inline void load_bin_as_float(const char *filename, float *&data, size_t &npts,
                              size_t &ndims, int part_num) {
    auto all_data = read_bin(filename);
    if (all_data.empty()) {
        std::cerr << "Error: No vectors found in file" << std::endl;
        return;
    }

    uint64_t start_id = (uint64_t)part_num * PARTSIZE;
    uint64_t end_id =
        (std::min)(start_id + PARTSIZE, (uint64_t)all_data.size());
    npts = end_id - start_id;
    ndims = all_data[0].size();

    std::cout << "#pts in part = " << npts << ", #dims = " << ndims
              << ", size = " << npts * ndims * sizeof(float) << "B"
              << std::endl;

    data = new float[npts * ndims];

    for (size_t i = 0; i < npts; i++) {
        for (size_t j = 0; j < ndims; j++) {
            data[i * ndims + j] = all_data[start_id + i][j];
        }
    }
    std::cout << "Finished reading part of the file." << std::endl;
}

inline void save_groundtruth_as_one_file(const std::string filename,
                                         int32_t *data, float *distances,
                                         size_t npts, size_t ndims) {
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    writer.exceptions(
        std::ios::failbit |
        std::ios::badbit);  // Enable exceptions for file operations

    int npts_i32 = (int)npts, ndims_i32 = (int)ndims;
    writer.write(reinterpret_cast<char *>(&npts_i32), sizeof(int));
    writer.write(reinterpret_cast<char *>(&ndims_i32), sizeof(int));
    std::cout << "Saving truthset in one file (npts, dim, npts*dim id-matrix, "
                 "npts*dim dist-matrix) with npts = "
              << npts << ", dim = " << ndims << ", size = "
              << 2 * npts * ndims * sizeof(uint32_t) + 2 * sizeof(int) << "B"
              << std::endl;

    writer.write(reinterpret_cast<char *>(data),
                 npts * ndims * sizeof(uint32_t));
    writer.write(reinterpret_cast<char *>(distances),
                 npts * ndims * sizeof(float));
    writer.close();
    std::cout << "Finished writing truthset" << std::endl;
}

template <typename T>
std::vector<std::vector<std::pair<uint32_t, float>>> processUnfilteredParts(
    const std::string &base_file, size_t &nqueries, size_t &npoints,
    size_t &dim, size_t &k, float *query_data) {
    float *base_data = nullptr;
    int num_parts = get_num_parts<T>(base_file.c_str());
    std::vector<std::vector<std::pair<uint32_t, float>>> res(nqueries);
    for (int p = 0; p < num_parts; p++) {
        size_t start_id = (size_t)p * PARTSIZE;
        load_bin_as_float<T>(base_file.c_str(), base_data, npoints, dim, p);
        if (p == 0) {
            std::cout << "Loaded base data: " << npoints << " vectors, " << dim
                      << " dimensions" << std::endl;
        }

        size_t *closest_points_part = new size_t[nqueries * k];
        float *dist_closest_points_part = new float[nqueries * k];

        auto part_k = k < npoints ? k : npoints;
        exact_knn(dim, part_k, closest_points_part, dist_closest_points_part,
                  npoints, base_data, nqueries, query_data);

        for (size_t i = 0; i < nqueries; i++) {
            for (size_t j = 0; j < part_k; j++) {
                res[i].push_back(std::make_pair(
                    (uint32_t)(closest_points_part[i * k + j] + start_id),
                    dist_closest_points_part[i * part_k + j]));
            }
        }

        delete[] closest_points_part;
        delete[] dist_closest_points_part;

        delete[] base_data;
    }
    return res;
}

template <typename T>
int aux_main_logic(const std::string &base_file, const std::string &query_file,
                   const std::string &gt_file, size_t k, int64_t query_limit) {
    size_t npoints, nqueries, dim;

    float *query_data;

    load_bin_as_float<T>(query_file.c_str(), query_data, nqueries, dim, 0);
    std::cout << "Loaded query data: " << nqueries << " vectors, " << dim
              << " dimensions" << std::endl;
    if (query_limit > 0 && static_cast<size_t>(query_limit) < nqueries) {
        std::cout << "Limiting queries to first " << query_limit << std::endl;
        nqueries = static_cast<size_t>(query_limit);
    }

    if (nqueries > PARTSIZE)
        std::cerr << "WARNING: #Queries provided (" << nqueries
                  << ") is greater than " << PARTSIZE
                  << ". Computing GT only for the first " << PARTSIZE
                  << " queries." << std::endl;

    int *closest_points = new int[nqueries * k];
    float *dist_closest_points = new float[nqueries * k];

    std::vector<std::vector<std::pair<uint32_t, float>>> results =
        processUnfilteredParts<T>(base_file, nqueries, npoints, dim, k,
                                  query_data);  // Removed metric parameter

    for (size_t i = 0; i < nqueries; i++) {
        std::vector<std::pair<uint32_t, float>> &cur_res = results[i];
        std::sort(cur_res.begin(), cur_res.end(), custom_dist);
        size_t j = 0;
        for (auto iter : cur_res) {
            if (j == k) break;
            closest_points[i * k + j] = (int32_t)iter.first;
            dist_closest_points[i * k + j] = iter.second;
            ++j;
        }
        if (j < k)
            std::cout << "WARNING: found less than k GT entries for query " << i
                      << std::endl;
    }

    save_groundtruth_as_one_file(gt_file, closest_points, dist_closest_points,
                                 nqueries, k);
    delete[] closest_points;
    delete[] dist_closest_points;
    delete[] query_data;

    return 0;
}

int main(int argc, char **argv) {
    std::string base_file, query_file, gt_file;
    int K = 0;
    int64_t query_limit = -1;

    const char *const short_opts = "";
    const option long_opts[] = {{"base_file", required_argument, nullptr, 0},
                                {"query_file", required_argument, nullptr, 0},
                                {"gt_file", required_argument, nullptr, 0},
                                {"k", required_argument, nullptr, 0},
                                {"query_limit", required_argument, nullptr, 0},
                                {nullptr, 0, nullptr, 0}};
    int opt_idx = 0;
    while (true) {
        int opt = getopt_long(argc, argv, "", long_opts, &opt_idx);
        if (opt == -1) break;
        if (opt == 0) {
            const std::string opt_name(long_opts[opt_idx].name);
            if (opt_name == "base_file")
                base_file = optarg;
            else if (opt_name == "query_file")
                query_file = optarg;
            else if (opt_name == "gt_file")
                gt_file = optarg;
            else if (opt_name == "k")
                K = std::stoi(optarg);
            else if (opt_name == "query_limit")
                query_limit = std::stoll(optarg);
        }
    }

    if (base_file.empty() || query_file.empty() || gt_file.empty() || K <= 0) {
        std::cout << "Usage: ./compute_gt --base_file BASE --query_file QUERY "
                     "--gt_file GT --k K [--query_limit N]"
                  << std::endl;
        return 1;
    }

    aux_main_logic<float>(base_file, query_file, gt_file, K, query_limit);

    std::cout << "Done. Saved groundtruth to " << gt_file << std::endl;
    return 0;
}
