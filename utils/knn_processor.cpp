#include "knn_processor.h"

#include <immintrin.h>

#include <algorithm>
#include <iostream>
#include <thread>

std::vector<std::vector<PointPair>> CPUKNNProcessor::compute_incremental_knn(
    const std::vector<std::vector<float>>& base_vectors,
    const std::vector<std::vector<float>>& queries, int increment_size) {
    std::vector<std::vector<PointPair>> results(queries.size());
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    size_t chunk_size = (queries.size() + num_threads - 1) / num_threads;

    auto worker = [&](size_t start, size_t end) {
        CPUIncrementalKNN knn(k_);
        for (size_t i = start; i < end && i < queries.size(); ++i) {
            knn.reset();
            for (size_t current_size = increment_size;
                 current_size <= base_vectors.size();
                 current_size += increment_size) {
                knn.add_new_vectors_optimized(base_vectors, queries[i],
                                              current_size - increment_size,
                                              current_size);
            }
            results[i] = knn.get_topk();
        }
    };

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, queries.size());
        threads.emplace_back(worker, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return results;
}

void CPUIncrementalKNN::add_new_vectors_optimized(
    const std::vector<std::vector<float>>& vectors,
    const std::vector<float>& query, size_t start_idx, size_t end_idx) {
    std::vector<float> distances;
    euclidean_distance_batch_simd(query, vectors, start_idx, end_idx,
                                  distances);

    for (size_t i = 0; i < distances.size(); ++i) {
        int tag_id = static_cast<int>(start_idx + i);
        add_candidate(distances[i], tag_id);
    }
}

void CPUIncrementalKNN::euclidean_distance_batch_simd(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& vectors, size_t start_idx,
    size_t end_idx, std::vector<float>& distances) {
    if (vectors.empty()) return;

    size_t dim = query.size();
    size_t n_vectors = end_idx - start_idx;
    distances.resize(n_vectors);

    // 临时使用简单的非SIMD版本避免SIMD指令问题
    // TODO: 修复SIMD版本后恢复原来的代码
    for (size_t v = 0; v < n_vectors; ++v) {
        size_t vec_idx = start_idx + v;
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float diff = query[i] - vectors[vec_idx][i];
            sum += diff * diff;
        }
        distances[v] = sum;
    }

    /* 原来的SIMD代码（暂时注释掉）
    if (dim >= 16) {
        for (size_t v = 0; v < n_vectors; ++v) {
            size_t vec_idx = start_idx + v;
            __m256 sum_vec = _mm256_setzero_ps();
            size_t i = 0;

            for (; i <= dim - 8; i += 8) {
                __m256 vq = _mm256_loadu_ps(&query[i]);
                __m256 vb = _mm256_loadu_ps(&vectors[vec_idx][i]);
                __m256 diff = _mm256_sub_ps(vq, vb);
                sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
            }

            float sum = 0.0f;
            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            for (int j = 0; j < 8; ++j) {
                sum += temp[j];
            }

            for (; i < dim; ++i) {
                float diff = query[i] - vectors[vec_idx][i];
                sum += diff * diff;
            }

            distances[v] = sum;
        }
    } else if (dim >= 4) {
        for (size_t v = 0; v < n_vectors; ++v) {
            size_t vec_idx = start_idx + v;
            __m128 sum_vec = _mm_setzero_ps();
            size_t i = 0;

            for (; i <= dim - 4; i += 4) {
                __m128 vq = _mm_loadu_ps(&query[i]);
                __m128 vb = _mm_loadu_ps(&vectors[vec_idx][i]);
                __m128 diff = _mm_sub_ps(vq, vb);
                sum_vec = _mm_fmadd_ps(diff, diff, sum_vec);
            }

            float sum = 0.0f;
            float temp[4];
            _mm_storeu_ps(temp, sum_vec);
            for (int j = 0; j < 4; ++j) {
                sum += temp[j];
            }

            for (; i < dim; ++i) {
                float diff = query[i] - vectors[vec_idx][i];
                sum += diff * diff;
            }

            distances[v] = sum;
        }
    } else {
        for (size_t v = 0; v < n_vectors; ++v) {
            size_t vec_idx = start_idx + v;
            float sum = 0.0f;
            for (size_t i = 0; i < dim; ++i) {
                float diff = query[i] - vectors[vec_idx][i];
                sum += diff * diff;
            }
            distances[v] = sum;
        }
    }
    */
}

float CPUIncrementalKNN::euclidean_distance_simd(const std::vector<float>& a,
                                                 const std::vector<float>& b) {
    if (a.size() != b.size())
        throw std::runtime_error("Vector dimensions mismatch");

    size_t n = a.size();
    float sum = 0.0f;
    size_t i = 0;

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

    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}

void CPUIncrementalKNN::add_candidate(float dist, int id) {
    if (candidates.size() < static_cast<size_t>(k)) {
        candidates.emplace_back(dist, id);
        if (dist > max_distance) {
            max_distance = dist;
        }
    } else if (dist < max_distance) {
        auto max_it =
            std::max_element(candidates.begin(), candidates.end(),
                             [](const PointPair& a, const PointPair& b) {
                                 return a.first < b.first;
                             });
        *max_it = {dist, id};

        max_distance =
            std::max_element(candidates.begin(), candidates.end(),
                             [](const PointPair& a, const PointPair& b) {
                                 return a.first < b.first;
                             })
                ->first;
    }
}

std::vector<PointPair> CPUIncrementalKNN::get_topk() {
    std::sort(candidates.begin(), candidates.end(),
              [](const PointPair& a, const PointPair& b) {
                  return a.first < b.first;
              });
    return candidates;
}

void CPUIncrementalKNN::reset() {
    candidates.clear();
    current_size = 0;
    max_distance = std::numeric_limits<float>::max();
}

#if defined(UTILS_ENABLE_CUDA) && UTILS_ENABLE_CUDA
GPUKNNProcessor::GPUKNNProcessor(int k, int max_queries, int max_base_size,
                                 int vector_dim) {
    try {
        cuda_accelerator_ = std::make_unique<CudaKNNAccelerator>(
            k, max_queries, max_base_size, vector_dim);
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize GPU accelerator: " << e.what()
                  << std::endl;
        throw;
    }
}

std::vector<std::vector<PointPair>> GPUKNNProcessor::compute_incremental_knn(
    const std::vector<std::vector<float>>& base_vectors,
    const std::vector<std::vector<float>>& queries, int increment_size) {
    return cuda_accelerator_->compute_incremental_knn(base_vectors, queries,
                                                      increment_size);
}

std::vector<std::vector<PointPair>>
GPUKNNProcessor::compute_single_increment_knn(
    const std::vector<std::vector<float>>& base_vectors,
    const std::vector<std::vector<float>>& queries, int current_base_size) {
    return cuda_accelerator_->compute_single_increment_knn(
        base_vectors, queries, current_base_size);
}

void GPUKNNProcessor::reset() { cuda_accelerator_->reset(); }
#endif

std::unique_ptr<KNNProcessor> createKNNProcessor(bool use_gpu, int k,
                                                 int max_queries,
                                                 int max_base_size,
                                                 int vector_dim) {
#if defined(UTILS_ENABLE_CUDA) && UTILS_ENABLE_CUDA
    if (use_gpu && CudaKNNAccelerator::isCudaAvailable()) {
        try {
            return std::make_unique<GPUKNNProcessor>(k, max_queries,
                                                     max_base_size, vector_dim);
        } catch (const std::exception& e) {
            std::cerr << "GPU initialization failed, falling back to CPU: "
                      << e.what() << std::endl;
        }
    } else if (use_gpu) {
        std::cerr << "CUDA runtime not available. Falling back to CPU."
                  << std::endl;
    }
#else
    if (use_gpu) {
        std::cerr << "This build was configured without CUDA support. "
                     "Falling back to CPU."
                  << std::endl;
    }
#endif

    std::cout << "Using CPU implementation" << std::endl;
    return std::make_unique<CPUKNNProcessor>(k);
}
