#include "cuda_knn.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <algorithm>
#include <limits>

__global__ void distance_kernel(
    const float* queries,
    const float* base_vectors,
    float* distances,
    int vector_dim,
    int n_queries,
    int n_base
) {
    long long tid =
        static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x) +
        static_cast<long long>(threadIdx.x);
    long long total_threads =
        static_cast<long long>(blockDim.x) * static_cast<long long>(gridDim.x);
    long long total_work =
        static_cast<long long>(n_queries) * static_cast<long long>(n_base);

    for (long long idx = tid; idx < total_work; idx += total_threads) {
        int query_idx = static_cast<int>(idx / n_base);
        int base_idx = static_cast<int>(idx % n_base);
        
        float sum = 0.0f;
        
        int i = 0;
        for (; i <= vector_dim - 8; i += 8) {
            float4 q_vec1 = reinterpret_cast<const float4*>(queries + query_idx * vector_dim)[i/4];
            float4 b_vec1 = reinterpret_cast<const float4*>(base_vectors + base_idx * vector_dim)[i/4];
            float4 q_vec2 = reinterpret_cast<const float4*>(queries + query_idx * vector_dim)[i/4 + 1];
            float4 b_vec2 = reinterpret_cast<const float4*>(base_vectors + base_idx * vector_dim)[i/4 + 1];
            
            float4 diff1 = make_float4(
                q_vec1.x - b_vec1.x, q_vec1.y - b_vec1.y,
                q_vec1.z - b_vec1.z, q_vec1.w - b_vec1.w
            );
            float4 diff2 = make_float4(
                q_vec2.x - b_vec2.x, q_vec2.y - b_vec2.y,
                q_vec2.z - b_vec2.z, q_vec2.w - b_vec2.w
            );
            
            sum += diff1.x*diff1.x + diff1.y*diff1.y + diff1.z*diff1.z + diff1.w*diff1.w;
            sum += diff2.x*diff2.x + diff2.y*diff2.y + diff2.z*diff2.z + diff2.w*diff2.w;
        }
        
        for (; i <= vector_dim - 4; i += 4) {
            float4 q_vec = reinterpret_cast<const float4*>(queries + query_idx * vector_dim)[i/4];
            float4 b_vec = reinterpret_cast<const float4*>(base_vectors + base_idx * vector_dim)[i/4];
            
            float4 diff = make_float4(
                q_vec.x - b_vec.x, q_vec.y - b_vec.y,
                q_vec.z - b_vec.z, q_vec.w - b_vec.w
            );
            
            sum += diff.x*diff.x + diff.y*diff.y + diff.z*diff.z + diff.w*diff.w;
        }
        
        for (; i < vector_dim; ++i) {
            float diff = queries[query_idx * vector_dim + i] - 
                        base_vectors[base_idx * vector_dim + i];
            sum += diff * diff;
        }
        
        distances[idx] = sum;
    }
}

__global__ void knn_update_kernel(
    const float* distances,
    float* knn_distances,
    int* knn_indices,
    int* current_sizes,
    int k,
    int n_queries,
    int n_base
) {
    int query_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (query_idx >= n_queries) return;
    
    extern __shared__ float shared_distances[];
    extern __shared__ int shared_indices[];
    
    for (int i = tid; i < n_base; i += blockDim.x) {
        shared_distances[i] = distances[query_idx * n_base + i];
        shared_indices[i] = i;
    }
    __syncthreads();
    
    for (int i = 0; i < k && i < n_base; ++i) {
        int min_idx = i;
        float min_dist = shared_distances[i];
        
        for (int j = i + 1; j < n_base; ++j) {
            if (shared_distances[j] < min_dist) {
                min_dist = shared_distances[j];
                min_idx = j;
            }
        }
        
        if (min_idx != i) {
            float temp_dist = shared_distances[i];
            int temp_idx = shared_indices[i];
            shared_distances[i] = shared_distances[min_idx];
            shared_indices[i] = shared_indices[min_idx];
            shared_distances[min_idx] = temp_dist;
            shared_indices[min_idx] = temp_idx;
        }
        
        knn_distances[query_idx * k + i] = shared_distances[i];
        knn_indices[query_idx * k + i] = shared_indices[i];
    }
    
    if (tid == 0) {
        current_sizes[query_idx] = (k < n_base) ? k : n_base;
    }
}

__global__ void knn_update_kernel_simple(
    const float* distances,
    float* knn_distances,
    int* knn_indices,
    int* current_sizes,
    int k,
    int n_queries,
    int n_base
) {
    int query_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (query_idx >= n_queries) return;
    
    for (int i = 0; i < k && i < n_base; ++i) {
        int min_idx = -1;
        float min_dist = 1e30f;  
        
        for (int j = 0; j < n_base; ++j) {
            float dist = distances[query_idx * n_base + j];
            
            bool already_selected = false;
            for (int prev = 0; prev < i; ++prev) {
                if (knn_indices[query_idx * k + prev] == j) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && dist < min_dist) {
                min_dist = dist;
                min_idx = j;
            }
        }
        
        
        knn_distances[query_idx * k + i] = min_dist;
        knn_indices[query_idx * k + i] = min_idx;
    }
    
    if (tid == 0) {
        current_sizes[query_idx] = (k < n_base) ? k : n_base;
    }
}

__global__ void knn_update_kernel_optimized(
    const float* distances,
    float* knn_distances,
    int* knn_indices,
    int* current_sizes,
    int k,
    int n_queries,
    int n_base
) {
    int query_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (query_idx >= n_queries) return;
    
    extern __shared__ float shared_candidates[];
    float* shared_distances = shared_candidates;
    int* shared_indices = (int*)(shared_candidates + k);
    
    for (int i = tid; i < k && i < n_base; i += block_size) {
        shared_distances[i] = distances[query_idx * n_base + i];
        shared_indices[i] = i;
    }
    __syncthreads();
    
    for (int i = k; i < n_base; ++i) {
        float dist = distances[query_idx * n_base + i];
        
        int max_idx = 0;
        float max_dist = shared_distances[0];
        for (int j = 1; j < k; ++j) {
            if (shared_distances[j] > max_dist) {
                max_dist = shared_distances[j];
                max_idx = j;
            }
        }
        
        if (dist < max_dist) {
            shared_distances[max_idx] = dist;
            shared_indices[max_idx] = i;
        }
    }
    
    for (int i = 0; i < k - 1; ++i) {
        for (int j = i + 1; j < k; ++j) {
            if (shared_distances[i] > shared_distances[j]) {
                float temp_dist = shared_distances[i];
                int temp_idx = shared_indices[i];
                shared_distances[i] = shared_distances[j];
                shared_indices[i] = shared_indices[j];
                shared_distances[j] = temp_dist;
                shared_indices[j] = temp_idx;
            }
        }
    }
    
    for (int i = tid; i < k && i < n_base; i += block_size) {
        knn_distances[query_idx * k + i] = shared_distances[i];
        knn_indices[query_idx * k + i] = shared_indices[i];
    }
    
    if (tid == 0) {
        current_sizes[query_idx] = (k < n_base) ? k : n_base;
    }
}

extern "C" void launch_distance_kernel(
    const float* queries,
    const float* base_vectors,
    float* distances,
    int vector_dim,
    int n_queries,
    int n_base,
    cudaStream_t stream
) {
    long long total_elements =
        static_cast<long long>(n_queries) * static_cast<long long>(n_base);
    int block_size = 256;
    long long grid_size_ll =
        (total_elements + block_size - 1) / block_size;
    if (grid_size_ll <= 0) {
        std::cerr << "CUDA distance kernel error: invalid launch configuration "
                  << "(computed grid size " << grid_size_ll << ")."
                  << std::endl;
        return;
    }
    if (grid_size_ll > std::numeric_limits<int>::max()) {
        std::cerr << "CUDA distance kernel error: grid size exceeds int "
                     "limits."
                  << std::endl;
        return;
    }
    int grid_size = static_cast<int>(grid_size_ll);
    
    distance_kernel<<<grid_size, block_size, 0, stream>>>(
        queries, base_vectors, distances, vector_dim, n_queries, n_base
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA distance kernel error: " << cudaGetErrorString(err) << std::endl;
    }
}

extern "C" void launch_knn_update_kernel(
    const float* distances,
    float* knn_distances,
    int* knn_indices,
    int* current_sizes,
    int k,
    int n_queries,
    int n_base,
    cudaStream_t stream
) {
    int block_size = 256;
    
    static bool shared_mem_warning_printed = false;
    static int max_shared_mem = -1;
    
    if (max_shared_mem == -1) {
        cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    }
    
    int required_shared_mem = k * (sizeof(float) + sizeof(int));
    
    if (required_shared_mem > max_shared_mem) {
        if (!shared_mem_warning_printed) {
            std::cerr << "Info: Using optimized global memory kernel for better performance." << std::endl;
            shared_mem_warning_printed = true;
        }
        
        knn_update_kernel_simple<<<n_queries, block_size, 0, stream>>>(
            distances, knn_distances, knn_indices, current_sizes, k, n_queries, n_base
        );
    } else {
        knn_update_kernel_optimized<<<n_queries, block_size, required_shared_mem, stream>>>(
            distances, knn_distances, knn_indices, current_sizes, k, n_queries, n_base
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA KNN update kernel error: " << cudaGetErrorString(err) << std::endl;
    }
}

CudaKNNAccelerator::CudaKNNAccelerator(int k, int max_queries, int max_base_size, int vector_dim)
    : k_(k), max_queries_(max_queries), max_base_size_(max_base_size), vector_dim_(vector_dim) {
    
    if (!isCudaAvailable()) {
        throw std::runtime_error("CUDA is not available on this system");
    }
    
    allocateGpuMemory();
    cudaStreamCreate(&stream_);
    
    h_distances_.resize(max_queries_ * max_base_size_);
    h_knn_indices_.resize(max_queries_ * k_);
    h_knn_distances_.resize(max_queries_ * k_);
}

CudaKNNAccelerator::~CudaKNNAccelerator() {
    freeGpuMemory();
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

bool CudaKNNAccelerator::isCudaAvailable() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

void CudaKNNAccelerator::allocateGpuMemory() {
    size_t queries_size = max_queries_ * vector_dim_ * sizeof(float);
    size_t base_size = max_base_size_ * vector_dim_ * sizeof(float);
    size_t distances_size = max_queries_ * max_base_size_ * sizeof(float);
    size_t knn_size = max_queries_ * k_ * sizeof(float);
    size_t indices_size = max_queries_ * k_ * sizeof(int);
    size_t sizes_size = max_queries_ * sizeof(int);
    
    cudaMalloc(&d_queries_, queries_size);
    cudaMalloc(&d_base_vectors_, base_size);
    cudaMalloc(&d_distances_, distances_size);
    cudaMalloc(&d_knn_distances_, knn_size);
    cudaMalloc(&d_knn_indices_, indices_size);
    cudaMalloc(&d_current_sizes_, sizes_size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA memory allocation error: " << cudaGetErrorString(err) << std::endl;
        freeGpuMemory();
        throw std::runtime_error("Failed to allocate GPU memory");
    }
}

void CudaKNNAccelerator::freeGpuMemory() {
    if (d_queries_) cudaFree(d_queries_);
    if (d_base_vectors_) cudaFree(d_base_vectors_);
    if (d_distances_) cudaFree(d_distances_);
    if (d_knn_distances_) cudaFree(d_knn_distances_);
    if (d_knn_indices_) cudaFree(d_knn_indices_);
    if (d_current_sizes_) cudaFree(d_current_sizes_);
}

std::vector<std::vector<PointPair>> CudaKNNAccelerator::compute_incremental_knn(
    const std::vector<std::vector<float>>& base_vectors,
    const std::vector<std::vector<float>>& queries,
    int increment_size) {
    
    int n_queries = queries.size();
    int n_base = base_vectors.size();
    
    if (n_queries > max_queries_ || n_base > max_base_size_) {
        throw std::runtime_error("Input size exceeds maximum capacity");
    }
    
    copyQueriesToGpu(queries);
    
    std::vector<std::vector<PointPair>> results(n_queries);
    
    int total_increments = (n_base + increment_size - 1) / increment_size;
    int current_increment = 0;
    
    for (int current_size = increment_size; current_size <= n_base; current_size += increment_size) {
        current_increment++;
        
        int progress_percent = (current_increment * 100) / total_increments;
        std::cout << "GPU Processing: " << current_increment << "/" << total_increments 
                  << " (" << progress_percent << "%) - base size: " << current_size << std::endl;
        
        copyBaseVectorsToGpu(base_vectors, current_size);
        
        computeDistancesGpu(n_queries, current_size);
        
        updateKnnGpu(n_queries, current_size);
        
        copyResultsFromGpu(n_queries);
        
        for (int q = 0; q < n_queries; ++q) {
            results[q].clear();
            for (int i = 0; i < k_ && i < current_size; ++i) {
                int idx = q * k_ + i;
                results[q].emplace_back(h_knn_indices_[idx], h_knn_distances_[idx]);
            }
        }
    }
    
    std::cout << "GPU Processing completed successfully!" << std::endl;
    return results;
}

std::vector<std::vector<PointPair>> CudaKNNAccelerator::compute_single_increment_knn(
    const std::vector<std::vector<float>>& base_vectors,
    const std::vector<std::vector<float>>& queries,
    int current_base_size) {
    
    int n_queries = queries.size();
    int n_base = base_vectors.size();
    
    if (n_queries > max_queries_ || current_base_size > max_base_size_) {
        throw std::runtime_error("Input size exceeds maximum capacity");
    }
    
    if (current_base_size > n_base) {
        current_base_size = n_base;
    }
    
    copyQueriesToGpu(queries);
    copyBaseVectorsToGpu(base_vectors, current_base_size);
    
    computeDistancesGpu(n_queries, current_base_size);
    updateKnnGpu(n_queries, current_base_size);
    copyResultsFromGpu(n_queries);
    
    std::vector<std::vector<PointPair>> results(n_queries);
    for (int q = 0; q < n_queries; ++q) {
        for (int i = 0; i < k_ && i < current_base_size; ++i) {
            int idx = q * k_ + i;
            results[q].emplace_back(h_knn_indices_[idx], h_knn_distances_[idx]);
        }
    }
    
    return results;
}

void CudaKNNAccelerator::copyQueriesToGpu(const std::vector<std::vector<float>>& queries) {
    int n_queries = queries.size();
    std::vector<float> flat_queries(n_queries * vector_dim_);
    
    for (int i = 0; i < n_queries; ++i) {
        for (int j = 0; j < vector_dim_; ++j) {
            flat_queries[i * vector_dim_ + j] = queries[i][j];
        }
    }
    
    cudaMemcpyAsync(d_queries_, flat_queries.data(), 
                   n_queries * vector_dim_ * sizeof(float),
                   cudaMemcpyHostToDevice, stream_);
}

void CudaKNNAccelerator::copyBaseVectorsToGpu(const std::vector<std::vector<float>>& base_vectors, int current_size) {
    std::vector<float> flat_base(current_size * vector_dim_);
    
    for (int i = 0; i < current_size; ++i) {
        for (int j = 0; j < vector_dim_; ++j) {
            flat_base[i * vector_dim_ + j] = base_vectors[i][j];
        }
    }
    
    cudaMemcpyAsync(d_base_vectors_, flat_base.data(),
                   current_size * vector_dim_ * sizeof(float),
                   cudaMemcpyHostToDevice, stream_);
}

void CudaKNNAccelerator::computeDistancesGpu(int n_queries, int n_base) {
    launch_distance_kernel(d_queries_, d_base_vectors_, d_distances_,
                          vector_dim_, n_queries, n_base, stream_);
}

void CudaKNNAccelerator::updateKnnGpu(int n_queries, int n_base) {
    launch_knn_update_kernel(d_distances_, d_knn_distances_, d_knn_indices_,
                            d_current_sizes_, k_, n_queries, n_base, stream_);
}

void CudaKNNAccelerator::copyResultsFromGpu(int n_queries) {
    cudaMemcpyAsync(h_knn_distances_.data(), d_knn_distances_,
                   n_queries * k_ * sizeof(float),
                   cudaMemcpyDeviceToHost, stream_);
    
    cudaMemcpyAsync(h_knn_indices_.data(), d_knn_indices_,
                   n_queries * k_ * sizeof(int),
                   cudaMemcpyDeviceToHost, stream_);
    
    cudaStreamSynchronize(stream_);
}

void CudaKNNAccelerator::reset() {
    cudaMemsetAsync(d_knn_distances_, 0, max_queries_ * k_ * sizeof(float), stream_);
    cudaMemsetAsync(d_knn_indices_, 0, max_queries_ * k_ * sizeof(int), stream_);
    cudaMemsetAsync(d_current_sizes_, 0, max_queries_ * sizeof(int), stream_);
    cudaStreamSynchronize(stream_);
}
