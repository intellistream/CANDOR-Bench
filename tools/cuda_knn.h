#ifndef CUDA_KNN_H
#define CUDA_KNN_H

#include <vector>
#include <memory>
#include <limits>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
typedef void* cudaStream_t;
#endif

using PointPair = std::pair<int, float>;

class CudaKNNAccelerator {
public:
    CudaKNNAccelerator(int k, int max_queries = 10000, int max_base_size = 1000000, int vector_dim = 128);
    ~CudaKNNAccelerator();
    
    static bool isCudaAvailable();
    
    std::vector<std::vector<PointPair>> compute_incremental_knn(
        const std::vector<std::vector<float>>& base_vectors,
        const std::vector<std::vector<float>>& queries,
        int increment_size
    );
    
    std::vector<std::vector<PointPair>> compute_single_increment_knn(
        const std::vector<std::vector<float>>& base_vectors,
        const std::vector<std::vector<float>>& queries,
        int current_base_size
    );
    
    void reset();
    
private:
    int k_;
    int max_queries_;
    int max_base_size_;
    int vector_dim_;
    
    float* d_queries_;
    float* d_base_vectors_;
    float* d_distances_;
    int* d_knn_indices_;
    float* d_knn_distances_;
    int* d_current_sizes_;
    
    std::vector<float> h_distances_;
    std::vector<int> h_knn_indices_;
    std::vector<float> h_knn_distances_;
    
    cudaStream_t stream_;
    
    void allocateGpuMemory();
    void freeGpuMemory();
    void copyQueriesToGpu(const std::vector<std::vector<float>>& queries);
    void copyBaseVectorsToGpu(const std::vector<std::vector<float>>& base_vectors, int current_size);
    void computeDistancesGpu(int n_queries, int n_base);
    void updateKnnGpu(int n_queries, int current_size);
    void copyResultsFromGpu(int n_queries);
};

extern "C" {
    void launch_distance_kernel(
        const float* queries,
        const float* base_vectors,
        float* distances,
        int vector_dim,
        int n_queries,
        int n_base,
        cudaStream_t stream
    );
    
    void launch_knn_update_kernel(
        const float* distances,
        float* knn_distances,
        int* knn_indices,
        int* current_sizes,
        int k,
        int n_queries,
        int n_base,
        cudaStream_t stream
    );
}

#endif // CUDA_KNN_H
