#ifndef KNN_PROCESSOR_H
#define KNN_PROCESSOR_H

#include <limits>
#include <memory>
#include <vector>

#if defined(UTILS_ENABLE_CUDA) && UTILS_ENABLE_CUDA
#include "cuda_knn.h"
#endif

using PointPair = std::pair<int, float>;

class KNNProcessor {
public:
    virtual ~KNNProcessor() = default;
    virtual std::vector<std::vector<PointPair>> compute_incremental_knn(
        const std::vector<std::vector<float>>& base_vectors,
        const std::vector<std::vector<float>>& queries,
        int increment_size
    ) = 0;
    virtual void reset() = 0;
};

class CPUIncrementalKNN {
private:
    std::vector<PointPair> candidates;
    size_t current_size;
    int k;
    float max_distance;
    
public:
    CPUIncrementalKNN(int k) : current_size(0), k(k), max_distance(std::numeric_limits<float>::max()) {
        candidates.reserve(k + 100);
    }
    
    void add_new_vectors_optimized(const std::vector<std::vector<float>>& vectors,
                                  const std::vector<float>& query,
                                  size_t start_idx, size_t end_idx);
    
    std::vector<PointPair> get_topk();
    void reset();
    
private:
    void add_candidate(float dist, int id);
    float euclidean_distance_simd(const std::vector<float>& a, const std::vector<float>& b);
    void euclidean_distance_batch_simd(const std::vector<float>& query,
                                      const std::vector<std::vector<float>>& vectors,
                                      size_t start_idx, size_t end_idx,
                                      std::vector<float>& distances);
};

class CPUKNNProcessor : public KNNProcessor {
public:
    CPUKNNProcessor(int k) : k_(k) {}
    
    std::vector<std::vector<PointPair>> compute_incremental_knn(
        const std::vector<std::vector<float>>& base_vectors,
        const std::vector<std::vector<float>>& queries,
        int increment_size) override;
    
    void reset() override {}
    
private:
    int k_;
};

#if defined(UTILS_ENABLE_CUDA) && UTILS_ENABLE_CUDA
class GPUKNNProcessor : public KNNProcessor {
public:
    GPUKNNProcessor(int k, int max_queries = 10000, int max_base_size = 1000000, int vector_dim = 128);
    
    std::vector<std::vector<PointPair>> compute_incremental_knn(
        const std::vector<std::vector<float>>& base_vectors,
        const std::vector<std::vector<float>>& queries,
        int increment_size) override;
    
    // 计算单个增量的KNN结果
    std::vector<std::vector<PointPair>> compute_single_increment_knn(
        const std::vector<std::vector<float>>& base_vectors,
        const std::vector<std::vector<float>>& queries,
        int current_base_size);
    
    void reset() override;
    
private:
    std::unique_ptr<CudaKNNAccelerator> cuda_accelerator_;
};
#endif

std::unique_ptr<KNNProcessor> createKNNProcessor(
    bool use_gpu, 
    int k, 
    int max_queries = 10000, 
    int max_base_size = 1000000, 
    int vector_dim = 128
);

#endif // KNN_PROCESSOR_H
