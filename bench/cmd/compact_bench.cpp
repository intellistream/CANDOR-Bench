// Compact benchmark: build MVCC index, then measure compact() time.
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include "algorithms/annchor/ANNchor/src/hnswlib/hnswlib.h"
#include "algorithms/annchor/ANNchor/src/hnswlib/hnswalg.h"

int main() {
    const int dim = 128;
    const int max_elements = 1100000;
    const int total_elements = 1000000;
    const int M = 16;
    const int ef_construction = 200;
    const int batch_size = 20;

    std::cout << "=== Compact Benchmark ===" << std::endl;

    annchor::L2Space space(dim);
    annchor::HierarchicalNSW<float> index(&space, max_elements, M, ef_construction);
    index.setEnableMvcc(true);
    index.activateEUL();

    // Generate random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(total_elements * dim);
    for (auto &v : data) v = dist(rng);

    // Build index in batches (simulating MVCC batch inserts)
    std::cout << "Building index with " << total_elements << " points in batches of " << batch_size << "..." << std::flush;
    auto build_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < total_elements; i++) {
        index.addPoint(data.data() + i * dim, i);
    }
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    std::cout << " done (" << build_ms << " ms)" << std::endl;
    std::cout << "Total elements: " << index.cur_element_count.load() << std::endl;

    // Compact
    std::cout << "\n--- Compact ---" << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t compacted = index.compact();
    auto t1 = std::chrono::high_resolution_clock::now();
    double compact_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << compacted << " nodes compacted in " << compact_ms << " ms" << std::endl;

    // Second call (no-op)
    auto t2 = std::chrono::high_resolution_clock::now();
    size_t compacted2 = index.compact();
    auto t3 = std::chrono::high_resolution_clock::now();
    double compact2_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << compacted2 << " nodes (no-op) in " << compact2_ms << " ms" << std::endl;

    return 0;
}
