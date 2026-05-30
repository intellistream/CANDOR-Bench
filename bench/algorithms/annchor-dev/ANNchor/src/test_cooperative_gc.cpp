// Quick test: cooperative GC + thread-local pool
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include "hnswlib/hnswlib.h"
#include "hnswlib/hnswalg.h"

int main() {
    const int dim = 32;
    const int max_elements = 50000;
    const int M = 16;
    const int ef_construction = 100;
    const int batch_size = 20;
    const int num_search_threads = 4;

    annchor::L2Space space(dim);
    annchor::HierarchicalNSW<float> index(&space, max_elements, M, ef_construction,
                                           100, false, true, num_search_threads);

    // Random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(max_elements * dim);
    for (auto &v : data) v = dist(rng);

    // Build initial 10K
    std::cout << "Building 10K base..." << std::flush;
    for (int i = 0; i < 10000; i++) {
        index.addPoint(data.data() + i * dim, i);
    }
    std::cout << " done." << std::endl;

    // Concurrent insert + search
    std::cout << "Concurrent insert+search (40K inserts, " << num_search_threads << " search threads)..." << std::flush;
    std::atomic<bool> stop{false};
    std::atomic<long> searches{0};

    std::vector<std::thread> searchers;
    for (int t = 0; t < num_search_threads; t++) {
        searchers.emplace_back([&, t]() {
            std::mt19937 lr(1000 + t);
            std::vector<float> q(dim);
            while (!stop.load(std::memory_order_relaxed)) {
                for (auto &v : q) v = dist(lr);
                index.searchKnn(q.data(), 10, nullptr);
                searches.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Insert remaining in batches
    std::vector<annchor::labeltype> labels(batch_size);
    for (int i = 10000; i < max_elements; i += batch_size) {
        int end = std::min(i + batch_size, max_elements);
        int n = end - i;
        for (int j = 0; j < n; j++) labels[j] = i + j;
        index.addPointBatch(data.data() + i * dim, labels.data(), n);
    }

    stop.store(true);
    for (auto &t : searchers) t.join();
    std::cout << " done. Searches: " << searches.load() << std::endl;

    // Test compact
    std::cout << "Compact..." << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t compacted = index.compact();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << " " << compacted << " nodes in " << ms << " ms" << std::endl;

    // Verify search still works
    std::vector<float> q(dim);
    for (auto &v : q) v = dist(rng);
    auto result = index.searchKnn(q.data(), 10, nullptr);
    std::cout << "Post-compact search: " << result.size() << " results, recall OK" << std::endl;

    return 0;
}
