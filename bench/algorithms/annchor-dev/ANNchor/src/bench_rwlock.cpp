#include "hnswlib/hnswlib.h"
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <iostream>
#include <vector>
#include <cstring>

// Concurrent insert+search benchmark
// Compares: MVCC, NodeRWLock (no MVCC), NoLock

int main(int argc, char* argv[]) {
    int dim = 128;
    int max_elements = 500000;
    int M = 32;
    int ef_construction = 200;
    int search_threads = 16;
    int insert_threads = 4;
    int insert_batch_size = 1000;
    int search_k = 10;
    int ef_search = 200;
    int pre_build = 100000;  // pre-build this many before concurrent phase
    int concurrent_inserts = 400000;  // insert this many during concurrent phase
    int search_duration_sec = 10;

    // Parse mode: mvcc, rwlock, nolock
    std::string mode = "mvcc";
    if (argc > 1) mode = argv[1];
    if (argc > 2) search_threads = std::stoi(argv[2]);
    if (argc > 3) insert_threads = std::stoi(argv[3]);
    if (argc > 4) max_elements = std::stoi(argv[4]);
    if (argc > 5) pre_build = std::stoi(argv[5]);

    concurrent_inserts = max_elements - pre_build;

    bool enable_mvcc, use_node_lock;
    if (mode == "mvcc") {
        enable_mvcc = true;
        use_node_lock = false;
    } else if (mode == "rwlock") {
        enable_mvcc = false;
        use_node_lock = true;
    } else if (mode == "nolock") {
        enable_mvcc = false;
        use_node_lock = false;
    } else {
        std::cerr << "Usage: " << argv[0] << " [mvcc|rwlock|nolock] [search_threads] [insert_threads] [max_elements] [pre_build]\n";
        return 1;
    }

    std::cout << "=== Mode: " << mode << " | SearchThreads: " << search_threads
              << " | InsertThreads: " << insert_threads
              << " | MaxElements: " << max_elements
              << " | PreBuild: " << pre_build
              << " | ConcurrentInserts: " << concurrent_inserts << " ===" << std::endl;

    // Generate random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(dim * max_elements);
    for (auto& v : data) v = dist(rng);

    annchor::L2Space space(dim);
    annchor::HierarchicalNSW<float> alg(&space, max_elements, M, ef_construction,
                                         100, false, use_node_lock, insert_threads);
    alg.setEnableMVCC(enable_mvcc);
    alg.setEf(ef_search);

    // Phase 1: Pre-build index single-batch
    std::cout << "Pre-building " << pre_build << " elements..." << std::flush;
    {
        std::vector<annchor::labeltype> labels(pre_build);
        for (int i = 0; i < pre_build; i++) labels[i] = i;
        auto t0 = std::chrono::high_resolution_clock::now();
        alg.addPointBatch(data.data(), labels.data(), pre_build);
        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        std::cout << " done in " << sec << "s (" << pre_build / sec << " pts/s)" << std::endl;
    }

    // Phase 2: Concurrent insert + search
    std::atomic<bool> inserting{true};
    std::atomic<size_t> total_searches{0};
    std::atomic<size_t> total_search_time_ns{0};
    std::atomic<size_t> insert_idx{0};

    // Per-thread latency collection
    std::vector<std::vector<double>> per_thread_latencies(search_threads);

    // chasing mode: search queries target recently inserted points
    bool chasing = (argc > 6 && std::string(argv[6]) == "chasing");
    std::cout << "Query mode: " << (chasing ? "chasing" : "random") << std::endl;

    auto search_fn = [&](int tid) {
        std::mt19937 local_rng(tid * 1000 + 7);
        size_t count = 0;
        auto start = std::chrono::high_resolution_clock::now();

        while (inserting.load(std::memory_order_relaxed)) {
            int qi;
            if (chasing) {
                // Query near the insertion frontier
                size_t current_inserted = insert_idx.load(std::memory_order_relaxed);
                size_t frontier = pre_build + std::min(current_inserted, (size_t)concurrent_inserts);
                if (frontier <= 1) frontier = pre_build;
                // Pick from last 10% of inserted points
                size_t window = std::max<size_t>(frontier / 10, 100);
                size_t low = (frontier > window) ? frontier - window : 0;
                std::uniform_int_distribution<size_t> qdist(low, frontier - 1);
                qi = (int)qdist(local_rng);
            } else {
                std::uniform_int_distribution<int> qdist(0, pre_build - 1);
                qi = qdist(local_rng);
            }
            const void* query = data.data() + qi * dim;
            auto t0 = std::chrono::high_resolution_clock::now();
            auto result = alg.searchKnn(query, search_k);
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            per_thread_latencies[tid].push_back(us);
            count++;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        total_searches.fetch_add(count, std::memory_order_relaxed);
        total_search_time_ns.fetch_add(ns, std::memory_order_relaxed);
    };

    auto insert_fn = [&](int tid) {
        while (true) {
            size_t batch_start = insert_idx.fetch_add(insert_batch_size, std::memory_order_relaxed);
            if (batch_start >= (size_t)concurrent_inserts) break;
            size_t batch_end = std::min(batch_start + insert_batch_size, (size_t)concurrent_inserts);
            size_t actual_batch = batch_end - batch_start;

            size_t global_start = pre_build + batch_start;
            std::vector<annchor::labeltype> labels(actual_batch);
            for (size_t i = 0; i < actual_batch; i++) labels[i] = global_start + i;

            alg.addPointBatch(data.data() + global_start * dim, labels.data(), actual_batch);
        }
    };

    std::cout << "Starting concurrent phase..." << std::endl;
    auto phase2_start = std::chrono::high_resolution_clock::now();

    // Launch search threads
    std::vector<std::thread> searchers;
    for (int i = 0; i < search_threads; i++) {
        searchers.emplace_back(search_fn, i);
    }

    // Launch insert threads (each grabs batches)
    std::vector<std::thread> inserters;
    for (int i = 0; i < insert_threads; i++) {
        inserters.emplace_back(insert_fn, i);
    }

    // Wait for inserts to finish
    for (auto& t : inserters) t.join();
    inserting.store(false, std::memory_order_relaxed);
    for (auto& t : searchers) t.join();

    auto phase2_end = std::chrono::high_resolution_clock::now();
    double phase2_sec = std::chrono::duration<double>(phase2_end - phase2_start).count();

    size_t searches = total_searches.load();
    double search_qps = searches / phase2_sec;
    double avg_latency_us = (total_search_time_ns.load() / 1000.0) / searches;

    // Merge all latencies and compute percentiles
    std::vector<double> all_latencies;
    for (auto& v : per_thread_latencies) {
        all_latencies.insert(all_latencies.end(), v.begin(), v.end());
    }
    std::sort(all_latencies.begin(), all_latencies.end());
    auto pct = [&](double p) -> double {
        size_t idx = (size_t)(p * all_latencies.size());
        if (idx >= all_latencies.size()) idx = all_latencies.size() - 1;
        return all_latencies[idx];
    };

    std::cout << "\n=== Results (" << mode << ") ===" << std::endl;
    std::cout << "Duration: " << phase2_sec << "s" << std::endl;
    std::cout << "Total searches: " << searches << std::endl;
    std::cout << "Search QPS: " << (size_t)search_qps << std::endl;
    std::cout << "Avg search latency: " << avg_latency_us << " us" << std::endl;
    std::cout << "p50: " << pct(0.50) << " us" << std::endl;
    std::cout << "p95: " << pct(0.95) << " us" << std::endl;
    std::cout << "p99: " << pct(0.99) << " us" << std::endl;
    std::cout << "p999: " << pct(0.999) << " us" << std::endl;
    std::cout << "max: " << all_latencies.back() << " us" << std::endl;
    std::cout << "Insert QPS: " << concurrent_inserts / phase2_sec << std::endl;

    // Phase 3: Post-insert recall test (search-only, no concurrency)
    // Skip compact to avoid destructor bug for now

    std::cout << "\nRecall test (1-NN on first 10000 points)..." << std::endl;
    int recall_n = std::min(10000, max_elements);
    int correct = 0;
    for (int i = 0; i < recall_n; i++) {
        auto result = alg.searchKnn(data.data() + i * dim, 1);
        if (!result.empty() && (int)result.top().second == i) correct++;
    }
    std::cout << "Recall@1: " << (float)correct / recall_n << std::endl;

    return 0;
}
