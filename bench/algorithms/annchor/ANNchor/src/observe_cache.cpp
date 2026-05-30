/**
 * Observe Cache Behavior: Insert then Search
 *
 * Measure whether Insert warms CPU cache for nearby Search.
 *
 * Protocol:
 *   Phase A: Insert batch of B points → flush timing
 *   Phase B: Search near those B points (chasing) → measure time + cache miss
 *   Phase C: Search random points (unrelated) → measure time + cache miss
 *
 * If B is faster than C with same algorithmic cost (same ef_s, same k),
 * the difference is purely cache locality from Insert warming the cache.
 *
 * We use perf_event_open to read hardware counters inline.
 */

#include "hnswlib/hnswlib.h"
#include "hnswlib/hnswalg.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cstring>

// Linux perf_event for reading hardware counters
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

struct PerfCounter {
    int fd = -1;
    bool ok = false;

    PerfCounter(uint32_t type, uint64_t config) {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(pe));
        pe.type = type;
        pe.size = sizeof(pe);
        pe.config = config;
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;
        fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
        ok = (fd >= 0);
    }
    ~PerfCounter() { if (fd >= 0) close(fd); }
    void start() { if (ok) { ioctl(fd, PERF_EVENT_IOC_RESET, 0); ioctl(fd, PERF_EVENT_IOC_ENABLE, 0); } }
    void stop() { if (ok) ioctl(fd, PERF_EVENT_IOC_DISABLE, 0); }
    long long read_count() {
        long long count = 0;
        if (ok) ::read(fd, &count, sizeof(count));
        return count;
    }
};

void load_bin(const char* filename, float* data, int dim, int num) {
    std::ifstream file(filename, std::ios::binary);
    uint32_t fn, fd;
    file.read(reinterpret_cast<char*>(&fn), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&fd), sizeof(uint32_t));
    for (int i = 0; i < num; i++)
        file.read(reinterpret_cast<char*>(data + i * dim), dim * sizeof(float));
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: observe_cache <data_file> <max_elements>\n";
        return 1;
    }
    char* data_file = argv[1];
    int max_elements = std::stoi(argv[2]);

    int dim = 0;
    { std::ifstream f(data_file, std::ios::binary);
      uint32_t fn, fd; f.read(reinterpret_cast<char*>(&fn), 4);
      f.read(reinterpret_cast<char*>(&fd), 4);
      dim = fd; max_elements = std::min(max_elements, (int)fn); }

    std::cout << "================================================================\n"
              << " Cache Behavior: Does Insert warm cache for Search?\n"
              << "================================================================\n\n";

    float* data = new float[dim * max_elements];
    load_bin(data_file, data, dim, max_elements);
    std::cout << "Loaded " << max_elements << " points, dim=" << dim << "\n";

    double avg_dist = 0;
    { std::mt19937 rng(42); std::uniform_int_distribution<int> idx(0, max_elements-1);
      for (int i = 0; i < 1000; i++) {
          int a = idx(rng), b = idx(rng); float s = 0;
          for (int d = 0; d < dim; d++) { float df = data[a*dim+d]-data[b*dim+d]; s+=df*df; }
          avg_dist += std::sqrt(s);
      } avg_dist /= 1000; }
    float sd = std::sqrt((float)dim);
    std::cout << "Avg dist: " << std::fixed << std::setprecision(2) << avg_dist << "\n\n";

    // Setup perf counters
    PerfCounter l1_miss(PERF_TYPE_HW_CACHE,
        PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
    PerfCounter llc_miss(PERF_TYPE_HW_CACHE,
        PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
    PerfCounter cache_ref(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
    PerfCounter cache_miss(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);

    if (!cache_miss.ok) std::cout << "WARNING: perf counters not available (try running as root or set perf_event_paranoid)\n\n";

    // Build HNSW
    int M = 16, ef_c = 200, ef_s = 50;
    int warmup = max_elements * 2 / 3;
    annchor::L2Space space(dim);
    annchor::HierarchicalNSW<float>* hnsw =
        new annchor::HierarchicalNSW<float>(&space, max_elements, M, ef_c, 100, false, false, 1);

    std::cout << "Building index (warmup=" << warmup << ")...\n";
    for (int i = 0; i < warmup; i++)
        hnsw->addPoint(data + (size_t)i * dim, i);
    std::cout << "Done. Index: " << hnsw->getCurrentElementCount() << "\n\n";

    // ============================================================
    // Experiment: Batch insert then measure search latency
    // ============================================================
    int batch_size = 100;
    int n_batches = 5;
    float noise_scale = 0.1f * avg_dist / sd;

    std::cout << "=== Batch Insert → Search Cache Behavior ===\n";
    std::cout << "  batch_size=" << batch_size << ", n_batches=" << n_batches << "\n";
    std::cout << "  ef_search=" << ef_s << ", k=10\n\n";

    std::cout << std::left
              << std::setw(8) << "batch"
              << std::setw(16) << "insert_ms"
              << std::setw(16) << "srch_near_ms"
              << std::setw(16) << "srch_rand_ms"
              << std::setw(12) << "near/rand"
              << std::setw(14) << "cache_miss_N"
              << std::setw(14) << "cache_miss_R"
              << "\n" << std::string(96, '-') << "\n";

    std::mt19937 qrng(123);
    int cur_pt = warmup;

    for (int b = 0; b < n_batches; b++) {
        // Phase A: Insert batch
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<int> inserted_pts;
        for (int i = 0; i < batch_size && cur_pt < max_elements; i++, cur_pt++) {
            hnsw->addPoint(data + (size_t)cur_pt * dim, cur_pt);
            inserted_pts.push_back(cur_pt);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double insert_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Phase B: Search NEAR the just-inserted points (should benefit from warm cache)
        hnsw->setEf(ef_s);
        std::vector<float> query(dim);

        cache_miss.start(); cache_ref.start();
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int pt : inserted_pts) {
            std::normal_distribution<float> noise(0, noise_scale);
            for (int d = 0; d < dim; d++) query[d] = data[pt*dim+d] + noise(qrng);
            auto result = hnsw->searchKnn(query.data(), 10);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        cache_miss.stop(); cache_ref.stop();
        double near_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        long long near_cache_miss = cache_miss.read_count();

        // Phase C: Search RANDOM points (cold cache for those regions)
        std::uniform_int_distribution<int> ridx(0, (int)hnsw->getCurrentElementCount() - 1);

        cache_miss.start(); cache_ref.start();
        auto t4 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < inserted_pts.size(); i++) {
            int rpt = ridx(qrng);
            std::normal_distribution<float> noise(0, noise_scale);
            for (int d = 0; d < dim; d++) query[d] = data[rpt*dim+d] + noise(qrng);
            auto result = hnsw->searchKnn(query.data(), 10);
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        cache_miss.stop(); cache_ref.stop();
        double rand_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
        long long rand_cache_miss = cache_miss.read_count();

        std::cout << std::setw(8) << b
                  << std::setw(16) << std::setprecision(2) << insert_ms
                  << std::setw(16) << near_ms
                  << std::setw(16) << rand_ms
                  << std::setw(12) << std::setprecision(3) << near_ms / rand_ms
                  << std::setw(14) << near_cache_miss
                  << std::setw(14) << rand_cache_miss
                  << "\n";
    }

    // ============================================================
    // Control: Insert → wait → search near (cache should be cold by then)
    // ============================================================
    std::cout << "\n=== Control: Insert → clear cache → Search near ===\n";
    std::cout << "  (Flush cache between insert and search by doing unrelated work)\n\n";
    {
        // Insert a batch
        std::vector<int> pts;
        for (int i = 0; i < batch_size && cur_pt < max_elements; i++, cur_pt++) {
            hnsw->addPoint(data + (size_t)cur_pt * dim, cur_pt);
            pts.push_back(cur_pt);
        }

        // "Flush" cache by doing a bunch of random searches (pollute cache)
        hnsw->setEf(ef_s);
        std::vector<float> query(dim);
        std::uniform_int_distribution<int> ridx(0, (int)hnsw->getCurrentElementCount() - 1);
        for (int i = 0; i < 500; i++) {
            int rpt = ridx(qrng);
            std::normal_distribution<float> noise(0, noise_scale);
            for (int d = 0; d < dim; d++) query[d] = data[rpt*dim+d] + noise(qrng);
            hnsw->searchKnn(query.data(), 10);
        }

        // Now search near the inserted points (cache should be cold)
        cache_miss.start();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int pt : pts) {
            std::normal_distribution<float> noise(0, noise_scale);
            for (int d = 0; d < dim; d++) query[d] = data[pt*dim+d] + noise(qrng);
            hnsw->searchKnn(query.data(), 10);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        cache_miss.stop();
        double cold_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        long long cold_cache_miss = cache_miss.read_count();

        std::cout << "  Search near (after cache flush): "
                  << std::setprecision(2) << cold_ms << " ms"
                  << "  cache_miss=" << cold_cache_miss << "\n";
        std::cout << "  Compare with Phase B above (warm cache) to see the difference.\n";
    }

    delete[] data;
    delete hnsw;
    std::cout << "\nDone.\n";
    return 0;
}
