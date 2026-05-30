/**
 * Observe Insert-Search Sharing on REAL hnswlib
 *
 * NO changes to hnswalg.h — pure black-box observation.
 *
 * Approach:
 *   1. Build index with N points
 *   2. For each new point v to insert:
 *      a. BEFORE inserting: searchKnn(v, k=200, ef=ef_c) → "what insert would see"
 *      b. Do the actual insert
 *      c. searchKnn(q, k=200, ef=ef_s) where q=v+noise → "what search sees"
 *      d. Compare the two result sets
 *
 *   This is an approximation because (a) happens before v is in the graph.
 *   But it captures the same beam search trajectory that insert would follow.
 */

#include "hnswlib/hnswlib.h"
#include "hnswlib/hnswalg.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <numeric>

void load_bin(const char* filename, float* data, int dim, int num) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) { std::cerr << "Cannot open " << filename << "\n"; std::abort(); }
    uint32_t fn, fd;
    file.read(reinterpret_cast<char*>(&fn), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&fd), sizeof(uint32_t));
    std::cout << "File: num=" << fn << " dim=" << fd << "\n";
    for (int i = 0; i < num; i++)
        file.read(reinterpret_cast<char*>(data + i * dim), dim * sizeof(float));
}

// Extract result IDs from searchKnn output (labeltype = size_t)
using labeltype = size_t;
std::unordered_set<labeltype> resultSet(
    std::priority_queue<std::pair<float, labeltype>> pq) {
    std::unordered_set<labeltype> s;
    while (!pq.empty()) { s.insert(pq.top().second); pq.pop(); }
    return s;
}

std::vector<std::pair<float, labeltype>> resultVec(
    std::priority_queue<std::pair<float, labeltype>> pq) {
    std::vector<std::pair<float, labeltype>> v;
    while (!pq.empty()) { v.push_back(pq.top()); pq.pop(); }
    std::sort(v.begin(), v.end());
    return v;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: observe_sharing <data_file> <max_elements> [dim]\n";
        return 1;
    }
    char* data_file = argv[1];
    int max_elements = std::stoi(argv[2]);
    int dim = argc > 3 ? std::stoi(argv[3]) : 0;

    // Read dim from file
    {
        std::ifstream f(data_file, std::ios::binary);
        uint32_t fn, fd;
        f.read(reinterpret_cast<char*>(&fn), sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(&fd), sizeof(uint32_t));
        if (dim == 0) dim = fd;
        max_elements = std::min(max_elements, (int)fn);
    }

    std::cout << "================================================================\n"
              << " Observe Insert-Search Sharing (Real hnswlib)\n"
              << "================================================================\n\n";

    // Load data
    float* data = new float[dim * max_elements];
    load_bin(data_file, data, dim, max_elements);
    std::cout << "Loaded " << max_elements << " points, dim=" << dim << "\n";

    // Compute avg distance
    double avg_dist = 0;
    { std::mt19937 rng(42); std::uniform_int_distribution<int> idx(0, max_elements-1);
      for (int i = 0; i < 1000; i++) {
          int a = idx(rng), b = idx(rng);
          float s = 0;
          for (int d = 0; d < dim; d++) {
              float df = data[a*dim+d] - data[b*dim+d]; s += df*df;
          }
          avg_dist += std::sqrt(s);
      }
      avg_dist /= 1000;
    }
    float sd = std::sqrt((float)dim);
    std::cout << "Avg inter-point dist: " << std::fixed << std::setprecision(2) << avg_dist << "\n\n";

    // Build HNSW
    int M = 16;
    int ef_construction = 200;
    int warmup = max_elements * 2 / 3;
    int test_n = std::min(500, max_elements - warmup - 200);

    annchor::L2Space space(dim);
    annchor::HierarchicalNSW<float>* hnsw =
        new annchor::HierarchicalNSW<float>(&space, max_elements, M,
                                             ef_construction, 100, false, false, 1);

    std::cout << "Building index (warmup=" << warmup << ")...\n";
    for (int i = 0; i < warmup; i++)
        hnsw->addPoint(data + (size_t)i * dim, i);
    std::cout << "Warmup done. Index: " << hnsw->getCurrentElementCount() << " nodes\n\n";

    int ef_s = 50;
    int k_observe = 200;  // collect top-200 to see full candidate set

    // ============================================================
    // Observation 1: Per-pair result overlap (chasing vs random)
    // ============================================================
    std::cout << "=== Obs 1: Per-pair result overlap ===\n";
    std::cout << "  Compare: searchKnn(v, ef=ef_c) vs searchKnn(q, ef=ef_s)\n\n";

    std::cout << std::left << std::setw(14) << "mode"
              << std::setw(10) << "noise"
              << std::setw(12) << "shared/ef_s"
              << std::setw(14) << "shared_top10"
              << std::setw(10) << "ins_sz"
              << std::setw(10) << "srch_sz"
              << "\n" << std::string(70, '-') << "\n";

    for (auto& [mode, nf] : std::vector<std::pair<std::string, float>>{
        {"chasing", 0.0f}, {"chasing", 0.05f}, {"chasing", 0.1f},
        {"chasing", 0.5f}, {"chasing", 1.0f}, {"random", -1.0f}}) {

        std::mt19937 qrng(123);
        float npd = nf * avg_dist / sd;
        double sum_overlap_ef = 0, sum_overlap_10 = 0;
        int sum_ins_sz = 0, sum_srch_sz = 0;

        for (int i = 0; i < test_n; i++) {
            int pt = warmup + i;
            float* vec = data + (size_t)pt * dim;

            // "Insert's view": search with ef=ef_c BEFORE inserting
            hnsw->setEf(ef_construction);
            auto ins_result = hnsw->searchKnn(vec, k_observe);
            auto ins_set = resultSet(ins_result);

            // Actually insert the point
            hnsw->addPoint(vec, pt);

            // Generate query
            std::vector<float> query(dim);
            if (nf >= 0) {
                std::normal_distribution<float> noise(0, npd);
                for (int d = 0; d < dim; d++) query[d] = vec[d] + noise(qrng);
            } else {
                // Random: pick a random existing point + small noise
                std::uniform_int_distribution<int> ridx(0, (int)hnsw->getCurrentElementCount()-1);
                int rpt = ridx(qrng);
                float small_npd = 0.1f * avg_dist / sd;
                std::normal_distribution<float> noise(0, small_npd);
                for (int d = 0; d < dim; d++)
                    query[d] = data[rpt * dim + d] + noise(qrng);
            }

            // "Search's view": search with ef=ef_s
            hnsw->setEf(ef_s);
            auto srch_result_pq = hnsw->searchKnn(query.data(), ef_s);
            auto srch_set = resultSet(srch_result_pq);

            // Also get top-10 for the search
            hnsw->setEf(ef_s);
            auto srch_top10_pq = hnsw->searchKnn(query.data(), 10);
            auto srch_top10 = resultVec(srch_top10_pq);

            // Overlap: how many of search's ef_s results are in insert's ef_c results?
            int shared_ef = 0;
            for (uint32_t n : srch_set) if (ins_set.count(n)) shared_ef++;

            // Overlap of search's top-10 with insert's candidates
            int shared_10 = 0;
            for (auto& [d, n] : srch_top10) if (ins_set.count(n)) shared_10++;

            sum_overlap_ef += (double)shared_ef / srch_set.size();
            sum_overlap_10 += (double)shared_10 / std::min(10, (int)srch_top10.size());
            sum_ins_sz += ins_set.size();
            sum_srch_sz += srch_set.size();
        }

        double n = test_n;
        std::ostringstream nf_str;
        if (nf >= 0) nf_str << std::setprecision(2) << nf << "x";
        else nf_str << "random";

        std::cout << std::setw(14) << mode
                  << std::setw(10) << nf_str.str()
                  << std::setw(11) << std::setprecision(1) << (sum_overlap_ef/n)*100 << "%"
                  << std::setw(13) << std::setprecision(1) << (sum_overlap_10/n)*100 << "%"
                  << std::setw(10) << std::setprecision(0) << sum_ins_sz/n
                  << std::setw(10) << sum_srch_sz/n
                  << "\n";
    }

    // ============================================================
    // Observation 2: For random workload, how many inserts to cover?
    // ============================================================
    std::cout << "\n=== Obs 2: Coverage growth (random queries) ===\n";
    std::cout << "  How many insert candidates needed to cover a random search's top-10?\n\n";
    {
        // Collect recent insert candidate sets
        std::vector<std::unordered_set<labeltype>> recent_ins_sets;
        int coverage_inserts = std::min(500, max_elements - (int)hnsw->getCurrentElementCount() - 100);

        hnsw->setEf(ef_construction);
        for (int i = 0; i < coverage_inserts; i++) {
            int pt = hnsw->getCurrentElementCount();
            if (pt >= max_elements) break;
            float* vec = data + (size_t)pt * dim;
            auto ins_result = hnsw->searchKnn(vec, k_observe);
            recent_ins_sets.push_back(resultSet(ins_result));
            hnsw->addPoint(vec, pt);
        }

        // For random queries, measure coverage with increasing # of inserts
        int n_queries = 200;
        std::mt19937 qrng(456);

        std::cout << std::left << std::setw(12) << "N_inserts"
                  << std::setw(14) << "top10_cov%"
                  << std::setw(14) << "top50_cov%"
                  << "\n" << std::string(40, '-') << "\n";

        for (int N : {1, 5, 10, 50, 100, 200, 500}) {
            if (N > (int)recent_ins_sets.size()) break;
            double sum_cov10 = 0, sum_cov50 = 0;

            for (int qi = 0; qi < n_queries; qi++) {
                // Random query
                std::uniform_int_distribution<int> ridx(0, (int)hnsw->getCurrentElementCount()-1);
                int rpt = ridx(qrng);
                float small_npd = 0.1f * avg_dist / sd;
                std::normal_distribution<float> noise(0, small_npd);
                std::vector<float> query(dim);
                for (int d = 0; d < dim; d++)
                    query[d] = data[rpt * dim + d] + noise(qrng);

                hnsw->setEf(ef_s);
                auto srch_top50_pq = hnsw->searchKnn(query.data(), 50);
                auto srch_top50 = resultVec(srch_top50_pq);

                // Union of last N insert candidate sets
                std::unordered_set<labeltype> ins_union;
                int start = std::max(0, (int)recent_ins_sets.size() - N);
                for (int j = start; j < (int)recent_ins_sets.size(); j++)
                    for (auto n : recent_ins_sets[j]) ins_union.insert(n);

                // Coverage of top-10 and top-50
                int cov10 = 0, cov50 = 0;
                for (int j = 0; j < (int)srch_top50.size(); j++) {
                    if (ins_union.count(srch_top50[j].second)) {
                        cov50++;
                        if (j < 10) cov10++;
                    }
                }
                sum_cov10 += (double)cov10 / std::min(10, (int)srch_top50.size());
                sum_cov50 += (double)cov50 / srch_top50.size();
            }

            std::cout << std::setw(12) << N
                      << std::setw(13) << std::setprecision(1) << (sum_cov10/n_queries)*100 << "%"
                      << std::setw(13) << std::setprecision(1) << (sum_cov50/n_queries)*100 << "%"
                      << "\n";
        }
    }

    delete[] data;
    delete hnsw;
    std::cout << "\nDone.\n";
    return 0;
}
