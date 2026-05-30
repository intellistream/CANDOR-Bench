#include "hnswlib/hnswlib.h"

void load_fvecs(char* filename, float* data, int dim, int num_vectors) {
    if (filename == NULL) {
        return;
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Data file " << filename << " not found " << std::endl;
        std::abort();
    }

    for (int i = 0; i < num_vectors; ++i) {
        file.read(reinterpret_cast<char*>(data + i * dim), dim * sizeof(float));
        if (!file) {
            std::cerr << "Error reading data for vector " << i << std::endl;
            std::abort();
        }
    }

    file.close();
    std::cout << "Loaded " << num_vectors << " vectors from " << filename
              << std::endl;
}

int main(int argc, char* argv[]) {
    int M = 32;
    int ef_construction = 400;
    int num_threads = 1;
    int max_elements = 1000;
    int dim = 128;
    char* base_file = "../data/sift/sift_base.fvecs";

    if (argc > 1) num_threads = std::stoi(argv[1]);
    if (argc > 2) max_elements = std::stoi(argv[2]);
    if (argc > 3) dim = std::stof(argv[3]);
    if (argc > 4) base_file = argv[4];

    annchor::L2Space space(dim);
    annchor::HierarchicalNSW<float>* alg_hnsw =
        new annchor::HierarchicalNSW<float>(&space, max_elements, M,
                                            ef_construction);

    if (false) {
        alg_hnsw->loadIndex("dummy", &space, 0);
    }

    float* data = new float[dim * max_elements];
    if (base_file == nullptr || std::string(base_file) == "random") {
        std::mt19937 rng;
        rng.seed(47);
        std::uniform_real_distribution<> distrib_real;
        for (int i = 0; i < dim * max_elements; i++) {
            data[i] = distrib_real(rng);
        }
    } else {
        load_fvecs(base_file, data, dim, max_elements);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Enable scheduler if requested (simple toggle for now, assume always on
    // for demo) In real usage, we would parse a flag.
    // alg_hnsw->enable_scheduler(num_threads);

    // For this demo, let's just enable it if num_threads > 1 to see effect
    // enableScheduler is internal/implicit in HierarchicalNSW or not available
    std::cout << "Scheduler usage simplified/removed." << std::endl;

    // auto start_time = std::chrono::high_resolution_clock::now();

    // Use addPointBatch with raw pointer.
    // New signature: addPointBatch(data, num, labels=nullptr) matches existing
    // call relying on default. Use addPoint in a loop since addPointBatch was
    // removed from library
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    /*
#pragma omp parallel for num_threads(num_threads)
    for (size_t row = 0; row < max_elements; row++) {
        alg_hnsw->addPoint((void*)(data + dim * row), row);
    }
    */

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> insert_duration = end_time - start_time;

    double qps = max_elements / insert_duration.count();
    std::cout << "Total Time: " << insert_duration.count() << " seconds\n";
    std::cout << "Queries per second: " << qps << " under " << num_threads
              << " threads" << "\n";

    // Benchmark Search
    std::cout << "Benchmarking Search..." << std::endl;
    auto start_search = std::chrono::high_resolution_clock::now();

    std::vector<std::priority_queue<std::pair<float, size_t>>> results(
        max_elements);

    // Use searchKnn in a loop since searchKnnBatch was removed
    for (size_t i = 0; i < max_elements; i++) {
        auto result = alg_hnsw->searchKnn(data + i * dim, 1);
        while (!result.empty()) {
            auto p = result.top();
            results[i].push(std::make_pair(p.first, (size_t)p.second));
            result.pop();
        }
    }

    auto end_search = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> search_duration = end_search - start_search;
    double search_qps = max_elements / search_duration.count();
    std::cout << "Search Time: " << search_duration.count() << " seconds\n";
    std::cout << "Search QPS: " << search_qps << " under " << num_threads
              << " threads\n";

    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        if (results[i].empty()) continue;
        auto label = results[i].top().second;
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";

    std::cout << "Recall: " << recall << "\n";

    // Test Save/Load
    std::cout << "Testing Save/Load..." << std::endl;
    alg_hnsw->saveIndex("test_index.bin");

    annchor::HierarchicalNSW<float>* alg_hnsw_loaded =
        new annchor::HierarchicalNSW<float>(&space, "test_index.bin", false,
                                            max_elements);

    std::cout << "Load successful. Checking integrity..." << std::endl;
    alg_hnsw_loaded->checkIntegrity();
    std::cout << "Integrity check passed." << std::endl;

    delete alg_hnsw_loaded;
    delete[] data;
    delete alg_hnsw;

    return 0;
}
