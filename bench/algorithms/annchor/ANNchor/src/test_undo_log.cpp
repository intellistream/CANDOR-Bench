#include <iostream>
#include <vector>
#include <random>
#include "hnswlib/hnswalg.h"
#include "hnswlib/space_l2.h"

using namespace annchor;

int main() {
    const int DIM = 128;
    const int MAX_ELEMENTS = 10000;
    const int M = 16;
    const int EF_CONSTRUCTION = 200;

    L2Space space(DIM);
    HierarchicalNSW<float> index(&space, MAX_ELEMENTS, M, EF_CONSTRUCTION, 100, false, true, 1);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 1);

    auto gen_vec = [&]() {
        std::vector<float> v(DIM);
        for (int i = 0; i < DIM; i++) v[i] = dist(rng);
        return v;
    };

    // Insert batch 0: nodes 0-999
    std::cout << "Inserting batch 0 (nodes 0-999)..." << std::endl;
    std::vector<float> batch0_data(1000 * DIM);
    std::vector<labeltype> batch0_labels(1000);
    for (int i = 0; i < 1000; i++) {
        auto v = gen_vec();
        std::copy(v.begin(), v.end(), batch0_data.begin() + i * DIM);
        batch0_labels[i] = i;
    }
    index.addPointBatch(batch0_data.data(), batch0_labels.data(), 1000);

    // Insert batch 1: nodes 1000-1999
    std::cout << "Inserting batch 1 (nodes 1000-1999)..." << std::endl;
    std::vector<float> batch1_data(1000 * DIM);
    std::vector<labeltype> batch1_labels(1000);
    for (int i = 0; i < 1000; i++) {
        auto v = gen_vec();
        std::copy(v.begin(), v.end(), batch1_data.begin() + i * DIM);
        batch1_labels[i] = 1000 + i;
    }
    index.addPointBatch(batch1_data.data(), batch1_labels.data(), 1000);

    // Insert batch 2: nodes 2000-2999
    std::cout << "Inserting batch 2 (nodes 2000-2999)..." << std::endl;
    std::vector<float> batch2_data(1000 * DIM);
    std::vector<labeltype> batch2_labels(1000);
    for (int i = 0; i < 1000; i++) {
        auto v = gen_vec();
        std::copy(v.begin(), v.end(), batch2_data.begin() + i * DIM);
        batch2_labels[i] = 2000 + i;
    }
    index.addPointBatch(batch2_data.data(), batch2_labels.data(), 1000);

    // Check undo log stats
    std::cout << "\n=== Undo Log Stats ===" << std::endl;

    int nodes_with_log = 0;
    int total_pruned_edges = 0;
    int total_batches = 0;

    for (uint32_t i = 0; i < 3000; i++) {
        if (index.undo_log_->hasLog(i)) {
            nodes_with_log++;
            const auto* log = index.undo_log_->getLog(i);
            total_pruned_edges += log->edges.size();
            total_batches += log->batches.size();
        }
    }

    std::cout << "Nodes with undo log: " << nodes_with_log << std::endl;
    std::cout << "Total pruned edges: " << total_pruned_edges << std::endl;
    std::cout << "Total batch entries: " << total_batches << std::endl;

    // Print some examples
    std::cout << "\n=== Sample Undo Logs ===" << std::endl;
    int printed = 0;
    for (uint32_t i = 0; i < 3000 && printed < 5; i++) {
        if (index.undo_log_->hasLog(i)) {
            const auto* log = index.undo_log_->getLog(i);
            std::cout << "Node " << i << ":" << std::endl;
            std::cout << "  batches: [";
            for (size_t j = 0; j < log->batches.size(); j++) {
                std::cout << log->batches[j];
                if (j < log->batches.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "  offsets: [";
            for (size_t j = 0; j < log->offsets.size(); j++) {
                std::cout << log->offsets[j];
                if (j < log->offsets.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "  edges (B): [";
            for (size_t j = 0; j < std::min(log->edges.size(), (size_t)10); j++) {
                std::cout << log->edges[j];
                if (j < log->edges.size() - 1) std::cout << ", ";
            }
            if (log->edges.size() > 10) std::cout << "...";
            std::cout << "]" << std::endl;
            std::cout << "  causes (S): [";
            for (size_t j = 0; j < std::min(log->causes.size(), (size_t)10); j++) {
                std::cout << log->causes[j];
                if (j < log->causes.size() - 1) std::cout << ", ";
            }
            if (log->causes.size() > 10) std::cout << "...";
            std::cout << "]" << std::endl;
            printed++;
        }
    }

    // Test recovery
    std::cout << "\n=== Test Recovery ===" << std::endl;
    for (uint32_t i = 0; i < 3000 && printed < 10; i++) {
        if (index.undo_log_->hasLog(i)) {
            const auto* log = index.undo_log_->getLog(i);
            if (log->batches.size() >= 2) {
                std::cout << "Node " << i << ":" << std::endl;

                uint32_t edges_buf[64], causes_buf[64];

                // Recover for visible_batch = 999
                size_t n1 = index.undo_log_->getRecoverPointers(i, 999,
                    (const uint32_t**)&edges_buf, (const uint32_t**)&causes_buf);
                std::cout << "  visible_batch=999: recover " << n1 << " edges" << std::endl;

                // Recover for visible_batch = 1999
                size_t n2 = index.undo_log_->getRecoverPointers(i, 1999,
                    (const uint32_t**)&edges_buf, (const uint32_t**)&causes_buf);
                std::cout << "  visible_batch=1999: recover " << n2 << " edges" << std::endl;

                // Recover for visible_batch = 2999
                size_t n3 = index.undo_log_->getRecoverPointers(i, 2999,
                    (const uint32_t**)&edges_buf, (const uint32_t**)&causes_buf);
                std::cout << "  visible_batch=2999: recover " << n3 << " edges" << std::endl;

                break;
            }
        }
    }

    std::cout << "\nDone!" << std::endl;
    return 0;
}
