#include "hnswlib/hnswlib.h"
#include "hnswlib/hnswalg.h"
#include <map>

void load_bin(char* filename, float* data, int dim, int num_vectors) {
    if (filename == NULL) {
        return;
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Data file " << filename << " not found " << std::endl;
        std::abort();
    }

    uint32_t file_num, file_dim;
    file.read(reinterpret_cast<char*>(&file_num), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&file_dim), sizeof(uint32_t));
    std::cout << "File header: num=" << file_num << ", dim=" << file_dim << std::endl;

    for (int i = 0; i < num_vectors; ++i) {
        file.read(reinterpret_cast<char*>(data + i * dim), dim * sizeof(float));
        if (!file) {
            std::cerr << "Error reading data for vector " << i << std::endl;
            std::abort();
        }
    }

    file.close();
    std::cout << "Loaded " << num_vectors << " vectors from " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    int M = 32;
    int ef_construction = 400;
    int num_threads = 1;
    int max_elements = 1000;
    int dim = 128;
    char* base_file = nullptr;

    if (argc > 1) num_threads = std::stoi(argv[1]);
    if (argc > 2) max_elements = std::stoi(argv[2]);
    if (argc > 3) dim = std::stof(argv[3]);
    if (argc > 4) base_file = argv[4];

    annchor::L2Space space(dim);
    annchor::HierarchicalNSW<float>* alg_hnsw =
        new annchor::HierarchicalNSW<float>(&space, max_elements, M,
                                            ef_construction, 100, false, true, num_threads);

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
        load_bin(base_file, data, dim, max_elements);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Enable scheduler if requested (simple toggle for now, assume always on
    // for demo) In real usage, we would parse a flag.
    // alg_hnsw->enable_scheduler(num_threads);

    // For this demo, let's just enable it if num_threads > 1 to see effect
    // enableScheduler is internal/implicit in HierarchicalNSW or not available
    std::cout << "Scheduler usage simplified/removed." << std::endl;

    // auto start_time = std::chrono::high_resolution_clock::now();

    // Use addPointBatch for batch insertion
    int batch_size = 1000;
    if (argc > 5) batch_size = std::stoi(argv[5]);
    char* query_file_arg = nullptr;
    if (argc > 6) query_file_arg = argv[6];
    std::cout << "Batch size: " << batch_size << std::endl;
    std::vector<annchor::labeltype> labels(batch_size);
    for (int batch_start = 0; batch_start < max_elements; batch_start += batch_size) {
        int this_batch = std::min(batch_size, max_elements - batch_start);
        for (int i = 0; i < this_batch; i++) {
            labels[i] = batch_start + i;
        }
        alg_hnsw->addPointBatch(data + batch_start * dim, labels.data(), this_batch);
    }

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

    // Undo log stats
    std::cout << "\n=== Undo Log Stats ===" << std::endl;
    int nodes_with_log = 0;
    int total_pruned_edges = 0;
    std::map<uint32_t, int> edges_per_batch;
    for (uint32_t i = 0; i < (uint32_t)max_elements; i++) {
        if (alg_hnsw->undo_log_->hasLog(i)) {
            nodes_with_log++;
            const auto* log = alg_hnsw->undo_log_->getLog(i);
            total_pruned_edges += log->edges.size();
            for (size_t j = 0; j < log->batches.size(); j++) {
                uint32_t batch_id = log->batches[j];
                uint32_t start = (j == 0) ? 0 : log->offsets[j - 1];
                uint32_t end = log->offsets[j];
                edges_per_batch[batch_id] += (end - start);
            }
        }
    }
    std::cout << "Nodes with undo log: " << nodes_with_log << " / " << max_elements << std::endl;
    std::cout << "Total pruned edges: " << total_pruned_edges << std::endl;
    std::cout << "Avg pruned edges per node with log: " << (nodes_with_log > 0 ? (float)total_pruned_edges / nodes_with_log : 0) << std::endl;
    std::cout << "\nEdges pruned per batch (first 10 and last 10):" << std::endl;
    int cnt = 0;
    for (auto& [batch_id, count] : edges_per_batch) {
        if (cnt < 10) std::cout << "  Batch " << batch_id << ": " << count << " edges" << std::endl;
        cnt++;
    }
    if (edges_per_batch.size() > 20) {
        std::cout << "  ..." << std::endl;
        auto it = edges_per_batch.end();
        std::advance(it, -10);
        for (; it != edges_per_batch.end(); ++it) {
            std::cout << "  Batch " << it->first << ": " << it->second << " edges" << std::endl;
        }
    }

    std::cout << "\n=== Version Distribution ===" << std::endl;
    std::map<int, int> version_count_dist;
    int max_versions = 0;
    int max_versions_node = -1;
    for (uint32_t i = 0; i < (uint32_t)max_elements; i++) {
        if (alg_hnsw->undo_log_->hasLog(i)) {
            const auto* log = alg_hnsw->undo_log_->getLog(i);
            int num_versions = log->batches.size();
            version_count_dist[num_versions]++;
            if (num_versions > max_versions) {
                max_versions = num_versions;
                max_versions_node = i;
            }
        }
    }
    std::cout << "Version count distribution:" << std::endl;
    for (auto& [v, c] : version_count_dist) {
        std::cout << "  " << v << " versions: " << c << " nodes" << std::endl;
    }
    std::cout << "Max versions: " << max_versions << " (node " << max_versions_node << ")" << std::endl;

    // Memory overhead analysis
    std::cout << "\n=== Memory Overhead ===" << std::endl;
    size_t undo_log_bytes = 0;
    for (uint32_t i = 0; i < (uint32_t)max_elements; i++) {
        if (alg_hnsw->undo_log_->hasLog(i)) {
            const auto* log = alg_hnsw->undo_log_->getLog(i);
            undo_log_bytes += log->batches.capacity() * sizeof(uint32_t);
            undo_log_bytes += log->offsets.capacity() * sizeof(uint32_t);
            undo_log_bytes += log->edges.capacity() * sizeof(uint32_t);
            undo_log_bytes += log->causes.capacity() * sizeof(uint32_t);
        }
    }
    std::cout << "Undo log memory: " << undo_log_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "Per-node overhead: " << (double)undo_log_bytes / max_elements << " bytes" << std::endl;
    std::cout << "Per-pruned-edge overhead: " << (double)undo_log_bytes / total_pruned_edges << " bytes" << std::endl;

    // Test version recovery query latency
    std::cout << "\n=== Version Recovery Query Latency ===" << std::endl;
    int num_batches = (max_elements + batch_size - 1) / batch_size;
    std::vector<int> versions_back = {10, 20, 50, 100, 200};

    // Load query data
    int num_queries = 10000;
    float* query_data = new float[dim * num_queries];
    std::string query_file = query_file_arg ? query_file_arg : "";
    {
        std::ifstream qf(query_file, std::ios::binary);
        if (qf.is_open()) {
            uint32_t qnum, qdim;
            qf.read(reinterpret_cast<char*>(&qnum), sizeof(uint32_t));
            qf.read(reinterpret_cast<char*>(&qdim), sizeof(uint32_t));
            num_queries = std::min(num_queries, (int)qnum);
            for (int i = 0; i < num_queries; i++) {
                qf.read(reinterpret_cast<char*>(query_data + i * dim), dim * sizeof(float));
            }
            qf.close();
            std::cout << "Loaded " << num_queries << " queries from " << query_file << std::endl;
        } else {
            std::cout << "Query file not found, using base data as queries" << std::endl;
            num_queries = std::min(num_queries, max_elements);
            memcpy(query_data, data, dim * num_queries * sizeof(float));
        }
    }

    std::cout << "versions_back, visible_batch, avg_us, p50_us, p95_us, p99_us, qps, nodes_visited, vis_checks, undo_lookups, recovered, useful, hit%" << std::endl;
    for (int vb : versions_back) {
        if (vb >= num_batches) continue;

        size_t visible_batch_id = (num_batches - vb) * batch_size - 1;
        if (visible_batch_id >= (size_t)max_elements) visible_batch_id = max_elements - 1;

        // Reset all metrics before test
        alg_hnsw->metric_nodes_visited.store(0);
        alg_hnsw->metric_visibility_checks.store(0);
        alg_hnsw->metric_undo_log_lookups.store(0);
        alg_hnsw->metric_recovered_edges_total.store(0);
        alg_hnsw->metric_recovered_edges_useful.store(0);

        std::vector<double> latencies(num_queries);
        for (int q = 0; q < num_queries; q++) {
            auto t1 = std::chrono::high_resolution_clock::now();
            auto result = alg_hnsw->searchKnn(query_data + q * dim, 10, nullptr, visible_batch_id + 1);
            auto t2 = std::chrono::high_resolution_clock::now();
            latencies[q] = std::chrono::duration<double, std::micro>(t2 - t1).count();
        }
        std::sort(latencies.begin(), latencies.end());
        double avg = 0;
        for (auto l : latencies) avg += l;
        avg /= num_queries;
        double p50 = latencies[num_queries * 50 / 100];
        double p95 = latencies[num_queries * 95 / 100];
        double p99 = latencies[num_queries * 99 / 100];
        double qps_val = num_queries / (avg * num_queries / 1e6);

        size_t nodes_vis = alg_hnsw->metric_nodes_visited.load();
        size_t vis_checks = alg_hnsw->metric_visibility_checks.load();
        size_t undo_lookups = alg_hnsw->metric_undo_log_lookups.load();
        size_t total_rec = alg_hnsw->metric_recovered_edges_total.load();
        size_t useful_rec = alg_hnsw->metric_recovered_edges_useful.load();
        double hit_rate = total_rec > 0 ? (double)useful_rec / total_rec * 100.0 : 0;

        std::cout << vb << ", " << visible_batch_id << ", " << avg << ", " << p50 << ", " << p95 << ", " << p99
                  << ", " << qps_val << ", " << nodes_vis << ", " << vis_checks << ", " << undo_lookups
                  << ", " << total_rec << ", " << useful_rec << ", " << hit_rate << "%" << std::endl;
    }

    // Also test current version (no recovery)
    {
        // Reset all metrics
        alg_hnsw->metric_nodes_visited.store(0);
        alg_hnsw->metric_visibility_checks.store(0);
        alg_hnsw->metric_undo_log_lookups.store(0);
        alg_hnsw->metric_recovered_edges_total.store(0);
        alg_hnsw->metric_recovered_edges_useful.store(0);

        std::vector<double> latencies(num_queries);
        for (int q = 0; q < num_queries; q++) {
            auto t1 = std::chrono::high_resolution_clock::now();
            auto result = alg_hnsw->searchKnn(query_data + q * dim, 10);
            auto t2 = std::chrono::high_resolution_clock::now();
            latencies[q] = std::chrono::duration<double, std::micro>(t2 - t1).count();
        }
        std::sort(latencies.begin(), latencies.end());
        double avg = 0;
        for (auto l : latencies) avg += l;
        avg /= num_queries;
        double p50 = latencies[num_queries * 50 / 100];
        double p95 = latencies[num_queries * 95 / 100];
        double p99 = latencies[num_queries * 99 / 100];
        double qps_val = num_queries / (avg * num_queries / 1e6);

        size_t nodes_vis = alg_hnsw->metric_nodes_visited.load();
        size_t vis_checks = alg_hnsw->metric_visibility_checks.load();
        size_t undo_lookups = alg_hnsw->metric_undo_log_lookups.load();

        std::cout << "current, " << max_elements - 1 << ", " << avg << ", " << p50 << ", " << p95 << ", " << p99
                  << ", " << qps_val << ", " << nodes_vis << ", " << vis_checks << ", " << undo_lookups
                  << ", 0, 0, 0%" << std::endl;
    }

    // Output per-query averages for understanding overhead
    std::cout << "\n=== MVCC Overhead Analysis (per query) ===" << std::endl;

    // Re-run 200 versions back to get detailed metrics
    size_t vb200 = 200;
    if (vb200 < (size_t)num_batches) {
        size_t visible_batch_id = (num_batches - vb200) * batch_size - 1;
        alg_hnsw->metric_nodes_visited.store(0);
        alg_hnsw->metric_visibility_checks.store(0);
        alg_hnsw->metric_undo_log_lookups.store(0);
        alg_hnsw->metric_recovered_edges_total.store(0);
        alg_hnsw->metric_recovered_edges_useful.store(0);
        alg_hnsw->metric_distance_computations.store(0);

        double total_time_us = 0;
        for (int q = 0; q < num_queries; q++) {
            auto t1 = std::chrono::high_resolution_clock::now();
            alg_hnsw->searchKnn(query_data + q * dim, 10, nullptr, visible_batch_id + 1);
            auto t2 = std::chrono::high_resolution_clock::now();
            total_time_us += std::chrono::duration<double, std::micro>(t2 - t1).count();
        }

        double avg_time_us = total_time_us / num_queries;
        double nodes_per_q = (double)alg_hnsw->metric_nodes_visited.load() / num_queries;
        double vis_per_q = (double)alg_hnsw->metric_visibility_checks.load() / num_queries;
        double undo_per_q = (double)alg_hnsw->metric_undo_log_lookups.load() / num_queries;
        double rec_per_q = (double)alg_hnsw->metric_recovered_edges_total.load() / num_queries;
        double useful_per_q = (double)alg_hnsw->metric_recovered_edges_useful.load() / num_queries;
        double dist_per_q = (double)alg_hnsw->metric_distance_computations.load() / num_queries;

        std::cout << "200 versions back:" << std::endl;
        std::cout << "  Avg latency: " << avg_time_us << " us" << std::endl;
        std::cout << "  Nodes visited/q: " << nodes_per_q << std::endl;
        std::cout << "  Visibility checks/q: " << vis_per_q << std::endl;
        std::cout << "  Undo log lookups/q: " << undo_per_q << std::endl;
        std::cout << "  Recovered edges/q: " << rec_per_q << std::endl;
        std::cout << "  Useful recovered/q: " << useful_per_q << std::endl;
        std::cout << "  Distance comps/q: " << dist_per_q << std::endl;

        // Now baseline without MVCC
        alg_hnsw->metric_nodes_visited.store(0);
        alg_hnsw->metric_distance_computations.store(0);
        double baseline_time_us = 0;
        for (int q = 0; q < num_queries; q++) {
            auto t1 = std::chrono::high_resolution_clock::now();
            alg_hnsw->searchKnn(query_data + q * dim, 10);
            auto t2 = std::chrono::high_resolution_clock::now();
            baseline_time_us += std::chrono::duration<double, std::micro>(t2 - t1).count();
        }
        double baseline_avg_us = baseline_time_us / num_queries;
        double baseline_nodes_per_q = (double)alg_hnsw->metric_nodes_visited.load() / num_queries;
        double baseline_dist_per_q = (double)alg_hnsw->metric_distance_computations.load() / num_queries;

        std::cout << "\nCurrent version (no MVCC):" << std::endl;
        std::cout << "  Avg latency: " << baseline_avg_us << " us" << std::endl;
        std::cout << "  Nodes visited/q: " << baseline_nodes_per_q << std::endl;
        std::cout << "  Distance comps/q: " << baseline_dist_per_q << std::endl;

        std::cout << "\n=== MVCC Overhead Breakdown ===" << std::endl;
        double overhead_us = avg_time_us - baseline_avg_us;
        double overhead_pct = (avg_time_us / baseline_avg_us - 1) * 100;
        std::cout << "Total overhead: " << overhead_us << " us (" << overhead_pct << "%)" << std::endl;

        // Estimate per-operation costs
        double per_vis_check_ns = overhead_us * 1000 / vis_per_q;
        double per_undo_lookup_ns = overhead_us * 1000 / undo_per_q;
        std::cout << "If overhead is all visibility checks: " << per_vis_check_ns << " ns per check" << std::endl;
        std::cout << "If overhead is all undo lookups: " << per_undo_lookup_ns << " ns per lookup" << std::endl;

        // What percentage of time is distance computation vs overhead
        // Assume ~100ns per L2 distance (typical for 128-dim)
        double estimated_dist_time_us = baseline_dist_per_q * 0.05; // ~50ns per distance
        std::cout << "\nEstimated breakdown (assuming 50ns per distance comp):" << std::endl;
        std::cout << "  Distance computation: ~" << estimated_dist_time_us << " us (" << (estimated_dist_time_us/baseline_avg_us*100) << "% of baseline)" << std::endl;
        std::cout << "  Other (memory, queue ops): ~" << (baseline_avg_us - estimated_dist_time_us) << " us" << std::endl;
        std::cout << "  MVCC overhead: ~" << overhead_us << " us (" << overhead_pct << "% added)" << std::endl;
    }

    // Exact recovery-on/off comparison on the same historical snapshot.
    {
        struct CompareRow {
            double on_us{0};
            double off_us{0};
            size_t undo_lookups{0};
            size_t recovered_total{0};
            size_t useful_total{0};
            size_t max_single{0};
        };
        struct LatencySummary {
            double avg{0};
            double p50{0};
            double p95{0};
            double p99{0};
        };

        auto reset_recovery_metrics = [&]() {
            alg_hnsw->metric_undo_log_lookups.store(0);
            alg_hnsw->metric_recovered_edges_total.store(0);
            alg_hnsw->metric_recovered_edges_useful.store(0);
            alg_hnsw->metric_recovered_edges_single_node_max.store(0);
        };

        auto run_historical_query = [&](const void* query, size_t view_limit,
                                        bool enable_recovery, CompareRow* row,
                                        bool record_metrics) -> double {
            reset_recovery_metrics();
            alg_hnsw->setEnableUndoRecovery(enable_recovery);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto result = alg_hnsw->searchKnn(query, 10, nullptr, view_limit);
            auto t2 = std::chrono::high_resolution_clock::now();
            (void)result;
            if (record_metrics && row != nullptr) {
                row->undo_lookups = alg_hnsw->metric_undo_log_lookups.load();
                row->recovered_total = alg_hnsw->metric_recovered_edges_total.load();
                row->useful_total = alg_hnsw->metric_recovered_edges_useful.load();
                row->max_single =
                    alg_hnsw->metric_recovered_edges_single_node_max.load();
            }
            return std::chrono::duration<double, std::micro>(t2 - t1).count();
        };

        auto summarize = [&](const std::vector<double>& values) -> LatencySummary {
            LatencySummary summary;
            if (values.empty()) return summary;
            std::vector<double> sorted = values;
            std::sort(sorted.begin(), sorted.end());
            for (double value : sorted) summary.avg += value;
            summary.avg /= sorted.size();
            summary.p50 = sorted[sorted.size() * 50 / 100];
            summary.p95 = sorted[sorted.size() * 95 / 100];
            summary.p99 = sorted[sorted.size() * 99 / 100];
            return summary;
        };

        auto print_bucket = [&](const char* label, const std::vector<CompareRow>& rows,
                                size_t min_recovered, size_t min_single) {
            std::vector<double> on_values;
            std::vector<double> off_values;
            size_t recovered_sum = 0;
            size_t useful_sum = 0;
            size_t undo_sum = 0;
            size_t max_single_seen = 0;
            for (const auto& row : rows) {
                if (row.recovered_total < min_recovered) continue;
                if (row.max_single < min_single) continue;
                on_values.push_back(row.on_us);
                off_values.push_back(row.off_us);
                recovered_sum += row.recovered_total;
                useful_sum += row.useful_total;
                undo_sum += row.undo_lookups;
                if (row.max_single > max_single_seen) {
                    max_single_seen = row.max_single;
                }
            }
            if (on_values.empty()) {
                std::cout << "  " << label << ": no matching queries" << std::endl;
                return;
            }
            LatencySummary on_summary = summarize(on_values);
            LatencySummary off_summary = summarize(off_values);
            double avg_saved = on_summary.avg - off_summary.avg;
            double p95_saved = on_summary.p95 - off_summary.p95;
            double p99_saved = on_summary.p99 - off_summary.p99;
            double avg_saved_pct = on_summary.avg > 0 ? avg_saved / on_summary.avg * 100.0 : 0;
            double p99_saved_pct = on_summary.p99 > 0 ? p99_saved / on_summary.p99 * 100.0 : 0;
            double avg_undo = static_cast<double>(undo_sum) / on_values.size();
            double avg_recovered = static_cast<double>(recovered_sum) / on_values.size();
            double avg_useful = static_cast<double>(useful_sum) / on_values.size();
            std::cout << "  " << label
                      << ": n=" << on_values.size()
                      << ", avg_on=" << on_summary.avg << "us"
                      << ", avg_off=" << off_summary.avg << "us"
                      << ", avg_saved=" << avg_saved << "us (" << avg_saved_pct << "%)"
                      << ", p95_on=" << on_summary.p95 << "us"
                      << ", p95_off=" << off_summary.p95 << "us"
                      << ", p95_saved=" << p95_saved << "us"
                      << ", p99_on=" << on_summary.p99 << "us"
                      << ", p99_off=" << off_summary.p99 << "us"
                      << ", p99_saved=" << p99_saved << "us (" << p99_saved_pct << "%)"
                      << ", avg_undo=" << avg_undo
                      << ", avg_recovered=" << avg_recovered
                      << ", avg_useful=" << avg_useful
                      << ", max_single_seen=" << max_single_seen
                      << std::endl;
        };

        std::cout << "\n=== Exact Recovery On/Off Comparison ===" << std::endl;
        std::vector<int> compare_versions_back = {200, 400, 600, 800};
        for (int vb : compare_versions_back) {
            if (vb >= num_batches) continue;

            size_t visible_batch_id = (num_batches - vb) * batch_size - 1;
            if (visible_batch_id >= (size_t)max_elements) {
                visible_batch_id = max_elements - 1;
            }
            size_t view_limit = visible_batch_id + 1;
            std::vector<CompareRow> rows(num_queries);
            for (int q = 0; q < num_queries; q++) {
                CompareRow row;
                const void* query = query_data + q * dim;
                if ((q & 1) == 0) {
                    row.on_us = run_historical_query(query, view_limit, true, &row, true);
                    row.off_us = run_historical_query(query, view_limit, false, nullptr, false);
                } else {
                    row.off_us = run_historical_query(query, view_limit, false, nullptr, false);
                    row.on_us = run_historical_query(query, view_limit, true, &row, true);
                }
                rows[q] = row;
            }
            alg_hnsw->setEnableUndoRecovery(true);

            const CompareRow* max_log_row = nullptr;
            const CompareRow* max_saved_row = nullptr;
            for (const auto& row : rows) {
                if (max_log_row == nullptr || row.max_single > max_log_row->max_single) {
                    max_log_row = &row;
                }
                if (max_saved_row == nullptr ||
                    (row.on_us - row.off_us) > (max_saved_row->on_us - max_saved_row->off_us)) {
                    max_saved_row = &row;
                }
            }

            std::cout << "versions_back=" << vb
                      << ", visible_batch=" << visible_batch_id
                      << ", compare_queries=" << num_queries << std::endl;
            print_bucket("all queries", rows, 0, 0);
            print_bucket("recovered>0", rows, 1, 0);
            print_bucket("max_single>=8", rows, 0, 8);
            print_bucket("max_single>=16", rows, 0, 16);
            print_bucket("max_single>=32", rows, 0, 32);
            print_bucket("max_single>=64", rows, 0, 64);
            if (max_log_row != nullptr) {
                std::cout << "  max_log_query: max_single=" << max_log_row->max_single
                          << ", recovered_total=" << max_log_row->recovered_total
                          << ", undo_lookups=" << max_log_row->undo_lookups
                          << ", useful_total=" << max_log_row->useful_total
                          << ", on=" << max_log_row->on_us << "us"
                          << ", off=" << max_log_row->off_us << "us"
                          << ", saved=" << (max_log_row->on_us - max_log_row->off_us) << "us"
                          << std::endl;
            }
            if (max_saved_row != nullptr) {
                std::cout << "  max_saved_query: max_single=" << max_saved_row->max_single
                          << ", recovered_total=" << max_saved_row->recovered_total
                          << ", undo_lookups=" << max_saved_row->undo_lookups
                          << ", useful_total=" << max_saved_row->useful_total
                          << ", on=" << max_saved_row->on_us << "us"
                          << ", off=" << max_saved_row->off_us << "us"
                          << ", saved=" << (max_saved_row->on_us - max_saved_row->off_us) << "us"
                          << std::endl;
            }
        }
        alg_hnsw->setEnableUndoRecovery(true);
    }

    // === Compact Benchmark ===
    {
        std::cout << "\n=== Compact Benchmark ===" << std::endl;
        std::cout << "Elements: " << alg_hnsw->cur_element_count.load() << std::endl;

        auto t0 = std::chrono::high_resolution_clock::now();
        size_t compacted = alg_hnsw->compact();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Compact: " << compacted << " nodes in " << ms << " ms" << std::endl;

        auto t2 = std::chrono::high_resolution_clock::now();
        size_t compacted2 = alg_hnsw->compact();
        auto t3 = std::chrono::high_resolution_clock::now();
        double ms2 = std::chrono::duration<double, std::milli>(t3 - t2).count();
        std::cout << "Compact (no-op): " << compacted2 << " nodes in " << ms2 << " ms" << std::endl;
    }

    delete[] query_data;
    delete[] data;
    delete alg_hnsw;

    return 0;
}
