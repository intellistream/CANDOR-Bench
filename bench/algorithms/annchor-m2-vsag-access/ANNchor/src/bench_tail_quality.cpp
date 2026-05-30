#include "hnswlib/hnswlib.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

using Clock = std::chrono::steady_clock;

static uint64_t nsSince(const Clock::time_point& start,
                        const Clock::time_point& end) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

static float l2(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

int main(int argc, char** argv) {
    const int initial = argc > 1 ? std::stoi(argv[1]) : 10000;
    const int dim = argc > 2 ? std::stoi(argv[2]) : 128;
    const int queries = argc > 3 ? std::stoi(argv[3]) : 500;
    const int ef = argc > 4 ? std::stoi(argv[4]) : 64;
    const int k = argc > 5 ? std::stoi(argv[5]) : 10;

    std::vector<float> data(static_cast<size_t>(initial) * dim);
    std::vector<float> query_data(static_cast<size_t>(queries) * dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : data) x = dist(rng);
    for (auto& x : query_data) x = dist(rng);

    annchor::L2Space space(dim);
    annchor::HierarchicalNSW<float> index(&space, initial, 32, 400, 100,
                                          false, true, 1);
    index.setEf(static_cast<size_t>(ef));
    for (int i = 0; i < initial; ++i) {
        index.addPoint(data.data() + static_cast<size_t>(i) * dim, i);
    }

    uint64_t correct = 0;
    uint64_t total_latency_ns = 0;
    std::vector<std::pair<float, int>> exact(static_cast<size_t>(initial));
    const int exact_k = std::min(k, initial);

    for (int q = 0; q < queries; ++q) {
        const float* query = query_data.data() + static_cast<size_t>(q) * dim;
        for (int i = 0; i < initial; ++i) {
            exact[static_cast<size_t>(i)] = {
                l2(query, data.data() + static_cast<size_t>(i) * dim, dim), i};
        }
        std::nth_element(exact.begin(), exact.begin() + exact_k, exact.end());
        std::unordered_set<annchor::labeltype> truth;
        truth.reserve(static_cast<size_t>(exact_k) * 2);
        for (int i = 0; i < exact_k; ++i) {
            truth.insert(static_cast<annchor::labeltype>(exact[static_cast<size_t>(i)].second));
        }

        const auto begin = Clock::now();
        auto result = index.searchKnn(query, static_cast<size_t>(k));
        const auto end = Clock::now();
        total_latency_ns += nsSince(begin, end);

        while (!result.empty()) {
            if (truth.find(result.top().second) != truth.end()) {
                ++correct;
            }
            result.pop();
        }
    }

    const double denom = static_cast<double>(queries) * static_cast<double>(exact_k);
    std::cout << "queries=" << queries << "\n";
    std::cout << "initial=" << initial << "\n";
    std::cout << "dim=" << dim << "\n";
    std::cout << "ef=" << ef << "\n";
    std::cout << "k=" << k << "\n";
    std::cout << "recall_at_k=" << (denom > 0 ? static_cast<double>(correct) / denom : 0.0)
              << "\n";
    std::cout << "avg_search_ns="
              << (queries > 0 ? static_cast<double>(total_latency_ns) / queries : 0.0)
              << "\n";
    std::cout << "tail_reader_urgent_ops=" << index.getTailIssueReaderUrgentOps() << "\n";
    std::cout << "tail_reader_beam_clamps=" << index.getTailIssueReaderBeamClamps()
              << "\n";
    std::cout << "tail_reader_beam_trimmed=" << index.getTailIssueReaderBeamTrimmed()
              << "\n";
    std::cout << "tail_reader_beam_clamps2=" << index.getTailIssueReaderBeamClamps2()
              << "\n";
    std::cout << "tail_reader_beam_trimmed2=" << index.getTailIssueReaderBeamTrimmed2()
              << "\n";
    std::cout << "tail_reader_expansion_stops="
              << index.getTailIssueReaderExpansionStops() << "\n";
    std::cout << "tail_reader_margin_stops="
              << index.getTailIssueReaderMarginStops() << "\n";
    std::cout << "tail_reader_confidence_checks="
              << index.getTailIssueReaderConfidenceChecks() << "\n";
    std::cout << "tail_reader_confidence_grants="
              << index.getTailIssueReaderConfidenceGrants() << "\n";
    std::cout << "tail_reader_confidence_rejects="
              << index.getTailIssueReaderConfidenceRejects() << "\n";
    std::cout << "tail_reader_batch_prefetches="
              << index.getTailIssueReaderBatchPrefetches() << "\n";
    std::cout << "tail_reader_lookahead_deferred="
              << index.getTailIssueReaderLookaheadDeferred() << "\n";
    std::cout << "tail_reader_lookahead_skipped="
              << index.getTailIssueReaderLookaheadSkipped() << "\n";
    std::cout << "tail_reader_frontier_checks="
              << index.getTailIssueReaderFrontierChecks() << "\n";
    std::cout << "tail_reader_frontier_eligible="
              << index.getTailIssueReaderFrontierEligible() << "\n";
    std::cout << "tail_reader_frontier_stops="
              << index.getTailIssueReaderFrontierStops() << "\n";
    std::cout << "tail_reader_frontier_continue_checks="
              << index.getTailIssueReaderFrontierContinueChecks() << "\n";
    std::cout << "tail_reader_frontier_continue_grants="
              << index.getTailIssueReaderFrontierContinueGrants() << "\n";
    std::cout << "tail_reader_retry_checks="
              << index.getTailIssueReaderRetryChecks() << "\n";
    std::cout << "tail_reader_retry_runs="
              << index.getTailIssueReaderRetryRuns() << "\n";
    std::cout << "tail_reader_retry_confident_skips="
              << index.getTailIssueReaderRetryConfidentSkips() << "\n";
    std::cout << "tail_reader_continue_checks="
              << index.getTailIssueReaderContinueChecks() << "\n";
    std::cout << "tail_reader_continue_runs="
              << index.getTailIssueReaderContinueRuns() << "\n";
    std::cout << "tail_reader_continue_confident_stops="
              << index.getTailIssueReaderContinueConfidentStops() << "\n";
    std::cout << "tail_reader_bounded_distance_ops="
              << index.getTailIssueReaderBoundedDistanceOps() << "\n";
    std::cout << "tail_reader_bounded_distance_early="
              << index.getTailIssueReaderBoundedDistanceEarly() << "\n";
    std::cout << "tail_reader_prefix_guard_ops="
              << index.getTailIssueReaderPrefixGuardOps() << "\n";
    std::cout << "tail_reader_prefix_guard_early="
              << index.getTailIssueReaderPrefixGuardEarly() << "\n";
    std::cout << "tail_reader_prefix_guard_full="
              << index.getTailIssueReaderPrefixGuardFull() << "\n";
    std::cout << "tail_reader_prefix_estimate_ops="
              << index.getTailIssueReaderPrefixEstimateOps() << "\n";
    std::cout << "tail_reader_prefix_estimate_skips="
              << index.getTailIssueReaderPrefixEstimateSkips() << "\n";
    std::cout << "tail_reader_prefix_estimate_full="
              << index.getTailIssueReaderPrefixEstimateFull() << "\n";
    std::cout << "tail_reader_current_prefetch_lines="
              << index.getTailIssueReaderCurrentPrefetchLines() << "\n";
    std::cout << "tail_reader_refine_runs="
              << index.getTailIssueReaderRefineRuns() << "\n";
    std::cout << "tail_reader_refine_edges="
              << index.getTailIssueReaderRefineEdges() << "\n";
    std::cout << "tail_reader_refine_inserted="
              << index.getTailIssueReaderRefineInserted() << "\n";
    std::cout << "tail_writer_ef_budget_granted="
              << index.getTailIssueWriterEfBudgetGranted() << "\n";
    std::cout << "tail_writer_ef_limited_searches="
              << index.getTailIssueWriterEfLimitedSearches() << "\n";
    std::cout << "tail_writer_ef_trimmed="
              << index.getTailIssueWriterEfTrimmed() << "\n";
    std::cout << "tail_writer_repair_defer_granted="
              << index.getTailIssueWriterRepairDeferGranted() << "\n";
    std::cout << "tail_writer_repair_deferred_points="
              << index.getTailIssueWriterRepairDeferredPoints() << "\n";
    std::cout << "tail_writer_repair_deferred_edges="
              << index.getTailIssueWriterRepairDeferredEdges() << "\n";
    std::cout << "tail_writer_repair_defer_dropped="
              << index.getTailIssueWriterRepairDeferDropped() << "\n";
    std::cout << "tail_writer_repair_drained="
              << index.getTailIssueWriterRepairDrained() << "\n";
    std::cout << "tail_writer_repair_drain_runs="
              << index.getTailIssueWriterRepairDrainRuns() << "\n";
    std::cout << "tail_writer_repair_queue_max="
              << index.getTailIssueWriterRepairQueueMax() << "\n";
    std::cout << "tail_writer_lowvalue_skip_granted="
              << index.getTailIssueWriterLowvalueSkipGranted() << "\n";
    std::cout << "tail_writer_lowvalue_skip_checks="
              << index.getTailIssueWriterLowvalueSkipChecks() << "\n";
    std::cout << "tail_writer_lowvalue_skipped="
              << index.getTailIssueWriterLowvalueSkipped() << "\n";
    std::cout << "tail_writer_level0_granted="
              << index.getTailIssueWriterLevel0Granted() << "\n";
    std::cout << "tail_writer_level0_forced="
              << index.getTailIssueWriterLevel0Forced() << "\n";
    std::cout << "tail_writer_dist_shaped_ops="
              << index.getTailIssueWriterDistShapedOps() << "\n";
    std::cout << "tail_writer_dist_bounded_early="
              << index.getTailIssueWriterDistBoundedEarly() << "\n";
    std::cout << "tail_writer_dist_prefix_full="
              << index.getTailIssueWriterDistPrefixFull() << "\n";
    std::cout << "tail_writer_dist_prefix_estimate_skips="
              << index.getTailIssueWriterDistPrefixEstimateSkips() << "\n";
    std::cout << "tail_writer_dist_prefix_estimate_full="
              << index.getTailIssueWriterDistPrefixEstimateFull() << "\n";
    std::cout << "tail_writer_dist_admission_skips="
              << index.getTailIssueWriterDistAdmissionSkips() << "\n";
    std::cout << "tail_writer_lead_hint_granted="
              << index.getTailIssueWriterLeadHintGranted() << "\n";
    std::cout << "tail_writer_lead_prefetch_hinted="
              << index.getTailIssueWriterLeadPrefetchHinted() << "\n";
    std::cout << "tail_writer_frontier_granted="
              << index.getTailIssueWriterFrontierGranted() << "\n";
    std::cout << "tail_writer_frontier_checks="
              << index.getTailIssueWriterFrontierChecks() << "\n";
    std::cout << "tail_writer_frontier_skipped="
              << index.getTailIssueWriterFrontierSkipped() << "\n";
    std::cout << "tail_writer_shared_scan_granted="
              << index.getTailIssueWriterSharedScanGranted() << "\n";
    std::cout << "tail_writer_shared_scan_used="
              << index.getTailIssueWriterSharedScanUsed() << "\n";
    return 0;
}
