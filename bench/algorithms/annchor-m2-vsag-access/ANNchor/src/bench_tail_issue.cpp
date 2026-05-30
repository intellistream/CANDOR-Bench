#include "hnswlib/hnswlib.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

using Clock = std::chrono::steady_clock;

static uint64_t nsSince(const Clock::time_point& start, const Clock::time_point& end) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

static uint64_t percentile(std::vector<uint64_t>& values, double pct) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const double rank = pct * static_cast<double>(values.size() - 1);
    return values[static_cast<size_t>(std::llround(rank))];
}

struct SearchSample {
    uint64_t ns{0};
    uint8_t phase_mask{0};
};

static uint64_t countMaskAtOrAbove(const std::vector<SearchSample>& samples,
                                   uint64_t threshold, uint8_t mask) {
    uint64_t count = 0;
    for (const auto& sample : samples) {
        if (sample.ns >= threshold && (sample.phase_mask & mask) != 0) {
            ++count;
        }
    }
    return count;
}

static uint64_t countAtOrAbove(const std::vector<SearchSample>& samples,
                               uint64_t threshold) {
    uint64_t count = 0;
    for (const auto& sample : samples) {
        if (sample.ns >= threshold) {
            ++count;
        }
    }
    return count;
}

static std::vector<int> parseCpuList(const char* value) {
    std::vector<int> cpus;
    if (value == nullptr || *value == '\0') return cpus;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        const auto dash = token.find('-');
        if (dash == std::string::npos) {
            cpus.push_back(std::stoi(token));
            continue;
        }
        int first = std::stoi(token.substr(0, dash));
        int last = std::stoi(token.substr(dash + 1));
        if (last < first) std::swap(first, last);
        for (int cpu = first; cpu <= last; ++cpu) cpus.push_back(cpu);
    }
    return cpus;
}

static bool pinCurrentThreadToCpu(int cpu) {
#ifdef __linux__
    if (cpu < 0) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)cpu;
    return false;
#endif
}

int main(int argc, char** argv) {
    int initial = argc > 1 ? std::stoi(argv[1]) : 20000;
    int inserts = argc > 2 ? std::stoi(argv[2]) : 5000;
    int dim = argc > 3 ? std::stoi(argv[3]) : 128;
    int search_threads = argc > 4 ? std::stoi(argv[4]) : 4;
    int insert_threads = argc > 5 ? std::stoi(argv[5]) : 1;
    int duration_ms = argc > 6 ? std::stoi(argv[6]) : 3000;
    int ef = argc > 7 ? std::stoi(argv[7]) : 64;

    // controlled root-cause knobs:
    //  ANNCHOR_TI_VIEW_LIMIT pins the committed/searchable set, so foreground search
    //  WORK is identical whether or not a writer runs (only concurrent mutation of the
    //  same nodes differs) -> any p99 delta is pure interference, not extra search work.
    //  ANNCHOR_TI_USE_NODE_LOCK toggles reader per-node shared locks (lock-wait vs coherence).
    long pin_view = -1;
    if (const char* e = std::getenv("ANNCHOR_TI_VIEW_LIMIT")) pin_view = std::atol(e);
    bool use_node_lock = true;
    if (const char* e = std::getenv("ANNCHOR_TI_USE_NODE_LOCK")) use_node_lock = std::atoi(e) != 0;
    // ANNCHOR_TI_MVCC=0 + use_node_lock=0 approximates plain hnswlib (lockless
    // reads, no snapshot) -> aligns this rig with the C3 "plain HNSW" setup.
    int ti_mvcc = -1;
    if (const char* e = std::getenv("ANNCHOR_TI_MVCC")) ti_mvcc = std::atoi(e);

    const int total = initial + inserts;
    std::vector<float> data(static_cast<size_t>(total) * dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : data) x = dist(rng);

    annchor::L2Space space(dim);
    annchor::HierarchicalNSW<float> index(&space, total, 32, 400, 100, false, use_node_lock,
                                          search_threads + insert_threads);
    index.setEf(static_cast<size_t>(ef));
    if (ti_mvcc >= 0) index.setEnableMvcc(ti_mvcc != 0);

    for (int i = 0; i < initial; ++i) {
        index.addPoint(data.data() + static_cast<size_t>(i) * dim, i);
    }

    // Collision-localization layout dump (stderr, parsed by /tmp/cl_bucket.py).
    // Capacity == total at construction => data_level0_memory_ never reallocs
    // during the measured phase, so this base address is valid for the whole run.
    std::cerr << "LAYOUT base=" << static_cast<const void*>(index.data_level0_memory_)
              << " size_per_elem=" << index.size_data_per_element_
              << " links_l0=" << index.size_links_level0_
              << " data_size=" << index.data_size_
              << " label_off=" << index.label_offset_
              << " offset_data=" << index.offsetData_
              << " maxM0=" << index.maxM0_
              << " max_elem=" << index.max_elements_
              << " lls=" << sizeof(unsigned int) << std::endl;

    std::atomic<bool> start{false};
    std::atomic<bool> stop{false};
    std::atomic<int> next_insert{initial};
    std::atomic<uint64_t> inserted{0};
    std::vector<std::vector<SearchSample>> samples(search_threads);
    // per-thread search-work accumulators (rigor: verify foreground search WORK is
    // identical writer-off vs writer-on, and measure reader lock-wait directly).
    struct WorkAcc { uint64_t dist=0,l0_edges=0,l0_exp=0,l0_lock_ns=0,snap_ns=0,n=0; };
    std::vector<WorkAcc> work(search_threads);
    std::vector<std::thread> threads;
    const std::vector<int> search_cpus =
        parseCpuList(std::getenv("ANNCHOR_TAIL_ISSUE_SEARCH_CPUS"));
    const std::vector<int> insert_cpus =
        parseCpuList(std::getenv("ANNCHOR_TAIL_ISSUE_INSERT_CPUS"));
    std::atomic<uint64_t> pin_success{0};
    std::atomic<uint64_t> pin_fail{0};

    for (int t = 0; t < search_threads; ++t) {
        threads.emplace_back([&, t]() {
            if (!search_cpus.empty()) {
                const int cpu = search_cpus[static_cast<size_t>(t) % search_cpus.size()];
                if (pinCurrentThreadToCpu(cpu)) {
                    pin_success.fetch_add(1, std::memory_order_relaxed);
                } else {
                    pin_fail.fetch_add(1, std::memory_order_relaxed);
                }
            }
            std::mt19937 local_rng(1000 + t);
            while (!start.load(std::memory_order_acquire)) {
            }
            while (!stop.load(std::memory_order_acquire)) {
                int visible = (pin_view > 0)
                                  ? static_cast<int>(pin_view)
                                  : next_insert.load(std::memory_order_acquire);
                if (visible <= 0) visible = initial;
                int q = static_cast<int>(local_rng() % visible);
                const size_t vlim = (pin_view > 0)
                                        ? static_cast<size_t>(pin_view)
                                        : std::numeric_limits<size_t>::max();
                auto begin = Clock::now();
                annchor::SearchWorkStats stats;
                auto result = index.searchKnnWithStats(
                    data.data() + static_cast<size_t>(q) * dim, 10, nullptr,
                    vlim, &stats);
                auto end = Clock::now();
                (void)result;
                uint8_t phase_mask = 0;
                if (stats.phase_existing_update_hit) phase_mask |= 1u << 0;
                if (stats.phase_link_critical_hit) phase_mask |= 1u << 1;
                if (stats.phase_load_scan_hit) phase_mask |= 1u << 2;
                if (stats.phase_append_hit) phase_mask |= 1u << 3;
                if (stats.phase_prune_hit) phase_mask |= 1u << 4;
                if (stats.phase_undo_record_hit) phase_mask |= 1u << 5;
                if (stats.phase_rewrite_hit) phase_mask |= 1u << 6;
                samples[t].push_back({nsSince(begin, end), phase_mask});
                work[t].dist += stats.distance_computations;
                work[t].l0_edges += stats.level0_edges_scanned;
                work[t].l0_exp += stats.level0_expansions;
                work[t].l0_lock_ns += stats.level0_lock_wait_ns;
                work[t].snap_ns += stats.snapshot_guard_ns;
                work[t].n += 1;
            }
        });
    }

    for (int t = 0; t < insert_threads; ++t) {
        threads.emplace_back([&, t]() {
            if (!insert_cpus.empty()) {
                const int cpu = insert_cpus[static_cast<size_t>(t) % insert_cpus.size()];
                if (pinCurrentThreadToCpu(cpu)) {
                    pin_success.fetch_add(1, std::memory_order_relaxed);
                } else {
                    pin_fail.fetch_add(1, std::memory_order_relaxed);
                }
            }
            while (!start.load(std::memory_order_acquire)) {
            }
            while (!stop.load(std::memory_order_acquire)) {
                int id = next_insert.fetch_add(1, std::memory_order_acq_rel);
                if (id >= total) {
                    stop.store(true, std::memory_order_release);
                    break;
                }
                index.addPoint(data.data() + static_cast<size_t>(id) * dim, id);
                inserted.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    const auto run_start = Clock::now();
    start.store(true, std::memory_order_release);
    while (nsSince(run_start, Clock::now()) <
           static_cast<uint64_t>(duration_ms) * 1000000ull) {
        if (stop.load(std::memory_order_acquire)) break;
    }
    stop.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();
    const auto run_end = Clock::now();

    std::vector<SearchSample> all_samples;
    for (auto& v : samples) {
        all_samples.insert(all_samples.end(), v.begin(), v.end());
    }
    std::vector<uint64_t> all;
    all.reserve(all_samples.size());
    for (const auto& sample : all_samples) {
        all.push_back(sample.ns);
    }
    double seconds = static_cast<double>(nsSince(run_start, run_end)) / 1e9;
    uint64_t ops = static_cast<uint64_t>(all.size());
    uint64_t ins = inserted.load(std::memory_order_relaxed);

    std::cout << "search_ops=" << ops << "\n";
    std::cout << "insert_ops=" << ins << "\n";
    std::cout << "pin_success=" << pin_success.load(std::memory_order_relaxed)
              << "\n";
    std::cout << "pin_fail=" << pin_fail.load(std::memory_order_relaxed) << "\n";
    std::cout << "search_qps=" << (seconds > 0 ? static_cast<double>(ops) / seconds : 0.0)
              << "\n";
    std::cout << "insert_qps=" << (seconds > 0 ? static_cast<double>(ins) / seconds : 0.0)
              << "\n";
    {
        WorkAcc W;
        for (const auto& w : work) {
            W.dist += w.dist; W.l0_edges += w.l0_edges; W.l0_exp += w.l0_exp;
            W.l0_lock_ns += w.l0_lock_ns; W.snap_ns += w.snap_ns; W.n += w.n;
        }
        const double n = W.n ? static_cast<double>(W.n) : 1.0;
        std::cout << "dist_comps_per_query=" << static_cast<double>(W.dist) / n << "\n";
        std::cout << "l0_edges_per_query=" << static_cast<double>(W.l0_edges) / n << "\n";
        std::cout << "l0_expansions_per_query=" << static_cast<double>(W.l0_exp) / n << "\n";
        std::cout << "l0_lock_wait_ns_per_query=" << static_cast<double>(W.l0_lock_ns) / n << "\n";
        std::cout << "snapshot_guard_ns_per_query=" << static_cast<double>(W.snap_ns) / n << "\n";
    }
    // reverse-link repair op breakdown (which WRITE dirties reader lines)
    std::cout << "rev_appends=" << index.getExistingNeighborAppends() << "\n";
    std::cout << "rev_prunes=" << index.getExistingNeighborPrunes() << "\n";
    std::cout << "rev_edges_written=" << index.getExistingNeighborEdgesWritten() << "\n";
    std::cout << "rev_edges_pruned=" << index.getExistingNeighborEdgesPruned() << "\n";
    const uint64_t p50 = percentile(all, 0.50);
    const uint64_t p95 = percentile(all, 0.95);
    const uint64_t p99 = percentile(all, 0.99);
    const uint64_t p999 = percentile(all, 0.999);
    std::cout << "p50_ns=" << p50 << "\n";
    std::cout << "p95_ns=" << p95 << "\n";
    std::cout << "p99_ns=" << p99 << "\n";
    std::cout << "p999_ns=" << p999 << "\n";
    std::cout << "tail_reader_urgent_ops=" << index.getTailIssueReaderUrgentOps() << "\n";
    std::cout << "tail_reader_urgent_prefetch_lines="
              << index.getTailIssueReaderUrgentPrefetchLines() << "\n";
    std::cout << "tail_reader_window_prefetches="
              << index.getTailIssueReaderWindowPrefetches() << "\n";
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
    std::cout << "tail_writer_budget_granted_lines="
              << index.getTailIssueWriterBudgetGrantedLines() << "\n";
    std::cout << "tail_writer_dist_budget_granted="
              << index.getTailIssueWriterDistBudgetGranted() << "\n";
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
    std::cout << "tail_writer_lead_skip_granted="
              << index.getTailIssueWriterLeadSkipGranted() << "\n";
    std::cout << "tail_writer_lead_prefetch_skipped="
              << index.getTailIssueWriterLeadPrefetchSkipped() << "\n";
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
    std::cout << "tail_writer_demote_granted="
              << index.getTailIssueWriterDemoteGranted() << "\n";
    std::cout << "tail_writer_demote_used="
              << index.getTailIssueWriterDemoteUsed() << "\n";
    std::cout << "tail_writer_demote_lines="
              << index.getTailIssueWriterDemoteLines() << "\n";
    std::cout << "tail_writer_offlock_prune_granted="
              << index.getTailIssueWriterOfflockPruneGranted() << "\n";
    std::cout << "tail_writer_offlock_prune_attempts="
              << index.getTailIssueWriterOfflockPruneAttempts() << "\n";
    std::cout << "tail_writer_offlock_prune_success="
              << index.getTailIssueWriterOfflockPruneSuccess() << "\n";
    std::cout << "tail_writer_offlock_prune_validate_fail="
              << index.getTailIssueWriterOfflockPruneValidateFail() << "\n";
    std::cout << "tail_writer_offlock_prune_skipped="
              << index.getTailIssueWriterOfflockPruneSkipped() << "\n";
    std::cout << "phase_overlap_samples="
              << index.getSearchPhaseOverlapSamples() << "\n";
    std::cout << "phase_existing_update_samples="
              << index.getSearchPhaseExistingUpdateSamples() << "\n";
    std::cout << "phase_existing_update_queries="
              << index.getSearchPhaseExistingUpdateQueries() << "\n";
    std::cout << "phase_link_critical_samples="
              << index.getSearchPhaseLinkCriticalSamples() << "\n";
    std::cout << "phase_link_critical_queries="
              << index.getSearchPhaseLinkCriticalQueries() << "\n";
    std::cout << "phase_load_scan_samples="
              << index.getSearchPhaseLoadScanSamples() << "\n";
    std::cout << "phase_load_scan_queries="
              << index.getSearchPhaseLoadScanQueries() << "\n";
    std::cout << "phase_append_samples="
              << index.getSearchPhaseAppendSamples() << "\n";
    std::cout << "phase_append_queries="
              << index.getSearchPhaseAppendQueries() << "\n";
    std::cout << "phase_prune_samples="
              << index.getSearchPhasePruneSamples() << "\n";
    std::cout << "phase_prune_queries="
              << index.getSearchPhasePruneQueries() << "\n";
    std::cout << "phase_undo_record_samples="
              << index.getSearchPhaseUndoRecordSamples() << "\n";
    std::cout << "phase_undo_record_queries="
              << index.getSearchPhaseUndoRecordQueries() << "\n";
    std::cout << "phase_rewrite_samples="
              << index.getSearchPhaseRewriteSamples() << "\n";
    std::cout << "phase_rewrite_queries="
              << index.getSearchPhaseRewriteQueries() << "\n";
    const uint64_t p99_count = countAtOrAbove(all_samples, p99);
    const uint64_t p999_count = countAtOrAbove(all_samples, p999);
    std::cout << "p99_window_count=" << p99_count << "\n";
    std::cout << "p99_phase_existing_update=" << countMaskAtOrAbove(all_samples, p99, 1u << 0) << "\n";
    std::cout << "p99_phase_link_critical=" << countMaskAtOrAbove(all_samples, p99, 1u << 1) << "\n";
    std::cout << "p99_phase_load_scan=" << countMaskAtOrAbove(all_samples, p99, 1u << 2) << "\n";
    std::cout << "p99_phase_append=" << countMaskAtOrAbove(all_samples, p99, 1u << 3) << "\n";
    std::cout << "p99_phase_prune=" << countMaskAtOrAbove(all_samples, p99, 1u << 4) << "\n";
    std::cout << "p99_phase_undo_record=" << countMaskAtOrAbove(all_samples, p99, 1u << 5) << "\n";
    std::cout << "p99_phase_rewrite=" << countMaskAtOrAbove(all_samples, p99, 1u << 6) << "\n";
    std::cout << "p999_window_count=" << p999_count << "\n";
    std::cout << "p999_phase_existing_update=" << countMaskAtOrAbove(all_samples, p999, 1u << 0) << "\n";
    std::cout << "p999_phase_link_critical=" << countMaskAtOrAbove(all_samples, p999, 1u << 1) << "\n";
    std::cout << "p999_phase_load_scan=" << countMaskAtOrAbove(all_samples, p999, 1u << 2) << "\n";
    std::cout << "p999_phase_append=" << countMaskAtOrAbove(all_samples, p999, 1u << 3) << "\n";
    std::cout << "p999_phase_prune=" << countMaskAtOrAbove(all_samples, p999, 1u << 4) << "\n";
    std::cout << "p999_phase_undo_record=" << countMaskAtOrAbove(all_samples, p999, 1u << 5) << "\n";
    std::cout << "p999_phase_rewrite=" << countMaskAtOrAbove(all_samples, p999, 1u << 6) << "\n";
    std::cout << "tail_writer_hits=" << index.getTailIssueWriterTailHits() << "\n";
    std::cout << "tail_writer_hint_lines=" << index.getTailIssueWriterHintLines() << "\n";
    std::cout << "tail_writer_suppressed_lines=" << index.getTailIssueWriterSuppressedLines()
              << "\n";
    return 0;
}
