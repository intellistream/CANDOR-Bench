#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>

#ifdef __x86_64__
#include <immintrin.h>
#endif

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <tbb/concurrent_queue.h>

#include <boost/context/fiber.hpp>
#include <boost/context/fixedsize_stack.hpp>

#include "../index.hpp"
#include "../index_cgo.hpp"
#include "../annchor-m1/annchor_m1.hpp"

// Cooperative preemption variant.
//
// Each addPoint runs inside a boost::context::fiber within a TBB parallel_for task.
// At convergence-gated safe points, writer fiber yields.
// The TBB worker thread services pending search queries from a shared queue,
// then resumes the writer fiber. TBB handles thread scheduling (work-stealing).

struct alignas(64) SearchRequest {
    const void* query{nullptr};
    size_t k{0};
    size_t visible_ts{0};
    uint32_t* result_buf{nullptr};
    size_t result_count{0};
    std::atomic<bool> done{false};  // CAS coordination: writer or search claims ownership
    uint64_t epoch{0};             // batch epoch: writer skips stale requests
};

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class ANNchorPreempt : public IndexBase<T, TagT, LabelT> {
   public:
    ANNchorPreempt(size_t max_elements, size_t dim, size_t num_threads, size_t M,
                   size_t ef_construction, bool use_node_lock_in_search = true,
                   MetricType metric = METRIC_L2)
        : dim_(dim),
          num_threads_(num_threads),
          M_(M),
          ef_c_(ef_construction),
          ef_s_(0),
          max_elements_(max_elements),
          use_node_lock_in_search_(use_node_lock_in_search),
          metric_(metric),
          arena_(num_threads) {
        if (metric_ == METRIC_IP || metric_ == METRIC_COSINE) {
            space_.reset(new annchor_m1::InnerProductSpace(dim));
        } else {
            space_.reset(new annchor_m1::L2Space(dim));
        }
        index_ = new annchor_m1::HierarchicalNSW<T>(
            space_.get(), max_elements, M, ef_construction, 100, false,
            use_node_lock_in_search, num_threads_);
        index_->setMidInsertServicePressureCallback([]() {
            typename annchor_m1::HierarchicalNSW<T>::MidInsertServicePressure pressure;
            const int inflight = runtime_inflight_search_queries_.load(std::memory_order_relaxed);
            const int queued = runtime_search_backlog_.load(std::memory_order_relaxed);
            pressure.search_backlog = std::max(inflight, queued);
            pressure.priority_searches = runtime_priority_searches_.load(std::memory_order_relaxed);
            return pressure;
        });

        starvation_alpha_ = read_env_double("ANN_PREEMPT_STARVATION_ALPHA", starvation_alpha_);
        max_service_per_yield_ = read_env_int("ANN_PREEMPT_MAX_SERVICE_PER_YIELD", max_service_per_yield_);
        fiber_stack_size_ = read_env_int("ANN_PREEMPT_FIBER_STACK_SIZE", fiber_stack_size_);
        search_budget_ = read_env_int("ANN_PREEMPT_SEARCH_BUDGET", search_budget_);
        drain_alpha_ = read_env_double("ANN_PREEMPT_DRAIN_ALPHA", starvation_alpha_);  // default = same as fiber alpha
        if (read_env_int("ANN_NON_PREFIX_VISIBILITY", 0) != 0) {
            index_->setEnableNonPrefixVisibility(true);
        }

        // ── Per-node yield callback ──
        // Called after each candidate-node expansion in searchBaseLayer,
        // right after the node lock is released.  No convergence gate —
        // yield decision is purely L-based (PreemptDB-style starvation level).
        // High-dim search is expensive (~3-4 ms on gist-960d), so after one
        // yield L shoots up and subsequent nodes auto-skip — self-adaptive.
        index_->setFineGrainYieldCallback([this]() {
            if (!enable_preempt_m2_.load(std::memory_order_relaxed)) return;

            // (0) No pending searches → skip (~2 ns atomic load)
            if (pending_search_count_.load(std::memory_order_relaxed) <= 0) return;

            writer_safe_point_checks_.fetch_add(1, std::memory_order_relaxed);

            // (1) Starvation level L = donated / (own + donated)
            //     Shared across fiber yield + drain via thread-local accumulators.
            auto now_ts = std::chrono::steady_clock::now();
            if (tl_last_ts_.time_since_epoch().count() != 0) {
                tl_time_own_ns_ += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now_ts - tl_last_ts_).count());
            }
            tl_last_ts_ = now_ts;

            uint64_t total_ns = tl_time_own_ns_ + tl_time_donated_ns_;
            if (total_ns > 0) {
                double L = static_cast<double>(tl_time_donated_ns_) / static_cast<double>(total_ns);
                if (L >= starvation_alpha_) {
                    fiber_starvation_refuses_.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
            }

            // (2) Yield: suspend fiber → service ONE search → resume
            fiber_yield_count_.fetch_add(1, std::memory_order_relaxed);
            auto yield_start = std::chrono::steady_clock::now();
            fiber_yield();
            auto yield_end = std::chrono::steady_clock::now();
            tl_time_donated_ns_ += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(yield_end - yield_start).count());
            tl_last_ts_ = yield_end;
        });
    }

    ~ANNchorPreempt() {
        delete index_;
    }

    void build(const T* data, const TagT* tags, size_t num_points) override {
        (void)batch_insert(data, tags, num_points);
    }

    int insert(const T* data, const TagT tag) override {
        if (metric_ == METRIC_COSINE) {
            std::vector<T> temp_vec(data, data + dim_);
            normalize_vector(temp_vec.data(), 1);
            index_->addPoint(temp_vec.data(), tag);
        } else {
            index_->addPoint(data, tag);
        }
        return 0;
    }

    int batch_insert(const T* batch_data, const TagT* batch_tags,
                     size_t num_points) override {
        if (num_points == 0) return 0;
        bool mvcc = index_->isMvccEnabled();

        // Record bench start for utilization tracking
        if (!bench_started_.exchange(true, std::memory_order_relaxed)) {
            bench_start_ts_ = std::chrono::steady_clock::now();
        }

        // Non-preempt path: identical to stable
        if (!enable_preempt_m2_) {
            arena_.execute([&] {
                tbb::parallel_for(size_t(0), num_points, [&](size_t i) {
                    auto t0 = std::chrono::steady_clock::now();
                    const T* point = get_point(batch_data, i);
                    auto id = index_->addPoint(
                        point, static_cast<annchor_m1::labeltype>(batch_tags[i]), -1, mvcc);
                    if (mvcc) {
                        index_->markReady(id);
                        index_->flushMidInsertDeferredReady(id);
                    }
                    auto t1 = std::chrono::steady_clock::now();
                    total_insert_work_ns_.fetch_add(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                        std::memory_order_relaxed);
                });
            });
            post_insert_mvcc_cleanup();
            return 0;
        }

        // ── M2: Fiber + Budget Drain ──
        // Fiber yield (mid-insert): convergence gate + starvation alpha
        // Budget drain (inter-insert): service min(B, pending) searches after each insert
        index_->setFineGrainYieldActive(true);

        std::vector<T> norm_buf;
        const T* data_ptr = batch_data;
        if (metric_ == METRIC_COSINE) {
            norm_buf.assign(batch_data, batch_data + num_points * dim_);
            normalize_vector(norm_buf.data(), num_points);
            data_ptr = norm_buf.data();
        }

        arena_.execute([&] {
            tbb::parallel_for(size_t(0), num_points, [&](size_t i) {
                auto t0 = std::chrono::steady_clock::now();
                const T* point = data_ptr + i * dim_;
                auto tag = static_cast<annchor_m1::labeltype>(batch_tags[i]);

                // Reset per-insert L ledger
                tl_time_own_ns_ = 0;
                tl_time_donated_ns_ = 0;
                tl_last_ts_ = t0;

                boost::context::fiber writer_fib(
                    std::allocator_arg,
                    boost::context::fixedsize_stack(fiber_stack_size_),
                    [&, point, tag](boost::context::fiber&& caller) -> boost::context::fiber {
                        tl_caller_ = &caller;

                        auto id = index_->addPoint(point, tag, -1, mvcc);
                        if (mvcc) {
                            index_->markReady(id);
                            index_->flushMidInsertDeferredReady(id);
                        }

                        tl_caller_ = nullptr;
                        return std::move(caller);
                    });

                // Scheduler loop: resume fiber, service searches when it yields
                writer_fib = std::move(writer_fib).resume();
                while (writer_fib) {
                    service_search_inline();
                    writer_fib = std::move(writer_fib).resume();
                }

                // Inter-insert drain: service up to search_budget_ pending
                // searches, sharing the same L ledger as fiber yields.
                if (search_budget_ > 0) {
                    int pending = pending_search_count_.load(std::memory_order_relaxed);
                    int to_service = std::min(search_budget_, pending);
                    int serviced = 0;
                    SearchRequest* req = nullptr;
                    while (serviced < to_service && search_queue_.try_pop(req)) {
                        // L check before each search (same ledger as fiber callback)
                        auto now_drain = std::chrono::steady_clock::now();
                        if (tl_last_ts_.time_since_epoch().count() != 0) {
                            tl_time_own_ns_ += static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(now_drain - tl_last_ts_).count());
                        }
                        tl_last_ts_ = now_drain;
                        uint64_t total_ns = tl_time_own_ns_ + tl_time_donated_ns_;
                        if (total_ns > 0) {
                            double L = static_cast<double>(tl_time_donated_ns_) / static_cast<double>(total_ns);
                            if (L >= drain_alpha_) break;
                        }

                        bool expected = false;
                        if (!req->done.compare_exchange_strong(
                                expected, true, std::memory_order_acq_rel)) {
                            continue;
                        }
                        if (!req->query || !req->result_buf) {
                            pending_search_count_.fetch_sub(1, std::memory_order_relaxed);
                            continue;
                        }
                        pending_search_count_.fetch_sub(1, std::memory_order_relaxed);
                        if (index_->isMvccEnabled()) index_->advanceWatermark();
                        auto drain_start = std::chrono::steady_clock::now();
                        execute_search_query(
                            static_cast<const T*>(req->query),
                            req->k, req->visible_ts, req->result_buf);
                        auto drain_end = std::chrono::steady_clock::now();
                        tl_time_donated_ns_ += static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(drain_end - drain_start).count());
                        tl_last_ts_ = drain_end;
                        fiber_searches_serviced_.fetch_add(1, std::memory_order_relaxed);
                        serviced++;
                    }
                }

                auto t1 = std::chrono::steady_clock::now();
                // This includes both insert work and search work done via fiber
                total_insert_work_ns_.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                    std::memory_order_relaxed);
            });
        });

        index_->setFineGrainYieldActive(false);
        post_insert_mvcc_cleanup();
        return 0;
    }

    void set_query_params(const QParams& params) override {
        ef_s_ = params.ef_search;
        index_->setEf(ef_s_);
    }

    int search(const T* query, size_t k,
               std::vector<TagT>& result_tags) override {
        runtime_active_search_calls_.fetch_add(1, std::memory_order_relaxed);
        runtime_inflight_search_queries_.fetch_add(1, std::memory_order_relaxed);
        std::priority_queue<std::pair<T, annchor_m1::labeltype>> result;
        if (metric_ == METRIC_COSINE) {
            std::vector<T> temp_query(query, query + dim_);
            normalize_vector(temp_query.data(), 1);
            result = index_->searchKnn(temp_query.data(), k);
        } else {
            result = index_->searchKnn(query, k);
        }
        while (!result.empty()) {
            result_tags.push_back(result.top().second);
            result.pop();
        }
        runtime_inflight_search_queries_.fetch_sub(1, std::memory_order_relaxed);
        runtime_active_search_calls_.fetch_sub(1, std::memory_order_relaxed);
        return 0;
    }

    int batch_search(const T* batch_queries, size_t k, size_t num_queries,
                     TagT** batch_results, size_t* watermark_out = nullptr,
                     size_t visible_ts = std::numeric_limits<size_t>::max()) override {
        runtime_active_search_calls_.fetch_add(1, std::memory_order_relaxed);

        size_t wm = visible_ts;
        if (watermark_out) *watermark_out = wm;

        // Non-preempt path
        if (!enable_preempt_m2_.load(std::memory_order_relaxed)) {
            arena_.execute([&] {
                tbb::parallel_for(size_t(0), num_queries, [&](size_t i) {
                    auto t0 = std::chrono::steady_clock::now();
                    runtime_inflight_search_queries_.fetch_add(1, std::memory_order_relaxed);
                    execute_search_query(batch_queries + i * dim_, k, wm, batch_results[i]);
                    runtime_inflight_search_queries_.fetch_sub(1, std::memory_order_relaxed);
                    auto t1 = std::chrono::steady_clock::now();
                    total_search_work_ns_.fetch_add(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                        std::memory_order_relaxed);
                });
            });
            runtime_active_search_calls_.fetch_sub(1, std::memory_order_relaxed);
            return 0;
        }

        // ── M2: Deferred preemption (PreemptDB model adapted for graph ANN) ──
        //
        // 1. Post queries to shared queue — async signal to writers
        //    (analogous to PreemptDB's UINTR: notifies writers that search needs service)
        // 2. Search thread also executes queries itself via parallel_for
        //    (zero idle time — search thread is always productive)
        // 3. Writer fibers at safe points yield and service search queries
        //    (preemption benefit: scheduling granularity reduced from O(addPoint) to O(expansion))
        // 4. CAS on done flag: whoever claims first (writer or search) executes
        //    (zero spin, zero CV, zero futex — one atomic CAS is all coordination)
        std::vector<T> norm_buf;
        const T* q_ptr = batch_queries;
        if (metric_ == METRIC_COSINE) {
            norm_buf.assign(batch_queries, batch_queries + num_queries * dim_);
            normalize_vector(norm_buf.data(), num_queries);
            q_ptr = norm_buf.data();
        }

        uint64_t batch_epoch = search_epoch_.fetch_add(1, std::memory_order_relaxed);
        std::vector<SearchRequest> requests(num_queries);
        for (size_t i = 0; i < num_queries; i++) {
            requests[i].query = q_ptr + i * dim_;
            requests[i].k = k;
            requests[i].visible_ts = wm;
            requests[i].result_buf = batch_results[i];
            requests[i].result_count = 0;
            requests[i].done.store(false, std::memory_order_relaxed);
            requests[i].epoch = batch_epoch;
            search_queue_.push(&requests[i]);
            pending_search_count_.fetch_add(1, std::memory_order_release);
        }

        // Track queue high watermark
        int cur = pending_search_count_.load(std::memory_order_relaxed);
        int prev = pending_search_queue_max_.load(std::memory_order_relaxed);
        while (cur > prev && !pending_search_queue_max_.compare_exchange_weak(
                   prev, cur, std::memory_order_relaxed)) {}

        // Search thread executes queries itself — skip any already serviced by writer
        arena_.execute([&] {
            tbb::parallel_for(size_t(0), num_queries, [&](size_t i) {
                bool expected = false;
                if (requests[i].done.compare_exchange_strong(
                        expected, true, std::memory_order_acq_rel)) {
                    auto t0 = std::chrono::steady_clock::now();
                    // Writer didn't get it — execute ourselves
                    pending_search_count_.fetch_sub(1, std::memory_order_relaxed);
                    runtime_inflight_search_queries_.fetch_add(1, std::memory_order_relaxed);
                    execute_search_query(
                        static_cast<const T*>(requests[i].query),
                        requests[i].k, requests[i].visible_ts, requests[i].result_buf);
                    runtime_inflight_search_queries_.fetch_sub(1, std::memory_order_relaxed);
                    fiber_fallback_count_.fetch_add(1, std::memory_order_relaxed);
                    auto t1 = std::chrono::steady_clock::now();
                    total_search_work_ns_.fetch_add(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                        std::memory_order_relaxed);
                }
                // else: writer already executed via preemption — skip
            });
        });

        // Drain stale pointers: after parallel_for, all our requests have done==true.
        // Queue may still hold pointers to our stack-local requests.
        // Pop done requests (ours) to prevent writer from seeing dangling pointers.
        // Non-done requests (from other batches) get pushed back.
        {
            std::vector<SearchRequest*> requeue;
            SearchRequest* stale = nullptr;
            while (search_queue_.try_pop(stale)) {
                if (stale->done.load(std::memory_order_relaxed)) {
                    // Already handled — discard stale pointer
                    continue;
                }
                // Live request from another batch — keep it
                requeue.push_back(stale);
            }
            for (auto* r : requeue) {
                search_queue_.push(r);
            }
        }

        runtime_active_search_calls_.fetch_sub(1, std::memory_order_relaxed);
        return 0;
    }

    void dump_stats(std::string& str) override {
        std::stringstream ss;
        ss << "index_memory_mb:" << index_->indexFileSize() / (1024 * 1024)
           << ", dist_computations:" << index_->metric_distance_computations.load()
           << ", hops:" << index_->metric_hops.load()
           << ", preempt_m2_enabled:" << (enable_preempt_m2_.load(std::memory_order_relaxed) ? 1 : 0)
           << ", writer_safe_point_checks:" << writer_safe_point_checks_.load(std::memory_order_relaxed)
           << ", writer_convergence_gate_passes:" << writer_convergence_gate_passes_.load(std::memory_order_relaxed)
           << ", fiber_yield_count:" << fiber_yield_count_.load(std::memory_order_relaxed)
           << ", fiber_searches_serviced:" << fiber_searches_serviced_.load(std::memory_order_relaxed)
           << ", fiber_fallback_count:" << fiber_fallback_count_.load(std::memory_order_relaxed)
           << ", fiber_starvation_refuses:" << fiber_starvation_refuses_.load(std::memory_order_relaxed)
           << ", pending_search_queue_max:" << pending_search_queue_max_.load(std::memory_order_relaxed)
           << ", starvation_alpha:" << starvation_alpha_
           << ", max_service_per_yield:" << max_service_per_yield_
           << ", fiber_stack_size:" << fiber_stack_size_;

        // Thread utilization stats
        uint64_t ins_ns = total_insert_work_ns_.load(std::memory_order_relaxed);
        uint64_t srch_ns = total_search_work_ns_.load(std::memory_order_relaxed);
        uint64_t wall_ns = 0;
        if (bench_started_.load(std::memory_order_relaxed)) {
            wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - bench_start_ts_).count();
        }
        int nthreads = arena_.max_concurrency();
        uint64_t total_avail_ns = wall_ns * nthreads;
        uint64_t busy_ns = ins_ns + srch_ns;
        uint64_t idle_ns = (total_avail_ns > busy_ns) ? (total_avail_ns - busy_ns) : 0;
        double util_pct = (total_avail_ns > 0) ? (100.0 * busy_ns / total_avail_ns) : 0;
        ss << ", thread_util_insert_ms:" << (ins_ns / 1000000)
           << ", thread_util_search_ms:" << (srch_ns / 1000000)
           << ", thread_util_wall_ms:" << (wall_ns / 1000000)
           << ", thread_util_nthreads:" << nthreads
           << ", thread_util_avail_ms:" << (total_avail_ns / 1000000)
           << ", thread_util_idle_ms:" << (idle_ns / 1000000)
           << ", thread_util_pct:" << std::fixed << std::setprecision(1) << util_pct;

        ss << ", mid_insert_preempt_enabled:" << (index_->isMidInsertPreemptEnabled() ? 1 : 0)
           << ", mid_insert_preempt_k:" << index_->getMidInsertPreemptK()
           << ", mid_insert_preempt_revalidate:" << (index_->isMidInsertPreemptRevalidateEnabled() ? 1 : 0)
           << ", mid_insert_shadow_replan_enabled:" << (index_->isMidInsertShadowReplanEnabled() ? 1 : 0)
           << ", mid_insert_harm_guard_enabled:" << (index_->isMidInsertHarmGuardEnabled() ? 1 : 0)
           << ", mid_insert_shadow_snapshot_teacher_enabled:" << (index_->isMidInsertShadowSnapshotTeacherEnabled() ? 1 : 0)
           << ", mid_insert_harm_micro_replan_enabled:" << (index_->isMidInsertHarmMicroReplanEnabled() ? 1 : 0)
           << ", mid_insert_harm_micro_probe_ef:" << index_->getMidInsertHarmMicroProbeEf()
           << ", mid_insert_harm_online_policy_enabled:" << (index_->isMidInsertHarmOnlinePolicyEnabled() ? 1 : 0)
           << ", mid_insert_harm_busy_wait_commits:" << index_->getMidInsertHarmBusyWaitCommits()
           << ", mid_insert_harm_search_backlog_threshold:" << index_->getMidInsertHarmSearchBacklogThreshold()
           << ", mid_insert_harm_priority_search_threshold:" << index_->getMidInsertHarmPrioritySearchThreshold()
           << ", mid_insert_harm_full_foreign_outrank_threshold:" << index_->getMidInsertHarmFullForeignOutrankThreshold()
           << ", mid_insert_harm_full_selected_frontier_touched_threshold:" << index_->getMidInsertHarmFullSelectedFrontierTouchedThreshold()
           << ", mid_insert_harm_defer_enabled:" << (index_->isMidInsertHarmDeferEnabled() ? 1 : 0)
           << ", mid_insert_harm_defer_queue_cap:" << index_->getMidInsertHarmDeferQueueCap()
           << ", mid_insert_harm_defer_drain_budget:" << index_->getMidInsertHarmDeferDrainBudget()
           << ", mid_insert_harm_defer_high_watermark_pct:" << index_->getMidInsertHarmDeferHighWatermarkPct()
           << ", mid_insert_preempt_every_n:" << index_->getMidInsertPreemptEveryN()
           << ", mid_insert_preempt_max_inflight:" << index_->getMidInsertPreemptMaxInflight()
           << ", mid_insert_harm_micro_pool_cap:" << index_->getMidInsertHarmMicroPoolCap()
           << ", mid_insert_preempt_hits:" << index_->getMidInsertPreemptHits()
           << ", mid_insert_preempt_timeouts:" << index_->getMidInsertPreemptTimeouts()
           << ", mid_insert_preempt_wait_commits_sum:" << index_->getMidInsertPreemptWaitCommitsSum()
           << ", mid_insert_preempt_revalidations:" << index_->getMidInsertPreemptRevalidations()
           << ", mid_insert_shadow_replans:" << index_->getMidInsertShadowReplans()
           << ", mid_insert_preempt_slot_denied:" << index_->getMidInsertPreemptSlotDenied()
           << ", mid_insert_preempt_global_lock_skipped:" << index_->getMidInsertPreemptGlobalLockSkipped()
           << ", mid_insert_preempt_quiesce_breaks:" << index_->getMidInsertPreemptQuiesceBreaks()
           << ", mid_insert_candidate_changed:" << index_->getMidInsertCandidateChanged()
           << ", mid_insert_candidate_before_total:" << index_->getMidInsertCandidateBeforeTotal()
           << ", mid_insert_candidate_after_total:" << index_->getMidInsertCandidateAfterTotal()
           << ", mid_insert_candidate_intersection_total:" << index_->getMidInsertCandidateIntersectionTotal()
           << ", mid_insert_candidate_union_total:" << index_->getMidInsertCandidateUnionTotal()
           << ", mid_insert_selected_changed:" << index_->getMidInsertSelectedChanged()
           << ", mid_insert_selected_before_total:" << index_->getMidInsertSelectedBeforeTotal()
           << ", mid_insert_selected_after_total:" << index_->getMidInsertSelectedAfterTotal()
           << ", mid_insert_selected_intersection_total:" << index_->getMidInsertSelectedIntersectionTotal()
           << ", mid_insert_selected_union_total:" << index_->getMidInsertSelectedUnionTotal()
           << ", mid_insert_pause_low_slack_hits:" << index_->getMidInsertPauseLowSlackHits()
           << ", mid_insert_pause_mild_risk_hits:" << index_->getMidInsertPauseMildRiskHits()
           << ", mid_insert_pause_service_busy_hits:" << index_->getMidInsertPauseServiceBusyHits()
           << ", mid_insert_pause_search_pressure_busy_hits:" << index_->getMidInsertPauseSearchPressureBusyHits()
           << ", mid_insert_pause_severe_risk_hits:" << index_->getMidInsertPauseSevereRiskHits()
           << ", mid_insert_pause_selected_frontier_touched_hits:" << index_->getMidInsertPauseSelectedFrontierTouchedHits()
           << ", mid_insert_pause_selected_frontier_touched_nodes:" << index_->getMidInsertPauseSelectedFrontierTouchedNodes()
           << ", mid_insert_pause_candidate_frontier_touched_hits:" << index_->getMidInsertPauseCandidateFrontierTouchedHits()
           << ", mid_insert_pause_candidate_frontier_touched_nodes:" << index_->getMidInsertPauseCandidateFrontierTouchedNodes()
           << ", mid_insert_pause_foreign_commit_nodes:" << index_->getMidInsertPauseForeignCommitNodes()
           << ", mid_insert_pause_foreign_outrank_hits:" << index_->getMidInsertPauseForeignOutrankHits()
           << ", mid_insert_pause_foreign_outrank_nodes:" << index_->getMidInsertPauseForeignOutrankNodes()
           << ", mid_insert_pause_boundary_blocked_hits:" << index_->getMidInsertPauseBoundaryBlockedHits()
           << ", mid_insert_pause_boundary_blocked_candidates:" << index_->getMidInsertPauseBoundaryBlockedCandidates()
           << ", mid_insert_pause_boundary_blocker_touched_hits:" << index_->getMidInsertPauseBoundaryBlockerTouchedHits()
           << ", mid_insert_pause_boundary_blocker_touched_nodes:" << index_->getMidInsertPauseBoundaryBlockerTouchedNodes()
           << ", mid_insert_pause_boundary_witness_invalidated_hits:" << index_->getMidInsertPauseBoundaryWitnessInvalidatedHits()
           << ", mid_insert_pause_boundary_witness_invalidated_candidates:" << index_->getMidInsertPauseBoundaryWitnessInvalidatedCandidates()
           << ", mid_insert_pause_boundary_harm_risk_hits:" << index_->getMidInsertPauseBoundaryHarmRiskHits()
           << ", mid_insert_pause_boundary_harm_cert_true_positive:" << index_->getMidInsertPauseBoundaryHarmCertTruePositive()
           << ", mid_insert_pause_boundary_harm_cert_false_positive:" << index_->getMidInsertPauseBoundaryHarmCertFalsePositive()
           << ", mid_insert_pause_boundary_harm_cert_false_negative:" << index_->getMidInsertPauseBoundaryHarmCertFalseNegative()
           << ", mid_insert_pause_harm_risk_hits:" << index_->getMidInsertPauseHarmRiskHits()
           << ", mid_insert_pause_harm_guard_replans:" << index_->getMidInsertPauseHarmGuardReplans()
           << ", mid_insert_pause_action_continue:" << index_->getMidInsertPauseActionContinue()
           << ", mid_insert_pause_action_continue_mild:" << index_->getMidInsertPauseActionContinueMild()
           << ", mid_insert_pause_action_defer:" << index_->getMidInsertPauseActionDefer()
           << ", mid_insert_pause_action_micro:" << index_->getMidInsertPauseActionMicro()
           << ", mid_insert_pause_action_full:" << index_->getMidInsertPauseActionFull()
           << ", mid_insert_pause_deferred_enqueued:" << index_->getMidInsertPauseDeferredEnqueued()
           << ", mid_insert_pause_deferred_deduped:" << index_->getMidInsertPauseDeferredDeduped()
           << ", mid_insert_pause_deferred_dropped:" << index_->getMidInsertPauseDeferredDropped()
           << ", mid_insert_pause_deferred_drained:" << index_->getMidInsertPauseDeferredDrained()
           << ", mid_insert_pause_deferred_drain_runs:" << index_->getMidInsertPauseDeferredDrainRuns()
           << ", mid_insert_pause_deferred_queue_max:" << index_->getMidInsertPauseDeferredQueueMax()
           << ", mid_insert_pause_defer_backpressure_hits:" << index_->getMidInsertPauseDeferBackpressureHits()
           << ", mid_insert_pause_micro_replans:" << index_->getMidInsertPauseMicroReplans()
           << ", mid_insert_pause_micro_probe_hits:" << index_->getMidInsertPauseMicroProbeHits()
           << ", mid_insert_pause_micro_probe_candidates_total:" << index_->getMidInsertPauseMicroProbeCandidatesTotal()
           << ", mid_insert_pause_micro_probe_candidates_max:" << index_->getMidInsertPauseMicroProbeCandidatesMax()
           << ", mid_insert_pause_micro_replan_pool_total:" << index_->getMidInsertPauseMicroReplanPoolTotal()
           << ", mid_insert_pause_micro_replan_pool_max:" << index_->getMidInsertPauseMicroReplanPoolMax()
           << ", mid_insert_pause_harm_cert_true_positive:" << index_->getMidInsertPauseHarmCertTruePositive()
           << ", mid_insert_pause_harm_cert_false_positive:" << index_->getMidInsertPauseHarmCertFalsePositive()
           << ", mid_insert_pause_harm_cert_false_negative:" << index_->getMidInsertPauseHarmCertFalseNegative()
           << ", mid_insert_shadow_snapshot_teacher_hits:" << index_->getMidInsertShadowSnapshotTeacherHits()
           << ", mid_insert_shadow_snapshot_teacher_future_entrypoint_hits:" << index_->getMidInsertShadowSnapshotTeacherFutureEntrypointHits()
           << ", mid_insert_shadow_snapshot_teacher_postwait_commit_drift_hits:" << index_->getMidInsertShadowSnapshotTeacherPostwaitCommitDriftHits()
           << ", mid_insert_shadow_snapshot_teacher_postwait_commit_drift_sum:" << index_->getMidInsertShadowSnapshotTeacherPostwaitCommitDriftSum()
           << ", mid_insert_shadow_snapshot_teacher_postwait_watermark_drift_hits:" << index_->getMidInsertShadowSnapshotTeacherPostwaitWatermarkDriftHits()
           << ", mid_insert_shadow_snapshot_teacher_postwait_watermark_drift_sum:" << index_->getMidInsertShadowSnapshotTeacherPostwaitWatermarkDriftSum();

        ss << ", undo_recovery_enabled:" << (index_->isUndoRecoveryEnabled() ? 1 : 0)
           << ", undo_recovery_mode:" << index_->getM2Mode()
           << ", recovery_triggers:" << index_->m2_triggered_.load()
           << ", recovered_edges_total:" << index_->m2_recovered_edges_.load()
           << ", recovered_edges_useful:" << index_->m2_useful_edges_.load();

        str = ss.str();
    }

    // MVCC mechanism 1
    void set_enable_mvcc(bool enable) { index_->setEnableMvcc(enable); }
    bool is_mvcc_enabled() const { return index_->isMvccEnabled(); }
    void set_enable_undo_recovery(bool enable) {
        index_->setM2Mode(1);
        index_->resetM2Stats();
        index_->setEnableUndoRecovery(enable);
    }
    bool is_undo_recovery_enabled() const { return index_->isUndoRecoveryEnabled(); }

    // Preemptive mechanism 2
    void set_enable_preempt_m2(bool enable) {
        enable_preempt_m2_.store(enable, std::memory_order_relaxed);
    }
    // Legacy setters (no-ops, kept for CGO compatibility)
    void set_preempt_quantum_points(int) {}
    void set_preempt_search_backlog_threshold(int) {}
    void set_preempt_max_yields_per_batch(int) {}
    void set_preempt_budget_window_us(int) {}
    void set_preempt_budget_pct(double) {}
    void set_preempt_priority_cap(int) {}
    void set_fine_grain_yield_cooldown_us(int) {}

    static void set_runtime_search_backlog(int backlog) {
        runtime_search_backlog_.store(std::max(0, backlog), std::memory_order_relaxed);
    }
    static void set_runtime_priority_searches(int priority_searches) {
        runtime_priority_searches_.store(std::max(0, priority_searches), std::memory_order_relaxed);
    }

    size_t compact(long safe_ts = -1) { return index_->compact(safe_ts); }
    bool supports_snapshot() const override { return true; }

    int snapshot(std::vector<uint8_t>& out) override {
        std::string tmp_file = "/tmp/annchor_preempt_snapshot_" + std::to_string((uintptr_t)this) + ".bin";
        try { index_->saveIndex(tmp_file); } catch (...) { return -1; }
        std::ifstream file(tmp_file, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return -1;
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        out.resize(size);
        if (!file.read((char*)out.data(), size)) return -1;
        file.close();
        std::remove(tmp_file.c_str());
        return 0;
    }

    int restore(const uint8_t* data, size_t size) override {
        std::string tmp_file = "/tmp/annchor_preempt_restore_" + std::to_string((uintptr_t)this) + ".bin";
        std::ofstream file(tmp_file, std::ios::binary);
        if (!file.write((const char*)data, size)) return -1;
        file.close();
        try {
            delete index_;
            index_ = new annchor_m1::HierarchicalNSW<T>(
                space_.get(), tmp_file, false, max_elements_, false);
        } catch (...) { return -1; }
        std::remove(tmp_file.c_str());
        return 0;
    }

   private:
    size_t num_threads_;
    size_t dim_;
    size_t M_;
    size_t ef_c_;
    size_t ef_s_;
    size_t max_elements_;
    bool use_node_lock_in_search_;
    MetricType metric_;
    std::unique_ptr<annchor_m1::SpaceInterface<T>> space_;
    annchor_m1::HierarchicalNSW<T>* index_;

    std::atomic<bool> enable_preempt_m2_{false};
    double starvation_alpha_{0.20};         // max fraction of CPU donated to search
    int search_budget_{4};                  // inter-insert drain: service min(B, pending) after each insert
    double drain_alpha_{0.20};              // separate alpha for drain (default = same as fiber)
    int max_service_per_yield_{4};          // max searches per fiber yield
    int fiber_stack_size_{262144};          // 256KB

    tbb::task_arena arena_;

    // Shared search queue (MPMC)
    tbb::concurrent_queue<SearchRequest*> search_queue_;
    std::atomic<int> pending_search_count_{0};
    std::atomic<int> pending_search_queue_max_{0};
    std::atomic<uint64_t> search_epoch_{0};

    // Stats
    std::atomic<uint64_t> writer_safe_point_checks_{0};
    std::atomic<uint64_t> writer_convergence_gate_passes_{0};
    std::atomic<uint64_t> fiber_yield_count_{0};
    std::atomic<uint64_t> fiber_searches_serviced_{0};
    std::atomic<uint64_t> fiber_fallback_count_{0};
    std::atomic<uint64_t> fiber_starvation_refuses_{0};

    // Thread utilization tracking
    std::atomic<uint64_t> total_insert_work_ns_{0};
    std::atomic<uint64_t> total_search_work_ns_{0};
    std::atomic<uint64_t> total_wall_ns_{0};
    std::chrono::steady_clock::time_point bench_start_ts_{};
    std::atomic<bool> bench_started_{false};

    inline static std::atomic<int> runtime_search_backlog_{0};
    inline static std::atomic<int> runtime_inflight_search_queries_{0};
    inline static std::atomic<int> runtime_active_search_calls_{0};
    inline static std::atomic<int> runtime_priority_searches_{0};

    // Thread-local fiber caller for yielding from deep in the call stack
    inline static thread_local boost::context::fiber* tl_caller_ = nullptr;

    // Shared L-ledger: accumulated across fiber yields AND inter-insert drain.
    // Reset at the start of each insert (by the fiber callback on first check).
    inline static thread_local uint64_t tl_time_own_ns_ = 0;
    inline static thread_local uint64_t tl_time_donated_ns_ = 0;
    inline static thread_local std::chrono::steady_clock::time_point tl_last_ts_{};

    static void fiber_yield() {
        if (tl_caller_ && *tl_caller_) {
            *tl_caller_ = std::move(*tl_caller_).resume();
        }
    }

    // Service exactly one search request.  Called on each fiber yield so
    // that the L-check in the callback fires after every single search.
    void service_search_inline() {
        SearchRequest* req = nullptr;
        while (search_queue_.try_pop(req)) {
            bool expected = false;
            if (!req->done.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
                continue;  // already served
            if (!req->query || !req->result_buf) {
                pending_search_count_.fetch_sub(1, std::memory_order_relaxed);
                continue;
            }
            pending_search_count_.fetch_sub(1, std::memory_order_relaxed);
            if (index_->isMvccEnabled()) index_->advanceWatermark();
            execute_search_query(static_cast<const T*>(req->query), req->k, req->visible_ts, req->result_buf);
            fiber_searches_serviced_.fetch_add(1, std::memory_order_relaxed);
            return;  // one search per yield
        }
    }

    void execute_search_query(const T* q, size_t k, size_t wm, uint32_t* result_buf) {
        std::priority_queue<std::pair<T, annchor_m1::labeltype>> result;
        if (metric_ == METRIC_COSINE) {
            std::vector<T> temp(q, q + dim_);
            normalize_vector(temp.data(), 1);
            result = index_->searchKnn(temp.data(), k, nullptr, wm);
        } else {
            result = index_->searchKnn(q, k, nullptr, wm);
        }
        size_t sz = std::min(result.size(), k);
        for (size_t j = 0; j < sz; ++j) {
            result_buf[sz - j - 1] = static_cast<uint32_t>(result.top().second);
            result.pop();
        }
    }

    const T* get_point(const T* batch_data, size_t i) {
        if (metric_ == METRIC_COSINE) {
            thread_local std::vector<T> buf;
            buf.assign(batch_data + i * dim_, batch_data + (i + 1) * dim_);
            normalize_vector(buf.data(), 1);
            return buf.data();
        }
        return batch_data + i * dim_;
    }

    void post_insert_mvcc_cleanup() {
        if (index_->isMvccEnabled()) {
            index_->advanceWatermark();
            if (index_->isMidInsertHarmOnlinePolicyEnabled() &&
                index_->isMidInsertHarmDeferEnabled()) {
                index_->drainMidInsertDeferredRepairsBudgeted();
            }
        }
    }

    void normalize_vector(T* data, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            T* vec = data + i * dim_;
            float norm = 0;
            for (size_t j = 0; j < dim_; ++j) norm += vec[j] * vec[j];
            norm = 1.0f / (sqrt(norm) + 1e-30f);
            for (size_t j = 0; j < dim_; ++j) vec[j] *= norm;
        }
    }

    static int read_env_int(const char* name, int fallback) {
        const char* v = std::getenv(name);
        if (!v || !*v) return fallback;
        char* end = nullptr;
        long x = std::strtol(v, &end, 10);
        if (end == v) return fallback;
        return static_cast<int>(x);
    }

    static double read_env_double(const char* name, double fallback) {
        const char* v = std::getenv(name);
        if (!v || !*v) return fallback;
        char* end = nullptr;
        double x = std::strtod(v, &end);
        if (end == v) return fallback;
        return x;
    }
};
