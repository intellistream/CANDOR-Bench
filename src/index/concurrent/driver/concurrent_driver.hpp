#pragma once

// Native concurrent benchmark driver. Rate-limited producers feed insert
// and search worker pools; searches run against the committed snapshot
// offset; latency is captured per batch. The whole hot path runs in C++
// threads — no interpreter, scheduler, or GC sits in the measurement
// path, and each worker collects into private buffers so the collection
// itself never serializes the workers.

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "index.hpp"
#include "query_source.hpp"
#include "stats.hpp"
#include "sync.hpp"

namespace candor {
namespace driver {

struct Task {
    enum class Type { kInsert, kSearch };
    Type type = Type::kInsert;
    // Insert tasks borrow a contiguous slice of the base data; search tasks
    // own a flattened copy of their (possibly non-contiguous) queries.
    const float* data = nullptr;
    std::vector<float> owned;
    size_t count = 0;
    std::vector<uint32_t> tags;
    uint32_t recall_at = 0;
    uint64_t insert_offset = 0;
    Clock::time_point create_time;
};

// One recorded search stage: the committed offset it ran at and the
// workload-query tags it used.
struct ScheduleEntry {
    uint64_t insert_offset = 0;
    std::vector<uint32_t> query_tags;
};

struct DriverConfig {
    size_t begin_num = 0;     // index pre-built with [0, begin_num)
    size_t max_elements = 0;  // insert until this offset
    size_t batch_size = 100;
    size_t queue_size = 0;  // 0 = unbuffered rendezvous
    size_t num_threads = 4;
    double insert_event_rate = 0;  // points/sec, <= 0 = unlimited
    double search_event_rate = 0;  // points/sec, <= 0 = search disabled
    uint32_t recall_at = 0;        // 0 = search path disabled
    QueryMode query_mode = QueryMode::kRoundRobin;
    size_t stage_query_window = 0;  // 0 -> batch_size
    double zipfian_skew = 0.99;
    bool per_query_latency = false;
    bool external_rw_lock = false;
    int mem_sample_interval_ms = 0;  // 0 = no memory sampling
    uint64_t seed = 0;               // 0 = nondeterministic
    bool progress_log = true;
    // Samples (latency, QPS, lag) start counting only after this many
    // points have been inserted; lets the limiter burst and cold caches
    // warm up outside the measured window. 0 = measure everything.
    size_t warmup_points = 0;
    // Record/replay: record captures one entry per distinct committed
    // offset; replay re-issues those searches at the same offsets,
    // gating inserts so they never run ahead of the schedule.
    bool record_schedule = false;
    std::vector<ScheduleEntry> replay_schedule;
};

// Search records, flattened: query i of record j searched the index at
// committed offset rec_offsets[j]; its k result tags live at
// rec_result_tags[j*k .. (j+1)*k).
struct RunResult {
    DriverStats stats;
    uint32_t k = 0;
    std::vector<uint64_t> rec_offsets;
    std::vector<uint32_t> rec_query_tags;
    std::vector<uint32_t> rec_result_tags;
    std::vector<ScheduleEntry> recorded_schedule;  // when record_schedule
};

using LogFn = std::function<void(const std::string&)>;

namespace detail {

// Private per-worker collection buffers, merged after the workers join.
struct InsertBuf {
    std::vector<double> op, e2e;
};

struct SearchBuf {
    std::vector<double> op, e2e;
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> query_tags;
    std::vector<uint32_t> result_tags;
};

}  // namespace detail

inline Clock::time_point start_time_unset() { return Clock::time_point{}; }

inline RunResult run_concurrent_benchmark(IndexBase<float>& index,
                                          const float* data, size_t dim,
                                          const float* workload_queries,
                                          size_t num_workload_queries,
                                          const DriverConfig& cfg_in,
                                          const LogFn& log = nullptr) {
    DriverConfig cfg = cfg_in;
    if (cfg.batch_size == 0) cfg.batch_size = 1;
    if (cfg.num_threads == 0) cfg.num_threads = 1;
    if (cfg.stage_query_window == 0) cfg.stage_query_window = cfg.batch_size;

    auto say = [&](const std::string& s) {
        if (log) log(s);
    };

    RunResult result;
    result.k = cfg.recall_at;
    if (cfg.begin_num >= cfg.max_elements || dim == 0) return result;

    const size_t total_insert = cfg.max_elements - cfg.begin_num;
    const bool replay_active = !cfg.replay_schedule.empty();
    const bool search_enabled =
        cfg.recall_at > 0 &&
        (cfg.search_event_rate > 0 || replay_active) &&
        workload_queries != nullptr && num_workload_queries > 0;

    // --- shared state ---
    std::atomic<uint64_t> committed_offset{cfg.begin_num};
    std::atomic<int64_t> global_insert_cnt{0};
    std::atomic<size_t> insert_point_cnt{0}, search_point_cnt{0};
    std::atomic<int64_t> lag_sum{0}, lag_count{0}, lag_max{0};
    std::atomic<bool> insert_producer_done{false};
    std::atomic<bool> run_done{false};
    std::atomic<bool> warmup_done{cfg.warmup_points == 0};
    std::atomic<size_t> measured_insert_cnt{0}, measured_search_cnt{0};
    std::mutex warmup_mu;
    Clock::time_point warmup_end = start_time_unset();

    // In-order commit bookkeeping: inserts complete out of order, the
    // committed offset only advances contiguously.
    std::mutex completion_mu;
    std::vector<uint64_t> pending_offsets;
    size_t pending_idx = 0;
    std::unordered_set<uint64_t> completed_offsets;
    for (size_t off = cfg.begin_num + cfg.batch_size; off <= cfg.max_elements;
         off += cfg.batch_size) {
        pending_offsets.push_back(off);
    }
    if (pending_offsets.empty() || pending_offsets.back() != cfg.max_elements) {
        pending_offsets.push_back(cfg.max_elements);
    }

    std::shared_mutex rw_mu;

    BoundedQueue<Task> insert_queue(cfg.queue_size);
    BoundedQueue<Task> search_queue(cfg.queue_size);
    auto insert_limiter = build_limiter(cfg.insert_event_rate, cfg.batch_size);
    auto search_limiter = build_limiter(cfg.search_event_rate, cfg.batch_size);

    std::unique_ptr<QuerySource> query_source;
    if (search_enabled) {
        query_source = std::make_unique<QuerySource>(
            cfg.query_mode, workload_queries, num_workload_queries, dim,
            cfg.batch_size, cfg.stage_query_window, cfg.zipfian_skew,
            cfg.seed);
    }

    if (cfg.insert_event_rate <= 0 && cfg.search_event_rate <= 0) {
        say("Limit test mode enabled - filling queue to capacity");
    }

    const auto start_time = Clock::now();

    // --- memory monitor ---
    std::thread mem_thread;
    uint64_t mem_peak = 0;
    double mem_sum = 0;
    size_t mem_samples = 0;
    if (cfg.mem_sample_interval_ms > 0) {
        mem_thread = std::thread([&] {
            while (!run_done.load()) {
                uint64_t rss = detail::current_rss_bytes();
                if (rss > mem_peak) mem_peak = rss;
                mem_sum += static_cast<double>(rss);
                ++mem_samples;
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(cfg.mem_sample_interval_ms));
            }
        });
    }

    // --- progress logger (every completed 10% decade) ---
    std::thread progress_thread;
    if (cfg.progress_log) {
        progress_thread = std::thread([&] {
            int last_decade = -1;
            while (!run_done.load()) {
                auto current = static_cast<uint64_t>(global_insert_cnt.load());
                int percent =
                    static_cast<int>(current * 100 / total_insert);
                int decade = percent / 10;
                if (decade != last_decade) {
                    double elapsed = std::chrono::duration<double>(
                                         Clock::now() - start_time)
                                         .count();
                    int64_t lc = lag_count.load();
                    double lag_avg =
                        lc > 0 ? static_cast<double>(lag_sum.load()) /
                                     static_cast<double>(lc)
                               : 0.0;
                    int64_t offset_lag =
                        static_cast<int64_t>(current) -
                        static_cast<int64_t>(committed_offset.load()) +
                        static_cast<int64_t>(cfg.begin_num);
                    std::ostringstream os;
                    os << "Progress: " << current << "/" << total_insert
                       << " (" << percent << "%), Insert QPS: "
                       << static_cast<double>(current) / elapsed
                       << ", Search QPS: "
                       << static_cast<double>(search_point_cnt.load()) /
                              elapsed
                       << ", offset_lag: " << offset_lag
                       << ", search_snapshot_lag(avg/max): " << lag_avg
                       << "/" << lag_max.load() << " pts";
                    say(os.str());
                    last_decade = decade;
                }
                if (current >= total_insert) break;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
    }

    // Replay coordination: inserts may not pass max_allowed_offset, which
    // the search producer raises as it works through the schedule.
    std::mutex replay_mu;
    std::condition_variable replay_cv;
    uint64_t max_allowed_offset =
        replay_active ? cfg.replay_schedule.front().insert_offset
                      : std::numeric_limits<uint64_t>::max();

    // --- insert producer ---
    std::thread insert_producer([&] {
        size_t offset = cfg.begin_num;
        while (offset < cfg.max_elements) {
            size_t next = std::min(offset + cfg.batch_size, cfg.max_elements);
            if (replay_active) {
                std::unique_lock<std::mutex> lk(replay_mu);
                replay_cv.wait(lk, [&] {
                    return static_cast<uint64_t>(offset) < max_allowed_offset;
                });
                if (static_cast<uint64_t>(next) > max_allowed_offset) {
                    next = static_cast<size_t>(max_allowed_offset);
                }
                if (next <= offset) continue;
            }
            Task t;
            t.type = Task::Type::kInsert;
            t.data = data + offset * dim;
            t.count = next - offset;
            t.tags.resize(t.count);
            for (size_t i = 0; i < t.count; ++i) {
                t.tags[i] = static_cast<uint32_t>(offset + i);
            }
            t.insert_offset = next;
            // Stamped before the limiter wait, so insert E2E latency
            // includes throttle time.
            t.create_time = Clock::now();
            if (insert_limiter) insert_limiter->wait();
            insert_queue.push(std::move(t));
            offset = next;
        }
        insert_producer_done.store(true);
        insert_queue.close();
    });

    // --- search producer ---
    std::thread search_producer([&] {
        auto lift_gate = [&] {
            if (!replay_active) return;
            std::lock_guard<std::mutex> lk(replay_mu);
            max_allowed_offset = std::numeric_limits<uint64_t>::max();
            replay_cv.notify_all();
        };
        if (!search_enabled) {
            lift_gate();
            search_queue.close();
            return;
        }

        if (replay_active) {
            for (size_t i = 0; i < cfg.replay_schedule.size(); ++i) {
                const ScheduleEntry& entry = cfg.replay_schedule[i];
                // The producer flag flips while the last batches are still
                // inside batch_insert, so give the committed offset time to
                // settle before declaring the target unreachable.
                uint64_t last_seen = committed_offset.load();
                int stable_polls = 0;
                for (;;) {
                    uint64_t committed = committed_offset.load();
                    if (committed >= entry.insert_offset) break;
                    if (committed != last_seen) {
                        last_seen = committed;
                        stable_polls = 0;
                    } else if (insert_producer_done.load() &&
                               ++stable_polls > 2500) {  // ~250ms idle
                        say("Producer: replay - insert finished before "
                            "reaching target " +
                            std::to_string(entry.insert_offset));
                        lift_gate();
                        search_queue.close();
                        return;
                    }
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(100));
                }

                Task t;
                t.type = Task::Type::kSearch;
                for (uint32_t tag : entry.query_tags) {
                    size_t start = static_cast<size_t>(tag) * dim;
                    if (start + dim <= num_workload_queries * dim) {
                        const float* src = workload_queries + start;
                        t.owned.insert(t.owned.end(), src, src + dim);
                        t.tags.push_back(tag);
                    } else {
                        say("Producer: replay tag " + std::to_string(tag) +
                            " out of range");
                    }
                }
                if (!t.tags.empty()) {
                    t.count = t.tags.size();
                    t.recall_at = cfg.recall_at;
                    t.insert_offset = entry.insert_offset;
                    t.create_time = Clock::now();
                    search_queue.push(std::move(t));
                }

                {
                    std::lock_guard<std::mutex> lk(replay_mu);
                    max_allowed_offset =
                        i + 1 < cfg.replay_schedule.size()
                            ? cfg.replay_schedule[i + 1].insert_offset
                            : std::numeric_limits<uint64_t>::max();
                    replay_cv.notify_all();
                }
            }
            search_queue.close();
            return;
        }

        while (!insert_producer_done.load()) {
            if (search_limiter) search_limiter->wait();
            uint64_t current_max = committed_offset.load();
            Task t;
            t.type = Task::Type::kSearch;
            size_t n = 0;
            try {
                n = query_source->next_batch(current_max, t.owned, t.tags);
            } catch (const std::exception& e) {
                say(std::string("Producer: query generation failed: ") +
                    e.what());
                break;
            }
            if (n == 0) continue;
            if (cfg.record_schedule &&
                (result.recorded_schedule.empty() ||
                 result.recorded_schedule.back().insert_offset !=
                     current_max)) {
                result.recorded_schedule.push_back(
                    ScheduleEntry{current_max, t.tags});
            }
            t.count = n;
            t.recall_at = cfg.recall_at;
            t.insert_offset = current_max;
            t.create_time = Clock::now();
            search_queue.push(std::move(t));
        }
        search_queue.close();
    });

    // --- workers ---
    size_t insert_workers = cfg.num_threads / 2;
    if (insert_workers < 1) insert_workers = 1;
    size_t search_workers = cfg.num_threads - insert_workers;
    if (search_workers < 1) search_workers = 1;
    {
        std::ostringstream os;
        os << "Dispatchers: " << insert_workers << " insert workers, "
           << search_workers << " search workers";
        say(os.str());
    }

    std::vector<detail::InsertBuf> insert_bufs(insert_workers);
    std::vector<detail::SearchBuf> search_bufs(search_workers);

    auto handle_insert = [&](Task& t, detail::InsertBuf& buf) {
        if (t.count == 0) return;
        auto start = Clock::now();
        std::unique_lock<std::shared_mutex> rw_lk(rw_mu, std::defer_lock);
        if (cfg.external_rw_lock) rw_lk.lock();
        int rc = index.batch_insert(t.data, t.tags.data(), t.count);
        if (rw_lk.owns_lock()) rw_lk.unlock();
        if (rc != 0) {
            say("Insert error: code " + std::to_string(rc));
            return;
        }
        bool measured = warmup_done.load(std::memory_order_relaxed);
        if (measured) {
            buf.op.push_back(detail::ms_since(start));
            buf.e2e.push_back(detail::ms_since(t.create_time));
            measured_insert_cnt.fetch_add(t.count);
        }
        insert_point_cnt.fetch_add(t.count);
        int64_t now_cnt =
            global_insert_cnt.fetch_add(static_cast<int64_t>(t.count)) +
            static_cast<int64_t>(t.count);
        if (!measured &&
            now_cnt >= static_cast<int64_t>(cfg.warmup_points)) {
            std::lock_guard<std::mutex> lk(warmup_mu);
            if (!warmup_done.load()) {
                warmup_end = Clock::now();
                warmup_done.store(true);
            }
        }

        std::lock_guard<std::mutex> lk(completion_mu);
        completed_offsets.insert(t.insert_offset);
        while (pending_idx < pending_offsets.size()) {
            uint64_t off = pending_offsets[pending_idx];
            auto it = completed_offsets.find(off);
            if (it == completed_offsets.end()) break;
            completed_offsets.erase(it);
            committed_offset.store(off);
            ++pending_idx;
        }
    };

    auto handle_search = [&](Task& t, detail::SearchBuf& buf) {
        if (t.count == 0 || t.recall_at == 0) return;
        auto start = Clock::now();

        uint64_t view = committed_offset.load();
        if (view > t.insert_offset) view = t.insert_offset;

        bool measured = warmup_done.load(std::memory_order_relaxed);
        if (measured) {
            int64_t lag = global_insert_cnt.load() +
                          static_cast<int64_t>(cfg.begin_num) -
                          static_cast<int64_t>(view);
            if (lag < 0) lag = 0;
            lag_sum.fetch_add(lag);
            lag_count.fetch_add(1);
            int64_t old_max = lag_max.load();
            while (lag > old_max &&
                   !lag_max.compare_exchange_weak(old_max, lag)) {
            }
        }

        const size_t k = t.recall_at;
        std::vector<uint32_t> res(t.count * k, 0);
        std::vector<uint32_t*> ptrs(t.count);
        for (size_t i = 0; i < t.count; ++i) ptrs[i] = res.data() + i * k;
        size_t watermark = 0;

        std::shared_lock<std::shared_mutex> rw_lk(rw_mu, std::defer_lock);
        if (cfg.external_rw_lock) rw_lk.lock();
        int rc = index.batch_search(t.owned.data(), k, t.count, ptrs.data(),
                                    &watermark, view);
        if (rw_lk.owns_lock()) rw_lk.unlock();
        if (rc != 0) {
            say("Search error: code " + std::to_string(rc));
            return;
        }

        // Latencies are captured before any local bookkeeping so the
        // collection cost never contaminates the measurement.
        double op_ms = detail::ms_since(start);
        double e2e_ms = detail::ms_since(t.create_time);
        if (measured) {
            measured_search_cnt.fetch_add(t.count);
        }
        if (measured && cfg.per_query_latency) {
            double per_op = op_ms / static_cast<double>(t.count);
            double per_e2e = e2e_ms / static_cast<double>(t.count);
            for (size_t i = 0; i < t.count; ++i) {
                buf.op.push_back(per_op);
                buf.e2e.push_back(per_e2e);
            }
        } else if (measured) {
            buf.op.push_back(op_ms);
            buf.e2e.push_back(e2e_ms);
        }
        for (size_t i = 0; i < t.count; ++i) {
            buf.offsets.push_back(t.insert_offset);
            buf.query_tags.push_back(t.tags[i]);
        }
        buf.result_tags.insert(buf.result_tags.end(), res.begin(), res.end());
        search_point_cnt.fetch_add(t.count);
    };

    std::vector<std::thread> workers;
    workers.reserve(insert_workers + search_workers);
    for (size_t i = 0; i < insert_workers; ++i) {
        workers.emplace_back([&, i] {
            Task t;
            while (insert_queue.pop(t)) handle_insert(t, insert_bufs[i]);
        });
    }
    for (size_t i = 0; i < search_workers; ++i) {
        workers.emplace_back([&, i] {
            Task t;
            while (search_queue.pop(t)) handle_search(t, search_bufs[i]);
        });
    }

    insert_producer.join();
    search_producer.join();
    for (auto& w : workers) w.join();
    const auto end_time = Clock::now();
    double elapsed =
        std::chrono::duration<double>(end_time - start_time).count();
    run_done.store(true);
    if (progress_thread.joinable()) progress_thread.join();
    if (mem_thread.joinable()) mem_thread.join();

    // --- merge per-worker buffers ---
    std::vector<double> insert_op, insert_e2e, search_op, search_e2e;
    for (auto& b : insert_bufs) {
        insert_op.insert(insert_op.end(), b.op.begin(), b.op.end());
        insert_e2e.insert(insert_e2e.end(), b.e2e.begin(), b.e2e.end());
    }
    for (auto& b : search_bufs) {
        search_op.insert(search_op.end(), b.op.begin(), b.op.end());
        search_e2e.insert(search_e2e.end(), b.e2e.begin(), b.e2e.end());
        result.rec_offsets.insert(result.rec_offsets.end(),
                                  b.offsets.begin(), b.offsets.end());
        result.rec_query_tags.insert(result.rec_query_tags.end(),
                                     b.query_tags.begin(),
                                     b.query_tags.end());
        result.rec_result_tags.insert(result.rec_result_tags.end(),
                                      b.result_tags.begin(),
                                      b.result_tags.end());
    }

    // --- stats ---
    DriverStats& s = result.stats;
    double measured_elapsed = elapsed;
    if (cfg.warmup_points > 0) {
        if (!warmup_done.load()) {
            say("WARNING: warmup_points never reached; all samples "
                "discarded");
            measured_elapsed = 0;
        } else {
            measured_elapsed =
                std::chrono::duration<double>(end_time - warmup_end)
                    .count();
        }
    }
    s.elapsed_sec = elapsed;
    s.insert_points = cfg.warmup_points > 0 ? measured_insert_cnt.load()
                                            : insert_point_cnt.load();
    s.search_points = cfg.warmup_points > 0 ? measured_search_cnt.load()
                                            : search_point_cnt.load();
    double qps_window = cfg.warmup_points > 0 ? measured_elapsed : elapsed;
    if (s.insert_points > 0 && qps_window > 0) {
        s.insert_qps = static_cast<double>(s.insert_points) / qps_window;
        s.mean_insert_op = detail::mean(insert_op);
        s.p95_insert_op = detail::percentile(insert_op, 0.95);
        s.p99_insert_op = detail::percentile(insert_op, 0.99);
        s.mean_insert_e2e = detail::mean(insert_e2e);
        s.p95_insert_e2e = detail::percentile(insert_e2e, 0.95);
        s.p99_insert_e2e = detail::percentile(insert_e2e, 0.99);
    }
    if (s.search_points > 0 && qps_window > 0) {
        s.search_qps = static_cast<double>(s.search_points) / qps_window;
        s.mean_search_op = detail::mean(search_op);
        s.p95_search_op = detail::percentile(search_op, 0.95);
        s.p99_search_op = detail::percentile(search_op, 0.99);
        s.mean_search_e2e = detail::mean(search_e2e);
        s.p95_search_e2e = detail::percentile(search_e2e, 0.95);
        s.p99_search_e2e = detail::percentile(search_e2e, 0.99);
    }
    int64_t lc = lag_count.load();
    if (lc > 0) {
        s.search_lag_avg =
            static_cast<double>(lag_sum.load()) / static_cast<double>(lc);
    }
    s.search_lag_max = lag_max.load();
    if (mem_samples > 0) {
        s.peak_memory_mb =
            static_cast<double>(mem_peak) / (1024.0 * 1024.0);
        s.avg_memory_mb = mem_sum / static_cast<double>(mem_samples) /
                          (1024.0 * 1024.0);
    }
    return result;
}

}  // namespace driver
}  // namespace candor
