#include "segment.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <queue>
#include <sched.h>
#include <sstream>
#include <time.h>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
#include <immintrin.h>
#define HYBRID_HAS_X86_INTRIN 1
#else
#define HYBRID_HAS_X86_INTRIN 0
#endif

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "../hnsw/hnsw.hpp"
#include "../parlayann/parlay_hnsw.hpp"
#include "../parlayann/parlay_vamana.hpp"
#include "../vamana/vamana.hpp"

namespace {

bool read_env_bool(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
           value[0] == 'y' || value[0] == 'Y';
}

size_t read_env_size(const char* name, size_t default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || parsed == 0) {
        return default_value;
    }
    return static_cast<size_t>(parsed);
}

std::vector<int> read_env_cpu_list(const char* primary, const char* fallback) {
    const char* value = std::getenv(primary);
    if ((value == nullptr || value[0] == '\0') && fallback != nullptr) {
        value = std::getenv(fallback);
    }
    std::vector<int> cpus;
    if (value == nullptr || value[0] == '\0') {
        return cpus;
    }
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        auto dash = token.find('-');
        if (dash == std::string::npos) {
            char* end = nullptr;
            long cpu = std::strtol(token.c_str(), &end, 10);
            if (end != token.c_str() && cpu >= 0) {
                cpus.push_back(static_cast<int>(cpu));
            }
            continue;
        }
        const std::string begin_text = token.substr(0, dash);
        const std::string end_text = token.substr(dash + 1);
        char* begin_end = nullptr;
        char* finish_end = nullptr;
        long begin = std::strtol(begin_text.c_str(), &begin_end, 10);
        long finish = std::strtol(end_text.c_str(), &finish_end, 10);
        if (begin_end == begin_text.c_str() || finish_end == end_text.c_str()) {
            continue;
        }
        if (begin < 0 || finish < 0) {
            continue;
        }
        if (begin <= finish) {
            for (long cpu = begin; cpu <= finish; ++cpu) {
                cpus.push_back(static_cast<int>(cpu));
            }
        } else {
            for (long cpu = begin; cpu >= finish; --cpu) {
                cpus.push_back(static_cast<int>(cpu));
            }
        }
    }
    return cpus;
}

void pin_current_thread_to_env_cpu(const char* primary, const char* fallback,
                                   size_t ordinal) {
    const std::vector<int> cpus = read_env_cpu_list(primary, fallback);
    if (cpus.empty()) {
        return;
    }
    const int cpu = cpus[ordinal % cpus.size()];
    thread_local int current_cpu = -1;
    if (current_cpu == cpu) {
        return;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) == 0) {
        current_cpu = cpu;
    }
}

uint64_t elapsed_ns(const std::chrono::steady_clock::time_point& begin,
                    const std::chrono::steady_clock::time_point& end) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count());
}

uint64_t thread_cpu_elapsed_ns(const struct timespec& begin,
                               const struct timespec& end) {
    const int64_t ns = static_cast<int64_t>(end.tv_sec - begin.tv_sec) *
                           1000000000LL +
                       static_cast<int64_t>(end.tv_nsec - begin.tv_nsec);
    return ns > 0 ? static_cast<uint64_t>(ns) : 0;
}

}  // namespace

float l2_distance_simd(const float* a, const float* b, size_t dim) {
#if defined(__AVX512F__) && HYBRID_HAS_X86_INTRIN
    __m512 acc0 = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        acc0 = _mm512_fmadd_ps(diff, diff, acc0);
    }
    float sum = _mm512_reduce_add_ps(acc0);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
#elif defined(__AVX2__) && HYBRID_HAS_X86_INTRIN
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 diff0 = _mm256_sub_ps(va0, vb0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 diff1 = _mm256_sub_ps(va1, vb1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
    }
    if (i + 8 <= dim) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc0 = _mm256_fmadd_ps(diff, diff, acc0);
        i += 8;
    }
    alignas(32) float tmp0[8];
    alignas(32) float tmp1[8];
    _mm256_store_ps(tmp0, acc0);
    _mm256_store_ps(tmp1, acc1);
    float sum = 0.0f;
    for (int j = 0; j < 8; ++j) sum += tmp0[j] + tmp1[j];
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
#else
    float acc = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
#endif
}

SegmentIndex::SegmentIndex(const IndexParams& params)
    : params_(params),
      dim_(params.dim),
      max_elements_(params.max_elements),
      sealed_type_(resolve_sealed_type(params.sealed_index_type)),
      seal_threshold_(params.seal_threshold > 0 ? params.seal_threshold
                                               : 1000),
      builder_threads_(read_env_size("SEGMENT_BUILDER_THREADS",
                                     params.num_threads > 0
                                         ? params.num_threads
                                         : 1)),
      active_delta_hnsw_(
          read_env_bool("SEGMENT_ACTIVE_DELTA_HNSW", false)),
      search_arena_(read_env_size("SEGMENT_SEARCH_ARENA_THREADS",
                                  params.num_threads > 0 ? params.num_threads
                                                         : 1)),
      measured_search_arena_enabled_(
          read_env_bool("SEGMENT_MEASURED_SEARCH_ARENA", false)),
      measured_search_arena_(read_env_size(
          "SEGMENT_MEASURED_SEARCH_ARENA_THREADS",
          params.num_threads > 0 ? params.num_threads : 1)) {
    if (seal_threshold_ <= 0) {
        seal_threshold_ = 1000;
    }
    if (max_elements_ > 0 && dim_ > 0) {
        store_.resize(max_elements_ * dim_);
    }
    if (active_delta_hnsw_) {
        active_delta_ = create_active_delta_index();
    }
    builder_thread_ = std::thread(&SegmentIndex::builder_loop, this);
}

SegmentIndex::~SegmentIndex() {
    {
        std::lock_guard<std::mutex> guard(builder_mu_);
        stop_builder_ = true;
    }
    builder_cv_.notify_all();
    if (builder_thread_.joinable()) {
        builder_thread_.join();
    }
}

void SegmentIndex::build(const float* data, const uint32_t* tags,
                        size_t num_points) {
    cache_vectors(data, tags, num_points);
    auto sealed = build_segment(data, tags, num_points, params_.num_threads);
    if (sealed) {
        std::lock_guard<std::mutex> guard(mu_);
        sealed_segments_.push_back(std::move(sealed));
    }
}

int SegmentIndex::insert(const float* point, const uint32_t tag) {
    return batch_insert(point, &tag, 1);
}

int SegmentIndex::batch_insert(const float* batch_data,
                              const uint32_t* batch_tags,
                              size_t num_points) {
    if (num_points == 0) {
        return 0;
    }

    cache_vectors(batch_data, batch_tags, num_points);

    if (active_delta_hnsw_) {
        return batch_insert_active_delta(batch_data, batch_tags, num_points);
    }

    std::shared_ptr<std::vector<uint32_t>> seal_tags;

    {
        std::lock_guard<std::mutex> guard(mu_);
        for (size_t i = 0; i < num_points; ++i) {
            buffer_tags_.push_back(batch_tags[i]);
        }
        if (buffer_tags_.size() >= static_cast<size_t>(seal_threshold_)) {
            seal_tags = std::make_shared<std::vector<uint32_t>>();
            seal_tags->swap(buffer_tags_);
            raw_segments_.push_back(seal_tags);
            ++seal_jobs_enqueued_;
        }
    }

    if (seal_tags) {
        enqueue_seal(std::move(seal_tags));
    }

    return 0;
}

void SegmentIndex::set_query_params(const QParams& params) {
    std::lock_guard<std::mutex> guard(mu_);
    current_qparams_ = params;
    has_qparams_ = true;
    for (auto& seg : sealed_segments_) {
        if (seg) {
            seg->set_query_params(params);
        }
    }
    if (active_delta_) {
        active_delta_->set_query_params(params);
    }
}

int SegmentIndex::search(const float* query, size_t k,
                        std::vector<uint32_t>& result_tags) {
    if (k == 0) {
        return 0;
    }
    auto snap = take_snapshot();
    search_with_snapshot(query, k, snap, result_tags);
    return 0;
}

int SegmentIndex::batch_search(const float* batch_queries, size_t k,
                              size_t num_queries, uint32_t** batch_results, size_t* watermark_out,
                              size_t visible_ts) {
    (void)visible_ts;
    if (watermark_out) {
        *watermark_out = visible_ts;
    }
    return batch_search_impl(batch_queries, k, num_queries, batch_results,
                             nullptr, false);
}

int SegmentIndex::batch_search_measured_work(
    const float* batch_queries, size_t k, size_t num_queries,
    uint32_t** batch_results, SearchWorkStats* per_query_stats,
    size_t* watermark_out, size_t visible_ts) {
    (void)visible_ts;
    if (watermark_out) {
        *watermark_out = visible_ts;
    }
    return batch_search_impl(batch_queries, k, num_queries, batch_results,
                             per_query_stats, true);
}

void SegmentIndex::dump_stats(std::string& str) {
    size_t sealed_count = 0;
    size_t buffer_count = 0;
    size_t raw_count = 0;
    size_t raw_points = 0;
    size_t enqueued = 0;
    size_t completed = 0;
    {
        std::lock_guard<std::mutex> guard(mu_);
        sealed_count = sealed_segments_.size();
        buffer_count = buffer_tags_.size();
        raw_count = raw_segments_.size();
        if (active_delta_hnsw_) {
            buffer_count = active_delta_points_;
        }
        enqueued = seal_jobs_enqueued_;
        completed = seal_jobs_completed_;
        for (const auto& raw : raw_segments_) {
            if (raw) {
                raw_points += raw->size();
            }
        }
    }
    str = "sealed_segments:" + std::to_string(sealed_count) +
          ", raw_segments:" + std::to_string(raw_count) +
          ", raw_points:" + std::to_string(raw_points) +
          ", buffer_size:" + std::to_string(buffer_count) +
          ", seal_jobs_enqueued:" + std::to_string(enqueued) +
          ", seal_jobs_completed:" + std::to_string(completed) +
          ", segment_builder_threads:" + std::to_string(builder_threads_) +
          ", segment_active_delta_hnsw:" +
          std::to_string(active_delta_hnsw_ ? 1 : 0) +
          ", segment_active_delta_points:" +
          std::to_string(active_delta_points_) +
          ", segment_active_delta_rotations:" +
          std::to_string(active_delta_rotations_) +
          ", segment_search_queries:" +
          std::to_string(metric_search_queries_.load()) +
          ", segment_search_raw_points:" +
          std::to_string(metric_search_raw_points_.load()) +
          ", segment_search_buffer_points:" +
          std::to_string(metric_search_buffer_points_.load()) +
          ", segment_search_active_delta_points:" +
          std::to_string(metric_search_active_delta_points_.load()) +
          ", segment_search_sealed_result_rechecks:" +
          std::to_string(metric_search_sealed_result_rechecks_.load()) +
          ", segment_search_candidates:" +
          std::to_string(metric_search_candidates_.load()) +
          ", segment_search_sealed_segments:" +
          std::to_string(metric_search_sealed_segments_.load()) +
          ", segment_search_raw_segments:" +
          std::to_string(metric_search_raw_segments_.load()) +
          ", segment_search_sealed_search_ns:" +
          std::to_string(metric_search_sealed_search_ns_.load()) +
          ", segment_search_active_delta_search_ns:" +
          std::to_string(metric_search_active_delta_search_ns_.load()) +
          ", segment_search_exact_distance_ns:" +
          std::to_string(metric_search_exact_distance_ns_.load()) +
          ", segment_search_selection_ns:" +
          std::to_string(metric_search_selection_ns_.load());
}

IndexType SegmentIndex::resolve_sealed_type(IndexType requested) const {
    switch (requested) {
        case INDEX_TYPE_HNSW:
        case INDEX_TYPE_PARLAYHNSW:
        case INDEX_TYPE_PARLAYVAMANA:
        case INDEX_TYPE_VAMANA:
            return requested;
        default:
            return INDEX_TYPE_HNSW;
    }
}

const float* SegmentIndex::vector_for_tag(uint32_t tag) const {
    size_t offset = static_cast<size_t>(tag) * dim_;
    if (offset + dim_ <= store_.size()) {
        return store_.data() + offset;
    }
    return nullptr;
}

void SegmentIndex::cache_vectors(const float* data, const uint32_t* tags,
                                size_t num_points) {
    if (!data || !tags) return;
    std::lock_guard<std::mutex> guard(mu_);
    for (size_t i = 0; i < num_points; ++i) {
        size_t offset = static_cast<size_t>(tags[i]) * dim_;
        if (offset + dim_ > store_.size()) {
            continue;
        }
        float* dst = store_.data() + offset;
        const float* src = data + i * dim_;
        std::copy(src, src + dim_, dst);
    }
}

std::shared_ptr<IndexBase<float>> SegmentIndex::build_segment(
    const float* data, const uint32_t* tags, size_t num_points,
    size_t build_threads) {
    if (!data || !tags || num_points == 0) {
        return nullptr;
    }
    auto idx = create_base_index(num_points, build_threads, params_.use_node_lock);
    if (!idx) {
        return nullptr;
    }
    idx->build(data, tags, num_points);
    if (has_qparams_) {
        idx->set_query_params(current_qparams_);
    }
    return std::shared_ptr<IndexBase<float>>(std::move(idx));
}

std::shared_ptr<IndexBase<float>> SegmentIndex::build_segment_from_tags(
    const std::shared_ptr<std::vector<uint32_t>>& tags) {
    if (!tags || tags->empty()) {
        return nullptr;
    }

    std::vector<float> data(tags->size() * dim_);
    for (size_t i = 0; i < tags->size(); ++i) {
        const float* vec = vector_for_tag((*tags)[i]);
        if (!vec) {
            continue;
        }
        std::copy(vec, vec + dim_, data.data() + i * dim_);
    }

    QParams qparams;
    bool has_qparams = false;
    {
        std::lock_guard<std::mutex> guard(mu_);
        qparams = current_qparams_;
        has_qparams = has_qparams_;
    }

    auto idx = create_base_index(tags->size(), builder_threads_,
                                 params_.use_node_lock);
    if (!idx) {
        return nullptr;
    }
    idx->build(data.data(), tags->data(), tags->size());
    if (has_qparams) {
        idx->set_query_params(qparams);
    }
    return std::shared_ptr<IndexBase<float>>(std::move(idx));
}

std::unique_ptr<IndexBase<float>> SegmentIndex::create_base_index(
    size_t capacity, size_t threads, bool use_node_lock) const {
    size_t cap = std::max<size_t>(capacity, 1);
    size_t thread_count = std::max<size_t>(threads, 1);
    switch (sealed_type_) {
        case INDEX_TYPE_HNSW:
            return std::make_unique<HNSW<float>>(
                cap, dim_, thread_count, params_.M,
                params_.ef_construction, use_node_lock,
                params_.worker_scheduler);
        case INDEX_TYPE_PARLAYHNSW:
            return std::make_unique<ParlayHNSW<float>>(
                cap, dim_, thread_count, params_.M,
                params_.ef_construction, params_.level_m, params_.alpha,
                params_.visit_limit);
        case INDEX_TYPE_PARLAYVAMANA:
            return std::make_unique<ParlayVamana<float>>(
                cap, dim_, thread_count, params_.M,
                params_.ef_construction, params_.alpha);
        case INDEX_TYPE_VAMANA:
            return std::make_unique<Vamana<float>>(
                cap, dim_, thread_count, params_.M,
                params_.ef_construction, params_.alpha);
        default:
            return nullptr;
    }
}

std::shared_ptr<IndexBase<float>> SegmentIndex::create_active_delta_index() const {
    auto idx = create_base_index(static_cast<size_t>(seal_threshold_),
                                 builder_threads_, true);
    if (!idx) {
        return nullptr;
    }
    if (has_qparams_) {
        idx->set_query_params(current_qparams_);
    }
    return std::shared_ptr<IndexBase<float>>(std::move(idx));
}

void SegmentIndex::rotate_active_delta_locked() {
    if (!active_delta_hnsw_) {
        return;
    }
    if (active_delta_ && active_delta_points_ > 0) {
        sealed_segments_.push_back(active_delta_);
        ++active_delta_rotations_;
        ++seal_jobs_enqueued_;
        ++seal_jobs_completed_;
    }
    active_delta_ = create_active_delta_index();
    active_delta_points_ = 0;
}

int SegmentIndex::batch_insert_active_delta(const float* batch_data,
                                            const uint32_t* batch_tags,
                                            size_t num_points) {
    size_t offset = 0;
    while (offset < num_points) {
        std::lock_guard<std::mutex> insert_guard(active_delta_insert_mu_);
        std::shared_ptr<IndexBase<float>> target;
        size_t take = 0;
        {
            std::lock_guard<std::mutex> guard(mu_);
            if (!active_delta_) {
                active_delta_ = create_active_delta_index();
            }
            if (active_delta_points_ >= static_cast<size_t>(seal_threshold_)) {
                rotate_active_delta_locked();
            }
            const size_t remaining_capacity =
                static_cast<size_t>(seal_threshold_) - active_delta_points_;
            take = std::min(num_points - offset, remaining_capacity);
            target = active_delta_;
        }
        if (!target || take == 0) {
            return -1;
        }
        int rc = target->batch_insert(batch_data + offset * dim_,
                                      batch_tags + offset, take);
        if (rc != 0) {
            return rc;
        }
        {
            std::lock_guard<std::mutex> guard(mu_);
            active_delta_points_ += take;
            if (active_delta_points_ >= static_cast<size_t>(seal_threshold_)) {
                rotate_active_delta_locked();
            }
        }
        offset += take;
    }
    return 0;
}

void SegmentIndex::enqueue_seal(std::shared_ptr<std::vector<uint32_t>> tags) {
    {
        std::lock_guard<std::mutex> guard(builder_mu_);
        build_queue_.push_back(BuildJob{std::move(tags)});
    }
    builder_cv_.notify_one();
}

void SegmentIndex::builder_loop() {
    pin_current_thread_to_env_cpu("SEGMENT_BUILDER_CPU_LIST",
                                  "ANNCHOR_INSERT_CPU_LIST", 0);
    while (true) {
        BuildJob job;
        {
            std::unique_lock<std::mutex> lock(builder_mu_);
            builder_cv_.wait(lock, [&] {
                return stop_builder_ || !build_queue_.empty();
            });
            if (stop_builder_ && build_queue_.empty()) {
                return;
            }
            job = std::move(build_queue_.front());
            build_queue_.pop_front();
        }

        auto sealed = build_segment_from_tags(job.tags);
        if (!sealed) {
            continue;
        }

        std::lock_guard<std::mutex> guard(mu_);
        auto it = std::find(raw_segments_.begin(), raw_segments_.end(), job.tags);
        if (it != raw_segments_.end()) {
            raw_segments_.erase(it);
        }
        sealed_segments_.push_back(std::move(sealed));
        ++seal_jobs_completed_;
    }
}

SegmentIndex::Snapshot SegmentIndex::take_snapshot() const {
    std::lock_guard<std::mutex> guard(mu_);
    Snapshot snap;
    snap.buffer_tags = buffer_tags_;
    snap.raw_segments = raw_segments_;
    snap.sealed_segments = sealed_segments_;
    snap.active_delta = active_delta_;
    snap.active_delta_points = active_delta_points_;
    return snap;
}

void SegmentIndex::search_with_snapshot(const float* query, size_t k,
                                        const Snapshot& snap,
                                        std::vector<uint32_t>& result_tags,
                                        SegmentSearchStats* stats) const {
    using Candidate = std::pair<float, uint32_t>;
    std::vector<Candidate> candidates;
    candidates.reserve(k + snap.buffer_tags.size());

    // sealed segments
    if (stats) {
        stats->sealed_segments = snap.sealed_segments.size();
        stats->raw_segments = snap.raw_segments.size();
        stats->buffer_points = snap.buffer_tags.size();
        stats->active_delta_points = snap.active_delta_points;
    }
    auto search_ann_segment = [&](const std::shared_ptr<IndexBase<float>>& seg,
                                  bool active_delta) {
        if (!seg) return;
        std::vector<uint32_t> tmp;
        const auto begin = std::chrono::steady_clock::now();
        seg->search(query, k, tmp);
        const auto after_search = std::chrono::steady_clock::now();
        if (stats) {
            if (active_delta) {
                stats->active_delta_search_ns += elapsed_ns(begin, after_search);
            } else {
                stats->sealed_search_ns += elapsed_ns(begin, after_search);
            }
            stats->sealed_result_rechecks += tmp.size();
        }
        for (auto tag : tmp) {
            const float* vec = vector_for_tag(tag);
            if (!vec) continue;
            const auto dist_begin = std::chrono::steady_clock::now();
            candidates.emplace_back(l2_distance_simd(query, vec, dim_), tag);
            const auto dist_end = std::chrono::steady_clock::now();
            if (stats) {
                stats->exact_distance_ns += elapsed_ns(dist_begin, dist_end);
            }
        }
    };
    for (auto& seg : snap.sealed_segments) {
        search_ann_segment(seg, false);
    }
    if (snap.active_delta && snap.active_delta_points > 0) {
        search_ann_segment(snap.active_delta, true);
    }

    // growing buffer brute-force
    for (const auto& raw : snap.raw_segments) {
        if (!raw) continue;
        for (uint32_t tag : *raw) {
            const float* vec = vector_for_tag(tag);
            if (!vec) continue;
            const auto dist_begin = std::chrono::steady_clock::now();
            candidates.emplace_back(l2_distance_simd(query, vec, dim_), tag);
            const auto dist_end = std::chrono::steady_clock::now();
            if (stats) {
                ++stats->raw_points;
                stats->exact_distance_ns += elapsed_ns(dist_begin, dist_end);
            }
        }
    }

    for (uint32_t tag : snap.buffer_tags) {
        const float* vec = vector_for_tag(tag);
        if (!vec) continue;
        const auto dist_begin = std::chrono::steady_clock::now();
        candidates.emplace_back(l2_distance_simd(query, vec, dim_), tag);
        const auto dist_end = std::chrono::steady_clock::now();
        if (stats) {
            ++stats->raw_points;
            stats->exact_distance_ns += elapsed_ns(dist_begin, dist_end);
        }
    }
    if (stats) {
        stats->candidates = candidates.size();
    }

    const auto selection_begin = std::chrono::steady_clock::now();
    if (candidates.size() > k) {
        std::partial_sort(candidates.begin(), candidates.begin() + k,
                          candidates.end(),
                          [](const Candidate& a, const Candidate& b) {
                              return a.first < b.first;
                          });
        candidates.resize(k);
    } else {
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.first < b.first;
                  });
    }
    const auto selection_end = std::chrono::steady_clock::now();
    if (stats) {
        stats->selection_ns = elapsed_ns(selection_begin, selection_end);
    }

    result_tags.resize(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        result_tags[i] = candidates[i].second;
    }
}

tbb::task_arena& SegmentIndex::search_task_arena(bool measured) {
    return measured && measured_search_arena_enabled_ ? measured_search_arena_
                                                      : search_arena_;
}

int SegmentIndex::batch_search_impl(const float* batch_queries, size_t k,
                                    size_t num_queries,
                                    uint32_t** batch_results,
                                    SearchWorkStats* per_query_stats,
                                    bool measured) {
    if (num_queries == 0 || k == 0) {
        return 0;
    }
    auto snap = take_snapshot();
    search_task_arena(measured).execute([&] {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_queries),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    if (measured) {
                        pin_current_thread_to_env_cpu(
                            "SEGMENT_MEASURED_SEARCH_CPU_LIST",
                            "ANNCHOR_MEASURED_SEARCH_CPU_LIST", i);
                    } else {
                        pin_current_thread_to_env_cpu(
                            "SEGMENT_SEARCH_CPU_LIST",
                            "ANNCHOR_SEARCH_CPU_LIST", i);
                    }

                    std::vector<uint32_t> tmp;
                    SegmentSearchStats stats;
                    struct timespec cpu_start {};
                    if (per_query_stats) {
                        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_start);
                    }
                    const int start_cpu = per_query_stats ? sched_getcpu() : -1;
                    const auto begin = std::chrono::steady_clock::now();
                    search_with_snapshot(batch_queries + i * dim_, k, snap, tmp,
                                         per_query_stats ? &stats : nullptr);
                    const auto end = std::chrono::steady_clock::now();
                    const int end_cpu = per_query_stats ? sched_getcpu() : -1;
                    uint64_t thread_cpu_ns = 0;
                    if (per_query_stats) {
                        struct timespec cpu_end {};
                        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_end);
                        thread_cpu_ns =
                            thread_cpu_elapsed_ns(cpu_start, cpu_end);
                    }

                    size_t limit = std::min(k, tmp.size());
                    for (size_t j = 0; j < limit; ++j) {
                        batch_results[i][j] = tmp[j];
                    }

                    if (per_query_stats) {
                        auto& out = per_query_stats[i];
                        out.searchknn_ns = elapsed_ns(begin, end);
                        out.searchknn_thread_cpu_ns = thread_cpu_ns;
                        out.work_start_cpu = start_cpu;
                        out.work_end_cpu = end_cpu;
                        out.distance_computations =
                            stats.raw_points + stats.sealed_result_rechecks;
                        out.distance_compute_ns = stats.exact_distance_ns;
                        out.level0_edges_scanned = stats.raw_points;
                        out.level0_expansions =
                            stats.raw_segments + (stats.buffer_points > 0 ? 1 : 0);
                        out.candidate_pushes = stats.candidates;
                        out.result_pushes = limit;
                        out.upper_hops = stats.sealed_segments;
                        out.upper_distance_computations =
                            stats.sealed_result_rechecks;
                        out.level0_distance_computations = stats.raw_points;
                        out.upper_search_ns = stats.sealed_search_ns;
                        out.base_search_ns =
                            stats.exact_distance_ns + stats.selection_ns;
                    }

                    metric_search_queries_.fetch_add(1, std::memory_order_relaxed);
                    metric_search_raw_points_.fetch_add(stats.raw_points,
                                                        std::memory_order_relaxed);
                    metric_search_buffer_points_.fetch_add(stats.buffer_points,
                                                           std::memory_order_relaxed);
                    metric_search_active_delta_points_.fetch_add(
                        stats.active_delta_points, std::memory_order_relaxed);
                    metric_search_sealed_result_rechecks_.fetch_add(
                        stats.sealed_result_rechecks, std::memory_order_relaxed);
                    metric_search_candidates_.fetch_add(stats.candidates,
                                                        std::memory_order_relaxed);
                    metric_search_sealed_segments_.fetch_add(
                        stats.sealed_segments, std::memory_order_relaxed);
                    metric_search_raw_segments_.fetch_add(stats.raw_segments,
                                                          std::memory_order_relaxed);
                    metric_search_sealed_search_ns_.fetch_add(
                        stats.sealed_search_ns, std::memory_order_relaxed);
                    metric_search_active_delta_search_ns_.fetch_add(
                        stats.active_delta_search_ns, std::memory_order_relaxed);
                    metric_search_exact_distance_ns_.fetch_add(
                        stats.exact_distance_ns, std::memory_order_relaxed);
                    metric_search_selection_ns_.fetch_add(
                        stats.selection_ns, std::memory_order_relaxed);
                }
            });
    });
    return 0;
}
