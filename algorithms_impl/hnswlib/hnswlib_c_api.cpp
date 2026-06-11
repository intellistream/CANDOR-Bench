#include "hnswlib/hnswlib.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error;

enum MetricKind : int {
    METRIC_L2 = 0,
    METRIC_IP = 1,
};

struct HnswlibIndex {
    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    size_t dim = 0;
    MetricKind metric = METRIC_L2;
    std::mutex mutex;
    std::vector<hnswlib::labeltype> hint_table;
    std::vector<hnswlib::tableint> hint_internal_table;
    std::vector<uint32_t> hint_counts;
    std::vector<size_t> hint_owner_queries;
    size_t hint_table_slots = 0;
    size_t hint_slot_capacity = 0;

    HnswlibIndex(
        size_t dim_,
        size_t max_elements,
        size_t m,
        size_t ef_construction,
        size_t random_seed,
        bool allow_replace_deleted,
        MetricKind metric_)
        : dim(dim_), metric(metric_) {
        if (metric == METRIC_IP) {
            space.reset(new hnswlib::InnerProductSpace(dim));
        } else {
            space.reset(new hnswlib::L2Space(dim));
        }
        index.reset(new hnswlib::HierarchicalNSW<float>(
            space.get(),
            max_elements,
            m,
            ef_construction,
            random_seed,
            allow_replace_deleted));
    }
};

int set_error(const std::string& message) {
    g_last_error = message;
    return -1;
}

int set_error_from_exception(const std::exception& exc) {
    return set_error(exc.what());
}

HnswlibIndex* as_index(void* handle) {
    return reinterpret_cast<HnswlibIndex*>(handle);
}

void ensure_hint_table(HnswlibIndex* hi, size_t slots, size_t capacity) {
    if (slots == 0 || capacity == 0) {
        hi->hint_table.clear();
        hi->hint_internal_table.clear();
        hi->hint_counts.clear();
        hi->hint_owner_queries.clear();
        hi->hint_table_slots = 0;
        hi->hint_slot_capacity = 0;
        return;
    }
    if (hi->hint_table_slots != slots || hi->hint_slot_capacity != capacity) {
        hi->hint_table_slots = slots;
        hi->hint_slot_capacity = capacity;
        hi->hint_table.assign(slots * capacity, 0);
        hi->hint_internal_table.assign(slots * capacity, 0);
        hi->hint_counts.assign(slots, 0);
        hi->hint_owner_queries.assign(slots, std::numeric_limits<size_t>::max());
    }
}

template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t num_threads, Function fn) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    if (num_threads <= 1 || end <= start + 1) {
        for (size_t id = start; id < end; ++id) {
            fn(id, 0);
        }
        return;
    }

    std::vector<std::thread> threads;
    std::atomic<size_t> current(start);
    std::exception_ptr last_exception = nullptr;
    std::mutex exception_mutex;

    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        threads.emplace_back([&, thread_id] {
            while (true) {
                size_t id = current.fetch_add(1);
                if (id >= end) {
                    break;
                }
                try {
                    fn(id, thread_id);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    last_exception = std::current_exception();
                    current = end;
                    break;
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
    if (last_exception) {
        std::rethrow_exception(last_exception);
    }
}

void write_result(
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result,
    size_t out_offset,
    size_t k,
    uint64_t* out_labels,
    float* out_distances) {
    const uint64_t invalid_label = std::numeric_limits<uint64_t>::max();
    std::vector<std::pair<float, hnswlib::labeltype>> ordered;
    ordered.reserve(result.size());
    while (!result.empty()) {
        ordered.push_back(result.top());
        result.pop();
    }
    std::reverse(ordered.begin(), ordered.end());

    for (size_t j = 0; j < k; ++j) {
        const size_t out_pos = out_offset + j;
        if (j < ordered.size()) {
            out_distances[out_pos] = ordered[j].first;
            out_labels[out_pos] = static_cast<uint64_t>(ordered[j].second);
        } else {
            out_distances[out_pos] = std::numeric_limits<float>::infinity();
            out_labels[out_pos] = invalid_label;
        }
    }
}


bool label_to_internal(HnswlibIndex* hi, hnswlib::labeltype label, hnswlib::tableint& internal) {
    auto it = hi->index->label_lookup_.find(label);
    if (it == hi->index->label_lookup_.end()) {
        return false;
    }
    internal = it->second;
    if (internal >= hi->index->cur_element_count || hi->index->isMarkedDeleted(internal)) {
        return false;
    }
    return true;
}


void print_history_candidate_cover_histogram(
    HnswlibIndex* hi,
    size_t query_count,
    size_t k,
    const uint64_t* out_labels) {
    if (query_count == 0 || k == 0 || hi->hint_table_slots == 0 || hi->hint_slot_capacity == 0) {
        return;
    }

    const uint64_t invalid_label = std::numeric_limits<uint64_t>::max();
    std::vector<size_t> cover_hist(k + 1, 0);
    size_t comparable = 0;
    size_t no_history = 0;
    size_t total_candidate_size = 0;

    for (size_t qi = 0; qi < query_count; ++qi) {
        const size_t slot = qi % hi->hint_table_slots;
        const bool has_history =
            slot < hi->hint_counts.size() &&
            hi->hint_counts[slot] > 0 &&
            !hi->hint_owner_queries.empty() &&
            hi->hint_owner_queries[slot] == qi;
        if (!has_history) {
            ++no_history;
            continue;
        }

        const size_t seed_count = std::min<size_t>(hi->hint_counts[slot], hi->hint_slot_capacity);
        std::unordered_set<uint64_t> candidate_labels;
        candidate_labels.reserve(seed_count * 80 + 16);

        for (size_t i = 0; i < seed_count; ++i) {
            const size_t pos = slot * hi->hint_slot_capacity + i;
            const auto seed_label = hi->hint_table[pos];
            const hnswlib::tableint seed_internal = hi->hint_internal_table[pos];
            if (seed_internal >= hi->index->cur_element_count ||
                hi->index->isMarkedDeleted(seed_internal) ||
                hi->index->getExternalLabel(seed_internal) != seed_label) {
                continue;
            }

            candidate_labels.insert(static_cast<uint64_t>(seed_label));

            int* data = reinterpret_cast<int*>(hi->index->get_linklist0(seed_internal));
            const size_t size = hi->index->getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
            hnswlib::tableint* neighbors = reinterpret_cast<hnswlib::tableint*>(data + 1);
            for (size_t j = 0; j < size; ++j) {
                const hnswlib::tableint neighbor = neighbors[j];
                if (neighbor >= hi->index->cur_element_count || hi->index->isMarkedDeleted(neighbor)) {
                    continue;
                }
                candidate_labels.insert(static_cast<uint64_t>(hi->index->getExternalLabel(neighbor)));
            }
        }

        size_t covered = 0;
        for (size_t j = 0; j < k; ++j) {
            const uint64_t cur_label = out_labels[qi * k + j];
            if (cur_label != invalid_label && candidate_labels.find(cur_label) != candidate_labels.end()) {
                ++covered;
            }
        }
        if (covered > k) {
            covered = k;
        }
        ++cover_hist[covered];
        ++comparable;
        total_candidate_size += candidate_labels.size();
    }

    const double avg_candidate_size = comparable > 0
        ? static_cast<double>(total_candidate_size) / static_cast<double>(comparable)
        : 0.0;
    std::printf(
        "[hnswlib_streamseed] history_plus_1hop_cover batch_n=%zu comparable=%zu no_history=%zu avg_candidate_size=%.2f",
        query_count,
        comparable,
        no_history,
        avg_candidate_size);
    for (size_t cover = 0; cover <= k; ++cover) {
        const size_t pct_bin = (100 * cover + k / 2) / k;
        const double batch_pct = query_count > 0
            ? 100.0 * static_cast<double>(cover_hist[cover]) / static_cast<double>(query_count)
            : 0.0;
        std::printf(" %zu%%=%zu(%.2f%%)", pct_bin, cover_hist[cover], batch_pct);
    }
    std::printf("\n");
    std::fflush(stdout);
}


void print_history_overlap_histogram(
    HnswlibIndex* hi,
    size_t query_count,
    size_t k,
    const uint64_t* out_labels) {
    if (query_count == 0 || k == 0 || hi->hint_table_slots == 0 || hi->hint_slot_capacity == 0) {
        return;
    }

    const uint64_t invalid_label = std::numeric_limits<uint64_t>::max();
    std::vector<size_t> overlap_hist(k + 1, 0);
    size_t comparable = 0;
    size_t no_history = 0;

    for (size_t qi = 0; qi < query_count; ++qi) {
        const size_t slot = qi % hi->hint_table_slots;
        const bool has_history =
            slot < hi->hint_counts.size() &&
            hi->hint_counts[slot] > 0 &&
            !hi->hint_owner_queries.empty() &&
            hi->hint_owner_queries[slot] == qi;
        if (!has_history) {
            ++no_history;
            continue;
        }

        const size_t seed_count = std::min<size_t>(hi->hint_counts[slot], hi->hint_slot_capacity);
        size_t overlap = 0;
        for (size_t j = 0; j < k; ++j) {
            const uint64_t cur_label = out_labels[qi * k + j];
            if (cur_label == invalid_label) {
                continue;
            }
            bool found = false;
            for (size_t i = 0; i < seed_count; ++i) {
                const size_t pos = slot * hi->hint_slot_capacity + i;
                if (static_cast<uint64_t>(hi->hint_table[pos]) == cur_label) {
                    found = true;
                    break;
                }
            }
            if (found) {
                ++overlap;
            }
        }
        if (overlap > k) {
            overlap = k;
        }
        ++overlap_hist[overlap];
        ++comparable;
    }

    std::printf(
        "[hnswlib_streamseed] history_topk_overlap batch_n=%zu comparable=%zu no_history=%zu",
        query_count,
        comparable,
        no_history);
    for (size_t overlap = 0; overlap <= k; ++overlap) {
        const size_t pct_bin = (100 * overlap + k / 2) / k;
        const double batch_pct = query_count > 0
            ? 100.0 * static_cast<double>(overlap_hist[overlap]) / static_cast<double>(query_count)
            : 0.0;
        std::printf(" %zu%%=%zu(%.2f%%)", pct_bin, overlap_hist[overlap], batch_pct);
    }
    std::printf("\n");
    std::fflush(stdout);
}

void update_hint_slot(
    HnswlibIndex* hi,
    size_t slot,
    const uint64_t* labels,
    size_t count,
    size_t query_id) {
    if (hi->hint_table_slots == 0 || hi->hint_slot_capacity == 0 || slot >= hi->hint_table_slots) {
        return;
    }

    const uint64_t invalid_label = std::numeric_limits<uint64_t>::max();
    const size_t limit = std::min(count, hi->hint_slot_capacity);
    size_t store = 0;
    for (size_t i = 0; i < limit; ++i) {
        if (labels[i] == invalid_label) {
            continue;
        }
        hnswlib::tableint internal = 0;
        const auto label = static_cast<hnswlib::labeltype>(labels[i]);
        if (!label_to_internal(hi, label, internal)) {
            continue;
        }
        const size_t pos = slot * hi->hint_slot_capacity + store;
        hi->hint_table[pos] = label;
        hi->hint_internal_table[pos] = internal;
        ++store;
    }
    for (size_t i = store; i < hi->hint_slot_capacity; ++i) {
        const size_t pos = slot * hi->hint_slot_capacity + i;
        hi->hint_table[pos] = 0;
        hi->hint_internal_table[pos] = 0;
    }
    hi->hint_counts[slot] = static_cast<uint32_t>(store);
    hi->hint_owner_queries[slot] = query_id;
}



}  // namespace

extern "C" {

const char* hnswlib_last_error() {
    return g_last_error.c_str();
}

void* hnswlib_create(
    size_t dim,
    size_t max_elements,
    size_t m,
    size_t ef_construction,
    size_t random_seed,
    int allow_replace_deleted,
    int metric_kind) {
    try {
        if (dim == 0) {
            set_error("hnswlib dimension must be positive");
            return nullptr;
        }
        if (max_elements == 0) {
            set_error("hnswlib max_elements must be positive");
            return nullptr;
        }
        if (metric_kind != METRIC_L2 && metric_kind != METRIC_IP) {
            set_error("hnswlib metric_kind must be 0 (L2) or 1 (IP)");
            return nullptr;
        }
        g_last_error.clear();
        return new HnswlibIndex(
            dim,
            max_elements,
            m,
            ef_construction,
            random_seed,
            allow_replace_deleted != 0,
            static_cast<MetricKind>(metric_kind));
    } catch (const std::exception& exc) {
        set_error_from_exception(exc);
        return nullptr;
    } catch (...) {
        set_error("Unknown error in hnswlib_create");
        return nullptr;
    }
}

void hnswlib_destroy(void* handle) {
    delete as_index(handle);
}

int hnswlib_set_ef(void* handle, size_t ef) {
    try {
        HnswlibIndex* hi = as_index(handle);
        if (hi == nullptr) {
            return set_error("hnswlib_set_ef received null handle");
        }
        std::lock_guard<std::mutex> lock(hi->mutex);
        hi->index->setEf(ef);
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        return set_error_from_exception(exc);
    } catch (...) {
        return set_error("Unknown error in hnswlib_set_ef");
    }
}

int hnswlib_add_points(
    void* handle,
    const float* vectors,
    const uint64_t* labels,
    size_t count,
    size_t dim,
    int replace_deleted) {
    try {
        HnswlibIndex* hi = as_index(handle);
        if (hi == nullptr) {
            return set_error("hnswlib_add_points received null handle");
        }
        if ((vectors == nullptr || labels == nullptr) && count > 0) {
            return set_error("hnswlib_add_points received null pointer");
        }
        if (dim != hi->dim) {
            return set_error("hnswlib_add_points dimension mismatch");
        }
        std::lock_guard<std::mutex> lock(hi->mutex);
        for (size_t i = 0; i < count; ++i) {
            hi->index->addPoint(
                static_cast<const void*>(vectors + i * dim),
                static_cast<hnswlib::labeltype>(labels[i]),
                replace_deleted != 0);
        }
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        return set_error_from_exception(exc);
    } catch (...) {
        return set_error("Unknown error in hnswlib_add_points");
    }
}

int hnswlib_add_points_parallel(
    void* handle,
    const float* vectors,
    const uint64_t* labels,
    size_t count,
    size_t dim,
    int replace_deleted,
    int num_threads) {
    try {
        HnswlibIndex* hi = as_index(handle);
        if (hi == nullptr) {
            return set_error("hnswlib_add_points_parallel received null handle");
        }
        if ((vectors == nullptr || labels == nullptr) && count > 0) {
            return set_error("hnswlib_add_points_parallel received null pointer");
        }
        if (dim != hi->dim) {
            return set_error("hnswlib_add_points_parallel dimension mismatch");
        }
        std::lock_guard<std::mutex> lock(hi->mutex);
        ParallelFor(0, count, static_cast<size_t>(std::max(num_threads, 1)), [&](size_t i, size_t) {
            hi->index->addPoint(
                static_cast<const void*>(vectors + i * dim),
                static_cast<hnswlib::labeltype>(labels[i]),
                replace_deleted != 0);
        });
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        return set_error_from_exception(exc);
    } catch (...) {
        return set_error("Unknown error in hnswlib_add_points_parallel");
    }
}

int hnswlib_mark_delete(
    void* handle,
    const uint64_t* labels,
    size_t count) {
    try {
        HnswlibIndex* hi = as_index(handle);
        if (hi == nullptr) {
            return set_error("hnswlib_mark_delete received null handle");
        }
        if (labels == nullptr && count > 0) {
            return set_error("hnswlib_mark_delete received null labels pointer");
        }
        std::lock_guard<std::mutex> lock(hi->mutex);
        for (size_t i = 0; i < count; ++i) {
            hi->index->markDelete(static_cast<hnswlib::labeltype>(labels[i]));
        }
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        return set_error_from_exception(exc);
    } catch (...) {
        return set_error("Unknown error in hnswlib_mark_delete");
    }
}

int hnswlib_search(
    void* handle,
    const float* queries,
    size_t query_count,
    size_t dim,
    size_t k,
    int num_threads,
    uint64_t* out_labels,
    float* out_distances) {
    try {
        HnswlibIndex* hi = as_index(handle);
        if (hi == nullptr) {
            return set_error("hnswlib_search received null handle");
        }
        if ((queries == nullptr || out_labels == nullptr || out_distances == nullptr) && query_count > 0) {
            return set_error("hnswlib_search received null pointer");
        }
        if (dim != hi->dim) {
            return set_error("hnswlib_search dimension mismatch");
        }

        std::lock_guard<std::mutex> lock(hi->mutex);
        ParallelFor(0, query_count, static_cast<size_t>(std::max(num_threads, 1)), [&](size_t qi, size_t) {
            auto result = hi->index->searchKnn(
                static_cast<const void*>(queries + qi * dim),
                k);
            write_result(result, qi * k, k, out_labels, out_distances);
        });
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        return set_error_from_exception(exc);
    } catch (...) {
        return set_error("Unknown error in hnswlib_search");
    }
}

int hnswlib_search_warm(
    void* handle,
    const float* queries,
    size_t query_count,
    size_t dim,
    size_t k,
    int num_threads,
    int streamseed_mode,
    int hint_level1_only,
    int hint_adaptive_gate_mode,
    int hint_hops,
    int hint_max_candidates,
    float hint_gate,
    float hint_qual_gate,
    float hint_cons_gate,
    float hint_gate_m_quantile,
    float hint_gate_o_quantile,
    int hint_gate_min_samples,
    int hint_table_slots,
    int hint_slot_capacity,
    uint64_t* out_labels,
    float* out_distances) {
    try {
        (void)hint_level1_only;
        (void)hint_adaptive_gate_mode;
        (void)hint_gate_m_quantile;
        (void)hint_gate_o_quantile;
        (void)hint_gate_min_samples;

        HnswlibIndex* hi = as_index(handle);
        if (hi == nullptr) {
            return set_error("hnswlib_search_warm received null handle");
        }
        if ((queries == nullptr || out_labels == nullptr || out_distances == nullptr) && query_count > 0) {
            return set_error("hnswlib_search_warm received null pointer");
        }
        if (dim != hi->dim) {
            return set_error("hnswlib_search_warm dimension mismatch");
        }

        std::lock_guard<std::mutex> lock(hi->mutex);
        const bool use_hints = streamseed_mode != 0 && hint_table_slots > 0 && hint_slot_capacity > 0;
        if (use_hints) {
            ensure_hint_table(hi, static_cast<size_t>(hint_table_slots), static_cast<size_t>(hint_slot_capacity));
        } else {
            ensure_hint_table(hi, 0, 0);
        }

        ParallelFor(0, query_count, static_cast<size_t>(std::max(num_threads, 1)), [&](size_t qi, size_t thread_id) {
            (void)thread_id;
            const float* query = queries + qi * dim;
            const size_t out_offset = qi * k;
            const size_t slot = use_hints && hi->hint_table_slots > 0 ? qi % hi->hint_table_slots : 0;

            constexpr size_t stack_hint_capacity = 128;
            std::array<hnswlib::tableint, stack_hint_capacity> stack_hint_entries;
            std::vector<hnswlib::tableint> heap_hint_entries;
            hnswlib::tableint* hint_data = stack_hint_entries.data();
            size_t hint_count = 0;

            if (use_hints &&
                slot < hi->hint_counts.size() &&
                hi->hint_counts[slot] > 0 &&
                !hi->hint_owner_queries.empty() &&
                hi->hint_owner_queries[slot] == qi) {
                const size_t seed_count = std::min<size_t>(hi->hint_counts[slot], hi->hint_slot_capacity);
                if (seed_count > stack_hint_capacity) {
                    heap_hint_entries.reserve(seed_count);
                    hint_data = nullptr;
                }
                for (size_t i = 0; i < seed_count; ++i) {
                    const size_t pos = slot * hi->hint_slot_capacity + i;
                    const auto label = hi->hint_table[pos];
                    const hnswlib::tableint internal = hi->hint_internal_table[pos];
                    if (internal < hi->index->cur_element_count &&
                        !hi->index->isMarkedDeleted(internal) &&
                        hi->index->getExternalLabel(internal) == label) {
                        if (heap_hint_entries.capacity() > 0) {
                            heap_hint_entries.push_back(internal);
                        } else {
                            stack_hint_entries[hint_count] = internal;
                        }
                        ++hint_count;
                    }
                }
                if (!heap_hint_entries.empty()) {
                    hint_data = heap_hint_entries.data();
                }
            }

            auto result = hint_count == 0
                ? hi->index->searchKnn(static_cast<const void*>(query), k)
                : hi->index->searchKnnWarm(
                    static_cast<const void*>(query),
                    k,
                    hint_data,
                    hint_count,
                    hint_hops,
                    hint_max_candidates > 0 ? static_cast<size_t>(hint_max_candidates) : 256,
                    hint_gate,
                    hint_qual_gate,
                    hint_cons_gate,
                    true);

            write_result(result, out_offset, k, out_labels, out_distances);
        });
        if (use_hints) {
            print_history_overlap_histogram(hi, query_count, k, out_labels);
            print_history_candidate_cover_histogram(hi, query_count, k, out_labels);
            for (size_t qi = 0; qi < query_count; ++qi) {
                const size_t slot = hi->hint_table_slots > 0 ? qi % hi->hint_table_slots : 0;
                update_hint_slot(hi, slot, out_labels + qi * k, k, qi);
            }
        }
        g_last_error.clear();
        return 0;
    } catch (const std::exception& exc) {
        return set_error_from_exception(exc);
    } catch (...) {
        return set_error("Unknown error in hnswlib_search_warm");
    }
}

}  // extern "C"
