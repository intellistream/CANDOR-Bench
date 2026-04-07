/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <streamseed/StreamSeedCore.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/HNSWIncremental.h>

namespace faiss {
namespace streamseed {

bool OptimizationConfig::use_dictionary() const {
    return streamseed_mode == STREAMSEED_CORE && hint_table_slots > 0;
}

bool OptimizationConfig::streamseed_enabled() const {
    return streamseed_mode == STREAMSEED_CORE;
}

namespace {

using storage_idx_t = HNSWIncremental::storage_idx_t;

struct DictionarySeedSource : ISeedSource {
    std::vector<std::vector<idx_t>>& dictionary;
    std::vector<float>& dictionary_score;
    std::vector<uint64_t>& dictionary_age;
    std::vector<omp_lock_t>& dictionary_locks;
    uint64_t& dictionary_clock;

    DictionarySeedSource(
            std::vector<std::vector<idx_t>>& dictionary,
            std::vector<float>& dictionary_score,
            std::vector<uint64_t>& dictionary_age,
            std::vector<omp_lock_t>& dictionary_locks,
            uint64_t& dictionary_clock)
            : dictionary(dictionary),
              dictionary_score(dictionary_score),
              dictionary_age(dictionary_age),
              dictionary_locks(dictionary_locks),
              dictionary_clock(dictionary_clock) {}

    bool available(idx_t query_id) const override {
        if (dictionary.empty() || query_id < 0) {
            return false;
        }
        return true;
    }

    const std::vector<idx_t>& get(idx_t query_id) const override {
        return dictionary[query_id % dictionary.size()];
    }

    void writeback(const SeedWritebackRecord& record) override {
        if (!available(record.query_id) || record.k <= 0 || !record.idxi ||
            !record.simi || dictionary_locks.empty()) {
            return;
        }

        if (record.idxi[0] < 0 ||
            !std::isfinite(static_cast<double>(record.simi[0]))) {
            return;
        }

        const size_t slot =
                static_cast<size_t>(record.query_id) % dictionary.size();
        const float new_score = record.simi[0];
        constexpr uint64_t stale_window = 4096;

        uint64_t tick = 0;
#pragma omp atomic capture
        tick = ++dictionary_clock;

        omp_set_lock(&dictionary_locks[slot]);
        const bool empty_slot = dictionary[slot].empty();
        const float old_score = dictionary_score[slot];
        const uint64_t old_age = dictionary_age[slot];

        bool should_replace = empty_slot || new_score < old_score;
        if (!should_replace && tick > old_age + stale_window) {
            should_replace = true;
        }

        if (should_replace) {
            dictionary[slot].assign(record.idxi, record.idxi + record.k);
            dictionary_score[slot] = new_score;
            dictionary_age[slot] = tick;
        }
        omp_unset_lock(&dictionary_locks[slot]);
    }
};

struct NoopSeedSource : ISeedSource {
    mutable std::vector<idx_t> empty;

    bool available(idx_t query_id) const override {
        return false;
    }

    const std::vector<idx_t>& get(idx_t query_id) const override {
        return empty;
    }

    void writeback(const SeedWritebackRecord& record) override {}
};

struct StreamSeedCoreStrategy : IHintStrategy {
    StreamSeedCoreStrategy(int hint_hops, int hint_max_candidates, float hint_gate)
            : hint_hops(hint_hops),
              hint_max_candidates(hint_max_candidates),
              hint_gate(hint_gate) {}

    HintSearchResult apply(const HintSearchContext& ctx) const override {
        HintSearchResult result;
        if (ctx.cache_ids.empty()) {
            return result;
        }

        std::vector<storage_idx_t> frontier;
        std::vector<storage_idx_t> candidates;
        frontier.reserve(ctx.cache_ids.size());
        candidates.reserve(ctx.cache_ids.size() * 8);

        for (idx_t cached_id : ctx.cache_ids) {
            if (cached_id < 0) {
                continue;
            }
            storage_idx_t sid = static_cast<storage_idx_t>(cached_id);
            if (!ctx.vt.get(sid)) {
                ctx.vt.set(sid);
                result.touched = true;
                frontier.push_back(sid);
                candidates.push_back(sid);
                if (hint_max_candidates > 0 &&
                    candidates.size() >=
                            static_cast<size_t>(hint_max_candidates)) {
                    break;
                }
            }
        }

        if (!frontier.empty()) {
            std::vector<storage_idx_t> next_frontier;
            for (int level = 0; level < hint_hops; ++level) {
                next_frontier.clear();
                for (storage_idx_t u : frontier) {
                    size_t begin = 0, end = 0;
                    ctx.hnsw.neighbor_range(u, 0, &begin, &end);
                    for (size_t j = begin; j < end; j++) {
                        storage_idx_t v = ctx.hnsw.neighbors[j];
                        if (v < 0) {
                            break;
                        }
                        if (!ctx.vt.get(v)) {
                            ctx.vt.set(v);
                            result.touched = true;
                            next_frontier.push_back(v);
                            candidates.push_back(v);
                            if (hint_max_candidates > 0 &&
                                candidates.size() >= static_cast<size_t>(
                                                         hint_max_candidates)) {
                                break;
                            }
                        }
                    }
                    if (hint_max_candidates > 0 &&
                        candidates.size() >=
                                static_cast<size_t>(hint_max_candidates)) {
                        break;
                    }
                }
                if (next_frontier.empty()) {
                    break;
                }
                frontier.swap(next_frontier);
                if (hint_max_candidates > 0 &&
                    candidates.size() >=
                            static_cast<size_t>(hint_max_candidates)) {
                    break;
                }
            }
        }

        if (candidates.empty()) {
            return result;
        }

        std::vector<std::pair<float, storage_idx_t>> scored;
        scored.reserve(candidates.size());

        size_t j = 0;
        for (; j + 4 <= candidates.size(); j += 4) {
            float dis0 = 0.0f, dis1 = 0.0f, dis2 = 0.0f, dis3 = 0.0f;
            ctx.dis.distances_batch_4(
                    candidates[j],
                    candidates[j + 1],
                    candidates[j + 2],
                    candidates[j + 3],
                    dis0,
                    dis1,
                    dis2,
                    dis3);
            scored.emplace_back(dis0, candidates[j]);
            scored.emplace_back(dis1, candidates[j + 1]);
            scored.emplace_back(dis2, candidates[j + 2]);
            scored.emplace_back(dis3, candidates[j + 3]);
        }
        for (; j < candidates.size(); ++j) {
            scored.emplace_back(ctx.dis(candidates[j]), candidates[j]);
        }

        const size_t topn = std::min(static_cast<size_t>(ctx.k), scored.size());
        std::partial_sort(
                scored.begin(),
                scored.begin() + topn,
                scored.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

        if (topn > 0 && hint_gate >= 0.0f && scored[0].first > hint_gate) {
            return result;
        }

        for (idx_t j2 = 0; j2 < ctx.k; j2++) {
            if (static_cast<size_t>(j2) < topn) {
                ctx.simi[j2] = scored[j2].first;
                ctx.idxi[j2] = scored[j2].second;
            } else {
                ctx.simi[j2] = std::numeric_limits<float>::infinity();
                ctx.idxi[j2] = -1;
            }
        }

        result.used = true;
        return result;
    }

   private:
    int hint_hops;
    int hint_max_candidates;
    float hint_gate;
};

} // namespace

OptimizationConfig resolve_optimization_config(
        const HNSWIncremental& hnsw,
        const SearchParametersHNSWIncremental* params) {
    OptimizationConfig config;
    config.ef_search = hnsw.efSearch;
    if (!params) {
        return config;
    }

    config.ef_search = params->efSearch;
    config.streamseed_mode = params->streamseed_mode;
    config.hint_hops = params->hint_hops;
    config.hint_max_candidates = params->hint_max_candidates;
    config.hint_gate = params->hint_gate;
    config.hint_table_slots = params->hint_table_slots;

    if (config.streamseed_mode == STREAMSEED_CORE) {
        if (config.hint_hops <= 0) {
            config.hint_hops = 1;
        }
        if (config.hint_max_candidates <= 0) {
            config.hint_max_candidates = 256;
        }
        if (config.hint_table_slots <= 0) {
            config.hint_table_slots = 1024;
        }
    }

    return config;
}

std::unique_ptr<IHintStrategy> create_streamseed_strategy(
        const OptimizationConfig& config) {
    if (!config.streamseed_enabled()) {
        return nullptr;
    }
    return std::unique_ptr<IHintStrategy>(new StreamSeedCoreStrategy(
            std::max(1, config.hint_hops),
            config.hint_max_candidates,
            config.hint_gate));
}

void clear_dictionary_locks(std::vector<omp_lock_t>& locks) {
    for (size_t i = 0; i < locks.size(); ++i) {
        omp_destroy_lock(&locks[i]);
    }
    locks.clear();
}

void prepare_dictionary_if_needed(
        std::vector<std::vector<idx_t>>& warm_seed_dictionary,
        idx_t& warm_seed_dictionary_k,
        std::vector<float>& warm_seed_dictionary_score,
        std::vector<uint64_t>& warm_seed_dictionary_age,
        std::vector<omp_lock_t>& warm_seed_dictionary_locks,
        const OptimizationConfig& config,
        idx_t k) {
    if (!config.use_dictionary()) {
        return;
    }
    if (warm_seed_dictionary_k != k) {
        warm_seed_dictionary_k = k;
        warm_seed_dictionary.clear();
        warm_seed_dictionary_score.clear();
        warm_seed_dictionary_age.clear();
    }

    const size_t slots = static_cast<size_t>(config.hint_table_slots);
    if (warm_seed_dictionary.size() != slots) {
        clear_dictionary_locks(warm_seed_dictionary_locks);
        warm_seed_dictionary.assign(slots, {});
        warm_seed_dictionary_score.assign(
                slots,
                std::numeric_limits<float>::infinity());
        warm_seed_dictionary_age.assign(slots, 0);
        warm_seed_dictionary_locks.resize(slots);
        for (size_t i = 0; i < slots; ++i) {
            omp_init_lock(&warm_seed_dictionary_locks[i]);
        }
    }
    printf("Prepared seed dictionary with %zu slots for ef_search=%d\n",
           warm_seed_dictionary.size(),
           config.ef_search);
}

std::unique_ptr<ISeedSource> create_seed_source(
        const OptimizationConfig& config,
        std::vector<std::vector<idx_t>>& warm_seed_dictionary,
        std::vector<float>& warm_seed_dictionary_score,
        std::vector<uint64_t>& warm_seed_dictionary_age,
        std::vector<omp_lock_t>& warm_seed_dictionary_locks,
        uint64_t& warm_seed_dictionary_clock) {
    if (config.use_dictionary()) {
        return std::unique_ptr<ISeedSource>(
                new DictionarySeedSource(
                        warm_seed_dictionary,
                        warm_seed_dictionary_score,
                        warm_seed_dictionary_age,
                        warm_seed_dictionary_locks,
                        warm_seed_dictionary_clock));
    }
    return std::unique_ptr<ISeedSource>(new NoopSeedSource());
}

} // namespace streamseed
} // namespace faiss
