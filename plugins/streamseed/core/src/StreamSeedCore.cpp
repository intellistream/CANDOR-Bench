/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <streamseed/StreamSeedCore.h>

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <limits>
#include <unordered_set>
#include <utility>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/HNSWIncremental.h>

namespace faiss {
namespace streamseed {

uint64_t compute_semantic_signature(const float* query, idx_t dim) {
    if (!query || dim <= 0) {
        return 0;
    }

    constexpr int sample_dims = 8;
    constexpr float quant_scale = 16.0f;
    uint64_t signature = 0;
    const uint64_t u_dim = static_cast<uint64_t>(dim);

    for (int t = 0; t < sample_dims; ++t) {
        const uint64_t pick = (static_cast<uint64_t>(t) * 9973ULL + 13ULL) % u_dim;
        const float v = query[static_cast<idx_t>(pick)];
        const int q = static_cast<int>(std::lrint(v * quant_scale));
        const int clamped = std::max(-128, std::min(127, q));
        const uint8_t packed = static_cast<uint8_t>(clamped + 128);
        signature |= static_cast<uint64_t>(packed) << (t * 8);
    }

    return signature;
}

idx_t compute_semantic_slot_key(
        const float* query,
        idx_t dim,
        int hint_table_slots,
        idx_t fallback_query_id) {
    if (!query || dim <= 0 || hint_table_slots <= 0) {
        return fallback_query_id;
    }

    const uint64_t signature = compute_semantic_signature(query, dim);
    uint64_t h = 1469598103934665603ULL;

    for (int t = 0; t < 8; ++t) {
        const uint64_t token =
                ((signature >> (t * 8)) & 0xFFULL) ^
                (static_cast<uint64_t>(t + 1) * 11400714819323198485ULL);
        h ^= token;
        h *= 1099511628211ULL;
    }

    return static_cast<idx_t>(h % static_cast<uint64_t>(hint_table_slots));
}

bool OptimizationConfig::use_dictionary() const {
    return streamseed_mode == STREAMSEED_CORE && hint_table_slots > 0;
}

bool OptimizationConfig::streamseed_enabled() const {
    return streamseed_mode == STREAMSEED_CORE;
}

namespace {

using storage_idx_t = HNSWIncremental::storage_idx_t;

constexpr int ADAPTIVE_GATE_OFF = 0;
constexpr int ADAPTIVE_GATE_BATCH = 1;

float quantile_inplace(std::vector<float>& values, float q) {
    if (values.empty()) {
        return 0.0f;
    }
    const float q_clamped = std::max(0.0f, std::min(1.0f, q));
    const size_t n = values.size();
    const size_t kth = static_cast<size_t>(q_clamped * static_cast<float>(n - 1));
    std::nth_element(values.begin(), values.begin() + kth, values.end());
    return values[kth];
}

float signature_similarity(uint64_t a, uint64_t b) {
    int l1 = 0;
    for (int t = 0; t < 8; ++t) {
        const int av = static_cast<int>((a >> (t * 8)) & 0xFFULL);
        const int bv = static_cast<int>((b >> (t * 8)) & 0xFFULL);
        l1 += std::abs(av - bv);
    }
    constexpr float max_l1 = 8.0f * 255.0f;
    return 1.0f - static_cast<float>(l1) / max_l1;
}

struct DictionarySeedSource : ISeedSource {
    std::vector<std::vector<idx_t>>& dictionary;
    std::vector<std::vector<idx_t>>& dictionary_owner_query;
    std::vector<std::vector<uint64_t>>& dictionary_owner_signature;
    std::vector<std::vector<float>>& dictionary_score;
    std::vector<std::vector<uint64_t>>& dictionary_age;
    std::vector<omp_lock_t>& dictionary_locks;
    uint64_t& dictionary_clock;
    uint64_t& dictionary_round;
    float& adaptive_m_gate;
    float& adaptive_o_gate;
    bool level1_only;
    int adaptive_gate_mode;
    float gate_m_quantile;
    float gate_o_quantile;
    int gate_min_samples;
    int slot_capacity;
    mutable std::vector<float> batch_m_samples;
    mutable std::vector<float> batch_o_samples;
    mutable size_t batch_level2_attempt = 0;
    mutable size_t batch_level2_pass = 0;
    mutable size_t batch_level2_block = 0;
    mutable omp_lock_t adaptive_lock;

    DictionarySeedSource(
            std::vector<std::vector<idx_t>>& dictionary,
                        std::vector<std::vector<idx_t>>& dictionary_owner_query,
                        std::vector<std::vector<uint64_t>>& dictionary_owner_signature,
                        std::vector<std::vector<float>>& dictionary_score,
                        std::vector<std::vector<uint64_t>>& dictionary_age,
            std::vector<omp_lock_t>& dictionary_locks,
                        uint64_t& dictionary_clock,
                        uint64_t& dictionary_round,
                        float& adaptive_m_gate,
                        float& adaptive_o_gate,
            bool level1_only,
                        int adaptive_gate_mode,
                        float gate_m_quantile,
                        float gate_o_quantile,
                        int gate_min_samples,
                        int slot_capacity)
            : dictionary(dictionary),
                            dictionary_owner_query(dictionary_owner_query),
                            dictionary_owner_signature(dictionary_owner_signature),
              dictionary_score(dictionary_score),
              dictionary_age(dictionary_age),
              dictionary_locks(dictionary_locks),
                            dictionary_clock(dictionary_clock),
              dictionary_round(dictionary_round),
              adaptive_m_gate(adaptive_m_gate),
              adaptive_o_gate(adaptive_o_gate),
              level1_only(level1_only),
              adaptive_gate_mode(adaptive_gate_mode),
              gate_m_quantile(gate_m_quantile),
              gate_o_quantile(gate_o_quantile),
              gate_min_samples(std::max(1, gate_min_samples)),
                            slot_capacity(std::max(1, slot_capacity)) {
        omp_init_lock(&adaptive_lock);
    }

    ~DictionarySeedSource() override {
        omp_destroy_lock(&adaptive_lock);
    }

    bool available(idx_t query_id, idx_t slot_key) const override {
        if (dictionary.empty() || query_id < 0 || slot_key < 0) {
            return false;
        }
        return true;
    }

    const std::vector<idx_t>& get(
            idx_t query_id,
            idx_t slot_key,
            const float* query,
            idx_t dim,
            bool* level1_hit) const override {
        static thread_local std::vector<idx_t> tls_matched_ids;
        tls_matched_ids.clear();
        if (level1_hit) {
            *level1_hit = false;
        }
        if (!available(query_id, slot_key)) {
            return tls_matched_ids;
        }

        const uint64_t query_signature = compute_semantic_signature(query, dim);

        const size_t slot = static_cast<size_t>(slot_key) % dictionary.size();
        omp_set_lock(&dictionary_locks[slot]);
        const std::vector<idx_t>& slot_ids = dictionary[slot];
        const std::vector<idx_t>& slot_owners = dictionary_owner_query[slot];
        const std::vector<uint64_t>& slot_signatures =
                dictionary_owner_signature[slot];
        const std::vector<float>& slot_counts = dictionary_score[slot];

        if (slot_owners.empty()) {
            omp_unset_lock(&dictionary_locks[slot]);
            return tls_matched_ids;
        }

        if (slot_ids.size() % slot_owners.size() != 0 ||
            slot_signatures.size() != slot_owners.size() ||
            slot_counts.size() != slot_owners.size()) {
            omp_unset_lock(&dictionary_locks[slot]);
            return tls_matched_ids;
        }

        const size_t record_len = slot_ids.size() / slot_owners.size();
        if (record_len == 0) {
            omp_unset_lock(&dictionary_locks[slot]);
            return tls_matched_ids;
        }

        for (size_t i = 0; i < slot_owners.size(); ++i) {
            if (slot_owners[i] == query_id) {
                const size_t begin = i * record_len;
                tls_matched_ids.assign(
                        slot_ids.begin() + begin,
                        slot_ids.begin() + begin + record_len);
                if (level1_hit) {
                    *level1_hit = true;
                }
                omp_unset_lock(&dictionary_locks[slot]);
                return tls_matched_ids;
            }
        }

        if (level1_only) {
            omp_unset_lock(&dictionary_locks[slot]);
            return tls_matched_ids;
        }

        // Cold-start protection: disable secondary strategy in early rounds.
        constexpr uint64_t warmup_rounds = 5;
        uint64_t current_round = 0;
#pragma omp atomic read
        current_round = dictionary_round;
        if (current_round <= warmup_rounds) {
            omp_unset_lock(&dictionary_locks[slot]);
            return tls_matched_ids;
        }

        float max_count = 1.0f;
        for (float c : slot_counts) {
            if (c > max_count) {
                max_count = c;
            }
        }

        size_t best_i = 0;
        float best_score = -std::numeric_limits<float>::infinity();
        constexpr float alpha = 0.7f;
        for (size_t i = 0; i < slot_owners.size(); ++i) {
            const float sim = signature_similarity(query_signature, slot_signatures[i]);
            const float cnt = slot_counts[i] / max_count;
            const float s = alpha * sim + (1.0f - alpha) * cnt;
            if (s > best_score) {
                best_score = s;
                best_i = i;
            }
        }

        const size_t begin = best_i * record_len;

        tls_matched_ids.assign(
                slot_ids.begin() + begin,
                slot_ids.begin() + begin + record_len);

        omp_unset_lock(&dictionary_locks[slot]);
        return tls_matched_ids;
    }

    void writeback(const SeedWritebackRecord& record) override {
        if (!available(record.query_id, record.slot_key) || record.k <= 0 || !record.idxi ||
            !record.simi || dictionary_locks.empty()) {
            return;
        }

        if (record.idxi[0] < 0 ||
            !std::isfinite(static_cast<double>(record.simi[0]))) {
            return;
        }

        const size_t slot =
            static_cast<size_t>(record.slot_key) % dictionary.size();
        const float new_count = record.hint_used ? 1.0f : 0.0f;
        constexpr uint64_t stale_window = 4096;
        constexpr float age_weight = 0.25f;

        uint64_t tick = 0;
#pragma omp atomic capture
        tick = ++dictionary_clock;

        if (record.query_id == 0) {
    #pragma omp atomic
            dictionary_round++;
        }

        omp_set_lock(&dictionary_locks[slot]);
        std::vector<idx_t>& slot_ids = dictionary[slot];
        std::vector<idx_t>& slot_owners = dictionary_owner_query[slot];
        std::vector<uint64_t>& slot_signatures = dictionary_owner_signature[slot];
        std::vector<float>& slot_scores = dictionary_score[slot];
        std::vector<uint64_t>& slot_ages = dictionary_age[slot];
        const size_t record_len = static_cast<size_t>(record.k);

        if (record_len == 0) {
            omp_unset_lock(&dictionary_locks[slot]);
            return;
        }

        if (slot_scores.size() != slot_ages.size() ||
            slot_scores.size() != slot_owners.size() ||
            slot_scores.size() != slot_signatures.size() ||
            slot_ids.size() != slot_scores.size() * record_len) {
            slot_ids.clear();
            slot_owners.clear();
            slot_signatures.clear();
            slot_scores.clear();
            slot_ages.clear();
        }

        for (size_t i = 0; i < slot_owners.size(); ++i) {
            if (slot_owners[i] == record.query_id) {
                const size_t begin = i * record_len;
                std::copy(record.idxi, record.idxi + record.k, slot_ids.begin() + begin);
                if (record.hint_used) {
                    slot_scores[i] += 1.0f;
                }
                slot_signatures[i] = record.owner_signature;
                slot_ages[i] = tick;
                omp_unset_lock(&dictionary_locks[slot]);
                return;
            }
        }

        const size_t capacity = static_cast<size_t>(slot_capacity);
        if (slot_scores.size() < capacity) {
            slot_ids.insert(slot_ids.end(), record.idxi, record.idxi + record.k);
            slot_owners.push_back(record.query_id);
            slot_signatures.push_back(record.owner_signature);
            slot_scores.push_back(new_count);
            slot_ages.push_back(tick);
            omp_unset_lock(&dictionary_locks[slot]);
            return;
        }

        size_t victim = 0;
        float victim_key = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < slot_scores.size(); ++i) {
            uint64_t age_delta = tick > slot_ages[i] ? tick - slot_ages[i] : 0;
            if (age_delta > stale_window) {
                age_delta = stale_window;
            }
            const float age_term =
                    static_cast<float>(age_delta) / static_cast<float>(stale_window);
            const float keep_key = slot_scores[i] - age_weight * age_term;
            if (keep_key < victim_key) {
                victim_key = keep_key;
                victim = i;
            }
        }

        const bool victim_stale = tick > slot_ages[victim] + stale_window;
        const bool should_replace = victim_stale ||
                (new_count > slot_scores[victim]) ||
                (slot_scores[victim] <= 0.0f);
        if (should_replace) {
            const size_t begin = victim * record_len;
            std::copy(record.idxi, record.idxi + record.k, slot_ids.begin() + begin);
            slot_owners[victim] = record.query_id;
            slot_signatures[victim] = record.owner_signature;
            slot_scores[victim] = new_count;
            slot_ages[victim] = tick;
        }
        omp_unset_lock(&dictionary_locks[slot]);
    }

    void on_batch_end(bool verbose, int64_t i0, int64_t i1) override {
        (void)verbose;
        (void)i0;
        (void)i1;
    }
};

struct NoopSeedSource : ISeedSource {
    mutable std::vector<idx_t> empty;

    bool available(idx_t query_id, idx_t slot_key) const override {
        return false;
    }

    const std::vector<idx_t>& get(
            idx_t query_id,
            idx_t slot_key,
            const float* query,
            idx_t dim,
            bool* level1_hit) const override {
        if (level1_hit) {
            *level1_hit = false;
        }
        return empty;
    }

    void writeback(const SeedWritebackRecord& record) override {}

    void on_batch_end(bool verbose, int64_t i0, int64_t i1) override {}
};

struct StreamSeedCoreStrategy : IHintStrategy {
        StreamSeedCoreStrategy(
                        int hint_hops,
                        int hint_max_candidates,
                        float hint_gate,
                        float hint_qual_gate,
                        float hint_cons_gate)
            : hint_hops(hint_hops),
              hint_max_candidates(hint_max_candidates),
                            hint_gate(hint_gate),
                            hint_qual_gate(hint_qual_gate),
                            hint_cons_gate(hint_cons_gate) {}

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
            int effective_hops = std::max(0, hint_hops);
            if (!ctx.level1_hit) {
                // Secondary path uses one extra hop for broader candidate recovery.
                effective_hops += 1;
            }
            for (int level = 0; level < effective_hops; ++level) {
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

        if (!ctx.level1_hit && hint_qual_gate >= 0.0f) {
            // Quality validation is only for secondary-selected seeds.
            const float eps = 1e-6f;
            float qual = std::numeric_limits<float>::infinity();
            if (topn >= 2) {
                const float c1 = scored[0].first;
                const float c2 = scored[1].first;
                qual = (c2 - c1) / (std::fabs(c1) + eps);
            }
            if (qual < hint_qual_gate) {
                result.rejected_by_qual = true;
                return result;
            }
        }

        if (hint_cons_gate >= 0.0f) {
            float cons = 0.0f;
            if (ctx.k > 0) {
                std::unordered_set<idx_t> seed_set;
                seed_set.reserve(ctx.cache_ids.size());
                for (idx_t sid : ctx.cache_ids) {
                    if (sid >= 0) {
                        seed_set.insert(sid);
                    }
                }
                size_t overlap = 0;
                for (size_t t = 0; t < topn; ++t) {
                    if (seed_set.find(scored[t].second) != seed_set.end()) {
                        overlap += 1;
                    }
                }
                cons = static_cast<float>(overlap) / static_cast<float>(ctx.k);
            }
            if (cons < hint_cons_gate) {
                result.rejected_by_cons = true;
                return result;
            }
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
    float hint_qual_gate;
    float hint_cons_gate;
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
    config.hint_level1_only = params->hint_level1_only;
    config.hint_adaptive_gate_mode = params->hint_adaptive_gate_mode;
    config.hint_hops = params->hint_hops;
    config.hint_max_candidates = params->hint_max_candidates;
    config.hint_gate = params->hint_gate;
    config.hint_qual_gate = params->hint_qual_gate;
    config.hint_cons_gate = params->hint_cons_gate;
    config.hint_gate_m_quantile = params->hint_gate_m_quantile;
    config.hint_gate_o_quantile = params->hint_gate_o_quantile;
    config.hint_gate_min_samples = params->hint_gate_min_samples;
    config.hint_table_slots = params->hint_table_slots;
    config.hint_slot_capacity = params->hint_slot_capacity;

    if (config.streamseed_mode == STREAMSEED_CORE) {
        if (config.hint_hops < 0) {
            config.hint_hops = 0;
        }
        if (config.hint_max_candidates <= 0) {
            config.hint_max_candidates = 256;
        }
        if (config.hint_table_slots <= 0) {
            config.hint_table_slots = 1024;
        }
        if (config.hint_slot_capacity <= 0) {
            config.hint_slot_capacity = 2;
        }
        if (config.hint_level1_only < 0) {
            config.hint_level1_only = 0;
        }
        if (config.hint_adaptive_gate_mode < ADAPTIVE_GATE_OFF ||
            config.hint_adaptive_gate_mode > ADAPTIVE_GATE_BATCH) {
            config.hint_adaptive_gate_mode = ADAPTIVE_GATE_OFF;
        }
        config.hint_gate_m_quantile =
                std::max(0.0f, std::min(1.0f, config.hint_gate_m_quantile));
        config.hint_gate_o_quantile =
                std::max(0.0f, std::min(1.0f, config.hint_gate_o_quantile));
        if (config.hint_cons_gate > 1.0f) {
            config.hint_cons_gate = 1.0f;
        }
        if (config.hint_gate_min_samples <= 0) {
            config.hint_gate_min_samples = 128;
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
            std::max(0, config.hint_hops),
            config.hint_max_candidates,
            config.hint_gate,
            config.hint_qual_gate,
            config.hint_cons_gate));
}

void clear_dictionary_locks(std::vector<omp_lock_t>& locks) {
    for (size_t i = 0; i < locks.size(); ++i) {
        omp_destroy_lock(&locks[i]);
    }
    locks.clear();
}

void prepare_dictionary_if_needed(
        std::vector<std::vector<idx_t>>& warm_seed_dictionary,
        std::vector<std::vector<idx_t>>& warm_seed_dictionary_owner_query,
        std::vector<std::vector<uint64_t>>& warm_seed_dictionary_owner_signature,
        idx_t& warm_seed_dictionary_k,
        std::vector<std::vector<float>>& warm_seed_dictionary_score,
        std::vector<std::vector<uint64_t>>& warm_seed_dictionary_age,
        std::vector<omp_lock_t>& warm_seed_dictionary_locks,
        const OptimizationConfig& config,
        idx_t k) {
    if (!config.use_dictionary()) {
        return;
    }
    if (warm_seed_dictionary_k != k) {
        warm_seed_dictionary_k = k;
        warm_seed_dictionary.clear();
        warm_seed_dictionary_owner_query.clear();
        warm_seed_dictionary_owner_signature.clear();
        warm_seed_dictionary_score.clear();
        warm_seed_dictionary_age.clear();
    }

    const size_t slots = static_cast<size_t>(config.hint_table_slots);
    if (warm_seed_dictionary.size() != slots) {
        clear_dictionary_locks(warm_seed_dictionary_locks);
        warm_seed_dictionary.assign(slots, {});
        warm_seed_dictionary_owner_query.assign(slots, {});
        warm_seed_dictionary_owner_signature.assign(slots, {});
        warm_seed_dictionary_score.assign(slots, {});
        warm_seed_dictionary_age.assign(slots, {});
        warm_seed_dictionary_locks.resize(slots);
        for (size_t i = 0; i < slots; ++i) {
            omp_init_lock(&warm_seed_dictionary_locks[i]);
        }
    }
    printf("Prepared seed dictionary with %zu slots (cap=%d) for ef_search=%d\n",
           warm_seed_dictionary.size(),
           config.hint_slot_capacity,
           config.ef_search);
}

std::unique_ptr<ISeedSource> create_seed_source(
        const OptimizationConfig& config,
        std::vector<std::vector<idx_t>>& warm_seed_dictionary,
    std::vector<std::vector<idx_t>>& warm_seed_dictionary_owner_query,
    std::vector<std::vector<uint64_t>>& warm_seed_dictionary_owner_signature,
    std::vector<std::vector<float>>& warm_seed_dictionary_score,
    std::vector<std::vector<uint64_t>>& warm_seed_dictionary_age,
        std::vector<omp_lock_t>& warm_seed_dictionary_locks,
    uint64_t& warm_seed_dictionary_clock,
    uint64_t& warm_seed_dictionary_round,
    float& warm_seed_adaptive_m_gate,
    float& warm_seed_adaptive_o_gate) {
    if (config.use_dictionary()) {
        return std::unique_ptr<ISeedSource>(
                new DictionarySeedSource(
                        warm_seed_dictionary,
            warm_seed_dictionary_owner_query,
            warm_seed_dictionary_owner_signature,
                        warm_seed_dictionary_score,
                        warm_seed_dictionary_age,
                        warm_seed_dictionary_locks,
            warm_seed_dictionary_clock,
                        warm_seed_dictionary_round,
                        warm_seed_adaptive_m_gate,
                        warm_seed_adaptive_o_gate,
                        config.hint_level1_only != 0,
                    config.hint_adaptive_gate_mode,
                    config.hint_gate_m_quantile,
                    config.hint_gate_o_quantile,
                    config.hint_gate_min_samples,
            config.hint_slot_capacity));
    }
    return std::unique_ptr<ISeedSource>(new NoopSeedSource());
}

} // namespace streamseed
} // namespace faiss
