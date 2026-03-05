/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <omp.h>

#include <faiss/Index.h>

namespace faiss {

struct DistanceComputer;
struct VisitedTable;
struct HNSWIncremental;
struct SearchParametersHNSWIncremental;

namespace streamseed {

struct HintSearchResult {
    bool used = false;
    bool touched = false;
};

enum StreamSeedMode : int {
    STREAMSEED_OFF = 0,
    STREAMSEED_CORE = 1,
};

struct OptimizationConfig {
    int ef_search = 16;
    int streamseed_mode = STREAMSEED_CORE;
    int hint_hops = 1;
    int hint_max_candidates = 256;
    float hint_gate = -1.0f;
    int hint_table_slots = 1024;

    bool use_dictionary() const;
    bool streamseed_enabled() const;
};

struct HintSearchContext {
    idx_t k;
    idx_t* idxi;
    float* simi;
    const std::vector<idx_t>& cache_ids;
    VisitedTable& vt;
    DistanceComputer& dis;
    const HNSWIncremental& hnsw;
};

struct SeedWritebackRecord {
    idx_t query_id;
    idx_t k;
    const idx_t* idxi;
    const float* simi;
};

struct IHintStrategy {
    virtual ~IHintStrategy() = default;
    virtual HintSearchResult apply(const HintSearchContext& ctx) const = 0;
};

struct ISeedSource {
    virtual ~ISeedSource() = default;
    virtual bool available(idx_t query_id) const = 0;
    virtual const std::vector<idx_t>& get(idx_t query_id) const = 0;
    virtual void writeback(const SeedWritebackRecord& record) = 0;
};

OptimizationConfig resolve_optimization_config(
        const HNSWIncremental& hnsw,
        const SearchParametersHNSWIncremental* params);

std::unique_ptr<IHintStrategy> create_streamseed_strategy(
        const OptimizationConfig& config);

void clear_dictionary_locks(std::vector<omp_lock_t>& locks);

void prepare_dictionary_if_needed(
        std::vector<std::vector<idx_t>>& warm_seed_dictionary,
        idx_t& warm_seed_dictionary_k,
        std::vector<float>& warm_seed_dictionary_score,
        std::vector<uint64_t>& warm_seed_dictionary_age,
        std::vector<omp_lock_t>& warm_seed_dictionary_locks,
        const OptimizationConfig& config,
        idx_t k);

std::unique_ptr<ISeedSource> create_seed_source(
        const OptimizationConfig& config,
        std::vector<std::vector<idx_t>>& warm_seed_dictionary,
        std::vector<float>& warm_seed_dictionary_score,
        std::vector<uint64_t>& warm_seed_dictionary_age,
        std::vector<omp_lock_t>& warm_seed_dictionary_locks,
        uint64_t& warm_seed_dictionary_clock);

} // namespace streamseed
} // namespace faiss
