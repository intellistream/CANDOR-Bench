#pragma once

// Single construction point for every index type. Both the Python module
// and the legacy C ABI build indexes through here, so constructor
// signatures live in exactly one file.

#include <memory>
#include <stdexcept>
#include <string>

#include "hnsw/hnsw.hpp"
#include "index.hpp"
#include "index_types.hpp"
#include "parlayann/parlay_hnsw.hpp"
#include "parlayann/parlay_vamana.hpp"
#include "segment/segment.hpp"
#include "vamana/vamana.hpp"

namespace candor {

inline IndexType index_type_from_name(const std::string& name) {
    // Canonical names follow the YAML configs; underscore forms are
    // accepted as aliases.
    if (name == "hnsw") return INDEX_TYPE_HNSW;
    if (name == "parlayhnsw" || name == "parlay_hnsw")
        return INDEX_TYPE_PARLAYHNSW;
    if (name == "parlayvamana" || name == "parlay_vamana")
        return INDEX_TYPE_PARLAYVAMANA;
    if (name == "vamana") return INDEX_TYPE_VAMANA;
    if (name == "segmented" || name == "segment") return INDEX_TYPE_SEGMENT;
    throw std::invalid_argument("unknown index type: " + name);
}

inline std::unique_ptr<IndexBase<float>> make_index(IndexType type,
                                                    const IndexParams& p) {
    if (p.data_type != DATA_TYPE_FLOAT) {
        throw std::invalid_argument("only float data is supported");
    }
    switch (type) {
        case INDEX_TYPE_HNSW:
            return std::make_unique<HNSW<float>>(
                p.max_elements, p.dim, p.num_threads, p.M, p.ef_construction,
                p.use_node_lock, p.worker_scheduler);
        case INDEX_TYPE_PARLAYHNSW:
            return std::make_unique<ParlayHNSW<float>>(
                p.max_elements, p.dim, p.num_threads, p.M, p.ef_construction,
                p.level_m, p.alpha, p.visit_limit);
        case INDEX_TYPE_PARLAYVAMANA:
            return std::make_unique<ParlayVamana<float>>(
                p.max_elements, p.dim, p.num_threads, p.M, p.ef_construction,
                p.alpha);
        case INDEX_TYPE_VAMANA:
            return std::make_unique<Vamana<float>>(p.max_elements, p.dim,
                                                   p.num_threads, p.M,
                                                   p.ef_construction, p.alpha);
        case INDEX_TYPE_SEGMENT:
            return std::make_unique<SegmentIndex>(p);
        default:
            throw std::invalid_argument("index type not constructible here");
    }
}

}  // namespace candor
