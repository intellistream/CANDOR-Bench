// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "common.h"
#include "../include/index.h"
#include "parameters.h"

namespace py = pybind11;

namespace diskannpy
{

template <typename DT>
class DynamicMemoryIndex
{
  public:
    DynamicMemoryIndex(diskann::AlgoType algo, diskann::Metric m, size_t dimensions, size_t max_vectors, uint32_t complexity,
                       uint32_t graph_degree, bool saturate_graph, uint32_t max_occlusion_size, float alpha,
                       uint32_t num_threads, uint32_t filter_complexity, uint32_t num_frozen_points,
                       uint32_t initial_search_complexity, uint32_t initial_search_threads,
                       bool concurrent_consolidation);

    void load(const std::string &index_path);
    int insert(const py::array_t<DT, py::array::c_style | py::array::forcecast> &vector, DynamicIdType id);
    py::array_t<int> batch_insert(py::array_t<DT, py::array::c_style | py::array::forcecast> &vectors,
                                  py::array_t<DynamicIdType, py::array::c_style | py::array::forcecast> &ids, int32_t num_inserts,
                                  int num_threads = 0);
    int mark_deleted(DynamicIdType id);
    void save(const std::string &save_path, bool compact_before_save = false);
    NeighborsAndDistances<DynamicIdType> search(py::array_t<DT, py::array::c_style | py::array::forcecast> &query, uint64_t knn,
                                      uint64_t complexity);
    NeighborsAndDistances<DynamicIdType> batch_search(py::array_t<DT, py::array::c_style | py::array::forcecast> &queries,
                                            uint64_t num_queries, uint64_t knn, uint64_t complexity,
                                            uint32_t num_threads);
    NeighborsAndDistances<DynamicIdType> batch_search_warm(
        py::array_t<DT, py::array::c_style | py::array::forcecast> &queries, uint64_t num_queries, uint64_t knn,
        uint64_t complexity, uint32_t num_threads, int streamseed_mode, int hint_level1_only,
        int hint_adaptive_gate_mode, int hint_hops, int hint_max_candidates, float hint_gate, float hint_qual_gate,
        float hint_cons_gate, float hint_gate_m_quantile, float hint_gate_o_quantile, int hint_gate_min_samples,
        int hint_table_slots, int hint_slot_capacity, py::object query_ids = py::none());
    void consolidate_delete();
    py::array_t<DynamicIdType> get_neighbors(DynamicIdType id);

  private:
    void ensure_hint_table(size_t slots, size_t capacity, size_t knn);
    void update_hint_slot(size_t slot, const DynamicIdType *labels, size_t count, int64_t query_id,
                          uint64_t owner_signature, bool hint_used);
    uint64_t compute_semantic_signature(const DT *query) const;
    size_t compute_semantic_slot_key(const DT *query, size_t hint_table_slots, int64_t fallback_query_id) const;
    static float signature_similarity(uint64_t a, uint64_t b);

    const size_t _dimensions;
    const uint32_t _initial_search_complexity;
    const diskann::IndexWriteParameters _write_parameters;
    diskann::Index<DT, DynamicIdType, filterT> _index;
    std::vector<DynamicIdType> _hint_table;
    std::vector<uint32_t> _hint_counts;
    std::vector<int64_t> _hint_owner_queries;
    std::vector<uint64_t> _hint_owner_signatures;
    std::vector<float> _hint_scores;
    std::vector<uint64_t> _hint_ages;
    size_t _hint_table_slots = 0;
    size_t _hint_slot_capacity = 0;
    size_t _hint_record_k = 0;
    uint64_t _hint_clock = 0;
    uint64_t _hint_round = 0;
};

}; // namespace diskannpy