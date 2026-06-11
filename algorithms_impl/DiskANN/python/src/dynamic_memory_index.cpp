// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "parameters.h"
#include "dynamic_memory_index.h"

#include "pybind11/numpy.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace diskannpy
{

diskann::IndexWriteParameters dynamic_index_write_parameters(const uint32_t complexity, const uint32_t graph_degree,
                                                             const bool saturate_graph,
                                                             const uint32_t max_occlusion_size, const float alpha,
                                                             const uint32_t num_threads,
                                                             const uint32_t filter_complexity,
                                                             const uint32_t num_frozen_points)
{
    return diskann::IndexWriteParametersBuilder(complexity, graph_degree)
        .with_saturate_graph(saturate_graph)
        .with_max_occlusion_size(max_occlusion_size)
        .with_alpha(alpha)
        .with_num_threads(num_threads)
        .with_filter_list_size(filter_complexity)
        .with_num_frozen_points(num_frozen_points)
        .build();
}

template <class DT>
diskann::Index<DT, DynamicIdType, filterT> dynamic_index_builder(const diskann::Metric m,
                                                                 const diskann::IndexWriteParameters &write_params,
                                                                 const size_t dimensions, const size_t max_vectors,
                                                                 const uint32_t initial_search_complexity,
                                                                 const uint32_t initial_search_threads,
                                                                 const bool concurrent_consolidation)
{
    if(diskann::algo_type == diskann::AlgoType::CUFE) diskann::cout << "Farah is in dynamic_index_builder" << std::endl;
    const uint32_t _initial_search_threads =
        initial_search_threads != 0 ? initial_search_threads : omp_get_num_threads();
    return diskann::Index<DT, DynamicIdType, filterT>(
        m, dimensions, max_vectors,
        true,                      // dynamic_index
        write_params,              // used for insert
        initial_search_complexity, // used to prepare the scratch space for searching. can / may
                                   // be expanded if the search asks for a larger L.
        _initial_search_threads,   // also used for the scratch space
        true,                      // enable_tags
        concurrent_consolidation,
        false,  // pq_dist_build
        0,      // num_pq_chunks
        false); // use_opq = false
}

template <class DT>
DynamicMemoryIndex<DT>::DynamicMemoryIndex(const diskann::AlgoType algo,const diskann::Metric m, const size_t dimensions, const size_t max_vectors,
                                           const uint32_t complexity, const uint32_t graph_degree,
                                           const bool saturate_graph, const uint32_t max_occlusion_size,
                                           const float alpha, const uint32_t num_threads,
                                           const uint32_t filter_complexity, const uint32_t num_frozen_points,
                                           const uint32_t initial_search_complexity,
                                           const uint32_t initial_search_threads, const bool concurrent_consolidation)
    : _dimensions(dimensions),
      _initial_search_complexity(initial_search_complexity != 0 ? initial_search_complexity : complexity),
      _write_parameters(dynamic_index_write_parameters(complexity, graph_degree, saturate_graph, max_occlusion_size,
                                                       alpha, num_threads, filter_complexity, num_frozen_points)),
      _index(dynamic_index_builder<DT>(m, _write_parameters, dimensions, max_vectors, _initial_search_complexity,
                                       initial_search_threads, concurrent_consolidation))
{
    diskann::algo_type = algo;
}

template <class DT> void DynamicMemoryIndex<DT>::load(const std::string &index_path)
{
    const std::string tags_file = index_path + ".tags";
    if (!file_exists(tags_file))
    {
        throw std::runtime_error("tags file not found at expected path: " + tags_file);
    }
    _index.load(index_path.c_str(), _write_parameters.num_threads, _initial_search_complexity);
}

template <class DT>
int DynamicMemoryIndex<DT>::insert(const py::array_t<DT, py::array::c_style | py::array::forcecast> &vector,
                                   const DynamicIdType id)
{
    return _index.insert_point(vector.data(), id);
}

template <class DT>
py::array_t<int> DynamicMemoryIndex<DT>::batch_insert(
    py::array_t<DT, py::array::c_style | py::array::forcecast> &vectors,
    py::array_t<DynamicIdType, py::array::c_style | py::array::forcecast> &ids, const int32_t num_inserts,
    const int num_threads)
{
    if(diskann::algo_type == diskann::AlgoType::CUFE)diskann::cout << "Farah is in batch_insert" << std::endl;
    if (num_threads == 0)
        omp_set_num_threads(omp_get_num_procs());
    else
        omp_set_num_threads(num_threads);
    py::array_t<int> insert_retvals(num_inserts);

#pragma omp parallel for schedule(dynamic, 1) default(none) shared(num_inserts, insert_retvals, vectors, ids)
    for (int32_t i = 0; i < num_inserts; i++)
    {
        insert_retvals.mutable_data()[i] = _index.insert_point(vectors.data(i), *(ids.data(i)));
    }

    return insert_retvals;
}

template <class DT> int DynamicMemoryIndex<DT>::mark_deleted(const DynamicIdType id)
{
    return this->_index.lazy_delete(id);
}

template <class DT> void DynamicMemoryIndex<DT>::save(const std::string &save_path, const bool compact_before_save)
{
    if (save_path.empty())
    {
        throw std::runtime_error("A save_path must be provided");
    }
    _index.save(save_path.c_str(), compact_before_save);
}

template <class DT>
NeighborsAndDistances<DynamicIdType> DynamicMemoryIndex<DT>::search(
    py::array_t<DT, py::array::c_style | py::array::forcecast> &query, const uint64_t knn, const uint64_t complexity)
{
    py::array_t<DynamicIdType> ids(knn);
    py::array_t<float> dists(knn);
    std::vector<DT *> empty_vector;
    _index.search_with_tags(query.data(), knn, complexity, ids.mutable_data(), dists.mutable_data(), empty_vector);
    return std::make_pair(ids, dists);
}

template <class DT>
NeighborsAndDistances<DynamicIdType> DynamicMemoryIndex<DT>::batch_search(
    py::array_t<DT, py::array::c_style | py::array::forcecast> &queries, const uint64_t num_queries, const uint64_t knn,
    const uint64_t complexity, const uint32_t num_threads)
{
    py::array_t<DynamicIdType> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});
    std::vector<DT *> empty_vector;

    if (num_threads == 0)
        omp_set_num_threads(omp_get_num_procs());
    else
        omp_set_num_threads(static_cast<int32_t>(num_threads));

#pragma omp parallel for schedule(dynamic, 1) default(none)                                                            \
    shared(num_queries, queries, knn, complexity, ids, dists, empty_vector)
    for (int64_t i = 0; i < (int64_t)num_queries; i++)
    {
        _index.search_with_tags(queries.data(i), knn, complexity, ids.mutable_data(i), dists.mutable_data(i),
                                empty_vector);
    }

    return std::make_pair(ids, dists);
}

template <class DT> void DynamicMemoryIndex<DT>::ensure_hint_table(size_t slots, size_t capacity, size_t knn)
{
    if (slots == 0 || capacity == 0 || knn == 0)
    {
        _hint_table.clear();
        _hint_counts.clear();
        _hint_owner_queries.clear();
        _hint_owner_signatures.clear();
        _hint_scores.clear();
        _hint_ages.clear();
        _hint_table_slots = 0;
        _hint_slot_capacity = 0;
        _hint_record_k = 0;
        _hint_clock = 0;
        _hint_round = 0;
        return;
    }
    if (_hint_table_slots == slots && _hint_slot_capacity == capacity && _hint_record_k == knn)
    {
        return;
    }
    _hint_table_slots = slots;
    _hint_slot_capacity = capacity;
    _hint_record_k = knn;
    const size_t records = slots * capacity;
    _hint_table.assign(records * knn, 0);
    _hint_counts.assign(slots, 0);
    _hint_owner_queries.assign(records, -1);
    _hint_owner_signatures.assign(records, 0);
    _hint_scores.assign(records, 0.0f);
    _hint_ages.assign(records, 0);
    _hint_clock = 0;
    _hint_round = 0;
}

template <class DT>
void DynamicMemoryIndex<DT>::update_hint_slot(size_t slot, const DynamicIdType *labels, size_t count,
                                              int64_t query_id, uint64_t owner_signature, bool hint_used)
{
    if (_hint_table_slots == 0 || _hint_slot_capacity == 0 || _hint_record_k == 0 ||
        slot >= _hint_table_slots || labels == nullptr)
    {
        return;
    }
    if (count == 0 || labels[0] == 0)
    {
        return;
    }

    const uint64_t tick = ++_hint_clock;
    if (query_id == 0)
    {
        ++_hint_round;
    }

    uint32_t record_count = std::min<uint32_t>(_hint_counts[slot], static_cast<uint32_t>(_hint_slot_capacity));
    size_t target = static_cast<size_t>(-1);
    for (uint32_t r = 0; r < record_count; ++r)
    {
        const size_t rec = slot * _hint_slot_capacity + r;
        if (_hint_owner_queries[rec] == query_id)
        {
            target = rec;
            break;
        }
    }

    const float new_score = hint_used ? 1.0f : 0.0f;
    if (target == static_cast<size_t>(-1))
    {
        if (record_count < _hint_slot_capacity)
        {
            target = slot * _hint_slot_capacity + record_count;
            _hint_counts[slot] = record_count + 1;
        }
        else
        {
            constexpr uint64_t stale_window = 4096;
            constexpr float age_weight = 0.25f;
            float victim_key = std::numeric_limits<float>::infinity();
            size_t victim = slot * _hint_slot_capacity;
            for (uint32_t r = 0; r < record_count; ++r)
            {
                const size_t rec = slot * _hint_slot_capacity + r;
                uint64_t age_delta = tick > _hint_ages[rec] ? tick - _hint_ages[rec] : 0;
                if (age_delta > stale_window)
                {
                    age_delta = stale_window;
                }
                const float age_term = static_cast<float>(age_delta) / static_cast<float>(stale_window);
                const float keep_key = _hint_scores[rec] - age_weight * age_term;
                if (keep_key < victim_key)
                {
                    victim_key = keep_key;
                    victim = rec;
                }
            }
            const bool victim_stale = tick > _hint_ages[victim] + stale_window;
            const bool should_replace = victim_stale || new_score > _hint_scores[victim] || _hint_scores[victim] <= 0.0f;
            if (should_replace)
            {
                target = victim;
            }
        }
    }

    if (target == static_cast<size_t>(-1))
    {
        return;
    }

    const size_t begin = target * _hint_record_k;
    for (size_t i = 0; i < _hint_record_k; ++i)
    {
        _hint_table[begin + i] = i < count ? labels[i] : 0;
    }
    if (_hint_owner_queries[target] == query_id && hint_used)
    {
        _hint_scores[target] += 1.0f;
    }
    else
    {
        _hint_scores[target] = new_score;
    }
    _hint_owner_queries[target] = query_id;
    _hint_owner_signatures[target] = owner_signature;
    _hint_ages[target] = tick;
}

template <class DT> uint64_t DynamicMemoryIndex<DT>::compute_semantic_signature(const DT *query) const
{
    if (query == nullptr)
    {
        return 0;
    }
    constexpr int sample_dims = 8;
    constexpr float quant_scale = 16.0f;
    uint64_t signature = 0;
    const size_t dims = _dimensions;
    if (dims == 0)
    {
        return 0;
    }
    for (int t = 0; t < sample_dims; ++t)
    {
        const size_t pick = (static_cast<size_t>(t) * 9973ULL + 13ULL) % dims;
        const float v = static_cast<float>(query[pick]);
        const int q = static_cast<int>(std::lrint(v * quant_scale));
        const int clamped = std::max(-128, std::min(127, q));
        const uint8_t packed = static_cast<uint8_t>(clamped + 128);
        signature |= static_cast<uint64_t>(packed) << (t * 8);
    }
    return signature;
}

template <class DT>
size_t DynamicMemoryIndex<DT>::compute_semantic_slot_key(const DT *query, size_t hint_table_slots,
                                                         int64_t fallback_query_id) const
{
    if (hint_table_slots == 0)
    {
        return 0;
    }
    if (query == nullptr)
    {
        return fallback_query_id < 0 ? 0 : static_cast<size_t>(fallback_query_id) % hint_table_slots;
    }
    const uint64_t signature = compute_semantic_signature(query);
    uint64_t h = 1469598103934665603ULL;
    for (int t = 0; t < 8; ++t)
    {
        const uint64_t token = ((signature >> (t * 8)) & 0xFFULL) ^
                               (static_cast<uint64_t>(t + 1) * 11400714819323198485ULL);
        h ^= token;
        h *= 1099511628211ULL;
    }
    return static_cast<size_t>(h % hint_table_slots);
}

template <class DT> float DynamicMemoryIndex<DT>::signature_similarity(uint64_t a, uint64_t b)
{
    int l1 = 0;
    for (int t = 0; t < 8; ++t)
    {
        const int av = static_cast<int>((a >> (t * 8)) & 0xFFULL);
        const int bv = static_cast<int>((b >> (t * 8)) & 0xFFULL);
        l1 += std::abs(av - bv);
    }
    constexpr float max_l1 = 8.0f * 255.0f;
    return 1.0f - static_cast<float>(l1) / max_l1;
}

template <class DT>
NeighborsAndDistances<DynamicIdType> DynamicMemoryIndex<DT>::batch_search_warm(
    py::array_t<DT, py::array::c_style | py::array::forcecast> &queries, const uint64_t num_queries,
    const uint64_t knn, const uint64_t complexity, const uint32_t num_threads, const int streamseed_mode,
    const int hint_level1_only, const int hint_adaptive_gate_mode, const int hint_hops,
    const int hint_max_candidates, const float hint_gate, const float hint_qual_gate, const float hint_cons_gate,
    const float hint_gate_m_quantile, const float hint_gate_o_quantile, const int hint_gate_min_samples,
    const int hint_table_slots, const int hint_slot_capacity, py::object query_ids)
{
    (void)hint_adaptive_gate_mode;
    (void)hint_gate_m_quantile;
    (void)hint_gate_o_quantile;
    (void)hint_gate_min_samples;

    py::array_t<DynamicIdType> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});
    std::vector<DT *> empty_vector;

    py::array_t<int64_t, py::array::c_style | py::array::forcecast> query_ids_array;
    const int64_t *query_id_ptr = nullptr;
    if (!query_ids.is_none())
    {
        query_ids_array = py::array_t<int64_t, py::array::c_style | py::array::forcecast>(query_ids);
        auto id_buf = query_ids_array.request();
        if (id_buf.ndim != 1 || static_cast<uint64_t>(id_buf.shape[0]) != num_queries)
        {
            throw std::runtime_error("query_ids must be a 1D int64 array with length num_queries");
        }
        query_id_ptr = static_cast<const int64_t *>(id_buf.ptr);
    }

    const bool use_hints = streamseed_mode != 0 && hint_table_slots > 0 && hint_slot_capacity > 0 && knn > 0;
    if (use_hints)
    {
        ensure_hint_table(static_cast<size_t>(hint_table_slots), static_cast<size_t>(hint_slot_capacity),
                          static_cast<size_t>(knn));
    }
    else
    {
        ensure_hint_table(0, 0, 0);
    }

    std::vector<uint8_t> hint_used_flags(num_queries, 0);
    std::vector<size_t> query_slots(num_queries, 0);
    std::vector<uint64_t> query_signatures(num_queries, 0);
    std::vector<int64_t> owner_queries(num_queries, 0);

    for (uint64_t i = 0; i < num_queries; ++i)
    {
        owner_queries[i] = query_id_ptr ? query_id_ptr[i] : static_cast<int64_t>(i);
        if (use_hints && _hint_table_slots > 0)
        {
            query_signatures[i] = compute_semantic_signature(queries.data(i));
            query_slots[i] = compute_semantic_slot_key(queries.data(i), _hint_table_slots, owner_queries[i]);
        }
    }

    if (num_threads == 0)
        omp_set_num_threads(omp_get_num_procs());
    else
        omp_set_num_threads(static_cast<int32_t>(num_threads));

#pragma omp parallel for schedule(dynamic, 1) default(none)                                                            \
    shared(num_queries, queries, knn, complexity, ids, dists, empty_vector, use_hints, hint_level1_only, hint_hops,     \
           hint_max_candidates, hint_gate, hint_qual_gate, hint_cons_gate, hint_used_flags, query_slots,               \
           query_signatures, owner_queries)
    for (int64_t i = 0; i < (int64_t)num_queries; i++)
    {
        size_t found = 0;
        bool rejected = false;
        bool level1_hit = false;
        std::vector<DynamicIdType> seeds;

        if (use_hints && _hint_table_slots > 0 && _hint_record_k == knn)
        {
            const size_t slot = query_slots[static_cast<size_t>(i)];
            const uint32_t record_count = std::min<uint32_t>(_hint_counts[slot], static_cast<uint32_t>(_hint_slot_capacity));
            size_t selected_record = static_cast<size_t>(-1);
            const int64_t owner_query_id = owner_queries[static_cast<size_t>(i)];

            for (uint32_t r = 0; r < record_count; ++r)
            {
                const size_t rec = slot * _hint_slot_capacity + r;
                if (_hint_owner_queries[rec] == owner_query_id)
                {
                    selected_record = rec;
                    level1_hit = true;
                    break;
                }
            }

            if (selected_record == static_cast<size_t>(-1) && hint_level1_only == 0 && _hint_round > 5 && record_count > 0)
            {
                float max_score = 1.0f;
                for (uint32_t r = 0; r < record_count; ++r)
                {
                    const size_t rec = slot * _hint_slot_capacity + r;
                    if (_hint_scores[rec] > max_score)
                    {
                        max_score = _hint_scores[rec];
                    }
                }
                float best_score = -std::numeric_limits<float>::infinity();
                size_t best_record = static_cast<size_t>(-1);
                constexpr float alpha = 0.7f;
                for (uint32_t r = 0; r < record_count; ++r)
                {
                    const size_t rec = slot * _hint_slot_capacity + r;
                    if (_hint_owner_queries[rec] < 0)
                    {
                        continue;
                    }
                    const float sim = signature_similarity(query_signatures[static_cast<size_t>(i)], _hint_owner_signatures[rec]);
                    const float cnt = _hint_scores[rec] / max_score;
                    const float score = alpha * sim + (1.0f - alpha) * cnt;
                    if (score > best_score)
                    {
                        best_score = score;
                        best_record = rec;
                    }
                }
                selected_record = best_record;
            }

            if (selected_record != static_cast<size_t>(-1))
            {
                const size_t begin = selected_record * _hint_record_k;
                seeds.reserve(_hint_record_k);
                for (size_t j = 0; j < _hint_record_k; ++j)
                {
                    const DynamicIdType seed = _hint_table[begin + j];
                    if (seed != 0)
                    {
                        seeds.push_back(seed);
                    }
                }
            }
        }

        if (!seeds.empty())
        {
            const uint32_t hops = static_cast<uint32_t>(std::max(0, hint_hops) + (level1_hit ? 0 : 1));
            const uint32_t max_candidates = hint_max_candidates > 0 ? static_cast<uint32_t>(hint_max_candidates) : 256U;
            found = _index.search_with_tags_warm(queries.data(i), knn, static_cast<uint32_t>(complexity), seeds,
                                                 hops, max_candidates, ids.mutable_data(i), dists.mutable_data(i));
            if (found >= knn && hint_gate >= 0.0f && dists.data(i)[0] > hint_gate)
            {
                rejected = true;
            }
            if (found >= knn && !level1_hit && hint_qual_gate >= 0.0f)
            {
                const float eps = 1e-6f;
                float qual = std::numeric_limits<float>::infinity();
                if (knn >= 2)
                {
                    const float c1 = dists.data(i)[0];
                    const float c2 = dists.data(i)[1];
                    qual = (c2 - c1) / (std::fabs(c1) + eps);
                }
                if (qual < hint_qual_gate)
                {
                    rejected = true;
                }
            }
            if (found >= knn && hint_cons_gate >= 0.0f)
            {
                const float gate = std::min(hint_cons_gate, 1.0f);
                size_t overlap = 0;
                for (size_t r = 0; r < knn; ++r)
                {
                    const DynamicIdType label = ids.data(i)[r];
                    if (label == 0)
                    {
                        continue;
                    }
                    for (const auto seed : seeds)
                    {
                        if (label == seed)
                        {
                            ++overlap;
                            break;
                        }
                    }
                }
                const float cons = knn > 0 ? static_cast<float>(overlap) / static_cast<float>(knn) : 1.0f;
                if (cons < gate)
                {
                    rejected = true;
                }
            }
        }

        if (found < knn || rejected)
        {
            _index.search_with_tags(queries.data(i), knn, complexity, ids.mutable_data(i), dists.mutable_data(i),
                                    empty_vector);
        }
        else
        {
            hint_used_flags[static_cast<size_t>(i)] = 1;
        }
    }

    if (use_hints)
    {
        for (uint64_t i = 0; i < num_queries; ++i)
        {
            update_hint_slot(query_slots[i], ids.data(i), knn, owner_queries[i], query_signatures[i],
                             hint_used_flags[i] != 0);
        }
    }
    return std::make_pair(ids, dists);
}

template <class DT> void DynamicMemoryIndex<DT>::consolidate_delete()
{
    _index.consolidate_deletes(_write_parameters);
}

template <class DT> py::array_t<DynamicIdType> DynamicMemoryIndex<DT>::get_neighbors(DynamicIdType id)
{
    std::vector<DynamicIdType> neighbours;
    _index.get_neighbours_by_tag(id, neighbours);
    py::array_t<DynamicIdType> result(neighbours.size());
    if (!neighbours.empty())
    {
        std::copy(neighbours.begin(), neighbours.end(), result.mutable_data());
    }
    return result;
}

template class DynamicMemoryIndex<float>;
template class DynamicMemoryIndex<uint8_t>;
template class DynamicMemoryIndex<int8_t>;

}; // namespace diskannpy
