#pragma once

#include <omp.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <limits>
#include <iostream>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../common.hpp"
#include "../quantization/rabitq.hpp"
#include "../space/l2.hpp"
#include "../third/ngt/hashset.hpp"
#include "../third/svs/array.hpp"
#include "../utils/buffer.hpp"
#include "../utils/io.hpp"
#include "../utils/memory.hpp"
#include "../utils/rotator.hpp"
#include "./qg_query.hpp"
#include "./qg_scanner.hpp"

namespace symqg {
/**
 * @brief this Factor only for illustration, the true storage is continous
 * degree_bound_*triple_x + degree_bound_*factor_dq + degree_bound_*factor_vq
 *
 */
struct Factor {
    float triple_x;   // Sqr of distance to centroid + 2 * x * x1 / x0
    float factor_dq;  // Factor of delta * ||q_r|| * (FastScanRes - sum_q)
    float factor_vq;  // Factor of v_l * ||q_r||
};

class QuantizedGraph {
    friend class QGBuilder;

   private:
    size_t num_points_ = 0;    // num points
    size_t degree_bound_ = 0;  // degree bound
    size_t dimension_ = 0;     // dimension
    size_t padded_dim_ = 0;    // padded dimension
    PID entry_point_ = 0;      // Entry point of graph

    data::Array<
        float,
        std::vector<size_t>,
        memory::AlignedAllocator<
            float,
            1 << 22,
            true>>
        data_;  // vectors + graph + quantization codes
    QGScanner scanner_;
    FHTRotator rotator_;
    HashBasedBooleanSet visited_;
    buffer::SearchBuffer search_pool_;

    size_t hint_table_slots_ = 0;
    size_t hint_slot_capacity_ = 0;
    uint32_t hint_record_k_ = 0;
    std::vector<PID> hint_table_;
    std::vector<uint32_t> hint_counts_;
    std::vector<int64_t> hint_owner_query_;
    std::vector<uint64_t> hint_owner_signature_;
    std::vector<float> hint_scores_;
    std::vector<uint64_t> hint_ages_;
    uint64_t hint_clock_ = 0;
    uint64_t hint_round_ = 0;

    /*
     * Position of different data in each row
     *      RawData + QuantizationCodes + Factors + neighborIDs
     * Since we guarantee the degree for each vertex equals degree_bound (multiple of 32),
     * we do not need to store the degree for each vertex
     */
    size_t code_offset_ = 0;      // pos of packed code
    size_t factor_offset_ = 0;    // pos of Factor
    size_t neighbor_offset_ = 0;  // pos of Neighbors
    size_t row_offset_ = 0;       // length of entire row

    void initialize();

    // search on quantized graph
    void search_qg(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
    );

    void copy_vectors(const float*);

    [[nodiscard]] float* get_vector(PID data_id) {
        return &data_.at(row_offset_ * data_id);
    }

    [[nodiscard]] const float* get_vector(PID data_id) const {
        return &data_.at(row_offset_ * data_id);
    }

    [[nodiscard]] uint8_t* get_packed_code(PID data_id) {
        return reinterpret_cast<uint8_t*>(&data_.at((row_offset_ * data_id) + code_offset_)
        );
    }

    [[nodiscard]] const uint8_t* get_packed_code(PID data_id) const {
        return reinterpret_cast<const uint8_t*>(
            &data_.at((row_offset_ * data_id) + code_offset_)
        );
    }

    [[nodiscard]] float* get_factor(PID data_id) {
        return &data_.at((row_offset_ * data_id) + factor_offset_);
    }

    [[nodiscard]] const float* get_factor(PID data_id) const {
        return &data_.at((row_offset_ * data_id) + factor_offset_);
    }

    [[nodiscard]] PID* get_neighbors(PID data_id) {
        return reinterpret_cast<PID*>(&data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }

    [[nodiscard]] const PID* get_neighbors(PID data_id) const {
        return reinterpret_cast<const PID*>(
            &data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }

    void
    find_candidates(PID, size_t, std::vector<Candidate<float>>&, HashBasedBooleanSet&, const std::vector<uint32_t>&)
        const;

    void update_qg(PID, const std::vector<Candidate<float>>&);

    void update_results(buffer::ResultBuffer&, const float*);

    void update_hint_table(size_t slot, const uint32_t* results, uint32_t knn);

    void ensure_hint_table(size_t slots, size_t capacity, uint32_t knn);

    [[nodiscard]] uint64_t compute_semantic_signature(const float* query) const;

    [[nodiscard]] size_t compute_semantic_slot_key(
        const float* query,
        size_t hint_table_slots,
        int64_t fallback_query_id
    ) const;

    [[nodiscard]] static float signature_similarity(uint64_t a, uint64_t b);

    float scan_neighbors(
        const QGQuery& q_obj,
        const float* cur_data,
        float* appro_dist,
        buffer::SearchBuffer& search_pool,
        uint32_t cur_degree,
        HashBasedBooleanSet& visited
    ) const;

   public:
    explicit QuantizedGraph(size_t, size_t, size_t);

    [[nodiscard]] auto num_vertices() const { return this->num_points_; }
    [[nodiscard]] auto dimension() const { return this->dimension_; }
    [[nodiscard]] auto degree_bound() const { return this->degree_bound_; }
    [[nodiscard]] auto entry_point() const { return this->entry_point_; }
    void set_ep(PID entry) { this->entry_point_ = entry; }
    void save_index(const char*) const;

    void load_index(const char*);

    void set_ef(size_t);

    void insert(PID id, const float* vec, size_t ef_insert = 128);

    void insert_batch(const float* data, const PID* ids, size_t n, size_t ef_insert = 128);

    /* search and copy results to KNN */
    void search(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
    );

    void search_warm(
        const float* __restrict__ query,
        uint32_t knn,
        uint32_t* __restrict__ results,
        int64_t query_id,
        int streamseed_mode,
        int hint_level1_only,
        int hint_hops,
        int hint_max_candidates,
        float hint_gate,
        float hint_qual_gate,
        float hint_cons_gate,
        size_t hint_table_slots,
        size_t hint_slot_capacity
    );
};

inline QuantizedGraph::QuantizedGraph(size_t num_points, size_t degree, size_t dim)
    : num_points_(num_points)
    , degree_bound_(degree)
    , dimension_(dim)
    , padded_dim_(static_cast<size_t>(1ULL << ceil_log2(dim)))
    , entry_point_(0)
    , scanner_(padded_dim_, degree_bound_)
    , rotator_(dimension_) {
    initialize();
}

inline void QuantizedGraph::copy_vectors(const float* data) {
    if (data == nullptr) {
        return;
    }
    for (size_t i = 0; i < num_points_; ++i) {
        float* dst = &data_.at(row_offset_ * i);
        std::fill(dst, dst + row_offset_, 0.0F);
        const float* src = data + (i * dimension_);
        std::copy(src, src + dimension_, dst);
    }
}

inline void QuantizedGraph::save_index(const char* filename) const {

    std::cout << "Saving quantized graph to " << filename << '\n';
    std::ofstream output(filename, std::ios::binary);
    assert(output.is_open());

    /* Basic variants */
    output.write(reinterpret_cast<const char*>(&entry_point_), sizeof(PID));

    /* Data */
    data_.save(output);

    /* Rotator */
    this->rotator_.save(output);

    output.close();
    std::cout << "\tQuantized graph saved!\n";
}

inline void QuantizedGraph::load_index(const char* filename) {
    std::cout << "loading quantized graph " << filename << '\n';

    /* Check existence */
    if (!file_exists(filename)) {
        std::cerr << "Index does not exist!\n";
        abort();
    }

    /* Check file size */
    size_t filesize = get_filesize(filename);
    size_t correct_size = sizeof(PID) + (sizeof(float) * num_points_ * row_offset_) +
                          (sizeof(float) * padded_dim_);
    if (filesize != correct_size) {
        std::cerr << "Index file size error! Please make sure the index and "
                     "init parameters are correct\n";
        abort();
    }

    std::ifstream input(filename, std::ios::binary);
    assert(input.is_open());

    /* Basic variants */
    input.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));

    /* Data */
    data_.load(input);

    /* Rotator */
    this->rotator_.load(input);

    input.close();
    std::cout << "Quantized graph loaded!\n";
}

inline void QuantizedGraph::set_ef(size_t cur_ef) {
    this->search_pool_.resize(cur_ef);
    this->visited_ = HashBasedBooleanSet(std::min(this->num_points_ / 10, cur_ef * cur_ef));
}

inline void QuantizedGraph::insert(PID id, const float* vec, size_t ef_insert) {
    if (id >= num_points_) {
        std::cerr << "insert id out of range: " << id << " >= " << num_points_ << "\n";
        return;
    }
    if (vec == nullptr) {
        std::cerr << "insert vec is null\n";
        return;
    }

    float* dst = get_vector(id);
    std::copy(vec, vec + dimension_, dst);

    // Build local neighborhood for this node from current graph.
    std::vector<uint32_t> degrees(num_points_, static_cast<uint32_t>(degree_bound_));
    HashBasedBooleanSet vis(
        std::max<size_t>(num_points_, std::min(ef_insert * ef_insert, num_points_ / 10))
    );
    vis.clear();
    std::vector<Candidate<float>> candidates;
    candidates.reserve(std::min<size_t>(ef_insert * 2, num_points_));
    find_candidates(id, std::max<size_t>(ef_insert, degree_bound_), candidates, vis, degrees);

    std::sort(candidates.begin(), candidates.end());
    std::vector<Candidate<float>> pruned;
    pruned.reserve(degree_bound_);
    std::unordered_set<PID> seen;
    seen.reserve(degree_bound_ * 2);

    for (const auto& c : candidates) {
        if (c.id == id) {
            continue;
        }
        if (!seen.insert(c.id).second) {
            continue;
        }
        pruned.emplace_back(c);
        if (pruned.size() >= degree_bound_) {
            break;
        }
    }

    // Fill sparse neighborhoods with random fallback, consistent with builder behavior.
    while (pruned.size() < degree_bound_) {
        PID rand_id = rand_integer<PID>(0, static_cast<PID>(num_points_ - 1));
        if (rand_id == id || !seen.insert(rand_id).second) {
            continue;
        }
        pruned.emplace_back(
            rand_id,
            space::l2_sqr(get_vector(id), get_vector(rand_id), dimension_)
        );
    }

    std::sort(pruned.begin(), pruned.end());
    update_qg(id, pruned);

    // Add reverse edges locally and refresh affected nodes' quantization payloads.
    for (const auto& c : pruned) {
        const PID nid = c.id;
        PID* neigh = get_neighbors(nid);
        bool has_back_edge = false;
        for (size_t j = 0; j < degree_bound_; ++j) {
            if (neigh[j] == id) {
                has_back_edge = true;
                break;
            }
        }

        if (has_back_edge) {
            continue;
        }

        const float* nvec = get_vector(nid);
        float d_new = space::l2_sqr(nvec, get_vector(id), dimension_);

        size_t worst_pos = 0;
        float worst_dist = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < degree_bound_; ++j) {
            float dj = space::l2_sqr(nvec, get_vector(neigh[j]), dimension_);
            if (dj > worst_dist) {
                worst_dist = dj;
                worst_pos = j;
            }
        }

        if (d_new < worst_dist) {
            neigh[worst_pos] = id;
        }

        std::vector<Candidate<float>> refreshed;
        refreshed.reserve(degree_bound_);
        for (size_t j = 0; j < degree_bound_; ++j) {
            PID eid = neigh[j];
            refreshed.emplace_back(eid, space::l2_sqr(nvec, get_vector(eid), dimension_));
        }
        std::sort(refreshed.begin(), refreshed.end());
        update_qg(nid, refreshed);
    }
}

inline void QuantizedGraph::insert_batch(
    const float* data,
    const PID* ids,
    size_t n,
    size_t ef_insert
) {
    if (data == nullptr || ids == nullptr) {
        std::cerr << "insert_batch received null input\n";
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        insert(ids[i], data + i * dimension_, ef_insert);
    }
}

/*
 * search single query
 */
inline void QuantizedGraph::search(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    /* Init query matrix */
    this->visited_.clear();
    this->search_pool_.clear();
    search_qg(query, knn, results);
}

inline void QuantizedGraph::search_warm(
    const float* __restrict__ query,
    uint32_t knn,
    uint32_t* __restrict__ results,
    int64_t query_id,
    int streamseed_mode,
    int hint_level1_only,
    int hint_hops,
    int hint_max_candidates,
    float hint_gate,
    float hint_qual_gate,
    float hint_cons_gate,
    size_t hint_table_slots,
    size_t hint_slot_capacity
) {
    this->visited_.clear();
    this->search_pool_.clear();

    const bool use_hints = streamseed_mode != 0 && hint_table_slots > 0 &&
                           hint_slot_capacity > 0 && knn > 0;
    bool used_hint = false;
    bool touched = false;
    bool level1_hit = false;
    std::vector<PID> seed_ids;
    size_t slot = 0;

    if (use_hints) {
        ensure_hint_table(hint_table_slots, hint_slot_capacity, knn);
        if (hint_table_slots_ > 0 && hint_slot_capacity_ > 0 && hint_record_k_ == knn) {
            slot = compute_semantic_slot_key(query, hint_table_slots_, query_id);
            const uint32_t count = std::min<uint32_t>(
                hint_counts_[slot], static_cast<uint32_t>(hint_slot_capacity_)
            );
            const uint64_t query_signature = compute_semantic_signature(query);
            size_t selected_record = static_cast<size_t>(-1);

            for (uint32_t r = 0; r < count; ++r) {
                const size_t rec = slot * hint_slot_capacity_ + r;
                if (hint_owner_query_[rec] == query_id) {
                    selected_record = rec;
                    level1_hit = true;
                    break;
                }
            }

            if (selected_record == static_cast<size_t>(-1) && hint_level1_only == 0 &&
                hint_round_ > 5 && count > 0) {
                float max_score = 1.0f;
                for (uint32_t r = 0; r < count; ++r) {
                    const size_t rec = slot * hint_slot_capacity_ + r;
                    if (hint_scores_[rec] > max_score) {
                        max_score = hint_scores_[rec];
                    }
                }

                float best_score = -std::numeric_limits<float>::infinity();
                size_t best_record = static_cast<size_t>(-1);
                constexpr float alpha = 0.7f;
                for (uint32_t r = 0; r < count; ++r) {
                    const size_t rec = slot * hint_slot_capacity_ + r;
                    if (hint_owner_query_[rec] < 0) {
                        continue;
                    }
                    const float sim = signature_similarity(query_signature, hint_owner_signature_[rec]);
                    const float cnt = hint_scores_[rec] / max_score;
                    const float score = alpha * sim + (1.0f - alpha) * cnt;
                    if (score > best_score) {
                        best_score = score;
                        best_record = rec;
                    }
                }
                selected_record = best_record;
            }

            if (selected_record != static_cast<size_t>(-1)) {
                const size_t begin = selected_record * static_cast<size_t>(hint_record_k_);
                seed_ids.reserve(hint_record_k_);
                for (uint32_t i = 0; i < hint_record_k_; ++i) {
                    const PID seed = hint_table_[begin + i];
                    if (seed < num_points_) {
                        seed_ids.push_back(seed);
                    }
                }
            }
        }
    }

    if (!seed_ids.empty()) {
        std::vector<PID> frontier;
        std::vector<PID> candidates;
        frontier.reserve(seed_ids.size());
        candidates.reserve(seed_ids.size() * 8);

        const size_t max_candidates = hint_max_candidates > 0
            ? static_cast<size_t>(hint_max_candidates)
            : static_cast<size_t>(256);

        for (PID seed : seed_ids) {
            if (seed >= num_points_ || visited_.get(seed)) {
                continue;
            }
            visited_.set(seed);
            touched = true;
            frontier.push_back(seed);
            candidates.push_back(seed);
            if (candidates.size() >= max_candidates) {
                break;
            }
        }

        int effective_hops = std::max(0, hint_hops);
        if (!level1_hit) {
            effective_hops += 1;
        }

        for (int hop = 0; hop < effective_hops && !frontier.empty() &&
             candidates.size() < max_candidates; ++hop) {
            std::vector<PID> next_frontier;
            for (PID u : frontier) {
                const PID* neighbors = get_neighbors(u);
                for (uint32_t j = 0; j < degree_bound_; ++j) {
                    const PID v = neighbors[j];
                    if (v >= num_points_ || visited_.get(v)) {
                        continue;
                    }
                    visited_.set(v);
                    touched = true;
                    next_frontier.push_back(v);
                    candidates.push_back(v);
                    if (candidates.size() >= max_candidates) {
                        break;
                    }
                }
                if (candidates.size() >= max_candidates) {
                    break;
                }
            }
            frontier.swap(next_frontier);
        }

        if (!candidates.empty()) {
            std::vector<std::pair<float, PID>> scored;
            scored.reserve(candidates.size());
            for (PID candidate : candidates) {
                scored.emplace_back(
                    space::l2_sqr(query, get_vector(candidate), dimension_), candidate
                );
            }

            const size_t topn = std::min<size_t>(static_cast<size_t>(knn), scored.size());
            std::partial_sort(
                scored.begin(),
                scored.begin() + topn,
                scored.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; }
            );

            bool accepted = topn > 0;
            if (accepted && hint_gate >= 0.0f && scored[0].first > hint_gate) {
                accepted = false;
            }

            if (accepted && !level1_hit && hint_qual_gate >= 0.0f) {
                const float eps = 1e-6f;
                float qual = std::numeric_limits<float>::infinity();
                if (topn >= 2) {
                    const float c1 = scored[0].first;
                    const float c2 = scored[1].first;
                    qual = (c2 - c1) / (std::fabs(c1) + eps);
                }
                if (qual < hint_qual_gate) {
                    accepted = false;
                }
            }

            if (accepted && hint_cons_gate >= 0.0f) {
                std::unordered_set<PID> seed_set;
                seed_set.reserve(seed_ids.size());
                for (PID seed : seed_ids) {
                    seed_set.insert(seed);
                }
                size_t overlap = 0;
                for (size_t i = 0; i < topn; ++i) {
                    if (seed_set.find(scored[i].second) != seed_set.end()) {
                        ++overlap;
                    }
                }
                const float cons = static_cast<float>(overlap) / static_cast<float>(knn);
                const float gate = std::min(1.0f, hint_cons_gate);
                if (cons < gate) {
                    accepted = false;
                }
            }

            if (accepted) {
                for (uint32_t i = 0; i < knn; ++i) {
                    results[i] = i < topn ? scored[i].second : static_cast<uint32_t>(num_points_);
                }
                used_hint = true;
            }
        }
    }

    if (!used_hint) {
        if (touched) {
            this->visited_.clear();
            this->search_pool_.clear();
        }
        search_qg(query, knn, results);
    }

    if (use_hints && hint_table_slots_ > 0 && hint_slot_capacity_ > 0 &&
        hint_record_k_ == knn && results != nullptr && results[0] < num_points_) {
        const uint64_t query_signature = compute_semantic_signature(query);
        const uint64_t tick = ++hint_clock_;
        if (query_id == 0) {
            ++hint_round_;
        }

        uint32_t count = std::min<uint32_t>(
            hint_counts_[slot], static_cast<uint32_t>(hint_slot_capacity_)
        );
        size_t target = static_cast<size_t>(-1);
        for (uint32_t r = 0; r < count; ++r) {
            const size_t rec = slot * hint_slot_capacity_ + r;
            if (hint_owner_query_[rec] == query_id) {
                target = rec;
                break;
            }
        }

        const float new_score = used_hint ? 1.0f : 0.0f;
        if (target == static_cast<size_t>(-1)) {
            if (count < hint_slot_capacity_) {
                target = slot * hint_slot_capacity_ + count;
                hint_counts_[slot] = count + 1;
            } else {
                constexpr uint64_t stale_window = 4096;
                constexpr float age_weight = 0.25f;
                float victim_key = std::numeric_limits<float>::infinity();
                size_t victim = slot * hint_slot_capacity_;
                for (uint32_t r = 0; r < count; ++r) {
                    const size_t rec = slot * hint_slot_capacity_ + r;
                    uint64_t age_delta = tick > hint_ages_[rec] ? tick - hint_ages_[rec] : 0;
                    if (age_delta > stale_window) {
                        age_delta = stale_window;
                    }
                    const float age_term = static_cast<float>(age_delta) /
                                           static_cast<float>(stale_window);
                    const float keep_key = hint_scores_[rec] - age_weight * age_term;
                    if (keep_key < victim_key) {
                        victim_key = keep_key;
                        victim = rec;
                    }
                }
                const bool victim_stale = tick > hint_ages_[victim] + stale_window;
                const bool should_replace = victim_stale ||
                    new_score > hint_scores_[victim] || hint_scores_[victim] <= 0.0f;
                if (should_replace) {
                    target = victim;
                }
            }
        }

        if (target != static_cast<size_t>(-1)) {
            const size_t begin = target * static_cast<size_t>(hint_record_k_);
            for (uint32_t i = 0; i < hint_record_k_; ++i) {
                hint_table_[begin + i] = i < knn ? results[i] : static_cast<PID>(num_points_);
            }
            if (hint_owner_query_[target] == query_id && used_hint) {
                hint_scores_[target] += 1.0f;
            } else {
                hint_scores_[target] = new_score;
            }
            hint_owner_query_[target] = query_id;
            hint_owner_signature_[target] = query_signature;
            hint_ages_[target] = tick;
        }
    }
}

/**
 * @brief search on qg
 *
 * @param query     unrotated query vector, dimension_ elements
 * @param knn       num of nearest neighbors
 * @param results   searh res
 */
inline void QuantizedGraph::search_qg(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    // query preparation
    QGQuery q_obj(query, padded_dim_);
    q_obj.query_prepare(rotator_, scanner_);

    /* Searching pool initialization */
    search_pool_.insert(this->entry_point_, FLT_MAX);

    /* Result pool */
    buffer::ResultBuffer res_pool(knn);

    /* Current version of fast scan compute 32 distances */
    std::vector<float> appro_dist(degree_bound_);  // approximate dis

    while (search_pool_.has_next()) {
        PID cur_node = search_pool_.pop();
        if (visited_.get(cur_node)) {
            continue;
        }
        visited_.set(cur_node);

        float sqr_y = scan_neighbors(
            q_obj,
            get_vector(cur_node),
            appro_dist.data(),
            this->search_pool_,
            this->degree_bound_,
            this->visited_
        );
        res_pool.insert(cur_node, sqr_y);
    }

    update_results(res_pool, query);
    res_pool.copy_results(results);
}

// scan a data row (including data vec and quantization codes for its neighbors)
// return exact distnace for current vertex
inline float QuantizedGraph::scan_neighbors(
    const QGQuery& q_obj,
    const float* cur_data,
    float* appro_dist,
    buffer::SearchBuffer& search_pool,
    uint32_t cur_degree,
    HashBasedBooleanSet& visited
) const {
    float sqr_y = space::l2_sqr(q_obj.query_data(), cur_data, dimension_);

    /* Compute approximate distance by Fast Scan */
    const auto* packed_code = reinterpret_cast<const uint8_t*>(&cur_data[code_offset_]);
    const auto* factor = &cur_data[factor_offset_];
    this->scanner_.scan_neighbors(
        appro_dist,
        q_obj.lut().data(),
        sqr_y,
        q_obj.lower_val(),
        q_obj.width(),
        q_obj.sumq(),
        packed_code,
        factor
    );

    const PID* ptr_nb = reinterpret_cast<const PID*>(&cur_data[neighbor_offset_]);
    for (uint32_t i = 0; i < cur_degree; ++i) {
        PID cur_neighbor = ptr_nb[i];
        float tmp_dist = appro_dist[i];
#if defined(DEBUG)
        std::cout << "Neighbor ID " << cur_neighbor << '\n';
        std::cout << "Appro " << appro_dist[i] << '\t';
        float __gt_dist__ = l2_sqr(query, get_vector(cur_neighbor), dimension_);
        std::cout << "GT " << __gt_dist__ << '\t';
        std::cout << "Error " << (appro_dist[i] - __gt_dist__) / __gt_dist__ << '\t';
        std::cout << "sqr_y " << sqr_y << '\n';
#endif
        if (search_pool.is_full(tmp_dist) || visited.get(cur_neighbor)) {
            continue;
        }
        search_pool.insert(cur_neighbor, tmp_dist);
        memory::mem_prefetch_l2(
            reinterpret_cast<const char*>(get_vector(search_pool.next_id())), 10
        );
    }

    return sqr_y;
}

inline void QuantizedGraph::update_results(
    buffer::ResultBuffer& result_pool, const float* query
) {
    if (result_pool.is_full()) {
        return;
    }

    auto ids = result_pool.ids();
    for (PID data_id : ids) {
        PID* ptr_nb = get_neighbors(data_id);
        for (uint32_t i = 0; i < this->degree_bound_; ++i) {
            PID cur_neighbor = ptr_nb[i];
            if (!visited_.get(cur_neighbor)) {
                visited_.set(cur_neighbor);
                result_pool.insert(
                    cur_neighbor, space::l2_sqr(query, get_vector(cur_neighbor), dimension_)
                );
            }
        }
        if (result_pool.is_full()) {
            break;
        }
    }
}

inline void QuantizedGraph::update_hint_table(
    size_t slot,
    const uint32_t* results,
    uint32_t knn
) {
    if (hint_table_slots_ == 0 || hint_slot_capacity_ == 0 || hint_record_k_ == 0 ||
        results == nullptr || slot >= hint_table_slots_) {
        return;
    }

    const size_t rec = slot * hint_slot_capacity_;
    const size_t begin = rec * static_cast<size_t>(hint_record_k_);
    hint_counts_[slot] = std::max<uint32_t>(hint_counts_[slot], 1);
    for (uint32_t i = 0; i < hint_record_k_; ++i) {
        hint_table_[begin + i] = i < knn ? results[i] : static_cast<PID>(num_points_);
    }
}

inline void QuantizedGraph::ensure_hint_table(size_t slots, size_t capacity, uint32_t knn) {
    if (slots == 0 || capacity == 0 || knn == 0) {
        hint_table_.clear();
        hint_counts_.clear();
        hint_owner_query_.clear();
        hint_owner_signature_.clear();
        hint_scores_.clear();
        hint_ages_.clear();
        hint_table_slots_ = 0;
        hint_slot_capacity_ = 0;
        hint_record_k_ = 0;
        hint_clock_ = 0;
        hint_round_ = 0;
        return;
    }

    if (slots != hint_table_slots_ || capacity != hint_slot_capacity_ || knn != hint_record_k_) {
        hint_table_slots_ = slots;
        hint_slot_capacity_ = capacity;
        hint_record_k_ = knn;
        const size_t records = hint_table_slots_ * hint_slot_capacity_;
        hint_table_.assign(records * static_cast<size_t>(hint_record_k_), static_cast<PID>(num_points_));
        hint_counts_.assign(hint_table_slots_, 0);
        hint_owner_query_.assign(records, -1);
        hint_owner_signature_.assign(records, 0);
        hint_scores_.assign(records, 0.0f);
        hint_ages_.assign(records, 0);
        hint_clock_ = 0;
        hint_round_ = 0;
    }
}

inline uint64_t QuantizedGraph::compute_semantic_signature(const float* query) const {
    if (query == nullptr || dimension_ == 0) {
        return 0;
    }

    constexpr int sample_dims = 8;
    constexpr float quant_scale = 16.0f;
    uint64_t signature = 0;
    for (int t = 0; t < sample_dims; ++t) {
        const size_t pick = (static_cast<size_t>(t) * 9973ULL + 13ULL) % dimension_;
        const float v = query[pick];
        const int q = static_cast<int>(std::lrint(v * quant_scale));
        const int clamped = std::max(-128, std::min(127, q));
        const uint8_t packed = static_cast<uint8_t>(clamped + 128);
        signature |= static_cast<uint64_t>(packed) << (t * 8);
    }
    return signature;
}

inline size_t QuantizedGraph::compute_semantic_slot_key(
    const float* query,
    size_t hint_table_slots,
    int64_t fallback_query_id
) const {
    if (query == nullptr || dimension_ == 0 || hint_table_slots == 0) {
        if (fallback_query_id < 0) {
            return 0;
        }
        return static_cast<size_t>(fallback_query_id) % std::max<size_t>(hint_table_slots, 1);
    }

    const uint64_t signature = compute_semantic_signature(query);
    uint64_t h = 1469598103934665603ULL;
    for (int t = 0; t < 8; ++t) {
        const uint64_t token = ((signature >> (t * 8)) & 0xFFULL) ^
            (static_cast<uint64_t>(t + 1) * 11400714819323198485ULL);
        h ^= token;
        h *= 1099511628211ULL;
    }
    return static_cast<size_t>(h % hint_table_slots);
}

inline float QuantizedGraph::signature_similarity(uint64_t a, uint64_t b) {
    int l1 = 0;
    for (int t = 0; t < 8; ++t) {
        const int av = static_cast<int>((a >> (t * 8)) & 0xFFULL);
        const int bv = static_cast<int>((b >> (t * 8)) & 0xFFULL);
        l1 += std::abs(av - bv);
    }
    constexpr float max_l1 = 8.0f * 255.0f;
    return 1.0f - static_cast<float>(l1) / max_l1;
}

inline void QuantizedGraph::initialize() {
    /* check size */
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dimension_);

    this->code_offset_ = dimension_;  // Pos of packed code (aligned)
    this->factor_offset_ =
        code_offset_ + padded_dim_ / 64 * 2 * degree_bound_;  // Pos of Factor
    this->neighbor_offset_ =
        factor_offset_ + sizeof(Factor) * degree_bound_ / sizeof(float);
    this->row_offset_ = neighbor_offset_ + degree_bound_;

    /* Allocate memory of data*/
    data_ = data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 1 << 22, true>>(
            std::vector<size_t>{num_points_, row_offset_}
        );
}

// find candidate neighbors for cur_id, exclude the vertex itself
inline void QuantizedGraph::find_candidates(
    PID cur_id,
    size_t search_ef,
    std::vector<Candidate<float>>& results,
    HashBasedBooleanSet& vis,
    const std::vector<uint32_t>& degrees
) const {
    const float* query = get_vector(cur_id);
    QGQuery q_obj(query, padded_dim_);
    q_obj.query_prepare(rotator_, scanner_);

    /* Searching pool initialization */
    buffer::SearchBuffer tmp_pool(search_ef);
    tmp_pool.insert(this->entry_point_, 1e10);
    memory::mem_prefetch_l1(
        reinterpret_cast<const char*>(get_vector(this->entry_point_)), 10
    );

    /* Current version of fast scan compute 32 distances */
    std::vector<float> appro_dist(degree_bound_);  // approximate dis
    while (tmp_pool.has_next()) {
        auto cur_candi = tmp_pool.pop();
        if (vis.get(cur_candi)) {
            continue;
        }
        vis.set(cur_candi);
        auto cur_degree = degrees[cur_candi];
        auto sqr_y = scan_neighbors(
            q_obj,
            get_vector(cur_candi),
            appro_dist.data(),
            tmp_pool,
            cur_degree,
            vis
        );
        if (cur_candi != cur_id) {
            results.emplace_back(cur_candi, sqr_y);
        }
    }
}

inline void QuantizedGraph::update_qg(
    PID cur_id, const std::vector<Candidate<float>>& new_neighbors
) {
    size_t cur_degree = new_neighbors.size();

    if (cur_degree == 0) {
        return;
    }
    // copy neighbors
    PID* neighbor_ptr = get_neighbors(cur_id);
    for (size_t i = 0; i < cur_degree; ++i) {
        neighbor_ptr[i] = new_neighbors[i].id;
    }

    RowMatrix<float> x_pad(cur_degree, padded_dim_);  // padded neighbors mat
    RowMatrix<float> c_pad(1, padded_dim_);           // padded duplicate centroid mat
    x_pad.setZero();
    c_pad.setZero();

    /* Copy data */
    for (size_t i = 0; i < cur_degree; ++i) {
        auto neighbor_id = new_neighbors[i].id;
        const auto* cur_data = get_vector(neighbor_id);
        std::copy(cur_data, cur_data + dimension_, &x_pad(static_cast<long>(i), 0));
    }
    const auto* cur_cent = get_vector(cur_id);
    std::copy(cur_cent, cur_cent + dimension_, &c_pad(0, 0));

    /* rotate Matrix */
    RowMatrix<float> x_rotated(cur_degree, padded_dim_);
    RowMatrix<float> c_rotated(1, padded_dim_);
    for (long i = 0; i < static_cast<long>(cur_degree); ++i) {
        this->rotator_.rotate(&x_pad(i, 0), &x_rotated(i, 0));
    }
    this->rotator_.rotate(&c_pad(0, 0), &c_rotated(0, 0));

    // Get codes and factors for rabitq
    float* fac_ptr = get_factor(cur_id);
    float* triple_x = fac_ptr;
    float* factor_dq = triple_x + this->degree_bound_;
    float* factor_vq = factor_dq + this->degree_bound_;
    rabitq_codes(
        x_rotated, c_rotated, get_packed_code(cur_id), triple_x, factor_dq, factor_vq
    );
}
}  // namespace symqg
