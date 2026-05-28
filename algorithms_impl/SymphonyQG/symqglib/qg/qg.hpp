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
    std::vector<PID> hint_table_;
    std::vector<uint32_t> hint_counts_;

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

    void ensure_hint_table(size_t slots, size_t capacity);

    [[nodiscard]] uint64_t hash_query(const float* query) const;

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
        size_t query_id,
        int streamseed_mode,
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
    size_t query_id,
    int streamseed_mode,
    size_t hint_table_slots,
    size_t hint_slot_capacity
) {
    this->visited_.clear();
    this->search_pool_.clear();

    const bool use_hints = streamseed_mode != 0 && hint_table_slots > 0 &&
                           hint_slot_capacity > 0;
    bool has_history = false;
    size_t slot = 0;
    if (use_hints) {
        ensure_hint_table(hint_table_slots, hint_slot_capacity);
        if (hint_table_slots_ > 0) {
            slot = query_id % hint_table_slots_;
            has_history = hint_counts_[slot] > 0;
        }
    }

    if (!has_history) {
        search_qg(query, knn, results);
    } else {
        // Warm path: build candidates from history seeds and their 1-hop neighbors.
        buffer::ResultBuffer res_pool(knn);
        size_t unique_candidates = 0;

        const uint32_t count = hint_counts_[slot];
        for (uint32_t i = 0; i < count; ++i) {
            const PID seed = hint_table_[slot * hint_slot_capacity_ + i];
            if (seed >= num_points_ || visited_.get(seed)) {
                continue;
            }

            visited_.set(seed);
            ++unique_candidates;
            res_pool.insert(seed, space::l2_sqr(query, get_vector(seed), dimension_));

            const PID* neighbors = get_neighbors(seed);
            for (uint32_t j = 0; j < degree_bound_; ++j) {
                const PID nb = neighbors[j];
                if (nb >= num_points_ || visited_.get(nb)) {
                    continue;
                }
                visited_.set(nb);
                ++unique_candidates;
                res_pool.insert(nb, space::l2_sqr(query, get_vector(nb), dimension_));
            }
        }

        // Safety fallback: if warm candidates are too few, use baseline full search.
        if (unique_candidates < static_cast<size_t>(knn)) {
            search_qg(query, knn, results);
        } else {
            res_pool.copy_results(results);
        }
    }

    if (use_hints && hint_table_slots_ > 0 && hint_slot_capacity_ > 0) {
        const uint32_t store = std::min<uint32_t>(
            static_cast<uint32_t>(hint_slot_capacity_), knn
        );
        hint_counts_[slot] = store;
        for (uint32_t i = 0; i < store; ++i) {
            hint_table_[slot * hint_slot_capacity_ + i] = results[i];
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
    if (hint_table_slots_ == 0 || hint_slot_capacity_ == 0) {
        return;
    }

    const uint32_t store = std::min<uint32_t>(
        static_cast<uint32_t>(hint_slot_capacity_), knn
    );
    hint_counts_[slot] = store;
    for (uint32_t i = 0; i < store; ++i) {
        hint_table_[slot * hint_slot_capacity_ + i] = results[i];
    }
    for (uint32_t i = store; i < static_cast<uint32_t>(hint_slot_capacity_); ++i) {
        hint_table_[slot * hint_slot_capacity_ + i] = 0;
    }
}

inline void QuantizedGraph::ensure_hint_table(size_t slots, size_t capacity) {
    if (slots == 0 || capacity == 0) {
        hint_table_.clear();
        hint_counts_.clear();
        hint_table_slots_ = 0;
        hint_slot_capacity_ = 0;
        return;
    }

    if (slots != hint_table_slots_ || capacity != hint_slot_capacity_) {
        hint_table_slots_ = slots;
        hint_slot_capacity_ = capacity;
        hint_table_.assign(hint_table_slots_ * hint_slot_capacity_, 0);
        hint_counts_.assign(hint_table_slots_, 0);
    }
}

inline uint64_t QuantizedGraph::hash_query(const float* query) const {
    uint64_t h = 1469598103934665603ULL;
    const size_t limit = std::min<size_t>(dimension_, 8);
    for (size_t i = 0; i < limit; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, query + i, sizeof(uint32_t));
        h ^= bits;
        h *= 1099511628211ULL;
    }
    return h;
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
