#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define ALIGNMENT 64

namespace annchor {

using tableint = unsigned int;
using batch_t = uint32_t;

template <class T, std::size_t Align>
struct Allocator {
    using value_type = T;
    Allocator() noexcept = default;
    template <class U>
    Allocator(const Allocator<U, Align>&) noexcept {}

    T* allocate(std::size_t n) {
        void* p = ::operator new(n * sizeof(T), std::align_val_t(Align));
        return reinterpret_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete(p, std::align_val_t(Align));
    }
};

template <class T, class U, std::size_t A>
inline bool operator==(const Allocator<T, A>&, const Allocator<U, A>&) {
    return true;
}

template <class T, class U, std::size_t A>
inline bool operator!=(const Allocator<T, A>&, const Allocator<U, A>&) {
    return false;
}

struct alignas(8) Block64 {
    uint64_t w;
    static constexpr size_t bits() { return 64; }
    inline void clear() { w = 0; }
    inline void orWith(const Block64& o) { w |= o.w; }
    inline void atomicOrWith(const Block64& o) {
        __atomic_fetch_or(&w, o.w, __ATOMIC_RELAXED);
    }
};

struct alignas(16) Block128 {
    uint64_t w[2];
    static constexpr size_t bits() { return 128; }
    inline void clear() { std::memset(w, 0, sizeof(w)); }
    inline void orWith(const Block128& o) {
#if defined(__AVX2__)
        __m128i a = _mm_load_si128((const __m128i*)&w[0]);
        __m128i b = _mm_load_si128((const __m128i*)&o.w[0]);
        _mm_store_si128((__m128i*)&w[0], _mm_or_si128(a, b));
#else
        w[0] |= o.w[0];
        w[1] |= o.w[1];
#endif
    }
    inline void atomicOrWith(const Block128& o) {
        __atomic_fetch_or(&w[0], o.w[0], __ATOMIC_RELAXED);
        __atomic_fetch_or(&w[1], o.w[1], __ATOMIC_RELAXED);
    }
};

struct alignas(32) Block256 {
    uint64_t w[4];
    static constexpr size_t bits() { return 256; }
    inline void clear() { std::memset(w, 0, sizeof(w)); }
    inline void orWith(const Block256& o) {
#if defined(__AVX2__)
        __m256i a = _mm256_load_si256((const __m256i*)&w[0]);
        __m256i b = _mm256_load_si256((const __m256i*)&o.w[0]);
        _mm256_store_si256((__m256i*)&w[0], _mm256_or_si256(a, b));
#else
        for (int i = 0; i < 4; ++i) w[i] |= o.w[i];
#endif
    }
    inline void atomicOrWith(const Block256& o) {
        for (int i = 0; i < 4; ++i)
            __atomic_fetch_or(&w[i], o.w[i], __ATOMIC_RELAXED);
    }
};

struct alignas(64) Block512 {
    uint64_t w[8];
    static constexpr size_t bits() { return 512; }
    inline void clear() { std::memset(w, 0, sizeof(w)); }
    inline void orWith(const Block512& o) {
#if defined(__AVX2__)
        __m256i a0 = _mm256_load_si256((const __m256i*)&w[0]);
        __m256i b0 = _mm256_load_si256((const __m256i*)&o.w[0]);
        _mm256_store_si256((__m256i*)&w[0], _mm256_or_si256(a0, b0));

        __m256i a1 = _mm256_load_si256((const __m256i*)&w[4]);
        __m256i b1 = _mm256_load_si256((const __m256i*)&o.w[4]);
        _mm256_store_si256((__m256i*)&w[4], _mm256_or_si256(a1, b1));
#else
        for (int i = 0; i < 8; ++i) w[i] |= o.w[i];
#endif
    }
    inline void atomicOrWith(const Block512& o) {
        for (int i = 0; i < 8; ++i) {
            __atomic_fetch_or(&w[i], o.w[i], __ATOMIC_RELAXED);
        }
    }
};

template <typename BlockT>
static inline bool maybeContains(const BlockT& block, const uint64_t* mask) {
    constexpr size_t N = BlockT::bits() / 64;
#if defined(__AVX2__)
    if constexpr (N == 8) {
        const __m256i blockVec0 =
            _mm256_load_si256((const __m256i*)&block.w[0]);
        const __m256i maskVec0 = _mm256_loadu_si256((const __m256i*)&mask[0]);
        const __m256i missVec0 = _mm256_andnot_si256(blockVec0, maskVec0);
        if (!_mm256_testz_si256(missVec0, missVec0)) return false;

        const __m256i blockVec1 =
            _mm256_load_si256((const __m256i*)&block.w[4]);
        const __m256i maskVec1 = _mm256_loadu_si256((const __m256i*)&mask[4]);
        const __m256i missVec1 = _mm256_andnot_si256(blockVec1, maskVec1);
        return _mm256_testz_si256(missVec1, missVec1);
    } else if constexpr (N == 4) {
        const __m256i blockVec = _mm256_load_si256((const __m256i*)&block.w[0]);
        const __m256i maskVec = _mm256_loadu_si256((const __m256i*)&mask[0]);
        const __m256i missVec = _mm256_andnot_si256(blockVec, maskVec);
        return _mm256_testz_si256(missVec, missVec);
    } else if constexpr (N == 2) {
        const __m128i blockVec = _mm_load_si128((const __m128i*)&block.w[0]);
        const __m128i maskVec = _mm_loadu_si128((const __m128i*)&mask[0]);
        const __m128i missVec = _mm_andnot_si128(blockVec, maskVec);
        return _mm_testz_si128(missVec, missVec);
    } else {
        return (block.w & mask[0]) == mask[0];
    }
#else
    if constexpr (N == 1) {
        return (block.w & mask[0]) == mask[0];
    } else {
        for (size_t i = 0; i < N; ++i) {
            if ((block.w[i] & mask[i]) != mask[i]) return false;
        }
        return true;
    }
#endif
}

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline tableint ceilPow2(tableint x) {
    if (x <= 1) return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

struct BatchSpan {
    batch_t begin;
    batch_t end;
};

struct CheckResult {
    BatchSpan span;
    bool maybe_conflict;
};

class BatchTracker {
   public:
    BatchTracker() = default;

    void init(batch_t capacity) {
        bloom_finished_ = std::make_unique<std::atomic<bool>[]>(capacity);
        commit_finished_ = std::make_unique<std::atomic<bool>[]>(capacity);
        for (batch_t i = 0; i < capacity; ++i) {
            bloom_finished_[i].store(false, std::memory_order_relaxed);
            commit_finished_[i].store(false, std::memory_order_relaxed);
        }
    }

    batch_t beginBatch() {
        return started_.fetch_add(1, std::memory_order_relaxed);
    }

    void onPublished(batch_t id) {
        bloom_finished_[id].store(true, std::memory_order_release);
        updateWatermark(bloom_ready_, bloom_finished_.get(), bloom_mtx_);
    }

    void commitBatch(batch_t id) {
        commit_finished_[id].store(true, std::memory_order_release);
        updateWatermark(committed_, commit_finished_.get(), commit_mtx_);
    }

    batch_t mark() const { return committed_.load(std::memory_order_acquire); }

    BatchSpan getSpanSince(batch_t committed_at_mark) const {
        batch_t end = bloom_ready_.load(std::memory_order_acquire);
        batch_t start = committed_at_mark;
        if (start > end) start = end;
        return {start, end};
    }

   private:
    void updateWatermark(std::atomic<batch_t>& watermark,
                         std::atomic<bool>* flags, std::mutex& mtx) {
        batch_t cur = watermark.load(std::memory_order_relaxed);
        if (!flags[cur].load(std::memory_order_acquire)) return;

        std::lock_guard<std::mutex> lock(mtx);
        cur = watermark.load(std::memory_order_relaxed);
        while (flags[cur].load(std::memory_order_acquire)) {
            cur++;
        }
        watermark.store(cur, std::memory_order_release);
    }

    std::atomic<batch_t> started_{0};
    std::atomic<batch_t> bloom_ready_{0};
    std::atomic<batch_t> committed_{0};

    std::unique_ptr<std::atomic<bool>[]> bloom_finished_;
    std::unique_ptr<std::atomic<bool>[]> commit_finished_;

    std::mutex bloom_mtx_;
    std::mutex commit_mtx_;
};

template <typename BlockT = Block512>
class BatchBloomRange {
   public:
    BatchBloomRange(tableint capacity_batches, tableint est_items_per_batch,
                    double fp_rate, tableint k = 8,
                    uint64_t seed = 0x1234567890abcdefULL)
        : Bcap_(capacity_batches), k_(std::max<tableint>(1, k)), seed_(seed) {
        num_blocks_ = computeNumBlocks(est_items_per_batch, fp_rate);

        base_ = ceilPow2(std::max<tableint>(1, Bcap_));
        seg_.resize(std::size_t(2) * base_ * num_blocks_);
        for (auto& x : seg_) x.clear();

        tracker_.init(Bcap_ + 1024);
    }

    // For writer

    batch_t beginBatch() {
        batch_t id = tracker_.beginBatch();
        return id;
    }

    void buildBatch(batch_t id, const tableint* keys, std::size_t n) {
        if (id >= Bcap_) return;

        const tableint leaf = base_ + id;

        for (tableint blk = 0; blk < num_blocks_; ++blk) {
            nodeBlock(leaf, blk).clear();
        }
        for (std::size_t i = 0; i < n; ++i) {
            insertIntoLeafNode(leaf, keys[i]);
        }

        tableint curr = leaf;
        while (curr > 1) {
            tableint parent = curr >> 1;
            for (tableint blk = 0; blk < num_blocks_; ++blk) {
                nodeBlock(parent, blk).atomicOrWith(nodeBlock(curr, blk));
            }
            curr = parent;
        }

        tracker_.onPublished(id);
    }

    void commitBatch(batch_t id) { tracker_.commitBatch(id); }

    // For reader

    batch_t mark() const { return tracker_.mark(); }

    CheckResult check(batch_t committed_at_mark, const tableint* keys,
                      std::size_t nkeys) const {
        BatchSpan span = tracker_.getSpanSince(committed_at_mark);
        bool conflict = false;
        if (span.begin < span.end) {
            conflict = intersectsRange(span.begin, span.end - 1, keys, nkeys);
        }
        return {span, conflict};
    }

    CheckResult check(batch_t start_mark, batch_t& cursor,
                      const tableint* old_keys, std::size_t n_old,
                      const tableint* new_keys, std::size_t n_new) const {
        BatchSpan span = tracker_.getSpanSince(start_mark);
        bool conflict = false;

        if (cursor < span.end) {
            if (intersectsRange(cursor, span.end - 1, old_keys, n_old)) {
                conflict = true;
            }
        }

        if (!conflict && span.begin < span.end) {
            if (intersectsRange(span.begin, span.end - 1, new_keys, n_new)) {
                conflict = true;
            }
        }

        cursor = span.end;
        return {span, conflict};
    }

   private:
    inline bool intersectsRange(tableint l, tableint r, const tableint* keys,
                                std::size_t nkeys) const {
        if (nkeys == 0) return false;
        if (l > r) return false;

        struct Q {
            tableint blk;
            uint64_t h;
        };
        std::vector<Q> qs;
        qs.reserve(nkeys);

        for (std::size_t i = 0; i < nkeys; ++i) {
            const uint64_t h = splitmix64(uint64_t(keys[i]) ^ seed_);
            const tableint blk = tableint((h >> 32) & (num_blocks_ - 1));
            qs.push_back({blk, h});
        }

        std::sort(qs.begin(), qs.end(),
                  [](const Q& a, const Q& b) { return a.blk < b.blk; });

        std::size_t i = 0;
        while (i < qs.size()) {
            const tableint blk = qs[i].blk;
            const BlockT rangeBlk = rangeOrBlock(l, r, blk);

            do {
                uint64_t mask[8];
                makeMask(qs[i].h, mask);
                if (maybeContains(rangeBlk, mask)) return true;
                ++i;
            } while (i < qs.size() && qs[i].blk == blk);
        }

        return false;
    }

    tableint capacityBatches() const { return Bcap_; }
    tableint numBlocks() const { return num_blocks_; }

    static tableint computeNumBlocks(tableint n, double p) {
        const double ln2 = std::log(2.0);
        const double m_bits = -double(std::max<tableint>(1, n)) *
                              std::log(std::max(1e-12, p)) / (ln2 * ln2);
        const double blocks = std::ceil(m_bits / double(BlockT::bits()));
        tableint nb = (blocks < 1.0) ? 1U : tableint(blocks);
        return ceilPow2(nb);
    }

    inline void makeMask(uint64_t h, uint64_t mask[8]) const {
        std::memset(mask, 0, sizeof(uint64_t) * 8);
        uint64_t x = h;
        const uint64_t delta = (rotl64(h, 17) | 1ULL);
        for (tableint i = 0; i < k_; ++i) {
            const tableint bit = tableint(x & (BlockT::bits() - 1));
            mask[bit >> 6] |= (1ULL << (bit & 63));
            x += delta;
        }
    }

    inline void insertIntoLeafNode(tableint leaf_node, tableint key) {
        const uint64_t h = splitmix64(uint64_t(key) ^ seed_);
        const tableint block_idx = tableint((h >> 32) & (num_blocks_ - 1));

        uint64_t x = h;
        const uint64_t delta = (rotl64(h, 17) | 1ULL);

        BlockT& blk = nodeBlock(leaf_node, block_idx);
        for (tableint i = 0; i < k_; ++i) {
            const tableint bit = tableint(x & (BlockT::bits() - 1));
            if constexpr (BlockT::bits() == 64) {
                blk.w |= (1ULL << (bit & 63));
            } else {
                blk.w[bit >> 6] |= (1ULL << (bit & 63));
            }
            x += delta;
        }
    }

    inline BlockT rangeOrBlock(tableint l, tableint r, tableint blk) const {
        BlockT res;
        res.clear();

        tableint L = base_ + l;
        tableint R = base_ + r + 1;
        while (L < R) {
            if (L & 1) {
                res.orWith(nodeBlock(L, blk));
                ++L;
            }
            if (R & 1) {
                --R;
                res.orWith(nodeBlock(R, blk));
            }
            L >>= 1;
            R >>= 1;
        }
        return res;
    }

    inline BlockT& nodeBlock(tableint node, tableint blk) {
        return seg_[std::size_t(node) * num_blocks_ + blk];
    }

    inline const BlockT& nodeBlock(tableint node, tableint blk) const {
        return seg_[std::size_t(node) * num_blocks_ + blk];
    }

   private:
    tableint Bcap_{0};

    tableint base_{1};
    tableint num_blocks_{1};

    tableint k_{8};
    uint64_t seed_{0};

    std::vector<BlockT> seg_;
    BatchTracker tracker_;
};

}  // namespace annchor