#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace symqg::space {

inline auto popcount(size_t dim, const uint64_t* __restrict__ data) -> size_t {
    size_t ret = 0;
    for (size_t i = 0; i < dim / 64; ++i) {
        ret += __builtin_popcountll((*data));
        ++data;
    }
    return ret;
}

/* Change 0/1 data to uint64 */
inline void pack_binary(
    const int* __restrict__ bin_x, uint64_t* __restrict__ binary, size_t length
) {
    for (size_t i = 0; i < length; i += 64) {
        uint64_t cur = 0;
        for (size_t j = 0; j < 64; ++j) {
            cur |= (static_cast<uint64_t>(bin_x[i + j]) << (63 - j));
        }
        *binary = cur;
        ++binary;
    }
}

}  // namespace symqg::space