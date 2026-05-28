#pragma once

#include <immintrin.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace symqg::scalar {

inline void data_range(const float* __restrict__ vec, size_t dim, float& lo, float& hi) {
#if defined(__AVX512F__)
    __m512 max_q = _mm512_setzero_ps();
    __m512 min_q = _mm512_setzero_ps();
    size_t mul16 = dim - (dim & 0b1111);
    size_t i;
    for (i = 0; i < mul16; i += 16) {
        __m512 y1 = _mm512_load_ps(&vec[i]);
        max_q = _mm512_max_ps(y1, max_q);
        min_q = _mm512_min_ps(y1, min_q);
    }
    hi = _mm512_reduce_max_ps(max_q);
    lo = _mm512_reduce_min_ps(min_q);
    for (i = 0; i < dim; ++i) {
        float tmp = vec[i];
        lo = tmp < lo ? tmp : lo;
        hi = tmp > hi ? tmp : hi;
    }
#else
    lo = FLT_MAX;
    hi = FLT_MIN;
    for (size_t i = 0; i < dim; ++i) {
        float tmp = vec[i];
        lo = tmp < lo ? tmp : lo;
        hi = tmp > hi ? tmp : hi;
    }
#endif
}

template <typename T>
void quantize(
    T* __restrict__ result,
    const float* __restrict__ vec,
    size_t dim,
    float lo,
    float width,
    int32_t& sum_q
) {
    float one_over_width = 1.0F / width;
    int32_t sum = 0;
    T cur;
    for (size_t i = 0; i < dim; ++i) {
        cur = static_cast<T>(std::lround(((vec[i] - lo) * one_over_width) + 0.5));
        result[i] = cur;
        sum += cur;
    }
    sum_q = sum;
}
}  // namespace symqg::scalar