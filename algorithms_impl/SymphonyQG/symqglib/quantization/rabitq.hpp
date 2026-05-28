#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../common.hpp"
#include "../space/bitwise.hpp"
#include "./fastscan_impl.hpp"

namespace symqg {

static inline void rabitq_factors(
    const RowMatrix<float>& rotated_data,
    const RowMatrix<float>& rotated_centroid,
    const RowMatrix<int>& bin_x,
    float* triple_x,
    float* factor_dq,
    float* factor_vq
);

inline void rabitq_codes(
    RowMatrix<float>& rotated_data,
    const RowMatrix<float>& rotated_centroid,
    uint8_t* packed_code,
    float* triple_x,
    float* factor_dq,
    float* factor_vq
) {
    long num_points = rotated_data.rows();
    long dim = rotated_data.cols();

    // current dim should be a multiple of 64
    assert(dim % 64 == 0);

    // get residual
    for (long i = 0; i < num_points; ++i) {
        rotated_data.row(i) = rotated_data.row(i) - rotated_centroid;
    }

    // binary representation
    RowMatrix<int> bin_x(num_points, dim);
    for (long i = 0; i < num_points; ++i) {
        for (long j = 0; j < dim; ++j) {
            bin_x(i, j) = static_cast<int>(rotated_data(i, j) > 0);
        }
    }

    // change bin_x to uint64
    std::vector<uint64_t> binary(num_points * (dim / 64));
    space::pack_binary(bin_x.data(), binary.data(), num_points * dim);

    // compute codes of RaBitQ, store at packed_code
    pack_codes(dim, binary.data(), num_points, packed_code);

    // compute factors for RaBitQ
    rabitq_factors(rotated_data, rotated_centroid, bin_x, triple_x, factor_dq, factor_vq);
}

static inline void rabitq_factors(
    const RowMatrix<float>& rotated_data_residual,
    const RowMatrix<float>& rotated_centroid,
    const RowMatrix<int>& bin_x,
    float* triple_x,
    float* factor_dq,
    float* factor_vq
) {
    long num_points = rotated_data_residual.rows();
    long dim = rotated_data_residual.cols();

    float fac_norm = 1.F / std::sqrt(static_cast<float>(dim));

    // signed quantized vectors
    RowMatrix<float> ones(num_points, dim);
    ones.setOnes();
    RowMatrix<float> signed_x = 2 * bin_x.cast<float>() - ones;

    // fac_x0 (num_points, 1)
    RowMatrix<float> fac_x0 =
        (rotated_data_residual.array() * signed_x.array() * fac_norm).rowwise().sum();
    RowMatrix<float> x_rotated_norm = rotated_data_residual.rowwise().norm();
    for (long i = 0; i < num_points; ++i) {
        float cur_x0 = fac_x0(i, 0);
        fac_x0(i, 0) = cur_x0 / x_rotated_norm(i, 0);
    }

    // fac_x1: (num_points, 1)
    RowMatrix<float> fac_x1(num_points, 1);
    for (long i = 0; i < num_points; ++i) {
        fac_x1(i, 0) = static_cast<float>(
            (rotated_centroid.array() * signed_x.row(i).array()).sum() * fac_norm
        );
    }

    for (long j = 0; j < num_points; ++j) {
        double cur_x = x_rotated_norm(j, 0);  // current dist 2 centroid
        double cur_x0 = fac_x0(j, 0);         // current <o, o_bar>
        double cur_x1 = fac_x1(j, 0);         // current <c, o_bar>
        long double x_x0 = static_cast<long double>(cur_x) / cur_x0;

        triple_x[j] = static_cast<float>((cur_x * cur_x) + (2 * x_x0 * cur_x1));
        factor_dq[j] = static_cast<float>(-2 * x_x0 * fac_norm);
        factor_vq[j] = static_cast<float>(
            -2 * x_x0 * fac_norm * (bin_x.row(j).sum() * 2 - static_cast<int>(dim))
        );
    }
}
}  // namespace symqg