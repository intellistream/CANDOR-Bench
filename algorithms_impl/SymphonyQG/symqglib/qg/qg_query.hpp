#pragma once

#include <cstdint>

#include "../common.hpp"
#include "../utils/memory.hpp"
#include "../utils/rotator.hpp"
#include "../utils/scalar_quantize.hpp"
#include "./qg_scanner.hpp"

namespace symqg {
class QGQuery {
   private:
    const float* query_data_ = nullptr;
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> lut_;
    size_t padded_dim_ = 0;
    float width_ = 0;
    float lower_val_ = 0;
    float upper_val_ = 0;
    int32_t sumq_ = 0;

   public:
    explicit QGQuery(const float* q, size_t padded_dim)
        : query_data_(q)
        , lut_(padded_dim << 2)  // padded_dim / 4 * 16
        , padded_dim_(padded_dim) {}

    void query_prepare(const FHTRotator& rotator, const QGScanner& scanner) {
        // rotate query
        std::vector<float, memory::AlignedAllocator<float>> rd_query(padded_dim_);
        rotator.rotate(query_data_, rd_query.data());

        // quantize query
        std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> byte_query(padded_dim_);
        scalar::data_range(rd_query.data(), padded_dim_, lower_val_, upper_val_);
        width_ = (upper_val_ - lower_val_) / ((1 << QG_BQUERY) - 1);
        scalar::quantize(
            byte_query.data(), rd_query.data(), padded_dim_, lower_val_, width_, sumq_
        );

        // pack lut
        scanner.pack_lut(byte_query.data(), lut_.data());
    }

    [[nodiscard]] const float& width() const { return width_; }

    [[nodiscard]] const float& lower_val() const { return lower_val_; }

    [[nodiscard]] const int32_t& sumq() const { return sumq_; }

    [[nodiscard]] const std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>>& lut(
    ) const {
        return lut_;
    }

    [[nodiscard]] const float* query_data() const { return query_data_; }
};
}  // namespace symqg