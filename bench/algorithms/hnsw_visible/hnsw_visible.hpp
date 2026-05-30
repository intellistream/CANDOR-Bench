#pragma once

#include "../annchor-m1/annchor_m1.hpp"

enum class HNSWVisibilityMode : int {
    kResultFilter = 0,
    kPostFilterRefill = 1,
    kTraversalFilter = 2,
};

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class HNSWVisible : public ANNchorM1<T, TagT, LabelT> {
   public:
    HNSWVisible(size_t max_elements, size_t dim, size_t num_threads, size_t M,
                size_t ef_construction,
                bool use_node_lock_in_search = true,
                MetricType metric = METRIC_L2,
                HNSWVisibilityMode mode =
                    HNSWVisibilityMode::kResultFilter)
        : ANNchorM1<T, TagT, LabelT>(max_elements, dim, num_threads, M,
                                     ef_construction,
                                     use_node_lock_in_search, metric),
          mode_(mode) {
        configure();
    }

    int restore(const uint8_t* data, size_t size) override {
        const int rc = ANNchorM1<T, TagT, LabelT>::restore(data, size);
        if (rc == 0) configure();
        return rc;
    }

    void set_visibility_mode(int mode) {
        if (mode < 0 || mode > 2) mode = 0;
        mode_ = static_cast<HNSWVisibilityMode>(mode);
        ANNchorM1<T, TagT, LabelT>::set_visibility_mode(mode);
    }

    void dump_stats(std::string& str) override {
        ANNchorM1<T, TagT, LabelT>::dump_stats(str);
        str += ", hnsw_visible:1";
    }

   private:
    void configure() {
        // This fork is a HNSW visibility-control baseline. It uses the same
        // visible timestamp interface as ANNchor-M1 but deliberately disables
        // pruning recovery and MVCC undo recording.
        ANNchorM1<T, TagT, LabelT>::set_enable_mvcc(false);
        ANNchorM1<T, TagT, LabelT>::set_enable_undo_recovery(false);
        ANNchorM1<T, TagT, LabelT>::set_direction_capsule(false);
        ANNchorM1<T, TagT, LabelT>::set_visibility_by_label(true);
        ANNchorM1<T, TagT, LabelT>::set_visibility_mode(
            static_cast<int>(mode_));
    }

    HNSWVisibilityMode mode_;
};
