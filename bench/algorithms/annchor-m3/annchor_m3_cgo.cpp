#include "annchor_m3.hpp"

IndexBase<float>* create_annchor_m3_float(const IndexParams& params) {
    if (params.data_type != DATA_TYPE_FLOAT) return nullptr;
    return new ANNchorM3<float>(
        params.max_elements, params.dim, params.num_threads, params.M,
        params.ef_construction, params.use_node_lock,
        static_cast<MetricType>(params.metric));
}

bool annchor_m3_set_enable_mvcc_float(IndexBase<float>* base, bool enable) {
    if (auto a = dynamic_cast<ANNchorM3<float>*>(base)) {
        a->set_enable_mvcc(enable);
        return true;
    }
    return false;
}

bool annchor_m3_set_enable_undo_recovery_float(IndexBase<float>* base,
                                               bool enable) {
    if (auto a = dynamic_cast<ANNchorM3<float>*>(base)) {
        a->set_enable_undo_recovery(enable);
        return true;
    }
    return false;
}

int annchor_m3_fresh_merge_float(IndexBase<float>* base,
                                 const float* batch_queries,
                                 size_t k,
                                 size_t num_queries,
                                 const float* fresh_data,
                                 const uint32_t* fresh_tags,
                                 size_t fresh_count,
                                 const uint32_t* committed_results,
                                 uint32_t** batch_results,
                                 size_t committed_view_limit,
                                 const C_FreshJoinParams& params,
                                 C_FreshJoinStats* stats) {
    auto annchor = dynamic_cast<ANNchorM3<float>*>(base);
    if (!annchor) return -2;
    return annchor->fresh_merge_graph_region(
        batch_queries, k, num_queries, fresh_data, fresh_tags, fresh_count,
        committed_results, batch_results, committed_view_limit, params, stats);
}

int annchor_m3_fresh_merge_labels_float(IndexBase<float>* base,
                                        const float* batch_queries,
                                        size_t k,
                                        size_t num_queries,
                                        const uint32_t* fresh_tags,
                                        size_t fresh_count,
                                        const uint32_t* committed_results,
                                        uint32_t** batch_results,
                                        const C_FreshJoinParams& params,
                                        C_FreshJoinStats* stats) {
    auto annchor = dynamic_cast<ANNchorM3<float>*>(base);
    if (!annchor) return -2;
    return annchor->fresh_merge_graph_region_labels(
        batch_queries, k, num_queries, fresh_tags, fresh_count,
        committed_results, batch_results, params, stats);
}
