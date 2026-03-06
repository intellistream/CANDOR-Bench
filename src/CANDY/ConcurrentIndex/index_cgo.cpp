#include "index_cgo.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hnsw/hnsw.hpp"
#include "index.hpp"
#include "parlayann/parlay_hnsw.hpp"
#include "parlayann/parlay_vamana.hpp"
#include "vamana/vamana.hpp"

#define BUFFER_LEN 2048

extern "C" {

void* create_index(IndexType type, IndexParams params) {
    IndexBase<float>* index = nullptr;
    switch (type) {
        case INDEX_TYPE_HNSW:
            if (params.data_type == DATA_TYPE_FLOAT) {
                index = new HNSW<float>(params.max_elements, params.dim,
                                        params.num_threads, params.M,
                                        params.ef_construction,
                                        params.use_node_lock,
                                        params.worker_scheduler);
            }
            break;
        case INDEX_TYPE_PARLAYHNSW:
            if (params.data_type == DATA_TYPE_FLOAT) {
                index = new ParlayHNSW<float>(
                    params.max_elements, params.dim, params.num_threads,
                    params.M, params.ef_construction, params.level_m,
                    params.alpha, params.visit_limit);
            }
            break;
        case INDEX_TYPE_PARLAYVAMANA:
            if (params.data_type == DATA_TYPE_FLOAT) {
                index = new ParlayVamana<float>(
                    params.max_elements, params.dim, params.num_threads,
                    params.M, params.ef_construction, params.alpha);
            }
            break;
        case INDEX_TYPE_VAMANA:
            if (params.data_type == DATA_TYPE_FLOAT) {
                index = new Vamana<float>(params.max_elements, params.dim,
                                          params.num_threads, params.M,
                                          params.ef_construction, params.alpha);
            }
            break;
        default:
            return nullptr;
    }
    return static_cast<void*>(index);
}

void destroy_index(void* index_ptr) {
    if (index_ptr) {
        delete static_cast<IndexBase<float>*>(index_ptr);
    }
}

int build(void* index_ptr, float* data, uint32_t* tags, size_t num_points) {
    if (!index_ptr || !data || !tags) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    index->build(data, tags, num_points);
    return 0;
}

void set_query_params(void* index_ptr, C_QueryParams params) {
    if (!index_ptr) return;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    QParams qparams(params.ef_search, params.beam_width, params.alpha,
                    params.visit_limit);
    index->set_query_params(qparams);
}

int search(void* index_ptr, float* query, size_t k, uint32_t* res_tags) {
    if (!index_ptr || !query || !res_tags) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    std::vector<uint32_t> results;
    index->search(query, k, results);
    for (size_t i = 0; i < results.size(); ++i) {
        res_tags[i] = results[i];
    }
    return 0;
}

int batch_insert(void* index_ptr, float* batch_data, uint32_t* batch_tags,
                 size_t batch_size) {
    if (!index_ptr || !batch_data || !batch_tags) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    return index->batch_insert(batch_data, batch_tags, batch_size);
}

int batch_partial_insert(void* index_ptr, float* batch_data,
                              uint32_t* batch_tags, size_t batch_size,
                              size_t* partial_limits, size_t* update_counts) {
    if (!index_ptr || !batch_data || !batch_tags) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    if (auto hnsw = dynamic_cast<HNSW<float>*>(index)) {
        return hnsw->batch_partial_insert(batch_data, batch_tags,
                                               batch_size, partial_limits,
                                               update_counts);
    }
    return index->batch_insert(batch_data, batch_tags, batch_size);
}

int batch_search(void* index_ptr, float* batch_queries, size_t k,
                 size_t num_queries, uint32_t** batch_results, size_t* watermark_out,
                 size_t visible_ts) {
    if (!index_ptr || !batch_queries) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    return index->batch_search(batch_queries, k, num_queries, batch_results, watermark_out, visible_ts);
}

void dump_stats(void* index_ptr, char* str) {
    if (!index_ptr) return;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    std::string stats_str;
    index->dump_stats(stats_str);
    strncpy(str, stats_str.c_str(), BUFFER_LEN);
    str[BUFFER_LEN - 1] = '\0';
}

int snapshot_index(void* index_ptr, uint8_t** buffer, size_t* size) {
    if (!index_ptr || !buffer || !size) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    if (!index->supports_snapshot()) return -2;

    std::vector<uint8_t> tmp;
    if (index->snapshot(tmp) != 0) return -3;

    size_t bytes = tmp.size();
    uint8_t* data = nullptr;
    if (bytes > 0) {
        data = reinterpret_cast<uint8_t*>(malloc(bytes));
        if (!data) {
            return -4;
        }
        memcpy(data, tmp.data(), bytes);
    }
    *buffer = data;
    *size = bytes;
    return 0;
}

int restore_index(void* index_ptr, const uint8_t* buffer, size_t size) {
    if (!index_ptr || !buffer) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    if (!index->supports_snapshot()) return -2;
    return index->restore(buffer, size);
}

void free_snapshot_buffer(uint8_t* buffer) {
    if (buffer) free(buffer);
}

void hnsw_enable_insert_telemetry(void* index_ptr, bool enabled) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto h = dynamic_cast<HNSW<float>*>(base)) h->enable_insert_telemetry(enabled);
}

size_t hnsw_last_insert_updates(void* index_ptr) {
    if (!index_ptr) return 0;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto h = dynamic_cast<HNSW<float>*>(base)) return h->last_insert_update_count();
    return 0;
}

void hnsw_configure_partial_insert(void* index_ptr, size_t limit) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto h = dynamic_cast<HNSW<float>*>(base)) h->configure_partial_insert(limit);
}

void hnsw_disable_partial_insert(void* index_ptr) {
    if (!index_ptr) return;
    auto base = static_cast<IndexBase<float>*>(index_ptr);
    if (auto h = dynamic_cast<HNSW<float>*>(base)) h->disable_partial_insert();
}

#if defined(__linux__)
#include <malloc.h>
#endif

}  // extern "C"
