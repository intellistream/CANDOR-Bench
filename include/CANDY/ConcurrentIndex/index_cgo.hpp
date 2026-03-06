#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    INDEX_TYPE_HNSW = 0,
    INDEX_TYPE_PARLAYHNSW = 1,
    INDEX_TYPE_PARLAYVAMANA = 2,
    INDEX_TYPE_VAMANA = 3,
    // 4, 5, 6 reserved
} IndexType;

typedef enum {
    DATA_TYPE_FLOAT = 0,
    DATA_TYPE_INT8 = 1,
    DATA_TYPE_UINT8 = 2,
} DataType;

typedef enum {
    METRIC_L2 = 0,
    METRIC_IP = 1,
    METRIC_COSINE = 2,
} MetricType;

typedef struct {
    size_t dim;
    size_t max_elements;
    size_t M;
    size_t ef_construction;
    float level_m;
    float alpha;
    size_t visit_limit;
    size_t seal_threshold;
    size_t num_threads;
    DataType data_type;
    bool use_node_lock;
    int metric;
    size_t worker_scheduler;
    IndexType sealed_index_type;
} IndexParams;

typedef struct {
    size_t ef_search;
    size_t beam_width;
    float alpha;
    size_t visit_limit;
} C_QueryParams;

void* create_index(IndexType type, IndexParams params);
void destroy_index(void* index_ptr);

int build(void* index_ptr, float* data, uint32_t* tags, size_t num_points);
void set_query_params(void* index_ptr, C_QueryParams params);
int search(void* index_ptr, float* query, size_t k, uint32_t* res_tags);
int batch_insert(void* index_ptr, float* batch_data, uint32_t* batch_tags,
                 size_t batch_size);
int batch_partial_insert(void* index_ptr, float* batch_data,
                              uint32_t* batch_tags, size_t batch_size,
                              size_t* partial_limits, size_t* update_counts);
int batch_search(void* index_ptr, float* batch_queries, size_t k,
                 size_t num_queries, uint32_t** batch_results, size_t* watermark_out,
                 size_t visible_ts);

void save_stat(void* index_ptr, const char* filename);

void dump_stats(void* index_ptr, char* str);

int snapshot_index(void* index_ptr, uint8_t** buffer, size_t* size);
int restore_index(void* index_ptr, const uint8_t* buffer, size_t size);
void free_snapshot_buffer(uint8_t* buffer);

void hnsw_enable_insert_telemetry(void* index_ptr, bool enabled);
size_t hnsw_last_insert_updates(void* index_ptr);
void hnsw_configure_partial_insert(void* index_ptr, size_t limit);
void hnsw_disable_partial_insert(void* index_ptr);

#ifdef __cplusplus
}
#endif
