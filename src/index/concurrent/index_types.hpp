#pragma once

// Index type tags and construction parameters, shared by the factory,
// the segment index and the Python bindings.

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    INDEX_TYPE_HNSW = 0,
    INDEX_TYPE_PARLAYHNSW = 1,
    INDEX_TYPE_PARLAYVAMANA = 2,
    INDEX_TYPE_VAMANA = 3,
    INDEX_TYPE_ANNCHOR = 4,
    INDEX_TYPE_SEGMENT = 5,
    INDEX_TYPE_ANNCHOR_DEV = 6,
    INDEX_TYPE_ANNCHOR_M1 = 7,
    INDEX_TYPE_ANNCHOR_PREEMPT = 8,
    INDEX_TYPE_ANNCHOR_M3 = 9,
    INDEX_TYPE_ANNCHOR_TRIM = 10,
    INDEX_TYPE_ANNCHOR_M2 = 11,
    INDEX_TYPE_HNSW_VISIBLE = 12,
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
