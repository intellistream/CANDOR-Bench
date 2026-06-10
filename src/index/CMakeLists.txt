add_sources(
        candor_object.cpp
        abstract_index.cpp
        bucketed_flat_index.cpp
        buffered_congestion_drop_index.cpp
        flat_index.cpp
        #flat_ammip_index.cpp
        #flat_ammip_obj_index.cpp
        #parallel_partition_index.cpp
        online_pq_index.cpp
        #online_ivflsh_index.cpp
        #online_ivfl2_h_index.cpp
        index_table.cpp
        #pq_index.cpp
        hnsw_naive_index.cpp
        faiss_index.cpp
        #yin_yang_graph_index.cpp
        #yin_yang_graph_simple_index.cpp
        congestion_drop_index.cpp
        nn_descent_index.cpp
        flann_index.cpp
        dpg_index.cpp
        lshapg_index.cpp
        flat_gpu_index.cpp
)
add_subdirectory(hashing_models)
add_subdirectory(parallel_partition_index)
#add_subdirectory(pq_index)
add_subdirectory(online_pq_index)
add_subdirectory(hnsw_naive)
#add_subdirectory(yin_yang_graph_index)
add_subdirectory(congestion_drop_index)
add_subdirectory(flann_index)
add_subdirectory(lshapg_index)
add_subdirectory(flat_gpu_index)

#if (ENABLE_CUDA)
#    add_subdirectory(song)
#endif ()

if (ENABLE_RAY)
    add_subdirectory(distributed_partition_index)
    add_sources(distributed_partition_index.cpp)
endif ()
if (ENABLE_SPTAG)
    add_sources(sptag_index.cpp)
endif ()
