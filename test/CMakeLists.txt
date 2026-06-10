
#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -Werror=return-type ")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)
enable_testing()
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/torchscripts DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/datasets
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/src/cl
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
macro(add_catch_test appName SOURCE_FILES SOURCE_LIBS)
    get_property(SOURCE_FILES_LOCAL GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP")
    set(${SOURCE_FILES} ${SOURCE_FILES_LOCAL})
    add_executable(${appName} ${SOURCE_FILES})
    target_link_libraries(${appName} ${SOURCE_LIBS})
    target_link_options(${appName} PRIVATE "-Wl,-rpath,./")
    add_test(
            NAME catch_${appName}
            COMMAND $<TARGET_FILE:${appName}> --success
    )
endmacro()
macro(add_catch_test_ray appName SOURCE_FILES SOURCE_LIBS)
    get_property(SOURCE_FILES_LOCAL GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP")
    set(${SOURCE_FILES} ${SOURCE_FILES_LOCAL})
    add_library(${appName} SHARED ${SOURCE_FILES})
    target_link_libraries(${appName} ${SOURCE_LIBS})
    target_link_options(${appName} PRIVATE "-Wl,-rpath,./")
    add_test(
            NAME catch_${appName}
            COMMAND $<TARGET_FILE:${appName}> --success
    )
endmacro()
add_catch_test(hnsw_test index/hnsw_test.cpp CANDORBENCH)
add_catch_test(cpp_test index/simple_test.cpp CANDORBENCH)
add_catch_test(flatIndex_test index/flat_index_test.cpp CANDORBENCH)
add_catch_test(flatAMMIPIndex_test index/flat_ammip_index_test.cpp CANDORBENCH)
add_catch_test(flatAMMIPObjIndex_test index/flat_ammip_obj_index_test.cpp CANDORBENCH)
add_catch_test(ppIndex_test index/parallel_partition_index_test.cpp CANDORBENCH)
add_catch_test(pqIndex_Test index/pq_index_test.cpp CANDORBENCH)
add_catch_test(onlinePQ_test index/online_pq_index_test.cpp CANDORBENCH)
add_catch_test(onlineIVFLSH_test index/online_ivflsh_index_test.cpp CANDORBENCH)
add_catch_test(yinYang_test index/yin_yang_graph_index_test.cpp CANDORBENCH)
add_catch_test(yinYangSimple_test index/yin_yang_graph_simple_index_test.cpp CANDORBENCH)
add_catch_test(nnDescent_test index/nn_descent_index_test.cpp CANDORBENCH)
add_catch_test(kdTree_test index/kd_tree_test.cpp CANDORBENCH)
add_catch_test(dpgIndex_test index/dpg_index_test.cpp CANDORBENCH)
add_catch_test(lshAPGIndex_test index/lshapg_index_test.cpp CANDORBENCH)
#if(ENABLE_CUDA)
#    add_catch_test(song_test index/song_test.cu CANDORBENCH)
#endif ()
if (ENABLE_SPTAG)
    add_catch_test(sptagIndex_test index/sptag_index_test.cpp CANDORBENCH)
endif ()

if (ENABLE_OPENCL)
    add_catch_test(cl_test SystemTest/CLTest.cpp CANDORBENCH)
endif ()
if (ENABLE_RAY)
    add_catch_test(dp_test index/distributed_partition_index_test.cpp CANDORBENCH)
    add_catch_test_ray(dp_testL index/distributed_partition_index_test.cpp CANDORBENCH)
endif ()

