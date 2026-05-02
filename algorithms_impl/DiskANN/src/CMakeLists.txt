#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT                        license.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)

if(MSVC)
    add_subdirectory(dll)
else()
    message(STATUS  "Building static library..")
    #file(GLOB CPP_SOURCES *.cpp)
    set(CPP_SOURCES abstract_data_store.cpp ann_exception.cpp disk_utils.cpp 
        distance.cpp index.cpp in_mem_graph_store.cpp in_mem_data_store.cpp
        linux_aligned_file_reader.cpp math_utils.cpp natural_number_map.cpp
        in_mem_data_store.cpp in_mem_graph_store.cpp
        natural_number_set.cpp memory_mapper.cpp partition.cpp pq.cpp
        pq_flash_index.cpp scratch.cpp logger.cpp utils.cpp filter_utils.cpp index_factory.cpp abstract_index.cpp  )
    if (RESTAPI)
        list(APPEND CPP_SOURCES restapi/search_wrapper.cpp restapi/server.cpp)
    endif()
    add_library(${PROJECT_NAME} ${CPP_SOURCES})
    add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
    set(DISKANN_MKL_INCLUDE_PATH "${MKL_INCLUDE_PATH}")
    if (NOT DISKANN_MKL_INCLUDE_PATH AND DEFINED ENV{MKLROOT} AND EXISTS "$ENV{MKLROOT}/include/mkl.h")
        set(DISKANN_MKL_INCLUDE_PATH "$ENV{MKLROOT}/include")
    elseif (NOT DISKANN_MKL_INCLUDE_PATH AND EXISTS "/opt/intel/oneapi/mkl/latest/include/mkl.h")
        set(DISKANN_MKL_INCLUDE_PATH "/opt/intel/oneapi/mkl/latest/include")
    elseif (NOT DISKANN_MKL_INCLUDE_PATH AND EXISTS "/opt/intel/oneapi/mkl/2026.0/include/mkl.h")
        set(DISKANN_MKL_INCLUDE_PATH "/opt/intel/oneapi/mkl/2026.0/include")
    endif()
    if (DISKANN_MKL_INCLUDE_PATH)
        target_include_directories(${PROJECT_NAME} PRIVATE ${DISKANN_MKL_INCLUDE_PATH})
        target_include_directories(${PROJECT_NAME}_s PRIVATE ${DISKANN_MKL_INCLUDE_PATH})
    endif()
endif()

if (NOT MSVC)
    install(TARGETS ${PROJECT_NAME} LIBRARY)
endif()
