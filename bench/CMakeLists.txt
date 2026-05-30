cmake_minimum_required(VERSION 3.10)
project(ANN-CC-bench)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release 
        CACHE STRING "Default build type: Release (options: Debug, Release, RelWithDebInfo)" 
        FORCE
    )
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -ggdb -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)  
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)  

option(USE_ASAN_ANNCHOR "Enable AddressSanitizer ONLY for ANNchor (no impact on other modules)" OFF)
set(USE_ASAN_ANNCHOR ${USE_ASAN_ANNCHOR} 
    CACHE BOOL "Control ASAN for ANNchor submodule" 
)

find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)
find_package(Boost 1.74 REQUIRED COMPONENTS context)     

add_subdirectory(algorithms/annchor/ANNchor annchor)    
add_subdirectory(algorithms/hnsw/hnswlib hnsw)         
add_subdirectory(algorithms/parlayann/parlayann parlayann)  
add_subdirectory(algorithms/vamana/diskann diskann)

if(TARGET diskann)
    set_property(TARGET diskann PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
if(TARGET diskann_s)
    set_property(TARGET diskann_s PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

add_library(index SHARED
    algorithms/index_cgo.cpp
    algorithms/annchor-m2/annchor_m2_cgo.cpp
    algorithms/annchor-m3/annchor_m3_cgo.cpp
    algorithms/segment/segment.cpp
)

target_include_directories(index PRIVATE
    ${CMAKE_SOURCE_DIR}                          
    ${CMAKE_SOURCE_DIR}/algorithms               
    ${CMAKE_SOURCE_DIR}/algorithms/parlayann/parlayann  
    ${CMAKE_SOURCE_DIR}/algorithms/vamana/diskann/include  
    ${CMAKE_SOURCE_DIR}/algorithms/annchor-dev
    ${TBB_INCLUDE_DIRS}                     
)

target_link_libraries(index PUBLIC
    hnswlib::hnswlib
    annchor::annchor
    ParlayANN
    diskann
    OpenMP::OpenMP_CXX
    TBB::tbb
    Boost::context
)
