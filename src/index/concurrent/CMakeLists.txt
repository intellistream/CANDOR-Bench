cmake_minimum_required(VERSION 3.10)
project(concurrency-index CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -ggdb -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)

add_subdirectory(hnsw/hnswlib hnsw)
add_subdirectory(parlayann/parlayann parlayann)
add_subdirectory(vamana/diskann diskann)

if(TARGET diskann)
    set_property(TARGET diskann PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
if(TARGET diskann_s)
    set_property(TARGET diskann_s PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

set(CC_INDEX_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/parlayann/parlayann
    ${CMAKE_CURRENT_SOURCE_DIR}/parlayann/parlayann/parlaylib/include
    ${CMAKE_CURRENT_SOURCE_DIR}/parlayann/parlayann/algorithms
    ${CMAKE_CURRENT_SOURCE_DIR}/parlayann/parlayann/algorithms/bench
    ${CMAKE_CURRENT_SOURCE_DIR}/vamana/diskann/include
    ${CMAKE_CURRENT_SOURCE_DIR}/hnsw/hnswlib
    ${TBB_INCLUDE_DIRS}
)

set(CC_INDEX_LIBS
    hnswlib::hnswlib
    ParlayANN
    diskann
    OpenMP::OpenMP_CXX
    TBB::tbb
)

# Python module: index bindings plus the native concurrent driver.
# Requires pybind11: cmake -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(concurrency_native python/bindings.cpp segment/segment.cpp)
    target_include_directories(concurrency_native PRIVATE ${CC_INDEX_INCLUDES})
    target_link_libraries(concurrency_native PRIVATE ${CC_INDEX_LIBS})
else()
    message(STATUS "pybind11 not found - skipping concurrency_native python module")
endif()
