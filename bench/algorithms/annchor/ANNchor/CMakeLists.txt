cmake_minimum_required(VERSION 3.0...3.26)
project(annchor LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

add_library(annchor INTERFACE)
add_library(annchor::annchor ALIAS annchor)
target_include_directories(annchor INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
find_package(TBB REQUIRED)
target_link_libraries(annchor INTERFACE TBB::tbb)

add_executable(bench src/bench.cpp)
add_executable(observe_sharing src/observe_sharing.cpp)
add_executable(observe_cache src/observe_cache.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(bench PRIVATE annchor TBB::tbb OpenMP::OpenMP_CXX)
target_link_libraries(observe_sharing PRIVATE annchor TBB::tbb OpenMP::OpenMP_CXX)
target_link_libraries(observe_cache PRIVATE annchor TBB::tbb OpenMP::OpenMP_CXX)

if(USE_ASAN_ANNCHOR)
    set(BENCH_OPTIMIZATION "-O1")
    message(STATUS "[ANNchor] ASAN enabled: -O1")
else()
    set(BENCH_OPTIMIZATION "-Ofast")
    message(STATUS "[ANNchor] Release mode: -Ofast")
endif()

set(COMMON_FLAGS ${BENCH_OPTIMIZATION} -std=c++17 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -mavx2)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(bench PRIVATE ${COMMON_FLAGS})
    target_compile_options(observe_sharing PRIVATE ${COMMON_FLAGS})
    if(USE_ASAN_ANNCHOR)
        target_compile_options(bench PRIVATE -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(bench PRIVATE -fsanitize=address)
    endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(bench PRIVATE ${COMMON_FLAGS} -ftree-vectorizer-verbose=0)
    target_compile_options(observe_sharing PRIVATE ${COMMON_FLAGS})
    target_compile_options(observe_cache PRIVATE ${COMMON_FLAGS})
    if(USE_ASAN_ANNCHOR)
        target_compile_options(bench PRIVATE -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(bench PRIVATE -fsanitize=address)
    endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(bench PRIVATE /O2 /DHAVE_CXX0X /W1 /openmp /EHsc)
    target_compile_options(observe_sharing PRIVATE /O2 /DHAVE_CXX0X /W1 /openmp /EHsc)
endif()
