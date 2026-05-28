cmake_minimum_required (VERSION 3.10)

project(SymphonyQG LANGUAGES CXX)

set(CXX_STANDARD 17)


# AVX512 required
include(CheckCXXSourceRuns)
set(CMAKE_REQUIRED_FLAGS "-march=native")
set(AVX512_CXX_CODE "
#include <immintrin.h>
int main() {
    __m512 a = _mm512_set1_ps(1.0f);  // AVX512 intrinsic
    return 0;
}")
check_cxx_source_runs("${AVX512_CXX_CODE}" AVX512_RUNS)
if(AVX512_RUNS)
    message(STATUS "AVX512 is supported by the compiler and runs successfully.")
    add_compile_options(-mavx512f)
else()
    message(FATAL_ERROR "AVX512 is either not supported by the compiler or cannot run successfully on the target system.")
endif()
unset(CMAKE_REQUIRED_FLAGS)



include_directories(${PROJECT_SOURCE_DIR}/symqglib)
include_directories(/usr/include/python3.11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EIGEN "-DEIGEN_DONT_PARALLELIZE")

SET(CMAKE_CXX_FLAGS  "-Wall -Wextra -Ofast -lrt -march=native -fpic -fopenmp -ftree-vectorize -fexceptions ${TEMPLATE_CONSTANTS} ${EIGEN}")

add_subdirectory(test)