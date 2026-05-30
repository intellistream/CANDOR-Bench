cmake_minimum_required(VERSION 3.10)
project(utils)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

option(ENABLE_CUDA "Enable optional CUDA support" ON)

set(UTILS_HAVE_CUDA OFF)

if(ENABLE_CUDA)
    cmake_policy(SET CMP0146 NEW)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(UTILS_HAVE_CUDA ON)
            set(CMAKE_CUDA_STANDARD 14)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER_VERSION}")
            message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
            cmake_policy(SET CMP0104 NEW)
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
        else()
            message(WARNING "CUDA compiler detected but CUDAToolkit not found - disabling CUDA support for utils")
        endif()
    else()
        message(STATUS "CUDA compiler not found - GPU acceleration for utils will be disabled")
    endif()
else()
    message(STATUS "CUDA support disabled via ENABLE_CUDA=OFF")
endif()

set(UTILS_ENABLE_CUDA_DEFINE 0)
if(UTILS_HAVE_CUDA)
    set(UTILS_ENABLE_CUDA_DEFINE 1)
endif()
add_compile_definitions(UTILS_ENABLE_CUDA=${UTILS_ENABLE_CUDA_DEFINE})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mfma -O3 -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -funroll-loops -ffast-math")
endif()

if(UTILS_HAVE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -use_fast_math")
endif()

option(ENABLE_DEBUG_INFO "Enable GDB debug information" OFF)

if(ENABLE_DEBUG_INFO)
    message(STATUS "GDB debug info enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

add_executable(calc_recall calc_recall.cpp)
add_executable(calc_incr_recall calc_incr_recall.cpp)
add_executable(compute_gt compute_gt.cpp)

if(UTILS_HAVE_CUDA)
    add_executable(compute_incr_gt 
        compute_incr_gt.cpp 
        knn_processor.cpp
        cuda_knn.cu
    )
    target_link_libraries(compute_incr_gt CUDA::cudart)
else()
    add_executable(compute_incr_gt 
        compute_incr_gt.cpp 
        knn_processor.cpp
    )
endif()

target_compile_definitions(compute_incr_gt PRIVATE UTILS_ENABLE_CUDA=${UTILS_ENABLE_CUDA_DEFINE})

add_executable(crop crop.cpp)
add_executable(fvecs_to_bin fvecs_to_bin.cpp)

# Debug utilities
add_executable(debug_incr_gt debug/debug_incr_gt.cpp)
add_executable(debug_incr_results debug/debug_incr_results.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(calc_recall PRIVATE -Wall -Wextra)
    target_compile_options(calc_incr_recall PRIVATE -Wall -Wextra)
    target_compile_options(compute_gt PRIVATE -Wall -Wextra)
    target_compile_options(compute_incr_gt PRIVATE -Wall -Wextra)
    target_compile_options(crop PRIVATE -Wall -Wextra)
    target_compile_options(fvecs_to_bin PRIVATE -Wall -Wextra)
    target_compile_options(debug_incr_gt PRIVATE -Wall -Wextra)
    target_compile_options(debug_incr_results PRIVATE -Wall -Wextra)
endif()

if(UTILS_HAVE_CUDA)
    target_compile_options(compute_incr_gt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wall>)
endif()

install(TARGETS calc_recall calc_incr_recall compute_gt compute_incr_gt crop fvecs_to_bin debug_incr_gt debug_incr_results
        RUNTIME DESTINATION bin)
