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

find_package(OpenMP REQUIRED)
target_link_libraries(bench PRIVATE
    annchor
    TBB::tbb
    OpenMP::OpenMP_CXX
)

if(USE_ASAN_ANNCHOR)  
    set(BENCH_OPTIMIZATION "-O1")  
    message(STATUS "[ANNchor] ASAN enabled (controlled by main project: USE_ASAN_ANNCHOR=ON): -O1")
else()
    set(BENCH_OPTIMIZATION "-Ofast")
    message(STATUS "[ANNchor] ASAN disabled + Release mode: -Ofast")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(bench PRIVATE
        ${BENCH_OPTIMIZATION}
        -std=c++17
        -DHAVE_CXX0X
        -fopenmp
        -fpic
        -ftree-vectorize
        -mavx2
    )

    if(USE_ASAN_ANNCHOR)
        target_compile_options(bench PRIVATE
            -fsanitize=address
            -fno-omit-frame-pointer
        )
        target_link_options(bench PRIVATE
            -fsanitize=address
        )
    endif()

    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
    if(COMPILER_SUPPORT_NATIVE_FLAG)
        target_compile_options(bench PRIVATE -march=native)
        message("[ANNchor] 'bench' (Clang): enabled -march=native")
    else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
            target_compile_options(bench PRIVATE -mcpu=apple-m1)
            message("[ANNchor] 'bench' (Clang): enabled -mcpu=apple-m1")
        endif()
    endif()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(bench PRIVATE
        ${BENCH_OPTIMIZATION}
        -lrt
        -std=c++17
        -DHAVE_CXX0X
        -march=native
        -fpic
        -w
        -fopenmp
        -ftree-vectorize
        -ftree-vectorizer-verbose=0
        -mavx2
    )

    if(USE_ASAN_ANNCHOR)
        target_compile_options(bench PRIVATE
            -fsanitize=address
            -fno-omit-frame-pointer
        )
        target_link_options(bench PRIVATE
            -fsanitize=address
        )
    endif()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(bench PRIVATE
        /O2
        /DHAVE_CXX0X
        /W1
        /openmp
        /EHsc
    )

    if(USE_ASAN_ANNCHOR)
        message(WARNING "[ANNchor] MSVC does not support -fsanitize=address; use Visual Studio's built-in ASAN instead.")
    endif()
endif()
