macro(get_log_level_value ${CMAKE_PROJECT_NAME}_LOGGING_VALUE)
    if (${${CMAKE_PROJECT_NAME}_LOGGING_LEVEL} STREQUAL "TRACE")
        message("-- Log level is set to TRACE!")
        set(${CMAKE_PROJECT_NAME}_LOGGING_VALUE 6)
    elseif (${${CMAKE_PROJECT_NAME}_LOGGING_LEVEL} STREQUAL "DEBUG")
        message("-- Log level is set to DEBUG!")
        set(${CMAKE_PROJECT_NAME}_LOGGING_VALUE 5)

    elseif (${${CMAKE_PROJECT_NAME}_LOGGING_LEVEL} STREQUAL "INFO")
        message("-- Log level is set to INFO!")
        set(${CMAKE_PROJECT_NAME}_LOGGING_VALUE 4)
    elseif (${${CMAKE_PROJECT_NAME}_LOGGING_LEVEL} STREQUAL "WARN")
        message("-- Log level is set to WARN!")
        set(${CMAKE_PROJECT_NAME}_LOGGING_VALUE 3)

    elseif (${${CMAKE_PROJECT_NAME}_LOGGING_LEVEL} STREQUAL "ERROR")
        message("-- Log level is set to ERROR!")
        set(${CMAKE_PROJECT_NAME}_LOGGING_VALUE 2)

    elseif (${${CMAKE_PROJECT_NAME}_LOGGING_LEVEL} STREQUAL "FATAL_ERROR")
        message("-- Log level is set to FATAL_ERROR!")
        set(${CMAKE_PROJECT_NAME}_LOGGING_VALUE 1)

    else ()
        message(WARNING "-- Could not set ${CMAKE_PROJECT_NAME}_LOGGING_VALUE as ${${CMAKE_PROJECT_NAME}_LOGGING_LEVEL} did not equal any logging level!!!  Defaulting to debug!")
        set(${CMAKE_PROJECT_NAME}_LOGGING_VALUE 5)
    endif ()
endmacro(get_log_level_value ${CMAKE_PROJECT_NAME}_LOGGING_VALUE)

macro(add_source PROP_NAME SOURCE_FILES)
    set(SOURCE_FILES_ABSOLUTE)
    foreach (it ${SOURCE_FILES})
        get_filename_component(ABSOLUTE_PATH ${it} ABSOLUTE)
        set(SOURCE_FILES_ABSOLUTE ${SOURCE_FILES_ABSOLUTE} ${ABSOLUTE_PATH})
    endforeach ()

    get_property(OLD_PROP_VAL GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP")
    set_property(GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP" ${SOURCE_FILES_ABSOLUTE} ${OLD_PROP_VAL})
endmacro()

macro(add_sources)
    add_source(${CMAKE_PROJECT_NAME} "${ARGN}")
endmacro()

macro(get_source PROP_NAME SOURCE_FILES)
    get_property(SOURCE_FILES_LOCAL GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP")
    set(${SOURCE_FILES} ${SOURCE_FILES_LOCAL})
endmacro()

macro(get_sources SOURCE_FILES)
    get_source(${CMAKE_PROJECT_NAME} SOURCE_FILES_LOCAL)
    set(${SOURCE_FILES} ${SOURCE_FILES_LOCAL})
endmacro()

macro(get_headers HEADER_FILES)
    #    file(GLOB_RECURSE ${HEADER_FILES} "include/*.h" "include/*.hpp")
    file(GLOB_RECURSE ${HEADER_FILES} "include/*.h" "include/*.hpp" "include/*.cuh")
endmacro()

# Define the function to detect AVX-512 support
function(detect_avx512_support result_var)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-mavx512f")
    check_cxx_source_compiles("
    #include <immintrin.h>
    int main() {
        __m512i vec = _mm512_set1_epi32(1);  // AVX-512 intrinsic
        return 0;
    }
" HAVE_AVX512)

    if(HAVE_AVX512)
        #message(STATUS "AVX-512 support detected.")
        set(${result_var} 1 PARENT_SCOPE)
    else()
        # message(STATUS "AVX-512 support NOT detected.")
        set(${result_var} 0 PARENT_SCOPE)
    endif()
endfunction()

function(detect_avx2_support result_var)
    include(CheckCXXSourceCompiles)
    # Save the current compiler flags to restore them later
    set(saved_flags "${CMAKE_CXX_FLAGS}")
    # Test AVX2 intrinsic support by compiling a minimal test program
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main() {
            __m256i vec = _mm256_set1_epi32(1);  // AVX2 intrinsic
            return 0;
        }
    " HAVE_AVX2)

    # Restore the original compiler flags
    set(CMAKE_CXX_FLAGS "${saved_flags}" PARENT_SCOPE)
    # Return TRUE or FALSE based on the test result
    if(HAVE_AVX2)
        set(${result_var} 1 PARENT_SCOPE)
    else()
        set(${result_var} 0 PARENT_SCOPE)
    endif()
endfunction()