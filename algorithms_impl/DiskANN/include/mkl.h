#pragma once

#if defined(__has_include_next)
#if __has_include_next(<mkl.h>)
#include_next <mkl.h>
#elif __has_include("/opt/intel/oneapi/mkl/latest/include/mkl.h")
#include "/opt/intel/oneapi/mkl/latest/include/mkl.h"
#elif __has_include("/opt/intel/oneapi/mkl/2026.0/include/mkl.h")
#include "/opt/intel/oneapi/mkl/2026.0/include/mkl.h"
#else
#error "Unable to locate the Intel MKL headers for DiskANN."
#endif
#else
#include "/opt/intel/oneapi/mkl/latest/include/mkl.h"
#endif
