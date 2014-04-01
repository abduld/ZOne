

project(libzrts)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
find_package(CUDA REQUIRED)

if(NOT CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif(NOT CUDA_FOUND)

set (CUDA_VERBOSE_BUILD ON)

if (UNIX)
    include(CheckLibraryExists)
    check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME )
    if(NOT HAVE_CLOCK_GETTIME)
        message(FATAL_ERROR "clock_gettime not found")
    endif(NOT HAVE_CLOCK_GETTIME)
endif(UNIX)

set(CUDA_NVCC_FP_FLAGS
    "-ftz=false;"
    "-prec-div=true;"
    "-prec-sqrt=true;"
)

set(CUDA_NVCC_FLAGS_RELEASE 
    "-O3;"
    "-lm;"
)

if (UNIX)
    set(CUDA_NVCC_FLAGS_RELEASE
        ${CUDA_NVCC_FLAGS_RELEASE}
        "-Xcompiler;-malign-double;"
    )
else (UNIX)
    set(CUDA_NVCC_FLAGS_RELEASE
        ${CUDA_NVCC_FLAGS_RELEASE}
        "-Xcompiler;-bigobj;"
    )
endif (UNIX)

file( GLOB  zhdr *.hpp *.h )
file( GLOB  zsrc *.cpp *.c *.cu)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

cuda_add_library(${name} ${zhdr} ${zsrc})
target_link_libraries(${name} uv)

if(UNIX)
    target_link_libraries(${name} ${LIBRT_LIBRARIES})
endif(UNIX)


