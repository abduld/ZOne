

project(libzrts)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake )

find_package(TBB REQUIRED)
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

set(TBB_LIBRARIES
    "tbb"
    "tbbmalloc"
)

include_directories( ${TBB_INCLUDE_DIRS} )

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

set (name "zrt")
cuda_add_library(${name} ${zsrc} SHARED)
target_link_libraries(${name} ${TBB_LIBRARIES})

if(UNIX)
    target_link_libraries(${name} ${LIBRT_LIBRARIES})
endif(UNIX)


