

project(libzrts CXX)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH} )

find_package(TBB REQUIRED)
find_package(CUDA REQUIRED)

if(NOT CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif(NOT CUDA_FOUND)

set (CUDA_VERBOSE_BUILD ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CUDA_NVCC_FP_FLAGS
    "-ftz=false;"
    "-prec-div=true;"
    "-prec-sqrt=true;"
)

set(CUDA_NVCC_FLAGS
    "-O0;"
    "-g;"
    "-lm;"
)

set(TBB_LIBRARIES
    "tbb"
    "tbbmalloc"
)

include_directories( ${TBB_INCLUDE_DIRS} )

if (UNIX)
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}
        #  "--compiler-options -std=c++11"
    )
else (UNIX)
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS_RELEASE}
        "-Xcompiler;-bigobj;"
    )
endif (UNIX)
list( APPEND CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -g -ftest-coverage -fprofile-arcs"
)

file( GLOB  zhdr *.hpp *.h )
file( GLOB  zsrc *.cpp *.c *.cu)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})


cuda_include_directories(${CMAKE_SOURCE_DIR})
cuda_include_directories(${CUDA_INCLUDE_DIRS})

set_source_files_properties(${zsrc} PROPERTIES COMPILE_FLAGS -std=c++11)

set (zrtLib "zrt")
add_library(${zrtLib} ${zsrc})
target_link_libraries(${zrtLib} ${TBB_LIBRARIES})

if(UNIX)
    target_link_libraries(${zrtLib} ${LIBRT_LIBRARIES})
endif(UNIX)


cuda_add_executable(conv1D examples/convolution1D.cu OPTIONS -arch sm_20)
target_link_libraries(conv1D ${zrtLib} ${TBB_LIBRARIES})


