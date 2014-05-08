
#ifndef __ZCUDA_H__
#define __ZCUDA_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include "zTypes.h"

#define zCUDA_check(stmt)                                                      \
	_zCUDA_check(zFile, zFunction, zLine, stmt)

#define _zCUDA_check(file, fun, line, stmt)                                                     \
  do {                                                                         \
    cudaError_t p_err = stmt;                                                    \
    if (p_err != cudaSuccess) {                                                  \
      printf("%s::%s(%d)::%s : %s -- %d\n", file, fun, line, #stmt, cudaGetErrorString(p_err), p_err);					   \
    }                                                                          \
  } while (0)

void zCUDA_malloc(zMemoryGroup_t mg);
void zCUDA_free(void *mem);
void zCUDA_copyToDevice(zMemory_t mem);
void zCUDA_copyToHost(zMemory_t mem);
void zCUDA_copyToDevice(zMemoryGroup_t mem);
void zCUDA_copyToHost(zMemoryGroup_t mem);

#endif /* __ZCUDA_H__ */
