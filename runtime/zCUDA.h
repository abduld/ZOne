
#ifndef __ZCUDA_H__
#define __ZCUDA_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include "zTypes.h"

#define zCUDA_check(stmt)                                                      \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      zError("Failed to run stmt ", #stmt);                                    \
      return -1;                                                               \
    }                                                                          \
  } while (0)

void zCUDA_malloc(zMemoryGroup_t mg);
void zCUDA_free(void *mem);
void zCUDA_copyToDevice(zMemory_t mem);
void zCUDA_copyToHost(zMemory_t mem);

#endif /* __ZCUDA_H__ */
