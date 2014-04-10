
#ifndef __ZCUDA_H__
#define __ZCUDA_H__

#include <cuda.h>
#include <cuda_runtime.h>

#define zCUDA_check(stmt)                                                       \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      zError("Failed to run stmt ", #stmt);                                    \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#endif /* __ZCUDA_H__ */
