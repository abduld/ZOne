

#include "z.h"

static zMemoryGroup_t zReadArray(zState_t st, const char *file,
                                 zMemoryType_t typ, int rank, size_t *dims) {
  zBool deviceMemoryAllocatedQ;
  size_t byteCount;
  size_t chunkByteCount;
  uv_work_t work_req;
  ReadArrayWork_t workData;
  uv_mutex_t mutexDeviceMemoryAllocatedQ;

  uv_loop_t *loop = zState_getLoop(st);

  zMemoryGroup_t mg = zMemoryGroup_new(st, type, rank, dims);
  deviceMemoryAllocatedQ = zFalse;

  zFile_t file = zFile_open(st, file, 'r');
  // TODO: one can interleave the host memory alloc with the file read here
  // TODO: need to know if malloc is thread safe...

  for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
      zMemory_t mem = zMemoryGroup_getMemory(mg, ii);
      zFile_readChunk(st, file, zMemory_getHostMemory(mem), zMemory_getByteCount(mem));
      if (zSuccessQ(zErr)) {
        if (deviceMemoryAllocatedQ == zFalse) {
          // block until memory has been allocated
          while (zMemoryGroup_getDeviceMemoryStatus(mg) == zMemoryStatus_unallocated && zSuccessQ(zErr)) {}
          deviceMemoryAllocatedQ = zTrue;
        }
        // at this point we know that device memory has been allocated
        zMemory_copyToDevice(st, mem);
      }
  }

  return mg;
}

zMemoryGroup_t zReadBit8Array(zState_t st, const char *file, int rank,
                              size_t *dims) {
  return zReadArray(st, file, zMemoryType_bit8, rank, dims);
}

zMemoryGroup_t zReadInt32Array(zState_t st, const char *file, int rank,
                               size_t *dims) {
  return zReadArray(st, file, zMemoryType_int32, rank, dims);
}

zMemoryGroup_t zReadInt64Array(zState_t st, const char *file, int rank,
                               size_t *dims) {
  return zReadArray(st, file, zMemoryType_int64, rank, dims);
}

zMemoryGroup_t zReadFloatArray(zState_t st, const char *file, int rank,
                               size_t *dims) {
  return zReadArray(st, file, zMemoryType_float, rank, dims);
}

zMemoryGroup_t zReadDoubleArray(zState_t st, const char *file, int rank,
                                size_t *dims) {
  return zReadArray(st, file, zMemoryType_double, rank, dims);
}
