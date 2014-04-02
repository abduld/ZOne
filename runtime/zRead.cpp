

#include "z.h"

static zMemoryGroup_t zReadArray(zState_t st, const char *file,
                                 zMemoryType_t typ, int rank, size_t *dims) {
  size_t byteCount;
  uv_work_t work_req;

  uv_loop_t *loop = zState_getLoop(st);

  zMemoryGroup_t mem = zMemoryGroup_new(st, type, rank, dims);

  work->data = { st, mem };

  byteCount = computeByteCount(typ, rank, dims);

  uv_queue_work(loop, &work, allocDeviceMemory, setDeviceMemoryAllocated);

  // for each chunk read the data and copy it to the memory in the group

   
}

zMemoryGroup_t zReadBit8Array(zState_t st, const char *file, int rank,
                              size_t *dims) {
  return zReadArray(st, file, zMemoryType_bit8);
}

zMemoryGroup_t zReadInt32Array(zState_t st, const char *file, int rank,
                               size_t *dims) {
  return zReadArray(st, file, zMemoryType_int32);
}

zMemoryGroup_t zReadInt64Array(zState_t st, const char *file, int rank,
                               size_t *dims) {
  return zReadArray(st, file, zMemoryType_int64);
}

zMemoryGroup_t zReadFloatArray(zState_t st, const char *file, int rank,
                               size_t *dims) {
  return zReadArray(st, file, zMemoryType_float);
}

zMemoryGroup_t zReadDoubleArray(zState_t st, const char *file, int rank,
                                size_t *dims) {
  return zReadArray(st, file, zMemoryType_double);
}
