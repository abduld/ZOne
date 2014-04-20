

#include "z.h"

void afterReadChunk(uv_fs_t * req) {
  zMemory_t * pmem = (zMemory_t *) req->data;
  zAssert(pmem != NULL);
  zMemory_t mem = *pmem;
  zState_t st = zMemory_getState(mem);
  uv_loop_t * loop = zState_getLoop(st);
  if (req->result < 0) {
    uv_err_t uvErr = uv_last_error(loop);
    zState_setError(st, uvErr);
  } else if (req->result == 0) {
    // TODO: think about what if not all have finished reading
    // zFile_close(fileName);
  }
  // spin until memory has been allocated
  while (!zMemory_deviceMemoryAllocatedQ(mg)) {}
  zMemory_copyToDevice(mem);
  uv_fs_req_cleanup(req);
}

static zMemoryGroup_t zReadArray(zState_t st, const char *fileName,
                                 zMemoryType_t typ, int rank, size_t *dims) {

  uv_loop_t *loop = zState_getLoop(st);

  zMemoryGroup_t mg = zMemoryGroup_new(st, type, rank, dims);

  zFile_t file = zFile_open(st, fileName, S_IREAD);
  // TODO: one can interleave the host memory alloc with the fileName read here
  // TODO: need to know if malloc is thread safe...

  size_t offset = 0;
  for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
      zMemory_t mem = zMemoryGroup_getMemory(mg, ii);
      size_t memBytecount = zMemory_getByteCount(mem);
      zFile_readChunk(file, zMemory_getHostMemory(mem), memBytecount, offset,
        afterReadChunk, &zMemoryGroup_getMemory(mg, ii));
      offset += memBytecount;
  }

  return mg;
}

zMemoryGroup_t zReadBit8Array(zState_t st, const char *fileName, int rank,
                              size_t *dims) {
  return zReadArray(st, fileName, zMemoryType_bit8, rank, dims);
}

zMemoryGroup_t zReadInt32Array(zState_t st, const char *fileName, int rank,
                               size_t *dims) {
  return zReadArray(st, fileName, zMemoryType_int32, rank, dims);
}

zMemoryGroup_t zReadInt64Array(zState_t st, const char *fileName, int rank,
                               size_t *dims) {
  return zReadArray(st, fileName, zMemoryType_int64, rank, dims);
}

zMemoryGroup_t zReadFloatArray(zState_t st, const char *fileName, int rank,
                               size_t *dims) {
  return zReadArray(st, fileName, zMemoryType_float, rank, dims);
}

zMemoryGroup_t zReadDoubleArray(zState_t st, const char *fileName, int rank,
                                size_t *dims) {
  return zReadArray(st, fileName, zMemoryType_double, rank, dims);
}
