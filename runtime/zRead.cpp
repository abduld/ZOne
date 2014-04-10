

#include "z.h"

void afterRead(uv_fs_t * req) {
  uv_fs_req_cleanup(req);
  if (req->result < 0) {
    fprintf(stderr, "Read error: %s\n", uv_strerror(uv_last_error(uv_default_loop())));
  } else if (req->result == 0) {
      zFile_close(file);
  }
  zMemory_t mem = req->data;
  // spin until memory has been allocated
  while (!zMemory_deviceMemoryAllocatedQ(mg)) {}
  zMemory_copyToDevice(mem);
}

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

  zFile_t file = zFile_open(st, file, S_IREAD);
  // TODO: one can interleave the host memory alloc with the file read here
  // TODO: need to know if malloc is thread safe...

  size_t offset = 0;
  for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
      zMemory_t mem = zMemoryGroup_getMemory(mg, ii);
      size_t memBytecount = zMemory_getByteCount(mem);
      zFile_readChunk(file, zMemory_getHostMemory(mem), memBytecount, offset, afterRead, mem);
      offset += memByteCount;
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
