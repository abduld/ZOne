

#include "z.h"

static zMemoryGroup_t zReadArray(zState_t st, const char *fileName,
                                 zMemoryType_t typ, int rank, size_t *dims) {

  zMemoryGroup_t mg = zMemoryGroup_new(st, typ, rank, dims);
  size_t memBytecount = zMemoryGroup_getByteCount(mg);
  int nMems = zMemoryGroup_getMemoryCount(mg);

  tbb::parallel_for(0, nMems, [=](int ii) {
    size_t offset = ii * (memBytecount / nMems);
    size_t end = zMin((ii + 1) * (memBytecount / nMems), memBytecount);
    size_t bufferSize = end - offset;
    zMemory_t mem = zMemoryGroup_getMemory(mg, ii);
    zFile_t file = zFile_new(st, fileName, O_RDONLY);
    zFile_readChunk(file, zMemory_getHostMemory(mem), bufferSize, offset);
    zMemory_copyToDevice(mem);
    zFile_delete(file);
  });

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
