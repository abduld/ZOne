

#include "z.h"

static zMemoryGroup_t zReadArray(zState_t st, const char *fileName,
                                 zMemoryType_t typ, int rank, size_t *dims) {

  zMemoryGroup_t mg = zMemoryGroup_new(st, type, rank, dims);
  size_t memBytecount = zMemoryGroup_getByteCount(mem);
  int nMems = zMemoryGroup_getMemoryCount(mg);

  tbb::parallel_for(size_t(0), nMems, [=](size_t ii) {
    size_t offset = ii * (memBytecount / nMems);
    size_t end =
        zMin((ii + 1) * (memBytecount / nMems), zMemoryGroup_getByteCount(mg));
    size_t bufferSize = end - offset;
    zMemory_t mem = zMemoryGroup_getMemory(mg, ii);
    zFile_t file = zFile_open(fileName, 'r');
    zFile_readChunk(file, zMemory_getHostMemory(mem), bufferSize, offset);
    zMemory_copyToDevice(mem);
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
