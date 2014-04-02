

#include "z.h"

static zMemoryGroup_t zReadArray(zState_t st, const char *file,
                                 zMemoryType_t typ, int rank, size_t *dims) {}

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
