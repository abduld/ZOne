
#ifndef __ZTYPES_H__
#define __ZTYPES_H__

#include <stdint.h>

typedef enum {
  zMemoryType_unknown = -99,
  zMemoryType_void = 0,
  zMemoryType_bit8,
  zMemoryType_int32,
  zMemoryType_int64,
  zMemoryType_float,
  zMemoryType_double
} zMemoryType_t;

static inline size_t zMemoryType_size(zMemoryType_t typ) {
  switch (typ) {
  case zMemoryType_void:
    return sizeof(void);
  case zMemoryType_bit8:
    return sizeof(int8_t);
  case zMemoryType_int32:
    return sizeof(int32_t);
  case zMemoryType_int64:
    return sizeof(int64_t);
  case zMemoryType_float:
    return sizeof(float);
  case zMemoryType_double:
    return sizeof(double);
  }
  return (size_t)(-1);
}

static size_t flattenedLength(int rank, size_t *dims) {
  int ii = 0;
  size_t sz = 1;

  while (ii < rank) {
    sz *= dims[ii];
  }
  return sz;
}

static size_t flattenedLength(int rank, size_t *dims, zMemoryType_t typ) {
  return flattenedLength(rank, dims) * zMemoryType_size(typ);
}

typedef struct st_zMemory_t zMemory_t;
typedef struct st_zMemoryGroup_t *zMemoryGroup_t;
typedef struct st_zState_t *zState_t;

#endif /* __ZTYPES_H__ */
