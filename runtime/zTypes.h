
#ifndef __ZTYPES_H__
#define __ZTYPES_H__

#include <stdint.h>
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

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
    return sizeof(unsigned char);
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

static size_t computeFlattenedLength(int rank, size_t *dims) {
  size_t sz = 1;

  for (int ii = 0; ii < rank; ii++) {
    sz *= dims[ii];
  }
  return sz;
}

static size_t computeByteCount(zMemoryType_t typ, int rank, size_t *dims) {
  return computeFlattenedLength(rank, dims) * zMemoryType_size(typ);
}

typedef struct st_zMemory_t *zMemory_t;
typedef struct st_zMemoryGroup_t *zMemoryGroup_t;
typedef struct st_zState_t *zState_t;
typedef struct st_zFunctionInformation_t *zFunctionInformation_t;
typedef struct st_zLogEntry_t *zLogEntry_t;
typedef struct st_zLogger_t *zLogger_t;
typedef struct st_zStream_t zStream_t;
typedef struct st_zTimer_t *zTimer_t;
typedef struct st_zTimerNode_t *zTimerNode_t;
typedef struct st_zError_t *zError_t;
typedef struct st_zFile_t *zFile_t;
typedef struct st_zStringBuffer_t *zStringBuffer_t;

typedef vector<zMemoryGroup_t> zMemoryGroupList_t;
typedef vector<zStream_t> zStreamList_t;
typedef unordered_map<string, zFunctionInformation_t> zFunctionInformationMap_t;

typedef enum en_zMemoryStatus_t {
  zMemoryStatus_unallocated = 0,
  zMemoryStatus_allocatedHost,
  zMemoryStatus_allocatedDevice,
  zMemoryStatus_dirtyDevice,
  zMemoryStatus_dirtyHost,
  zMemoryStatus_cleanDevice,
  zMemoryStatus_cleanHost,
} zMemoryStatus_t;

#endif /* __ZTYPES_H__ */
