
#ifndef __ZTYPES_H__
#define __ZTYPES_H__

typedef enum {
  zMemoryType_unknown = -99,
  zMemoryType_void = 0,
  zMemoryType_bit8,
  zMemoryType_int32,
  zMemoryType_int64,
  zMemoryType_float,
  zMemoryType_double
} zMemoryType_t;

typedef struct st_zMemory_t * zMemory_t;

#endif /* __ZTYPES_H__ */
