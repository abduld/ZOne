
#ifndef __ZMEMORY_H__
#define __ZMEMORY_H__

typedef enum en_zMemoryStatus_t {
  zMemoryStatus_unallocated = 0,
  zMemoryStatus_allocated
} zMemoryStatus_t;

struct st_zMemory_t {
  size_t sz;
  int rank;
  size_t *dims;
  zMemoryType_t typ;
  void *hostMemory;
  void *deviceMemory;
  zMemoryStatus_t hostMemoryStatus;
  zMemoryStatus_t deviceMemoryStatus;
  zMemoryGroup_t *group;
};

#define zMemory_getSize(mem) ((mem).sz)
#define zMemory_getType(mem) ((mem).typ)
#define zMemory_getHostMemory(mem) ((mem).hostMemory)
#define zMemory_getHostMemoryStatus(mem) ((mem).hostMemoryStatus)
#define zMemory_getDeviceMemory(mem) ((mem).deviceMemory)
#define zMemory_getDeviceMemoryStatus(mem) ((mem).deviceMemoryStatus)
#define zMemory_getMemoryGroup(mem) ((mem).group)

#define zMemory_setSize(mem, val) (zMemory_getSize(mem) = val)
#define zMemory_setType(mem, val) (zMemory_getType(mem) = val)
#define zMemory_setHostMemory(mem, val) (zMemory_getHostMemory(mem) = val)
#define zMemory_setHostMemoryStatus(mem, val)                                  \
  (zMemory_getHostMemoryStatus(mem) = val)
#define zMemory_setDeviceMemory(mem, val) (zMemory_getDeviceMemory(mem) = val)
#define zMemory_setDeviceMemoryStatus(mem, val)                                \
  (zMemory_getDeviceMemoryStatus(mem) = val)
#define zMemory_setMemoryGroup(mem, val) (zMemory_getMemoryGroup(mem) = val)

zMemory_t zMemory_new(zState_t st, size_t sz);
zMemory_t zMemory_new(zState_t st, void *data, size_t sz);
void zMemory_delete(zState_t st, zMemory_t mem);

void zMemory_copyToDevice(zState_t st, zMemory_t mem);
void zMemory_copyToHost(zState_t st, zMemory_t mem);

#endif /* __ZMEMORY_H__ */
