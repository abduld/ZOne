
#ifndef __ZMEMORY_H__
#define __ZMEMORY_H__

struct st_zMemory_t {
  zState_t st;
  size_t byteCount;
  zMemoryType_t typ;
  void *hostMemory;
  void *deviceMemory;
  zMemoryStatus_t hostMemoryStatus;
  zMemoryStatus_t deviceMemoryStatus;
  zMemoryGroup_t group;
};

#define zMemory_hostMemoryAllocatedQ(mem)                                      \
  (zMemory_getHostMemoryStatus(mem) >= zMemoryStatus_allocatedHost)
#define zMemory_deviceMemoryAllocatedQ(mem)                                    \
  (zMemory_getDeviceMemoryStatus(mem) >= zMemoryStatus_allocatedDevice)

#define zMemory_getState(mem) ((mem)->st)
#define zMemory_getByteCount(mem) ((mem)->sz)
#define zMemory_getType(mem) ((mem)->typ)
#define zMemory_getHostMemory(mem) ((mem)->hostMemory)
#define zMemory_getHostMemoryStatus(mem) ((mem)->hostMemoryStatus)
#define zMemory_getDeviceMemory(mem) ((mem)->deviceMemory)
#define zMemory_getDeviceMemoryStatus(mem) ((mem)->deviceMemoryStatus)
#define zMemory_getMemoryGroup(mem) ((mem)->group)

#define zMemory_setState(mem, val) (zMemory_getState(mem) = val)
#define zMemory_setByteCount(mem, val) (zMemory_getByteCount(mem) = val)
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
void zMemory_delete(zMemory_t mem);

void zMemory_copyToDevice(zMemory_t mem);
void zMemory_copyToHost(zMemory_t mem);

static inline int zMemory_getId(zMemory_t mem) {
  zMemoryGroup_t mg = zMemory_getMemoryGroup(mem);
  if (mg == NULL) {
    return -1;
  } else {
    return zMemoryGroup_getId(mg);
  }
}

#endif /* __ZMEMORY_H__ */
