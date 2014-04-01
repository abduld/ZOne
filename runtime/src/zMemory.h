
#ifndef __ZMEMORY_H__
#define __ZMEMORY_H__

struct st_zMemory_t {
  size_t sz;
  zMemoryType_t typ;
  void * hostMemory;
  void * deviceMemory;
};

#define zMemory_getSize(mem) ((mem)->sz)
#define zMemory_getType(mem) ((mem)->typ)
#define zMemory_getHostMemory(mem) ((mem)->hostMemory)
#define zMemory_getDeviceMemory(mem) ((mem)->deviceMemory)

#define zMemory_setSize(mem, val) (zMemory_getSize(mem) = val)
#define zMemory_setType(mem, val) (zMemory_getType(mem) = val)
#define zMemory_setHostMemory(mem, val) (zMemory_getHostMemory(mem) = val)
#define zMemory_setDeviceMemory(mem, val) (zMemory_getDeviceMemory(mem) = val)

zMemory_t zMemory_new(zState_t st, size_t sz);
zMemory_t zMemory_new(zState_t st, void * data, size_t sz);
void zMemory_delete(zState_t st, zMemory_t mem);

void zMemory_freeHostMemory(zState_t st, zMemory_t mem);
void zMemory_freeDeviceMemory(zState_t st, zMemory_t mem);

static void zMemory_freeMemory(zState_t st, zMemory_t mem) {
  zMemory_freeHostMemory(st, mem);
  zMemory_freeDeviceMemory(st, mem);
}

void zMemory_copyToDevice(zState_t st, zMemory_t mem);
void zMemory_copyToHost(zState_t st, zMemory_t mem);

#endif /* __ZMEMORY_H__ */

