
#ifndef __ZMEMORYGROUP_H__
#define __ZMEMORYGROUP_H__

/* The idea is that we can make partition the data into group
   that we can operate on the data seperatly even though they are
   contiguious
*/

struct st_zMemoryGroup_t {
  size_t sz;
  int rank;
  size_t *dims;
  void *deviceData;
  void *hostData;
  int nmems;
  zMemoryGroup_t *mems;
  zMemoryStatus_t hostMemoryStatus;
  zMemoryStatus_t deviceMemoryStatus;
};

#define zMemoryGroup_getSize(mem) ((mem)->sz)
#define zMemoryGroup_getType(mem) ((mem)->typ)
#define zMemoryGroup_getHostMemory(mem) ((mem)->hostMemory)
#define zMemoryGroup_getDeviceMemoryStatus(mem) ((mem)->deviceMemoryStatus)
#define zMemoryGroup_getDeviceMemory(mem) ((mem)->deviceMemory)
#define zMemoryGroup_getDeviceMemoryStatus(mem) ((mem)->deviceMemoryStatus)
#define zMemoryGroup_getMemories(mem) ((mem)->mems)
#define zMemoryGroup_getMemory(mem, ii) (zMemoryGroup_getMemories(mem)[ii])
#define zMemoryGroup_getMemoryCount(mem) ((mem)->nmems)

#define zMemoryGroup_setSize(mem, val) (zMemoryGroup_getSize(mem) = val)
#define zMemoryGroup_setType(mem, val) (zMemoryGroup_getType(mem) = val)
#define zMemoryGroup_setHostMemory(mem, val)                                   \
  (zMemoryGroup_getHostMemory(mem) = val)
#define zMemoryGroup_setDeviceMemoryStatus(mem, val)                           \
  (zMemoryGroup_getDeviceMemoryStatus(mem) = val)
#define zMemoryGroup_setDeviceMemory(mem, val)                                 \
  (zMemoryGroup_getDeviceMemory(mem) = val)
#define zMemoryGroup_setDeviceMemoryStatus(mem, val)                           \
  (zMemoryGroup_getDeviceMemoryStatus(mem) = val)
#define zMemoryGroup_setMemoryGroup(mem, val)                                  \
  (zMemoryGroup_getMemoryGroup(mem) = val)
#define zMemoryGroup_setMemories(mem, val) (zMemoryGroup_getMemories(mem) = val)
#define zMemoryGroup_setMemory(mem, ii, val)                                   \
  (zMemoryGroup_getMemory(mem, ii) = val)
#define zMemoryGroup_setMemoryCount(mem, val)                                  \
  (zMemoryGroup_getMemoryCount(mem) = val)

zMemoryGroup_t zMemoryGroup_new(zState_t st, size_t sz);
zMemoryGroup_t zMemoryGroup_new(zState_t st, void *data, size_t sz);
void zMemoryGroup_delete(zState_t st, zMemoryGroup_t mem);

void zMemoryGroup_freeHostMemory(zState_t st, zMemoryGroup_t mem);
void zMemoryGroup_freeDeviceMemory(zState_t st, zMemoryGroup_t mem);

void zMemoryGroup_copyToDevice(zState_t st, zMemoryGroup_t mem);
void zMemoryGroup_copyToHost(zState_t st, zMemoryGroup_t mem);

void zMemoryGroup_copyToDevice(zState_t st, zMemoryGroup_t mem, int elem);
void zMemoryGroup_copyToHost(zState_t st, zMemoryGroup_t mem, int elem);

#endif /* __ZMEMORYGROUP_H__ */
