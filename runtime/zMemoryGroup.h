
#ifndef __ZMEMORYGROUP_H__
#define __ZMEMORYGROUP_H__

/* The idea is that we can make partition the data into group
   that we can operate on the data seperatly even though they are
   contiguious
*/

struct st_zMemoryGroup_t {
  int mgid;
  zState_t mgst;
  size_t mgbyteCount;
  int mgrank;
  size_t *mgdims;
  void *mgdeviceMemory;
  void *mghostMemory;
  zMemoryType_t mgtyp;
  int mgnmems;
  zMemory_t *mgmems;
  zMemoryStatus_t mghostMemoryStatus;
  zMemoryStatus_t mgdeviceMemoryStatus;
  speculative_spin_mutex mgmutex;
};

#define zMemoryGroup_hostMemoryAllocatedQ(mem)                                 \
  (zMemoryGroup_getHostMemoryStatus(mem) >= zMemoryStatus_allocatedHost)
#define zMemoryGroup_deviceMemoryAllocatedQ(mem)                               \
  (zMemoryGroup_getDeviceMemoryStatus(mem) >= zMemoryStatus_allocatedDevice)

#define zMemoryGroup_getId(mem) ((mem)->mgid)
#define zMemoryGroup_getState(mem) ((mem)->mgst)
#define zMemoryGroup_getByteCount(mem) ((mem)->mgbyteCount)
#define zMemoryGroup_getRank(mem) ((mem)->mgrank)
#define zMemoryGroup_getDimensions(mem) ((mem)->mgdims)
#define zMemoryGroup_getType(mem) ((mem)->mgtyp)
#define zMemoryGroup_getHostMemory(mem) ((mem)->mghostMemory)
#define zMemoryGroup_getHostMemoryStatus(mem) ((mem)->mghostMemoryStatus)
#define zMemoryGroup_getDeviceMemory(mem) ((mem)->mgdeviceMemory)
#define zMemoryGroup_getDeviceMemoryStatus(mem) ((mem)->mgdeviceMemoryStatus)
#define zMemoryGroup_getMemories(mem) ((mem)->mgmems)
#define zMemoryGroup_getMemory(mem, ii) (zMemoryGroup_getMemories(mem)[ii])
#define zMemoryGroup_getMemoryCount(mem) ((mem)->mgnmems)
#define zMemoryGroup_getMutex(mem) ((mem)->mgmutex)

#define zMemoryGroup_setId(mem, val) (zMemoryGroup_getId(mem) = val)
#define zMemoryGroup_setState(mem, val) (zMemoryGroup_getState(mem) = val)
#define zMemoryGroup_setByteCount(mem, val)                                    \
  (zMemoryGroup_getByteCount(mem) = val)
#define zMemoryGroup_setDimensions(mem, val)                                   \
  (zMemoryGroup_getDimensions(mem) = val)
#define zMemoryGroup_setRank(mem, val) (zMemoryGroup_getRank(mem) = val)
#define zMemoryGroup_setType(mem, val) (zMemoryGroup_getType(mem) = val)
#define zMemoryGroup_setHostMemory(mem, val)                                   \
  (zMemoryGroup_getHostMemory(mem) = val)
#define zMemoryGroup__setHostMemoryStatus(mem, val)                            \
  (zMemoryGroup_getHostMemoryStatus(mem) = val)
#define zMemoryGroup_setDeviceMemory(mem, val)                                 \
  (zMemoryGroup_getDeviceMemory(mem) = val)
#define zMemoryGroup__setDeviceMemoryStatus(mem, val)                          \
  (zMemoryGroup_getDeviceMemoryStatus(mem) = val)
#define zMemoryGroup_setMemoryGroup(mem, val)                                  \
  (zMemoryGroup_getMemoryGroup(mem) = val)
#define zMemoryGroup_setMemories(mem, val) (zMemoryGroup_getMemories(mem) = val)
#define zMemoryGroup_setMemory(mem, ii, val)                                   \
  (zMemoryGroup_getMemory(mem, ii) = val)
#define zMemoryGroup_setMemoryCount(mem, val)                                  \
  (zMemoryGroup_getMemoryCount(mem) = val)
#define zMemoryGroup_setMutex(mem, val) (zMemoryGroup_getMutex(mem) = val)

zMemoryGroup_t zMemoryGroup_new(zState_t st, zMemoryType_t typ, int rank,
                                size_t *dims);
void zMemoryGroup_delete(zMemoryGroup_t mem);

void zMemoryGroup_freeHostMemory(zMemoryGroup_t mem);
void zMemoryGroup_freeDeviceMemory(zMemoryGroup_t mem);

size_t zMemoryGroup_getFlattenedLength(zMemoryGroup_t mem);

void zMemoryGroup_copyToDevice(zMemoryGroup_t mem);
void zMemoryGroup_copyToHost(zMemoryGroup_t mem);

void zMemoryGroup_copyToDevice(zMemoryGroup_t mem, int elem);
void zMemoryGroup_copyToHost(zMemoryGroup_t mem, int elem);

void zMemoryGroup_setDeviceMemoryStatus(zMemoryGroup_t mg, zMemoryStatus_t st);

void zMemoryGroup_setHostMemoryStatus(zMemoryGroup_t mg, zMemoryStatus_t st);
#endif /* __ZMEMORYGROUP_H__ */
