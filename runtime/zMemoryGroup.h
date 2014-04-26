
#ifndef __ZMEMORYGROUP_H__
#define __ZMEMORYGROUP_H__

/* The idea is that we can make partition the data into group
   that we can operate on the data seperatly even though they are
   contiguious
*/

struct st_zMemoryGroup_t {
  int id;
  zState_t st;
  size_t byteCount;
  int rank;
  size_t *dims;
  void *deviceData;
  void *hostData;
  zMemoryType_t typ;
  int nmems;
  zMemoryGroup_t *mems;
  zMemoryStatus_t hostMemoryStatus;
  zMemoryStatus_t deviceMemoryStatus;
  speculative_spin_mutex mutex;
};

#define zMemoryGroup_hostMemoryAllocatedQ(mem)                                 \
  (zMemoryGroup_getHostMemoryStatus(mem) >= zMemoryStatus_allocatedHost)
#define zMemoryGroup_deviceMemoryAllocatedQ(mem)                               \
  (zMemoryGroup_getDeviceMemoryStatus(mem) >= zMemoryStatus_allocatedDevice)

#define zMemoryGroup_getId(mem) ((mem)->id)
#define zMemoryGroup_getState(mem) ((mem)->st)
#define zMemoryGroup_getByteCount(mem) ((mem)->byteCount)
#define zMemoryGroup_getRank(mem) ((mem)->rank)
#define zMemoryGroup_getDimensions(mem) ((mem)->dims)
#define zMemoryGroup_getType(mem) ((mem)->typ)
#define zMemoryGroup_getHostMemory(mem) ((mem)->hostMemory)
#define zMemoryGroup_getHostMemoryStatus(mem) ((mem)->hostMemoryStatus)
#define zMemoryGroup_getDeviceMemory(mem) ((mem)->deviceMemory)
#define zMemoryGroup_getDeviceMemoryStatus(mem) ((mem)->deviceMemoryStatus)
#define zMemoryGroup_getMemories(mem) ((mem)->mems)
#define zMemoryGroup_getMemory(mem, ii) (zMemoryGroup_getMemories(mem)[ii])
#define zMemoryGroup_getMemoryCount(mem) ((mem)->nmems)
#define zMemoryGroup_getMutex(mem) ((mem)->mutex)

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
#define zMemoryGroup__sethostMemoryStatus(mem, val)                            \
  (zMemoryGroup_getDeviceMemoryStatus(mem) = val)
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
#define zMemoryGroup_setMutex(mem, val)                                  \
  (zMemoryGroup_getMutex(mem) = val)

zMemoryGroup_t zMemoryGroup_new(zState_t st, zMemoryType_t typ, int rank,
                                size_t *dims);
void zMemoryGroup_delete(zMemoryGroup_t mem);

void zMemoryGroup_freeHostMemory(zMemoryGroup_t mem);
void zMemoryGroup_freeDeviceMemory(zMemoryGroup_t mem);

void zMemoryGroup_copyToDevice(zMemoryGroup_t mem);
void zMemoryGroup_copyToHost(zMemoryGroup_t mem);

void zMemoryGroup_copyToDevice(zMemoryGroup_t mem, int elem);
void zMemoryGroup_copyToHost(zMemoryGroup_t mem, int elem);

static inline void zMemoryGroup_setDeviceMemoryStatus(zMemoryGroup_t mg,
                                                      zMemoryStatus_t st) {
  if (mg != NULL) {
    for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
      zMemory_t mem = zMemoryGroup_getMemory(mg, ii);
      if (mem != NULL) {
        zMemory_setDeviceMemoryStatus(mem, st);
      }
    }
    zMemoryGroup__setDeviceMemoryStatus(mem, st);
  }
}

static inline void zMemoryGroup_setHostMemoryStatus(zMemoryGroup_t mg,
                                                    zMemoryStatus_t st) {
  if (mg != NULL) {
    for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
      zMemory_t mem = zMemoryGroup_getMemory(mg, ii);
      if (mem != NULL) {
        zMemory_setHostMemoryStatus(mem, st);
      }
    }
    zMemoryGroup__setHostMemoryStatus(mem, st);
  }
}

#endif /* __ZMEMORYGROUP_H__ */
