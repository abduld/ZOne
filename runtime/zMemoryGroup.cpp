
#include "z.h"

zMemoryGroup_t zMemoryGroup_new(zState_t st, zMemoryType_t typ, int rank,
                                size_t *dims) {
  int nMems;
  zMemory_t *mems;
  zMemoryGroup_t mg;

  nMems = tbb::task_scheduler_init::automatic;

  mg = zNew(struct st_zMemoryGroup_t);
  mems = zNewArray(zMemory_t, nMems);

  int id = zState_getNextMemoryGroupId(st);
  size_t byteCount = computeByteCount(typ, rank, dims);
  size_t chunkSize = zCeil(byteCount, nMems);
  char *hostMem = zNewArray(char, byteCount);

  size_t *dimsCopy = zNewArray(size_t, rank);
  memcpy(dimsCopy, dims, rank * sizeof(size_t));

  size_t bytesLeft = byteCount;
  for (int ii = 0; ii < nMems; ii++) {
    size_t sz = zMin(bytesLeft, chunkSize);
    mems[ii] = zMemory_new(st, &hostMem[ii * chunkSize], sz);
    zMemory_setMemoryGroup(mems[ii], mg);
    bytesLeft -= chunkSize;
  }

  zMemoryGroup_setId(mg, id);
  zMemoryGroup_setState(mg, st);
  zMemoryGroup_setByteCount(mg, byteCount);
  zMemoryGroup_setType(mg, typ);
  zMemoryGroup_setRank(mg, rank);
  zMemoryGroup_setDimensions(mg, dimsCopy);
  zMemoryGroup_setHostMemory(mg, hostMem);
  zMemoryGroup_setDeviceMemory(mg, NULL);

  zMemoryGroup_setMemoryCount(mg, nMems);
  zMemoryGroup_setMemories(mg, mems);

  zMemoryGroup_setHostMemoryStatus(mg, zMemoryStatus_allocatedHost);
  zMemoryGroup_setDeviceMemoryStatus(mg, zMemoryStatus_unallocated);

  zState_addMemoryGroup(st, mg);

  zCUDA_malloc(mg);

  return mg;
}

void zMemoryGroup_delete(zMemoryGroup_t mem) {}

void zMemoryGroup_freeHostMemory(zMemoryGroup_t mem) {
  if (mem && zMemoryGroup_hostMemoryAllocatedQ(mem) && 
    zMemoryGroup_getHostMemory(mem) != NULL) {
    zFree(zMemoryGroup_getHostMemory(mem));
  zMemoryGroup_setHostMemoryStatus(mem, zMemoryStatus_unallocated);
  }
}

void zMemoryGroup_freeDeviceMemory(zMemoryGroup_t mem) {
  if (mem && zMemoryGroup_deviceMemoryAllocatedQ(mem) && 
    zMemoryGroup_getDeviceMemory(mem) != NULL) {
    zCUDA_free(zMemoryGroup_getDeviceMemory(mem));
  zMemoryGroup_setDeviceMemoryStatus(mem, zMemoryStatus_unallocated);
  }
}

void zMemoryGroup_copyToDevice(zMemoryGroup_t mem) {
  zCUDA_copyToDevice(mem);
}

void zMemoryGroup_copyToHost(zMemoryGroup_t mem) {
  zCUDA_copyToHost(mem);
}

void zMemoryGroup_setDeviceMemoryStatus(zMemoryGroup_t mg, zMemoryStatus_t st) {
  if (mg != NULL) {
    for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
      zMemory_t mem = zMemoryGroup_getMemory(mg, ii);
      if (mem != NULL) {
        zMemory_setDeviceMemoryStatus(mem, st);
      }
    }
    zMemoryGroup__setDeviceMemoryStatus(mg, st);
  }
}

void zMemoryGroup_setHostMemoryStatus(zMemoryGroup_t mg, zMemoryStatus_t st) {
  if (mg != NULL) {
    for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
      zMemory_t mem = zMemoryGroup_getMemory(mg, ii);
      if (mem != NULL) {
        zMemory_setHostMemoryStatus(mem, st);
      }
    }
    zMemoryGroup__setHostMemoryStatus(mg, st);
  }
}
