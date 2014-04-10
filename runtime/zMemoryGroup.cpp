
#include "z.h"


zMemoryGroup_t zMemoryGroup_new(zState_t st, zMemoryType_t typ, int rank, size_t *dims) {
	int nMems;
	zMemory_t *mems;
	zMemoryGroup_t mg;


#ifdef Z_CONFIG_MAX_CHUNKS
	nMems = Z_CONFIG_MAX_CHUNKS;
#else
	nMems = zState_getNCores(st);
#endif

	mg = zNew(struct st_zMemoryGroup_t);
	mems = zNewArray(zMemory_t, nMems);

  size_t byteCount = computeByteCount(typ, rank, dims);
  size_t chunkSize = zCeil(byteCount, nMems);
  char * hostMem = zNewArray(byteCount, char);

	dimsCopy = NewArray(size_t, rank);
	memcpy(dimsCopy, dims, rank * sizeof(size_t));

  size_t bytesLeft = byteCount;
	for (int ii = 0; ii < nMems; ii++) {
		size_t sz = zMin(bytesLeft, chunkSize);
		mems[ii] = zMemory_new(st, &hostMem[ii*chunkSize], sz);
		bytesLeft -= chunkSize;
	}

	zMemoryGroup_setByteCount(mem, bytecount);
	zMemoryGroup_setType(mem, typ);
	zMemoryGroup_setRank(mem, rank);
	zMemoryGroup_setDimensions(mem, dimsCopy);
	zMemoryGroup_setHostMemory(mem, hostMem);
	zMemoryGroup_setDeviceMemory(mem, NULL);

	zMemoryGroup_setMemoryCount(mem, nMems);
	zMemoryGroup_setMemories(mem, mems);

  zMemoryGroup_setHostMemoryStatus(mg, zMemoryStatus_allocated);
  zMemoryGroup_setDeviceMemoryStatus(mg, zMemoryStatus_unallocated);

	zCUDA_malloc(st, mg);

  return mg;
}


