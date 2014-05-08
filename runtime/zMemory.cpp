
#include "z.h"

zMemory_t zMemory_new(zState_t st, size_t sz) {

}

zMemory_t zMemory_new(zState_t st, void *data, size_t sz) {
	zMemory_t mem = zNew(struct st_zMemory_t);
	zMemory_setState(mem, st);
	zMemory_setByteCount(mem, sz);
	zMemory_setHostMemory(mem, data);
	zMemory_setHostMemoryStatus(mem, zMemoryStatus_allocatedHost);
	return mem;
}

void zMemory_delete(zMemory_t mem);

void zMemory_copyToDevice(zMemory_t mem) {}

void zMemory_copyToHost(zMemory_t mem) {}
