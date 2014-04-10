
#include "z.h"


zMemory_t zMemory_new(zState_t st, size_t sz);
zMemory_t zMemory_new(zState_t st, void *data, size_t sz);
void zMemory_delete(zMemory_t mem);

void zMemory_copyToDevice(zMemory_t mem);
void zMemory_copyToHost(zMemory_t mem);

