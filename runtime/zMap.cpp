
#include "z.h"

static inline dim3 optimalBlockDim(string fun, zMemory_t in) { return dim3(0); }

static inline dim3 optimalGridDim(string fun, zMemory_t in, dim3 blockDim) {
  return dim3(0);
}



zMapGroupFunction_t zMapGroupFunction_new(zState_t st, const char * name, void (*f)(zMemoryGroup_t, zMemoryGroup_t)) {
	zMapGroupFunction_t mpf = nNew(struct st_zMapGroupFunction_t);
	zMapGroupFunction_setName(mpf, zString_duplicate(name));
	zMapGroupFunction_setFunction(mpf, f);
	return mpf;
}


zMapFunction_t zMapFunction_new(zState_t st, const char * name, void (*f)(zMemory_t, zMemory_t)) {
	zMapFunction_t mpf = nNew(struct st_zMapFunction_t);
	zMapFunction_setName(mpf, zString_duplicate(name));
	zMapFunction_setFunction(mpf, f);
	return mpf;
}

void zMap(zState_t st, zMapGroupFunction_t mapFun, zMemoryGroup_t out, zMemoryGroup_t in) {
	zCUDA_copyToDevice(in);

	while(!zMemoryGroup_deviceMemoryAllocatedQ(out)) {
		continue ;
	}
    cudaStreamSynchronize(zState_getCopyToDeviceStream(st, zMemoryGroup_getId(in)));
	while(zMemoryGroup_getDeviceMemoryStatus(in) < zMemoryStatus_cleanDevice) {
		continue ;
	}
	//zCUDA_check(cudaDeviceSynchronize());
	zMemoryGroup_setDeviceMemoryStatus(out, zMemoryStatus_dirtyDevice);
	zMapFunction_getFunction(mapFun)(out, in);
	//zCUDA_check(cudaDeviceSynchronize());
	return ;
}
