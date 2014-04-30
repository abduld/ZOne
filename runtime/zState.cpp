

#include "z.h"

zState_t zState_new() {
	zState_t st = zNew(struct st_zState_t);
	zState_setNextMemoryGroupId(st, 0);
	for (int ii = 0; ii < zCUDAStream_count; ii++) {
		zState_setCUDAStreamInUse(st, ii, zFalse);
	}

	cudaStreamCreate(&zState_getComputeStream(st));
	cudaStreamCreate(&zState_getMallocStream(st));
	cudaStreamCreate(&zState_getCopyToDeviceStream(st));
	cudaStreamCreate(&zState_getCopyToHostStream(st));

	zState_setMemoryGroups(st, {});
	zState_setFunctionInformationMap(st, {});
	zState_setLogger(st, zLogger_new());
	zState_setError(st, zError_new());
	zState_setTimer(st, zTimer_new());
	zState_setCPUCount(st, tbb::task_scheduler_init::automatic);
	return st;
}

void zState_setError(zState_t st, zError_t err) {
  return; // todo
}

void zState_setError(zState_t st, zErrorCode_t errCode) {
  return; // todo
}

void zState_setError(zState_t st, cudaError cuErr) {
  return; // todo
}

void zState_addMemoryGroup(zState_t st, zMemoryGroup_t mg) {
	if (mg != NULL) {
		int id = zState_getNextMemoryGroupId(st);
		zState_setMemoryGroup(st, id, mg);
		zMemoryGroup_setId(mg, id);
	}
	return ;
}