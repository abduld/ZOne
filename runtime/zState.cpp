

#include "z.h"

zState_t zState_new() {
	zState_t st = zNew(struct st_zState_t);
	zState_setNextMemoryGroupId(st, 0);
	for (int ii = 0; ii < zCUDAStream_count; ii++) {
		zState_setCUDAStreamInUse(st, ii, zFalse);
	}

	cudaStreamCreate(&zState_getCUDAComputeStream(st));
	cudaStreamCreate(&zState_getCUDAMallocStream(st));
	cudaStreamCreate(&zState_getCUDADeviceToHostStream(st));
	cudaStreamCreate(&zState_getCUDAHostToDeviceStream(st));

	zState_setMemoryGroups(st, unordered_map<int, zMemoryGroup_t>());
	zState_setFunctionInformationMap(st, unordered_map<string, zFunctionInformation_t>());
	zState_setLogger(st, zLogger_new());
	zState_setError(st, zError_new());
	zState_setTimer(st, zTimer_new());
	return st;
}

void getCPUInfo(zState_t st) {
  uv_cpu_info(&zState_getCPUInfo(st), &zState_getCPUCount(st));
}

void freeCPUInfo(zState_t st) {
  uv_free_cpu_info(zState_getCPUInfo(st), zState_getCPUCount(st));
}

void zState_setError(zState_t st, zErrorCode_t errCode) {
  return; // todo
}

void zState_setError(zState_t st, cudaError cuErr) {
  return; // todo
}

void zState_addMemoryGroup(zState_t st, zMemoryGroup_t mg) {
	if (mg != NULL) {
		int id = zState_getNextMemoryGroupId(mg);
		zState_setMemoryGroup(st, id, mg);
		zMemoryGroup_setId(mg, id);
	}
	return ;
}