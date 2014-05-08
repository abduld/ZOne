

#include "z.h"

zState_t zState_new() {
  zState_t st = nNew(struct st_zState_t);
  zState_setNextMemoryGroupId(st, 0);

  zState_setMemoryGroups(st, {});
  zState_setFunctionMap(st, {});
  zState_setCUDAStreamsMap(st, {});
  zState_setLogger(st, zLogger_new());
  zState_setError(st, zError_new());
  zState_setTimer(st, zTimer_new());
  if (tbb::task_scheduler_init::automatic > 0) {
    zState_setCPUCount(st, tbb::task_scheduler_init::automatic);
  } else {
    zState_setCPUCount(st, Z_CONFIG_DEFAULT_CPU_COUNT);
  }
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
    zStreams_t strms;
    zState_setCUDAStreams(st, id, strms);
    zState_setMemoryGroup(st, id, mg);

    cudaStreamCreate(&zState_getComputeStream(st, id));
    cudaStreamCreate(&zState_getMallocStream(st, id));
    cudaStreamCreate(&zState_getCopyToDeviceStream(st, id));
    cudaStreamCreate(&zState_getCopyToHostStream(st, id));



    zMemoryGroup_setId(mg, id);
  }
  return;
}