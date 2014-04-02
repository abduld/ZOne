

#ifndef __ZSTATE_H__
#define __ZSTATE_H__

typedef enum {
  zCUDAStream_deviceToHost = 0,
  zCUDAStream_hostToDevice = 1,
  zCUDAStream_compute = 2,
  zCUDAStream_count = 3
} zCUDAStream_t;

typedef enum {
  zStateMutex_General = 0,
  zStateMutex_Logger,
  zStateMutex_Timer,
  zStateMutex_Memory,
  zStateMutex_Error,
  zStateMutex_IO,
  zStateMutex_Count
} zStateMutex_t;

struct st_zState_t {
  char cuStreamInUse[zCUDAStream_count];
  cudaStream_t cuStreams[zCUDAStream_count];
  zMemoryGroupList_t memoryGroups;
  zFunctionInformationMap_t fInfos;
  uv_mutex_t mutexs[zStateMutex_Count];
  zLogger_t logger;
  zTimer_t timer;
};

static inline void wbContext_lockMutex(wbContext_t ctx, zStateMutex_t lbl) {
  if (ctx != NULL) {
    uv_mutex_lock(&wbContext_getMutex(ctx, lbl));
  }
  return;
}

static inline void wbContext_unlockMutex(wbContext_t ctx, zStateMutex_t lbl) {
  if (ctx != NULL) {
    uv_mutex_unlock(&wbContext_getMutex(ctx, lbl));
  }
  return;
}

#define wbContext_mutexed(lbl, ctx, ...)                                       \
  do {                                                                         \
    wbContext_lockMutex(ctx, zStateMutex_##lbl);                               \
    { __VA_ARGS__; }                                                           \
    wbContext_unlockMutex(ctx, zStateMutex_##lbl);                             \
  } while (0)
#endif /* __ZSTATE_H__ */
