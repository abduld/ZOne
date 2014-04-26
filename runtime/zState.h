

#ifndef __ZSTATE_H__
#define __ZSTATE_H__

typedef enum {
  zCUDAStream_malloc = 0,
  zCUDAStream_deviceToHost = 1,
  zCUDAStream_hostToDevice = 2,
  zCUDAStream_compute = 3,
  zCUDAStream_count = 4
} zCUDAStream_t;

typedef enum {
  zStateLabel_General = 0,
  zStateLabel_Logger,
  zStateLabel_Timer,
  zStateLabel_Memory,
  zStateLabel_Error,
  zStateLabel_IO,
  zStateLabel_Count
} zStateLabel_t;

struct st_zStream_t {
  cudaStream_t compute;
  cudaStream_t malloc;
  cudaStream_t deviceToHost;
  cudaStream_t hostToDevice;
};

struct st_zState_t {
  zBool cuStreamInUse[zCUDAStream_count];
  zStreamList_t cuStreams;
  zMemoryGroupList_t memoryGroups;
  zFunctionInformationMap_t fInfos;
  uv_mutex_t mutexs[zStateLabel_Count];
  uv_thread_t threads[zStateLabel_Count];
  zLogger_t logger;
  zTimer_t timer;
  zError_t err;
  uv_cpu_info_t *cpuInfo;
  int cpuCount;
};

#define zErr zState_geError(st)

static inline void wbState_lockMutex(wbState_t st, zStateLabel_t lbl) {
  if (st != NULL) {
    uv_mutex_lock(&wbState_getMutex(st, lbl));
  }
  return;
}

static inline void wbState_unlockMutex(wbState_t st, zStateLabel_t lbl) {
  if (st != NULL) {
    uv_mutex_unlock(&wbState_getMutex(st, lbl));
  }
  return;
}

#define wbState_mutexed(lbl, ...)                                              \
  do {                                                                         \
    wbState_lockMutex(st, zStateLabel_##lbl);                                  \
    { __VA_ARGS__; }                                                           \
    wbState_unlockMutex(st, zStateLabel_##lbl);                                \
  } while (0)

#endif /* __ZSTATE_H__ */
