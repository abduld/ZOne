

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
  zBool_t cuStreamInUse[zCUDAStream_count];
  zStreamList_t cuStreams;
  zMemoryGroupList_t memoryGroups;
  zFunctionInformationMap_t fInfos;
  speculative_spin_mutex mutexs[zStateLabel_Count];
  zLogger_t logger;
  zTimer_t timer;
  zError_t err;
  int cpuCount;
};

#define zErr zState_geError(st)

#define wbState_mutexed(lbl, ...)                                              \
  do {                                                                         \
    speculative_spin_mutex mutex = wbState_getMutex(st, zStateLabel_##lbl);    \
    mutex::scoped_lock();                                                      \
    { __VA_ARGS__; }                                                           \
  } while (0)


#endif /* __ZSTATE_H__ */
