

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
  int nextMemId;
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

#define zState_getNextMemoryGroupId(st)  ((st)->nextMemId)
#define zState_getCUDAStreamsInUse(st)  ((st)->cuStreamInUse)
#define zState_getCUDAStreamInUse(st, ii) (zState_getCUDAStreamsInUse(st)[ii])
#define zState_getCUDAStreams(st)  ((st)->cuStreams)
#define zState_getCUDAStream(st, ii) (zState_getCUDAStreams(st)[ii])
#define zState_getMemoryGroups(st) ((st)->memoryGroups)
#define zState_getFunctionInformationMap(st) ((st)->fInfos)
#define zState_getMutexes(st) ((st)->mutexs)
#define zState_getMutex(st, ii) (zState_getMutexes(st)[ii])
#define zState_getLogger(st) ((st)->logger)
#define zState_getTimer(st) ((st)->timer)
#define zState_getError(st) ((st)->err)

#define zErr zState_geError(st)

#define wbState_mutexed(lbl, ...)                                              \
  do {                                                                         \
    speculative_spin_mutex mutex = zState_getMutex(st, zStateLabel_##lbl);    \
    mutex::scoped_lock();                                                      \
    { __VA_ARGS__; }                                                           \
  } while (0)


#endif /* __ZSTATE_H__ */
