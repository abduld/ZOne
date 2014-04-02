

#ifndef __ZSTATE_H__
#define __ZSTATE_H__

typedef enum {
  zCUDAStream_deviceToHost = 0,
  zCUDAStream_hostToDevice = 1,
  zCUDAStream_compute = 2,
  zCUDAStream_count = 3
} zCUDAStream_t;

struct st_zState_t {
  char cuStreamInUse[zCUDAStream_count];
  cudaStream_t cuStreams[zCUDAStream_count];
};

#endif /* __ZSTATE_H__ */
