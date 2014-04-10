

#include "z.h"

typedef struct st_ReadArrayWork_t {
  zState_t st;
  zMemoryGroup_t mg;
  uv_mutexDeviceMemoryAllocatedQ_t * mutexDeviceMemoryAllocatedQ;
} * ReadArrayWork_t;

#define ReadArrayWork_getState(obj) ((obj)->st)
#define ReadArrayWork_getMemoryGroup(obj) ((obj)->mg)
#define ReadArrayWork_getmutexDeviceMemoryAllocatedQ(obj) ((obj)->mutexDeviceMemoryAllocatedQ)

#define ReadArrayWork_setState(obj, val) (ReadArrayWork_getState(obj) = val)
#define ReadArrayWork_setMemoryGroup(obj, val) (ReadArrayWork_getMemoryGroup(obj) = val)
#define ReadArrayWork_setmutexDeviceMemoryAllocatedQ(obj, val) (ReadArrayWork_getmutexDeviceMemoryAllocatedQ(obj) = val)


void allocDeviceMemory(uv_work_t *req) {
  ReadArrayWork_t work = (ReadArrayWork_t) req->data;
  zCUDAMalloc(ReadArrayWork_getState(work), ReadArrayWork_getMemoryGroup(work));
}

void setDeviceMemoryAllocated(uv_work_t *req, int status) {
  ReadArrayWork_t work = (ReadArrayWork_t) req->data;
  zState_t st = ReadArrayWork_getState(work);
  if (status == -1) {
    zError(zErr, memoryAllocation);
  }
  uv_mutexDeviceMemoryAllocatedQ_unlock(ReadArrayWork_getmutexDeviceMemoryAllocatedQ(work));
}

static zMemoryGroup_t zReadArray(zState_t st, const char *file,
                                 zMemoryType_t typ, int rank, size_t *dims) {
  zBool tryLockmutexDeviceMemoryAllocatedQ;
  size_t byteCount;
  size_t chunkByteCount;
  uv_work_t work_req;
  ReadArrayWork_t workData;
  uv_mutex_t mutexDeviceMemoryAllocatedQ;

  uv_loop_t *loop = zState_getLoop(st);

  checkMutexDeviceMemoryAllocated = zTrue;
  uv_mutexDeviceMemoryAllocatedQ_init(&mutexDeviceMemoryAllocatedQ);

  zMemoryGroup_t mg = zMemoryGroup_new(st, type, rank, dims);

  workData = zNew(struct st_ReadArrayWork_t);
  ReadArrayWork_setState(workData, state);
  ReadArrayWork_setMemoryGroup(workData, mg);
  ReadArrayWork_setmutexDeviceMemoryAllocatedQ(mutexDeviceMemoryAllocatedQ);

  work->data = workData;

  uv_mutex_lock(&mutexDeviceMemoryAllocatedQ);
  uv_queue_work(loop, &work, allocDeviceMemory, setDeviceMemoryAllocated);

  zFile_t file = zFile_open(st, file, 'r');
  // TODO: one can interleave the host memory alloc with the file read here
  // TODO: need to know if malloc is thread safe...
  
  for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
      zMemory_t mem = zMemoryGroup_getMemory(mg, ii);
      zFile_readChunk(st, file, zMemory_getHostMemory(mem), zMemory_getByteCount(mem));
      if (zSuccessQ(zErr)) {
        if (checkMutexDeviceMemoryAllocated == zTrue) {
          uv_mutex_lock(&mutexDeviceMemoryAllocatedQ);
          checkMutexDeviceMemoryAllocated = zFalse;
          uv_mutex_unlock(&mutexDeviceMemoryAllocatedQ);
        }
        // at this point we know that device memory has been allocated
        zMemory_copyToDevice(st, mem);
      }
  }

  uv_mutex_destroy(&mutexDeviceMemoryAllocatedQ);
  zDelete(workData);

  return mg;
}

zMemoryGroup_t zReadBit8Array(zState_t st, const char *file, int rank,
                              size_t *dims) {
  return zReadArray(st, file, zMemoryType_bit8, rank, dims);
}

zMemoryGroup_t zReadInt32Array(zState_t st, const char *file, int rank,
                               size_t *dims) {
  return zReadArray(st, file, zMemoryType_int32, rank, dims);
}

zMemoryGroup_t zReadInt64Array(zState_t st, const char *file, int rank,
                               size_t *dims) {
  return zReadArray(st, file, zMemoryType_int64, rank, dims);
}

zMemoryGroup_t zReadFloatArray(zState_t st, const char *file, int rank,
                               size_t *dims) {
  return zReadArray(st, file, zMemoryType_float, rank, dims);
}

zMemoryGroup_t zReadDoubleArray(zState_t st, const char *file, int rank,
                                size_t *dims) {
  return zReadArray(st, file, zMemoryType_double, rank, dims);
}
