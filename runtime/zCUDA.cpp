
#include "z.h"

#define ENABLE_ASYNC_MALLOC 0

class cudaMallocTask : public task {
public:
  task *dummy;
  zMemoryGroup_t mg;

  cudaMallocTask(task *dummy_, zMemoryGroup_t mg_) : dummy(dummy_), mg(mg_) {}
  task *execute() {
    size_t offset = 0;
    zState_t st = zMemoryGroup_getState(mg);
    void *deviceMem = zMemoryGroup_getDeviceMemory(mg);
    size_t byteCount = zMemoryGroup_getByteCount(mg);
    speculative_spin_mutex::scoped_lock(zMemoryGroup_getMutex(mg));
    cudaError_t err = cudaMalloc(&deviceMem, byteCount);
    zCUDA_check(err);
    zState_setError(st, err);
    if (zSuccessQ(err)) {
      for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
        zMemory_t mem = zMemoryGroup_getMemory(mg, ii);
        zMemory_setDeviceMemory(mem, ((char *)deviceMem) + offset);
        offset += zMemory_getByteCount(mem);
      }
      zMemoryGroup_setDeviceMemoryStatus(mg, zMemoryStatus_dirtyDevice);
    }
#if ENABLE_ASYNC_MALLOC
    dummy->destroy(*dummy);
#endif
    return NULL;
  }
};

void zCUDA_malloc(zMemoryGroup_t mg) {
  // http://www.threadingbuildingblocks.org/docs/help/reference/task_scheduler/catalog_of_recommended_task_patterns.htm
  
#if ENABLE_ASYNC_MALLOC
  task *dummy = new (task::allocate_root()) empty_task;
  dummy->set_ref_count(2);
  task &tk = *new (dummy->allocate_child()) cudaMallocTask(dummy, mg);
  dummy->spawn(tk);
#else
    size_t offset = 0;
    zState_t st = zMemoryGroup_getState(mg);
    void * deviceMem = zMemoryGroup_getDeviceMemory(mg);
    size_t byteCount = zMemoryGroup_getByteCount(mg);
 cudaError_t err = cudaMalloc((void **) &deviceMem, byteCount);
 zMemoryGroup_setDeviceMemory(mg, deviceMem);
  zCUDA_check(err);
  //zState_setError(st, err);
  if (err != cudaSuccess) {
    printf("Cannot allocate memory %d\n", (int) byteCount);
  }
  if (zSuccessQ(err)) {
    for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
      zMemory_t mem = zMemoryGroup_getMemory(mg, ii);
      zMemory_setDeviceMemory(mem, ((char *) deviceMem) + offset);
      offset += zMemory_getByteCount(mem);
    }
    zMemoryGroup_setDeviceMemoryStatus(mg, zMemoryStatus_dirtyDevice);
  } else {
    printf("ERROR...allocating..\n");
  }
#endif
  return;
}

static void onCopyMemoryToDeviceStreamFinish(cudaStream_t stream, cudaError_t status,
                                       void *userData) {
  zMemory_t mem;
  assert(status == cudaSuccess);
  mem = (zMemory_t)userData;
  zMemory_setDeviceMemoryStatus(mem, zMemoryStatus_cleanDevice);
  zMemory_setHostMemoryStatus(mem, zMemoryStatus_cleanHost);
  zMemoryGroup_t mg = zMemory_getMemoryGroup(mem);
  for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
    zMemory_t mi = zMemoryGroup_getMemory(mg, ii);
    if (zMemory_getDeviceMemoryStatus(mi) != zMemoryStatus_cleanDevice) {
      return ;
    }
  }
  zMemoryGroup_setDeviceMemoryStatus(mg, zMemoryStatus_cleanDevice);
  return;
}

static void onCopyMemoryGroupToDeviceStreamFinish(cudaStream_t stream, cudaError_t status,
                                       void *userData) {
  zMemoryGroup_t mem;
  assert(status == cudaSuccess);
  mem = (zMemoryGroup_t)userData;
  zMemoryGroup_setDeviceMemoryStatus(mem, zMemoryStatus_cleanDevice);
  zMemoryGroup_setHostMemoryStatus(mem, zMemoryStatus_cleanHost);
  return;
}


void zCUDA_copyToDevice(zMemory_t mem) {
  zState_t st = zMemory_getState(mem);
  /*
  zMemoryGroup_t mg = zMemory_getMemoryGroup(mem);
  speculative_spin_mutex mutex = zMemoryGroup_getMutex(mg);

  while (!zMemoryStatus_allocatedHost(mem)) {
  }

  {
    mutex::scoped_lock();
    while (!zMemoryStatus_allocatedDevice(mem)) {
    }
  }
  */

  zAssert(zMemory_deviceMemoryAllocatedQ(mem));

  if (zMemory_hostMemoryAllocatedQ(mem) &&
      zMemory_getHostMemoryStatus(mem) == zMemoryStatus_dirtyHost ||
      zMemory_getDeviceMemoryStatus(mem) == zMemoryStatus_dirtyDevice) {
    zMemoryGroup_t mg = zMemory_getMemoryGroup(mem);
    cudaStream_t strm =
        zState_getCopyToDeviceStream(st, zMemoryGroup_getId(mg));
    zAssert(strm != NULL);
    //cudaStreamSynchronize(strm);
    cudaError_t err = cudaMemcpyAsync(
        zMemory_getDeviceMemory(mem), zMemory_getHostMemory(mem),
        zMemory_getByteCount(mem), cudaMemcpyHostToDevice, strm);
    zState_setError(st, err);
    cudaStreamAddCallback(strm, onCopyMemoryToDeviceStreamFinish, (void *)mem, 0);
  } else {
    zLog(TRACE, "Skipping recopy of data.");
  }
}


void zCUDA_copyToDevice(zMemoryGroup_t mem) {
  zState_t st = zMemoryGroup_getState(mem);
  /*
  zMemoryGroup_t mg = zMemory_getMemoryGroup(mem);
  speculative_spin_mutex mutex = zMemoryGroup_getMutex(mg);

  while (!zMemoryStatus_allocatedHost(mem)) {
  }

  {
    mutex::scoped_lock();
    while (!zMemoryStatus_allocatedDevice(mem)) {
    }
  }
  */

  zAssert(zMemoryGroup_deviceMemoryAllocatedQ(mem));

  if (zMemoryGroup_hostMemoryAllocatedQ(mem) &&
      zMemoryGroup_getHostMemoryStatus(mem) <= zMemoryStatus_dirtyHost) {
    cudaStream_t strm =
        zState_getCopyToDeviceStream(st, zMemoryGroup_getId(mem));
    zAssert(strm != NULL);
    cudaStreamAddCallback(strm, onCopyMemoryGroupToDeviceStreamFinish, (void *)mem, 0);
    cudaError_t err = cudaMemcpyAsync(
        zMemoryGroup_getDeviceMemory(mem), zMemoryGroup_getHostMemory(mem),
        zMemoryGroup_getByteCount(mem), cudaMemcpyHostToDevice, strm);
    zState_setError(st, err);
    cudaStreamSynchronize(strm);
  } else {
    zLog(TRACE, "Skipping recopy of data.");
  }

}


static void onCopyMemoryToHostStreamFinish(cudaStream_t stream, cudaError_t status,
                                     void *userData) {
  zMemory_t mem;
  assert(status == cudaSuccess);
  mem = (zMemory_t)userData;
  zMemory_setDeviceMemoryStatus(mem, zMemoryStatus_cleanDevice);
  zMemory_setHostMemoryStatus(mem, zMemoryStatus_cleanHost);
  zMemoryGroup_t mg = zMemory_getMemoryGroup(mem);
  for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
    zMemory_t mi = zMemoryGroup_getMemory(mg, ii);
    if (zMemory_getHostMemoryStatus(mi) != zMemoryStatus_cleanHost) {
      return ;
    }
  }
  zMemoryGroup_setHostMemoryStatus(mg, zMemoryStatus_cleanHost);
}

static void onCopyMemoryGroupToHostStreamFinish(cudaStream_t stream, cudaError_t status,
                                     void *userData) {
  zMemoryGroup_t mem;
  assert(status == cudaSuccess);
  mem = (zMemoryGroup_t)userData;
  zMemoryGroup_setDeviceMemoryStatus(mem, zMemoryStatus_cleanDevice);
  zMemoryGroup_setHostMemoryStatus(mem, zMemoryStatus_cleanHost);
  return;
}

void zCUDA_copyToHost(zMemory_t mem) {
  zState_t st = zMemory_getState(mem);

  zAssert(zMemory_hostMemoryAllocatedQ(mem));

  if (zMemory_hostMemoryAllocatedQ(mem) &&
      zMemory_deviceMemoryAllocatedQ(mem) &&
      zMemory_getDeviceMemoryStatus(mem) == zMemoryStatus_dirtyDevice ||
      zMemory_getHostMemoryStatus(mem) == zMemoryStatus_dirtyHost) {
    zMemoryGroup_t mg = zMemory_getMemoryGroup(mem);
    cudaStream_t strm = zState_getCopyToHostStream(st, zMemoryGroup_getId(mg));
    zAssert(strm != NULL);
    //cudaStreamSynchronize(strm);
    cudaError_t err = cudaMemcpyAsync(
        zMemory_getHostMemory(mem), zMemory_getDeviceMemory(mem),
        zMemory_getByteCount(mem), cudaMemcpyDeviceToHost, strm);
    zCUDA_check(err);
    zState_setError(st, err);
    cudaStreamAddCallback(strm, onCopyMemoryToHostStreamFinish, (void *)mem, 0);
  } else {
    zLog(TRACE, "Skipping recopy of data.");
  }
}

void zCUDA_copyToHost(zMemoryGroup_t mem) {
  zState_t st = zMemoryGroup_getState(mem);

#if 0
  zAssert(zMemoryGroup_hostMemoryAllocatedQ(mem));

  if (zMemoryGroup_hostMemoryAllocatedQ(mem) &&
      zMemoryGroup_deviceMemoryAllocatedQ(mem) &&
      zMemoryGroup_getDeviceMemoryStatus(mem) == zMemoryStatus_dirtyDevice) {
    cudaStream_t strm = zState_getCopyToHostStream(st, zMemoryGroup_getId(mem));
    zAssert(strm != NULL);
    //cudaStreamSynchronize(strm);
    cudaError_t err = cudaMemcpyAsync(
        zMemoryGroup_getHostMemory(mem), zMemoryGroup_getDeviceMemory(mem),
        zMemoryGroup_getByteCount(mem), cudaMemcpyDeviceToHost, strm);
    zCUDA_check(err);
    zState_setError(st, err);
    cudaStreamAddCallback(strm, onCopyMemoryGroupToHostStreamFinish, (void *)mem, 0);
  } else {
    zLog(TRACE, "Skipping recopy of data.");
  }
#else
  while (!zMemoryGroup_deviceMemoryAllocatedQ(mem)) {
    continue ;
  }
  char * host = (char*) zMemoryGroup_getHostMemory(mem);
  zCUDA_check(cudaMemcpy(host, zMemoryGroup_getDeviceMemory(mem),
        zMemoryGroup_getByteCount(mem), cudaMemcpyDeviceToHost));
#endif
}
void zCUDA_free(void *mem) {
  if (mem != NULL) {
    cudaFree(mem);
  }
}

void zCUDA_free(zMemory_t mem) {
  if (mem != NULL && zMemory_deviceMemoryAllocatedQ(mem)) {
    cudaFree(mem);
    zMemory_setDeviceMemoryStatus(mem, zMemoryStatus_unallocated);
  }
}


