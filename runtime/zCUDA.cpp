
#include "z.h"

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
    speculative_spin_mutex mutex = zMemoryGroup_getMutex(mg);
    //mutex::scoped_lock();
    cudaError_t err = cudaMalloc(&deviceMem, byteCount);
    zState_setError(st, err);
    if (zSuccessQ(err)) {
      for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
        zMemory_t mem = zMemoryGroup_getMemory(mg, ii);
        zMemory_setDeviceMemory(mg, ((char *)deviceMem) + offset);
        offset += zMemory_getByteCount(mem);
      }
      zMemoryGroup_setDeviceMemoryStatus(mg, zMemoryStatus_allocatedDevice);
    }
    dummy->destroy(*dummy);
    return NULL;
  }
};

void zCUDA_malloc(zMemoryGroup_t mg) {
  // http://www.threadingbuildingblocks.org/docs/help/reference/task_scheduler/catalog_of_recommended_task_patterns.htm
  task *dummy = new (task::allocate_root()) empty_task;
  dummy->set_ref_count(1);
  task &tk = *new (dummy->allocate_child()) cudaMallocTask(dummy, mg);
  dummy->spawn(tk);

  return;
}

static void onCopyToDeviceStreamFinish(cudaStream_t stream, cudaError_t status,
                                       void *userData) {
  zMemory_t mem;
  assert(status == cudaSuccess);
  mem = (zMemory_t)userData;
  zMemory_setDeviceMemoryStatus(mem, zMemoryStatus_cleanDevice);
  zMemory_setHostMemoryStatus(mem, zMemoryStatus_cleanHost);
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
      zMemory_getHostMemoryStatus(mem) == zMemoryStatus_dirtyHost) {
    zMemoryGroup_t mg = zMemory_getMemoryGroup(mem);
    cudaStream_t strm =
        zState_getCopyToDeviceStream(st, zMemoryGroup_getId(mg));
    zAssert(strm != NULL);
    cudaError_t err = cudaMemcpyAsync(
        zMemory_getDeviceMemory(mem), zMemory_getHostMemory(mem),
        zMemory_getByteCount(mem), cudaMemcpyHostToDevice, strm);
    zState_setError(st, err);
    cudaStreamAddCallback(strm, onCopyToDeviceStreamFinish, (void *)mem, 0);
  } else {
    zLog(TRACE, "Skipping recopy of data.");
  }
}

static void onCopyToHostStreamFinish(cudaStream_t stream, cudaError_t status,
                                     void *userData) {
  zMemory_t mem;
  assert(status == cudaSuccess);
  mem = (zMemory_t)userData;
  zMemory_setDeviceMemoryStatus(mem, zMemoryStatus_cleanDevice);
  zMemory_setHostMemoryStatus(mem, zMemoryStatus_cleanHost);
  return;
}

void zCUDA_copyToHost(zMemory_t mem) {
  zState_t st = zMemory_getState(mem);

  zAssert(zMemory_hostMemoryAllocatedQ(mem));

  if (zMemory_hostMemoryAllocatedQ(mem) &&
      zMemory_deviceMemoryAllocatedQ(mem) &&
      zMemory_getDeviceMemoryStatus(mem) == zMemoryStatus_dirtyDevice) {
    zMemoryGroup_t mg = zMemory_getMemoryGroup(mem);
    cudaStream_t strm = zState_getCopyToHostStream(st, zMemoryGroup_getId(mg));
    zAssert(strm != NULL);
    cudaError_t err = cudaMemcpyAsync(
        zMemory_getHostMemory(mem), zMemory_getDeviceMemory(mem),
        zMemory_getByteCount(mem), cudaMemcpyDeviceToHost, strm);
    zState_setError(st, err);
    cudaStreamAddCallback(strm, onCopyToHostStreamFinish, (void *)mem, 0);
  } else {
    zLog(TRACE, "Skipping recopy of data.");
  }
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
