
#include "z.h"


void allocDeviceMemory(uv_work_t *req) {
  zMemoryGroup_t mg = (zMemoryGroup_t) req->data;
  zState_t st = zMemoryGroup_getState(mg);
  void * deviceMem = zMemoryGroup_getDeviceMemory(mg);
  size_t byteCount = zMemoryGroup_getByteCount(mg);

  cudaError_t err = cudaMalloc(&deviceMem, byteCount);
  zState_setError(st, err);
}

void afterAllocDeviceMemory(uv_work_t *req, int status) {
  size_t offset = 0;
  zMemoryGroup_t mg = (zMemoryGroup_t) req->data;
  zState_t st = zMemoryGroup_getState(mg);
  void * deviceMem = zMemoryGroup_getDeviceMemory(mg);
  if (status == -1) {
    zState_setError(st, zError_memoryAllocation);
  } else {
	  for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
	  	zMemory_t mem = zMemoryGroup_getMemory(mg, ii);
	  	zMemory_setDeviceMemory(mg, deviceMem + offset);
	  	offset += zMemory_getByteCount(mem);
	  }
  	zMemoryGroup_setDeviceMemoryStatus(mg, zMemoryStatus_allocatedDevice);
	}
  // TODO: think about .... zDelete(req);
}

void zCUDA_malloc(zMemoryGroup_t mem) {
	uv_work_t * work = New(uv_work_t);

	zAssert(!zMemory_deviceMemoryAllocatedQ(mem));

  work->data = mem;

  uv_queue_work(loop, work, allocDeviceMemory, afterAllocDeviceMemory);

// should just use cudaMallocAsync, but code is kept here for reference (spent a lot of time figuring it out)
  return ;
}

void zCUDA_copyToDevice(zMemory_t mem) {
	zState_t st = zMemory_getState(mem);
	zMemoryStatus_t status = zMemory_getStatus(mem);

	zAssert(zMemory_deviceMemoryAllocatedQ(mem));

	if (status == zMemoryStatus_allocatedDevice || status = zMemoryStatus_dirtyHost) {
		cudaStream_t strm = zState_getCopyToDeviceStream(st, zMemory_getId(mem));
		zAssert(strm != NULL);
		cudaError_t err = cudaMemcpyAsync(zMemory_getDeviceMemory(mem), zMemory_getHostMemory(mem),
			zMemory_getByteCount(mem), cudaMemcpyHostToDevice, strm);
  	zState_setError(st, err);
  	// one cannot just set zMemoryStatus_copied here, since the copy has not happened yet
	} else {
		zLog(TRACE, "Skipping recopy of data.");
	}
}

void zCUDA_copyToHost(zMemory_t mem) {
	zState_t st = zMemory_getState(mem);
	zMemoryStatus_t status = zMemory_getStatus(mem);

	zAssert(zMemory_hostMemoryAllocatedQ(mem));

	if (status == zMemoryStatus_allocatedHost || status == zMemoryStatus_dirtyDevice) {
		cudaStream_t strm = zState_getCopyToHostStream(st, zMemory_getId(mem));
		zAssert(strm != NULL);
		cudaError_t err = cudaMemcpyAsync(zMemory_getHostMemory(mem), zMemory_getDeviceMemory(mem),
			zMemory_getByteCount(mem), cudaMemcpyDevicetoHost, strm);
  	zState_setError(st, err);
	} else {
		zLog(TRACE, "Skipping recopy of data.");
	}
}
