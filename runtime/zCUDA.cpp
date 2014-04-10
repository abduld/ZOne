
#include "z.h"


typedef struct st_MemoryAllocateWork_t {
  zState_t st;
  zMemoryGroup_t mg;
} * MemoryAllocateWork_t;

#define MemoryAllocateWork_getState(obj) ((obj)->st)
#define MemoryAllocateWork_getMemoryGroup(obj) ((obj)->mg)

#define MemoryAllocateWork_setState(obj, val) (MemoryAllocateWork_getState(obj) = val)
#define MemoryAllocateWork_setMemoryGroup(obj, val) (MemoryAllocateWork_getMemoryGroup(obj) = val)

void allocDeviceMemory(uv_work_t *req) {
  MemoryAllocateWork_t work = (MemoryAllocateWork_t) req->data;
  zState_t st = MemoryAllocateWork_getState(work);
  zMemoryGroup_t mg = MemoryAllocateWork_getMemoryGroup(work);
  void * deviceMem = zMemoryGroup_getDeviceMemory(mg);
  size_t byteCount = zMemoryGroup_getByteCount(mg);

  cudaError_t err = cudaMalloc(&deviceMem, byteCount);
  zState_setError(st, err);
}

void afterAllocDeviceMemory(uv_work_t *req, int status) {
  size_t offset = 0;
  MemoryAllocateWork_t work = (MemoryAllocateWork_t) req->data;
  zState_t st = MemoryAllocateWork_getState(work);
  zMemoryGroup_t mg = MemoryAllocateWork_getMemoryGroup(work);
  void * deviceMem = zMemoryGroup_getDeviceMemory(mg);
  if (status == -1) {
    zState_setError(st, zError_memoryAllocation);
  } else {
	  for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
	  	zMemory_t mem = zMemoryGroup_getMemory(mg, ii);
	  	zMemory_setDeviceMemory(mg, deviceMem + offset);
	  	offset += zMemory_getByteCount(mem);
	  }
  	zMemoryGroup_setDeviceMemoryStatus(mg, zMemoryStatus_allocated);
	}
  zDelete(req->data);
  zDelete(req);
}

void zCUDA_malloc(zState_t st, zMemoryGroup_t mem) {
	uv_work_t * work = New(uv_work_t);

	assert(zMemoryGroup_getDeviceMemoryStatus(mem) == zMemoryStatus_unallocated);

  MemoryAllocateWork_t workData = zNew(struct st_MemoryAllocateWork_t);
  MemoryAllocateWork_setState(workData, state);
  MemoryAllocateWork_setMemoryGroup(workData, mg);
  work->data = workData;

  uv_queue_work(loop, work, allocDeviceMemory, afterAllocDeviceMemory);

// should just use cudaMallocAsync, but code is kept here for reference (spent a lot of time figuring it out)
  return ;
}

void zCUDA_copyToDevice(zState_t st, zMemory_t mem) {
	zMemoryStatus_t status = zMemory_getStatus(mem);

	assert(zMemory_deviceMemoryAllocatedQ(mem));

	if (status == zMemoryStatus_allocated) {
		cudaStream_t strm = zState_getCopyToDeviceStream(st, zMemory_getId(mem));
		assert(strm != NULL);
		cudaError_t err = cudaMemcpyAsync(zMemory_getDeviceMemory(mem), zMemory_getHostMemory(mem),
			zMemory_getByteCount(mem), cudaMemcpyHostToDevice, strm);
  	zState_setError(st, err);
  	// one cannot just set zMemoryStatus_copied here, since the copy has not happened yet
	} else {
		zLog(TRACE, "Skipping recopy of data.");
	}
}

