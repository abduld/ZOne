
#include "z.h"

static void zWriteArray(zState_t st, const char * fileName, zMemoryGroup_t mg) {
	for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
		zMemory_t mem = zMemoryGroup_getMemory(mg, ii);

	    cudaStream_t strm =
	        zState_getCopyToHostStream(st, zMemoryGroup_getId(mg));
	    if (zMemory_getDeviceMemoryStatus(mem) == zMemoryStatus_dirtyDevice) {
	    	cudaStreamSynchronize(strm);
	    }
	    zFile_t file = zFile_new(st, fileName, O_WRONLY);
	    zFile_write(file, zMemory_getHostMemory(mem), zMemory_getByteCount(mem));
	    zFile_delete(file);
	}
}

void zWriteBit8Array(zState_t st, const char * fileName, zMemoryGroup_t mg) {
	zWriteArray(st, fileName, mg);
	return ;
}

void zWriteInt32Array(zState_t st, const char * fileName, zMemoryGroup_t mg) {
	zWriteArray(st, fileName, mg);
	return ;

}

void zWriteInt64Array(zState_t st, const char * fileName, zMemoryGroup_t mg) {
	zWriteArray(st, fileName, mg);
	return ;

}

void zWriteFloatArray(zState_t st, const char * fileName, zMemoryGroup_t mg) {
	zWriteArray(st, fileName, mg);
	return ;

}

void zWriteDoubleArray(zState_t st, const char * fileName, zMemoryGroup_t mg) {
	zWriteArray(st, fileName, mg);
	return ;

}

