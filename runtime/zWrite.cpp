
#include "z.h"

static void zWriteArray(zState_t st, const char * fileName, zMemoryGroup_t mg) {
	zFile_t file = zFile_new(st, fileName, O_CREAT | O_RDWR | O_TRUNC);
	cudaStream_t compStrm = zState_getComputeStream(st, zMemoryGroup_getId(mg));
	zCUDA_check(cudaStreamSynchronize(compStrm));
	zCUDA_copyToHost(mg);
	while (zMemoryGroup_getDeviceMemoryStatus(mg) <= zMemoryStatus_allocatedDevice) {
		continue ;
	}
	cudaStream_t memStrm = zState_getCopyToHostStream(st, zMemoryGroup_getId(mg));
	zCUDA_check(cudaStreamSynchronize(memStrm));
	/*
	for (int ii = 0; ii < zMemoryGroup_getMemoryCount(mg); ii++) {
		zMemory_t mem = zMemoryGroup_getMemory(mg, ii);

	    cudaStream_t strm =
	        zState_getCopyToHostStream(st, zMemoryGroup_getId(mg));
	    if (zMemory_getDeviceMemoryStatus(mem) == zMemoryStatus_dirtyDevice) {
	    	cudaStreamSynchronize(strm);
	    }
	}
	*/
	zFile_write(file, zMemoryGroup_getHostMemory(mg), zMemoryGroup_getByteCount(mg));
	zFile_delete(file);
}

void zWriteBit8Array(zState_t st, const char * fileName, zMemoryGroup_t mg) {
	zWriteArray(st, fileName, mg);
	/*
	const char * data = (const char *) zMemoryGroup_getHostMemory(mg);
	for (int ii = 0; ii < 10; ii++) {
		printf("output (%d) = %d\n", ii, data[ii]);
	}*/
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
	/*
	const float * data = (const float *) zMemoryGroup_getHostMemory(mg);
	for (int ii = 0; ii < 10; ii++) {
		printf("output (%d) = %f\n", ii, data[ii]);
	}*/
	return ;

}

void zWriteDoubleArray(zState_t st, const char * fileName, zMemoryGroup_t mg) {
	zWriteArray(st, fileName, mg);
	return ;

}


