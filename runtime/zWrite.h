

#ifndef __Z_WRITE_H__
#define __Z_WRITE_H__


void zWriteBit8Array(zState_t st, const char * fileName, zMemoryGroup_t mg);
void zWriteInt32Array(zState_t st, const char * fileName, zMemoryGroup_t mg);
void zWriteInt64Array(zState_t st, const char * fileName, zMemoryGroup_t mg);
void zWriteFloatArray(zState_t st, const char * fileName, zMemoryGroup_t mg);
void zWriteDoubleArray(zState_t st, const char * fileName, zMemoryGroup_t mg);

#endif /* __Z_WRITE_H__ */

