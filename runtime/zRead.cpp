

#include "z.h"


static zMemory_t zReadArray(zState_t st, const char * file, zMemoryType_t typ,
							int rank, size_t * dims) {

}

zMemory_t zReadBit8Array(zState_t st, const char * file, int rank, size_t * dims) {
	return zReadArray(st, file, zMemoryType_bit8);
}

zMemory_t zReadInt32Array(zState_t st, const char * file, int rank, size_t * dims) {
	return zReadArray(st, file, zMemoryType_int32);
}

zMemory_t zReadInt64Array(zState_t st, const char * file, int rank, size_t * dims) {
	return zReadArray(st, file, zMemoryType_int64);
}

zMemory_t zReadFloatArray(zState_t st, const char * file, int rank, size_t * dims) {
	return zReadArray(st, file, zMemoryType_float);
}

zMemory_t zReadDoubleArray(zState_t st, const char * file, int rank, size_t * dims) {
	return zReadArray(st, file, zMemoryType_double);
}

