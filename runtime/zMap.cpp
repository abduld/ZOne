
#include "z.h"

static inline dim3 optimalBlockDim(string fun, zMemory_t in) {
	return dim3(0);
}

static inline dim3 optimalGridDim(string fun, zMemory_t in, dim3 blockDim) {
	return dim3(0);
}
