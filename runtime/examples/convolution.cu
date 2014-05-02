

#include "z.h"

#define KERNEL_RADIUS 1
#define KERNEL_WIDTH (2*KERNEL_RADIUS + 1)
#define BLOCK_SIZE  16

__global__ void convolutionKernel(unsigned char * output, const char * input, float * kernel, int inputWidth, int inputHeight) {
	__shared__ float sKernel[KERNEL_WIDTH][KERNEL_WIDTH];
	__shared__ float sKernel[BLOCK_SIZE + 2 * KERNEL_RADIUS][BLOCK_SIZE + 2 * KERNEL_RADIUS];
	
}