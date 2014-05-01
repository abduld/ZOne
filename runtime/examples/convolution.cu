

#include "z.h"

typename <int kernelWidth>
__global__ void convolutionKernel(unsigned char * output, const char * input, float * kernel, int inputWidth, int inputHeight) {
	__shared__ float sKernel[kernelWidth][kernelWidth];
	
}