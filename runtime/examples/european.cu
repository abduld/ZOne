
#include "curand_kernel.h"
#define N(x)       (erf ((x)/sqrt(2.0))/2+0.5)
__global __ void europeanMonteCarlo(float * call, 
float * S, float * X, float * T, float * R, float * V, float * 
Q, int seed, int pathN, int length) {
	curandState rngState;
	int ii = threadIdx.x + blockIdx.x*blockDim.x;
	curand_init((unsigned long long)seed, ii, 0, &rngState);
	if (ii < length) {

		float sumS = 0, st, s = S[ii];
		float x = X[ii], t = T[ii];
		float r = R[ii], v = V[ii], q = Q[ii];
		float vByT = sqrt(t) * v;
		float muByT = t * (r - q - v*v/2);
		for (int jj = 0; jj < pathN; jj++) {
			st = s * exp(curand_normal(&rngState) * vByT + muByT);
			sumS += max(st - x, 0f);
		}
		call[ii] = exp(-r * t) * sumS/pathN;
	}
}