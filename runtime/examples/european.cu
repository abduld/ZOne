
#include "z.h"
#include "curand_kernel.h"

#define BLOCK_DIM_X 64

#define N(x) (erf((x) / sqrt(2.0)) / 2 + 0.5)

__global__ void gpuEuropeanMonteCarlo(float *call, float *S, float *X, float *T,
                                      float *R, float *V, float *Q, int seed,
                                      int pathN, int length) {
  curandState rngState;
  int ii = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init((unsigned long long)seed, ii, 0, &rngState);
  if (ii < length) {

    float sumS = 0, st, s = S[ii];
    float x = X[ii], t = T[ii];
    float r = R[ii], v = V[ii], q = Q[ii];
    float vByT = sqrt(t) * v;
    float muByT = t * (r - q - v * v / 2);
    for (int jj = 0; jj < pathN; jj++) {
      st = s * exp(curand_normal(&rngState) * vByT + muByT);
      sumS += max(st - x, 0.0f);
    }
    call[ii] = exp(-r * t) * sumS / pathN;
  }
}

void EuropeanMonteCarlo(zMemoryGroup_t out, zMemoryGroup_t S, zMemoryGroup_t X,
                        zMemoryGroup_t T, zMemoryGroup_t R, zMemoryGroup_t V,
                        zMemoryGroup_t Q) {
  size_t len = zMemoryGroup_getFlattenedLength(S);
  dim3 blockDim(BLOCK_DIM_X);
  dim3 gridDim(zCeil(len, blockDim.x));
  zState_t st = zMemoryGroup_getState(out);
  cudaStream_t strm = zState_getComputeStream(st, zMemoryGroup_getId(out));
  gpuEuropeanMonteCarlo << <gridDim, blockDim, 0, strm>>>
      ((float *)zMemoryGroup_getDeviceMemory(out),
       (float *)zMemoryGroup_getDeviceMemory(S),
       (float *)zMemoryGroup_getDeviceMemory(X),
       (float *)zMemoryGroup_getDeviceMemory(T),
       (float *)zMemoryGroup_getDeviceMemory(R),
       (float *)zMemoryGroup_getDeviceMemory(V),
       (float *)zMemoryGroup_getDeviceMemory(Q), 1, 2, len);
  return;
}

int main(int argc, char *argv[]) {
  size_t dim = 1 << 25;
  zState_t st = zState_new();
  zMemoryGroup_t S = zReadFloatArray(st, "data/S.dat", 1, &dim);
  zMemoryGroup_t X = zReadFloatArray(st, "data/X.dat", 1, &dim);
  zMemoryGroup_t T = zReadFloatArray(st, "data/T.dat", 1, &dim);
  zMemoryGroup_t R = zReadFloatArray(st, "data/r.dat", 1, &dim);
  zMemoryGroup_t V = zReadFloatArray(st, "data/r.dat", 1, &dim);
  zMemoryGroup_t Q = zReadFloatArray(st, "data/r.dat", 1, &dim);
  zMemoryGroup_t out = zMemoryGroup_new(st, zMemoryType_float, 1, &dim);
  zMapGroupFunction_t mapFun =
      zMapGroupFunction_new(st, "european", EuropeanMonteCarlo);
  zMap(st, mapFun, out, S, X, T, R, V, Q);
  zWriteFloatArray(st, "outputVector.dat", out);
  return 0;
}
