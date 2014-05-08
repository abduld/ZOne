
#include "z.h"

#define BLOCK_DIM_X 64

#define N(x) (erf((x) / sqrt(2.0f)) / 2 + 0.5f)

__global__ void gpuBlackScholes(float *call, float *S, float *X, float *T,
                                float *r, float *sigma, int len) {

  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  if (ii > len) {
    return;
  }
  float d1 =
      (log(S[ii] / X[ii]) + (r[ii] + (sigma[ii] * sigma[ii]) / 2) * T[ii]) /
      (sigma[ii] * sqrt(T[ii]));
  float d2 = d1 - sigma[ii] * sqrt(T[ii]);

  call[ii] = S[ii] * N(d1) - X[ii] * exp(-r[ii] * T[ii]) * N(d2);
}

void BlackSholes(zMemory_t out, zMemory_t S, zMemory_t X, zMemory_t T,
                 zMemory_t r, zMemory_t sigma) {
  size_t len = zMemory_getFlattenedLength(S);
  dim3 blockDim(BLOCK_DIM_X);
  dim3 gridDim(zCeil(len, blockDim.x));
  zState_t st = zMemory_getState(out);
  cudaStream_t strm = zState_getComputeStream(st, zMemory_getId(out));
  gpuBlackScholes << <gridDim, blockDim, 0, strm>>>
      ((float *)zMemory_getDeviceMemory(out),
       (float *)zMemory_getDeviceMemory(S), (float *)zMemory_getDeviceMemory(X),
       (float *)zMemory_getDeviceMemory(T), (float *)zMemory_getDeviceMemory(r),
       (float *)zMemory_getDeviceMemory(sigma), len);
  return;
}

int main(int argc, char *argv[]) {

  size_t dim = 1 << atoi(argv[1]);
  const char *sDataFileName = "data/S.dat";
  const char *xDataFileName = "data/S.dat";
  const char *tDataFileName = "data/S.dat";
  const char *rDataFileName = "data/S.dat";
  const char *qDataFileName = "data/S.dat";
  const char *outputDataFileName = "data/outputVector.dat";
  {
    tick_count tic = zTNow();
    zState_t st = zState_new();
    zMemoryGroup_t S = zReadFloatArray(st, sDataFileName, 1, &dim);
    zMemoryGroup_t X = zReadFloatArray(st, xDataFileName, 1, &dim);
    zMemoryGroup_t T = zReadFloatArray(st, tDataFileName, 1, &dim);
    zMemoryGroup_t r = zReadFloatArray(st, rDataFileName, 1, &dim);
    zMemoryGroup_t q = zReadFloatArray(st, qDataFileName, 1, &dim);
    zMemoryGroup_t out = zMemoryGroup_new(st, zMemoryType_float, 1, &dim);
    zMapGroupFunction_t mapFun =
        zMapGroupFunction_new(st, "blackScholes", BlackSholes);
    zMap(st, mapFun, out, S, X, T, r, q);
    zWriteFloatArray(st, outputDataFileName, out);
    tick_count toc = zTNow();
    printf("took %g seconds for optimized\n", (toc - tic).seconds());
  }
  {
    size_t inputLength = dim;
    tick_count tic = zTNow();

    FILE *sfdInput = fopen(sDataFileName, "r+");
    FILE *xfdInput = fopen(xDataFileName, "r+");
    FILE *tfdInput = fopen(tDataFileName, "r+");
    FILE *rfdInput = fopen(rDataFileName, "r+");
    FILE *qfdInput = fopen(qDataFileName, "r+");

    float *hSMem = zNewArray(float, inputLength);
    float *hXMem = zNewArray(float, inputLength);
    float *hTMem = zNewArray(float, inputLength);
    float *hRMem = zNewArray(float, inputLength);
    float *hQMem = zNewArray(float, inputLength);
    float *hOutputMem = zNewArray(float, inputLength);

    fread(hSMem, sizeof(float), inputLength, sfdInput);
    fread(hXMem, sizeof(float), inputLength, xfdInput);
    fread(hTMem, sizeof(float), inputLength, tfdInput);
    fread(hRMem, sizeof(float), inputLength, rfdInput);
    fread(hQMem, sizeof(float), inputLength, qfdInput);

    fclose(sfdInput);
    fclose(xfdInput);
    fclose(tfdInput);
    fclose(rfdInput);
    fclose(qfdInput);

    float *dSMem, *dXMem, *dTMem, *dRMem, *dQMem, *dOutputMem;
    cudaMalloc(&dSMem, sizeof(float) * inputLength);
    cudaMalloc(&dXMem, sizeof(float) * inputLength);
    cudaMalloc(&dTMem, sizeof(float) * inputLength);
    cudaMalloc(&dRMem, sizeof(float) * inputLength);
    cudaMalloc(&dQMem, sizeof(float) * inputLength);
    cudaMalloc(&dOutputMem, sizeof(float) * inputLength);

    cudaMemcpy(dSMem, hSMem, sizeof(float) * inputLength,
               cudaMemcpyHostToDevice);
    cudaMemcpy(dXMem, hXMem, sizeof(float) * inputLength,
               cudaMemcpyHostToDevice);
    cudaMemcpy(dTMem, hTMem, sizeof(float) * inputLength,
               cudaMemcpyHostToDevice);
    cudaMemcpy(dRMem, hRMem, sizeof(float) * inputLength,
               cudaMemcpyHostToDevice);
    cudaMemcpy(dQMem, hQMem, sizeof(float) * inputLength,
               cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_DIM_X);
    dim3 gridDim(zCeil(inputLength, blockDim.x));
    gpuBlackScholes << <gridDim, blockDim>>>
        (dOutputMem, dSMem, dXMem, dTMem, dRMem, dQMem, inputLength);

    cudaMemcpy(hOutputMem, dOutputMem, sizeof(float) * inputLength,
               cudaMemcpyDeviceToHost);

    FILE *fdOutput = fopen(outputDataFileName, "w");
    zAssert(fdOutput != NULL);
    fwrite(hOutputMem, sizeof(float), inputLength, fdOutput);
    fclose(fdOutput);

    cudaFree(dSMem);
    cudaFree(dXMem);
    cudaFree(dTMem);
    cudaFree(dRMem);
    cudaFree(dQMem);
    cudaFree(dOutputMem);
    zFree(hSMem);
    zFree(hXMem);
    zFree(hTMem);
    zFree(hRMem);
    zFree(hQMem);
    zFree(hOutputMem);

    tick_count toc = zTNow();
    printf("took %g seconds for unoptimized\n", (toc - tic).seconds());
  }
  return 0;
}
