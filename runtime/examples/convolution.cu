

#include "z.h"

#define Mask_width 5
#define Mask_radius Mask_width / 2

#define BLOCK_DIM_X 64

__global__ void dImage_convolve(char *out, char *in, int width) {

  __shared__ char sMask[Mask_width];
  __shared__ char sImage[BLOCK_DIM_X + Mask_width];

  int tidX = threadIdx.x;

  int ii = tidX + blockIdx.x * blockDim.x;

#define P(img, x)  (((x) >= 0 && (x) < width)  ? ((img)[x]) : 0)

    sImage[tidX + Mask_radius] = P(in, ii);

    if (tidX <= Mask_radius) {
      sImage[tidX + Mask_radius] = P(in, ii - Mask_radius);
      sImage[tidX + BLOCK_DIM_X + Mask_radius] = P(in, ii + BLOCK_DIM_X);
    }

#undef P

  if (tidX < Mask_width) {
    sMask[tidX] = pow(-1.0f, tidX % Mask_radius);//mask[tidX];
  }

  __syncthreads();

  if (ii < width) {
    int accum = 0;
    for (int x = -Mask_radius; x <= Mask_radius; x++) {
      int pixelValue;
      int maskValue;
      pixelValue = sImage[tidX + Mask_radius + x];
      maskValue = sMask[x + Mask_radius];
      accum += pixelValue * maskValue;
    }
    out[ii] = accum > 255 ? 255 : accum < 0 ? 0 : accum;
  }
}

void Image_convolve(zMemory_t out, zMemory_t in) {
  size_t len = zMemory_getFlattenedLength(in);
  dim3 blockDim(BLOCK_DIM_X);
  dim3 gridDim(zCeil(len, blockDim.x));
  zState_t st = zMemory_getState(out);
  cudaStream_t strm = zState_getComputeStream(st, zMemory_getId(out));
  dImage_convolve<<<gridDim,blockDim, 0, strm>>>(
    (char*)zMemory_getDeviceMemory(out),
    (char*)zMemory_getDeviceMemory(in),
    len
  );
  return ;
}

int main(int argc, char *argv[]) {
size_t inputLength = 1 << atoi(argv[1]);
const char * inputFile = "data/S.dat";
const char * outputFile = "data/outputVector.dat";

  {

  size_t dim = inputLength;
  tick_count tic = zTNow();
  zState_t st = zState_new();
  zMemoryGroup_t in = zReadBit8Array(st, inputFile, 1, &dim);
  zMemoryGroup_t out = zMemoryGroup_new(st, zMemoryType_bit8, 1, &dim);
  zMapGroupFunction_t mapFun = zMapGroupFunction_new(st, "imageConvolve", Image_convolve);
  zMap(st, mapFun, out, in);
  zWriteBit8Array(st, outputFile, out);
  tick_count toc = zTNow();
  printf("took %g seconds for optimized\n", (toc-tic).seconds());
  }



{
  
  tick_count tic = zTNow();
  
  FILE * fdInput = fopen(inputFile, "r+");
  zAssert(fdInput != NULL);
  
  char * hInputMem = zNewArray(char, inputLength);
  char * hOutputMem = zNewArray(char, inputLength);
  fread(hInputMem, sizeof(char), inputLength, fdInput);
  fclose(fdInput);

  char * dInputMem, * dOutputMem;
  cudaMalloc(&dInputMem, sizeof(char)*inputLength);
  cudaMalloc(&dOutputMem, sizeof(char)*inputLength);

  zCUDA_check(cudaMemcpy(dInputMem, hInputMem, sizeof(char)*inputLength, cudaMemcpyHostToDevice));

  dim3 blockDim(BLOCK_DIM_X);
  dim3 gridDim(zCeil(inputLength, blockDim.x));
  dImage_convolve<<<gridDim,blockDim>>>(dOutputMem, dInputMem, inputLength);

  cudaMemcpy(hOutputMem, dOutputMem, sizeof(char)*inputLength, cudaMemcpyDeviceToHost);

  FILE * fdOutput = fopen(outputFile, "w");
  zAssert(fdOutput != NULL);
  fwrite(hOutputMem, sizeof(char), inputLength, fdOutput);
  fclose(fdOutput);
  
  cudaFree(dInputMem);
  cudaFree(dOutputMem);
  zFree(hInputMem);
  zFree(hOutputMem);

  tick_count toc = zTNow();
  printf("took %g seconds for unoptimized\n", (toc-tic).seconds());
}

  return 0;
  
}
