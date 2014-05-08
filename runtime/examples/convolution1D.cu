

#include "z.h"

#define Mask_width 5
#define Mask_radius Mask_width / 2

#define BLOCK_DIM_X 64

__global__ void dImage_convolve(float *out, float *in, int width) {

  __shared__ float sMask[Mask_width];
  __shared__ float sImage[BLOCK_DIM_X + Mask_width];

  int tidX = threadIdx.x;

  int ii = tidX + blockIdx.x * BLOCK_DIM_X;

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
    float accum = 0;
    for (int x = -Mask_radius; x <= Mask_radius; x++) {
      float pixelValue;
      float maskValue;
      pixelValue = sImage[tidX + Mask_radius + x];
      maskValue = sMask[x + Mask_radius];
      accum += pixelValue * maskValue;
    }
    out[ii] = accum;
  }
}

void Image_convolve(zMemoryGroup_t out, zMemoryGroup_t in) {
  size_t len = zMemoryGroup_getFlattenedLength(in);
  dim3 blockDim(32);
  dim3 gridDim(zCeil(len, blockDim.x));
  dImage_convolve<<<gridDim,blockDim>>>(
    (float*)zMemoryGroup_getDeviceMemory(out),
    (float*)zMemoryGroup_getDeviceMemory(in),
    len
  );
  return ;
}

int main(int argc, char *argv[]) {
  size_t dim = 1024;
  zState_t st = zState_new();
  zMemoryGroup_t in = zReadBit8Array(st, "inputVector.dat", 1, &dim);
  zMemoryGroup_t out = zMemoryGroup_new(st, zMemoryType_bit8, 1, &dim);
  zMapGroupFunction_t mapFun = zMapGroupFunction_new(st, "imageConvolve", Image_convolve);
  zMap(st, mapFun, out, in);
  zWriteBit8Array(st, "outputVector.dat", out);
  return 0;
}
