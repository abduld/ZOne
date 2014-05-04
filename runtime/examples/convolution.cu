

#include "z.h"

#define Mask_width 5
#define Mask_radius Mask_width / 2

__global__ void Image_convolveGPU(float *out, float *in, float *mask, int width,
                                  int height) {
  int x, y;
  int xOffset, yOffset;
  int ii = threadIdx.y + blockIdx.y * blockDim.y;
  int jj = threadIdx.x + blockIdx.x * blockDim.x;

  if (ii < height && jj < width) {
    float accum = 0;
    for (y = -Mask_radius; y <= Mask_radius; y++) {
      yOffset = ii + y;
      if (yOffset >= 0 && yOffset < height) {
        for (x = -Mask_radius; x <= Mask_radius; x++) {
          xOffset = jj + x;
          if (xOffset >= 0 && xOffset < width) {
            float pixelValue;
            float maskValue;
            pixelValue = in[yOffset * width + xOffset];
            maskValue = mask[(y + Mask_radius) * Mask_width + x + Mask_radius];
            accum += pixelValue * maskValue;
          }
        }
      }
    }
    out[ii * width + jj] = accum;
  }
}

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8

__global__ void Image_convolveGPUShared(float *out, float *in, float *mask,
                                        int width, int height) {

  __shared__ float sMask[Mask_width][Mask_width];
  __shared__ float sImage[BLOCK_DIM_Y + Mask_width][BLOCK_DIM_X + Mask_width];

  int x, y;

  int tidX = threadIdx.x;
  int tidY = threadIdx.y;

  int ii = tidY + blockIdx.y * BLOCK_DIM_Y;
  int jj = tidX + blockIdx.x * BLOCK_DIM_X;

#define P(img, x, y)                                                           \
  ((x) >= 0 && (x) < height &&(y) >= 0 && (y) < width)                         \
      ? ((img)[(x) * width + (y)])                                             \
      : 0

        sImage[tidY + Mask_radius][tidX + Mask_radius] = P(in, ii, jj);

        if (tidX <= Mask_radius) {
          sImage[tidY + Mask_radius][tidX] = P(in, ii, jj - Mask_radius);
          sImage[tidY + Mask_radius][tidX + Mask_radius + BLOCK_DIM_X] =
              P(in, ii, jj + BLOCK_DIM_X);
        }
        if (tidY <= Mask_radius) {
          sImage[tidY][tidX + Mask_radius] = P(in, ii - Mask_radius, jj);
          sImage[tidY + Mask_radius + BLOCK_DIM_Y][tidX + Mask_radius] =
              P(in, ii + BLOCK_DIM_Y, jj);
        }
        if (tidX <= Mask_radius && tidY <= Mask_radius) {
          sImage[tidY][tidX * channels] =
              P(in, ii - Mask_radius, jj - Mask_radius);
          sImage[tidY][tidX + Mask_radius + BLOCK_DIM_X] =
              P(in, ii - Mask_radius, jj + BLOCK_DIM_X);
          sImage[tidY + Mask_radius + BLOCK_DIM_Y][tidX * channels] =
              P(in, ii + BLOCK_DIM_Y, jj - Mask_radius);
          sImage[tidY + Mask_radius + BLOCK_DIM_Y]
                [tidX + Mask_radius + BLOCK_DIM_X] =
                    P(in, ii + BLOCK_DIM_Y, jj + BLOCK_DIM_X);
        }

#undef P

  if (tidX < Mask_width && tidY < Mask_width) {
    sMask[tidY][tidX] = mask[tidY * Mask_width + tidX];
  }

  __syncthreads();

  if (ii < height && jj < width) {
    float accum = 0;
    for (y = -Mask_radius; y <= Mask_radius; y++) {
      for (x = -Mask_radius; x <= Mask_radius; x++) {
        float pixelValue;
        float maskValue;
        pixelValue = sImage[tidY + Mask_radius + y][tidX + Mask_radius + x];
        maskValue = sMask[y + Mask_radius][x + Mask_radius];
        accum += pixelValue * maskValue;
      }
    }
    out[ii * width + jj] = accum;
  }
}

void Image_convolveCPU(float *out, float *in, float *mask, int width,
                       int height, int channels) {
  float accum;
  int ii, jj, x, y;
  int xOffset, yOffset;

  for (ii = 0; ii < height; ii++) {
    for (jj = 0; jj < width; jj++) {
      accum = 0;
      for (y = -Mask_radius; y <= Mask_radius; y++) {
        yOffset = ii + y;
        if (yOffset >= 0 && yOffset < height) {
          for (x = -Mask_radius; x <= Mask_radius; x++) {
            xOffset = jj + x;
            if (xOffset >= 0 && xOffset < height) {
              float pixelValue;
              float maskValue;
              pixelValue = in[yOffset * width + xOffset];
              maskValue =
                  mask[(y + Mask_radius) * Mask_width + x + Mask_radius];
              accum += pixelValue * maskValue;
            }
          }
        }
      }
      out[ii * width + jj] = accum;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *outputImageFile;
  char *inputImageFile;
  char *inputMaskFile;
  char *expectedImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  wbImage_t expectedImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  inputImageFile = wbArg_getInputFile(arg, 0);
  inputMaskFile = wbArg_getInputFile(arg, 1);
  outputImageFile = wbArg_getOutputFile(arg);
  expectedImageFile = wbArg_getExpectedOutputFile(arg);

  inputImage = wbImport(inputImageFile);
  expectedImage = wbImport(expectedImageFile);
  hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(Compute, "Doing convolution on the CPU");
  //////////////////////////////////////
  // INSERT CODE HERE
  //@@ Call the CPU function that performs the convolution here
  Image_convolveCPU(hostOutputImageData, hostInputImageData, hostMaskData,
                    imageWidth, imageHeight, imageChannels);
  //////////////////////////////////////
  wbTime_stop(Compute, "Doing convolution on the CPU");

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceMaskData, maskRows * maskColumns * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData,
             maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //////////////////////////////////////
  // INSERT CODE HERE
  //@@
  dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 gridDim(
      ceil(static_cast<float>(imageWidth) / static_cast<float>(blockDim.x)),
      ceil(static_cast<float>(imageHeight) / static_cast<float>(blockDim.y)));
  // Image_convolveGPU<<<gridDim, blockDim>>>(deviceOutputImageData,
  // deviceInputImageData, deviceMaskData, imageWidth, imageHeight,
  // imageChannels);
  Image_convolveGPUShared << <gridDim, blockDim>>>
      (deviceOutputImageData, deviceInputImageData, deviceMaskData, imageWidth,
       imageHeight, imageChannels);
  //////////////////////////////////////
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbExport(outputImageFile, outputImage);

  wbImage_sameQ(outputImage, expectedImage);

  cout << wbTimer_toXML();

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);

  free(hostMaskData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
