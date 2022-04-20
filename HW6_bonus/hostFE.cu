#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"

__global__ void convolution(int filterWidth, float *filter, int imageHeight, int imageWidth, float *inputImage, float *outputImage) 
{
	//Thread gets its index within index space 
	const int j = blockIdx.x * blockDim.x + threadIdx.x;  
	
	const int i = blockIdx.y * blockDim.y + threadIdx.y; 
	// printf("(%d, %d)\n", i, j);
   	// Iterate over the rows of the source image
    int halffilterSize = filterWidth / 2;
    float sum;
    int k, l;

	// printf("========================\n");
	sum = 0; // Reset sum for new source pixel
    // Apply the filter to the neighborhood
    for (k = -halffilterSize; k <= halffilterSize; k++)
    {
        for (l = -halffilterSize; l <= halffilterSize; l++)
        {
            if (i + k >= 0 && i + k < imageHeight &&
                j + l >= 0 && j + l < imageWidth)
            {
                sum += inputImage[(i + k) * imageWidth + j + l] *
                       filter[(k + halffilterSize) * filterWidth +
                              l + halffilterSize];
            }
        }
    }
    outputImage[i * imageWidth + j] = sum;

}

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage)
{


    int filterSize = filterWidth * filterWidth * sizeof(float);

    dim3 blockSize(6, 4);
    dim3 numBlock(imageWidth/6, imageHeight/4);
	
	int data_size = imageWidth * imageHeight * sizeof(float);

    float *dev_filter;
    cudaMalloc((void **)&dev_filter, filterSize);
    float *dev_input;
    cudaMalloc((void **)&dev_input, data_size);
    float *dev_output;
    cudaMalloc((void **)&dev_output, data_size);


    convolution<<< numBlock, blockSize >>>(filterWidth, dev_filter, imageHeight, imageWidth, dev_input, dev_output);

    cudaDeviceSynchronize();
    printf("CUDA!!!!!!!!!!!!!!!!\n");
    cudaMemcpy(outputImage, dev_output, data_size, cudaMemcpyDeviceToHost);

    cudaFree(dev_output);


}