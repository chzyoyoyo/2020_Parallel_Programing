#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float stepX, float stepY, float lowerX, float lowerY, int resX, int count, int* dev_output) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    int j;
    for (j = 0; j < 100; j++)
    {    	
    	int thisX = j * blockDim.x + threadIdx.x;
	    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
	    float x = lowerX + thisX * stepX;
	    float y = lowerY + thisY * stepY;

	    int index = thisY * resX + thisX;

	    float z_re = x, z_im = y;
	    int i;
	    for (i = 0; i < count; ++i)
	    {
	    	if (z_re * z_re + z_im*z_im > 4.f)
	    		break;

	    	float new_re = z_re * z_re - z_im * z_im;
		    float new_im = 2.f * z_re * z_im;
		    z_re = x + new_re;
		    z_im = y + new_im;
	    }
	    dev_output[index] = i;

    }

    // printf("rounds:%d\n", threadIdx.y);

    
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
	
	float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    dim3 numBlock(1, resY/12);
    dim3 blockSize(16, 12);
    // printf("(%d, %d)\n", resX/16, resY/12);


    int data_size = resX * resY * sizeof(int);
    // int *output = (int *)malloc(data_size);
    int *output;
    cudaHostAlloc((void**)&output, data_size, cudaHostAllocDefault);
    int *dev_output;
    cudaMalloc((void **)&dev_output, data_size);

	size_t cols = resX;
	size_t rows = resY;
	size_t pitch = 0;
    cudaMallocPitch((void**)&dev_output, &pitch, cols*sizeof(int), rows);


	mandelKernel<<< numBlock, blockSize >>>(stepX, stepY, lowerX, lowerY, resX, maxIterations, dev_output);
    // stepX, stepY, lowerX, lowerY, resX, maxIterations, dev_output
    cudaDeviceSynchronize();
    cudaMemcpy(output, dev_output, data_size, cudaMemcpyDeviceToHost);

    cudaFree(dev_output);
    // printf("CUDA!!!!!!!!!!!!!!!!\n");
    for (int i = 0; i < resX*resY; ++i)
    {
    	img[i] = output[i];
    }
    cudaFreeHost(output);


}
