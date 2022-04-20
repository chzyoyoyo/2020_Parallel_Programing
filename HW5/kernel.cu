#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>


__constant__ int in_array[2];
__constant__ float flo_array[4];

__global__ void mandelKernel(int* dev_output) 
{
    // To avoid error caused by the floating number, use the following pseudo code
    //

    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    float x = flo_array[2] + thisX * flo_array[0];
    float y = flo_array[3] + thisY * flo_array[1];

    // x = flo_array[2] + (blockIdx.x * blockDim.x + threadIdx.x) * flo_array[0];
    // y = flo_array[3] + (blockIdx.y * blockDim.y + threadIdx.y) * flo_array[1];

    // int index = thisY * in_array[0] + thisX;

    float z_re = x, z_im = y;
    int i;
    for (i = 0; i < in_array[1]; ++i)
    {
        if (z_re * z_re + z_im*z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = flo_array[2] + thisX * flo_array[0] + new_re;
        z_im = flo_array[3] + thisY * flo_array[1] + new_im;
    }
    dev_output[thisY * in_array[0] + thisX] = i;
    // printf("(%d, %d)\n", thisX, thisY);


}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float test_flo[4];
    int test_in[2];
    test_flo[0] = (upperX - lowerX) / resX;
    test_flo[1] = (upperY - lowerY) / resY;

    dim3 numBlock(resX/16, resY/12);
    dim3 blockSize(16, 12);


    int data_size = resX * resY * sizeof(int);
    // int *output = (int *)malloc(data_size);
    int *output;
    cudaHostAlloc((void**)&output, data_size, cudaHostAllocDefault);
    int *dev_output;
    // cudaMalloc((void **)&dev_output, data_size);

	size_t cols = resX;
	size_t rows = resY;
	size_t pitch = 0;
    cudaMallocPitch((void**)&dev_output, &pitch, cols*sizeof(int), rows);
    test_flo[2] = lowerX;
    test_flo[3] = lowerY;

    test_in[0] = resX;
    test_in[1] = maxIterations;

    cudaMemcpyToSymbol(flo_array, test_flo, sizeof(float) * 4);
    cudaMemcpyToSymbol(in_array, test_in, sizeof(int) * 2);

    mandelKernel<<< numBlock, blockSize >>>(dev_output);
    // stepX, stepY, lowerX, lowerY, resX, maxIterations, dev_output
    // cudaDeviceSynchronize();
    cudaMemcpy(img, dev_output, data_size, cudaMemcpyDeviceToHost);

    cudaFree(dev_output);
    // printf("CUDA!!!!!!!!!!!!!!!!\n");
    // for (int i = 0; i < resX*resY; ++i)
    // {
    // 	img[i] = output[i];
    // }
    // cudaFreeHost(output);


}






