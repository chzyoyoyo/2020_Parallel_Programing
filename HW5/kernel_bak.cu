#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__constant__ int in_array[2];
__constant__ float flo_array[4];

// __global__ void mandelKernel(float stepX, float stepY, float lowerX, float lowerY, int resX, int count, int* dev_output) 
__global__ void mandelKernel(int* dev_output) 
{
    // To avoid error caused by the floating number, use the following pseudo code
    //

    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    float x = flo_array[2] + thisX * flo_array[0];
    float y = flo_array[3] + thisY * flo_array[1];

    int index = thisY * in_array[0] + thisX;

    float z_re = x, z_im = y;
    int i;
    for (i = 0; i < in_array[1]; ++i)
    {
        if (z_re * z_re + z_im*z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
    }
    dev_output[index] = i;
    // printf("(%d, %d)\n", thisX, thisY);


}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{

    float test_flo[4];
    int test_in[2];

    // float stepX = (upperX - lowerX) / resX;
    // float stepY = (upperY - lowerY) / resY;
    test_flo[0] = (upperX - lowerX) / resX;
    test_flo[1] = (upperY - lowerY) / resY;

    dim3 blockSize(16, 12);
    dim3 numBlock(resX/16, resY/12);


    int data_size = resX * resY * sizeof(int);
    int *output;
    output = (int *)malloc(data_size);
    // cudaHostAlloc((void**)&output, data_size, cudaHostAllocDefault);
    int *dev_output;
    // cudaMalloc((void **)&dev_output, data_size);
    cudaMallocManaged((void**)&dev_output, data_size);

    size_t cols = resX;
    size_t rows = resY;
    size_t pitch = 0;
    // cudaMallocPitch((void**)&dev_output, &pitch, cols*sizeof(int), rows);

    // float *test_flo = new float[4];
    // memset(test_flo, 0, sizeof(float) * 4);

    // int *test_in = new int[2];
    // memset(test_in, 0, sizeof(int) * 2);


    // test_flo[0] = stepX;
    // test_flo[1] = stepY;
    test_flo[2] = lowerX;
    test_flo[3] = lowerY;

    test_in[0] = resX;
    test_in[1] = maxIterations;



    cudaMemcpyToSymbol(flo_array, test_flo, sizeof(float) * 4);
    cudaMemcpyToSymbol(in_array, test_in, sizeof(int) * 2);

    // mandelKernel<<< numBlock, blockSize >>>(stepX, stepY, lowerX, lowerY, resX, maxIterations, dev_output);
    mandelKernel<<< numBlock, blockSize >>>(dev_output);
    // stepX, stepY, lowerX, lowerY, resX, maxIterations, dev_output
    cudaDeviceSynchronize();
    // cudaMemcpy(output, dev_output, data_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < resX*resY; ++i)
    {
        img[i] = dev_output[i];
    }
    cudaFree(dev_output);
    // printf("CUDA!!!!!!!!!!!!!!!!\n");
    cudaFreeHost(output);


}






