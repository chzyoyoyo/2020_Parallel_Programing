#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// float stepX, float stepY, float lowerX, float lowerY, int resX, int count, int* dev_output
__global__ void mandelKernel(float stepX, float stepY, float lowerX, float lowerY, int resX, int count, int* dev_output) {
    // To avoid error caused by the floating number, use the following pseudo code
    //

    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
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
    // printf("(%d, %d)\n", thisX, thisY);


}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    dim3 blockSize(16, 12);
    dim3 numBlock(resX/16, resY/12);

    // int *
    // float  *dev_A,  *dev_B,  *dev_C;
    // cudaMalloc( (void**)&dev_A, data_size );
    int data_size = resX * resY * sizeof(int);
    int *output = (int *)malloc(data_size);
    int *dev_output;
    cudaMalloc((void **)&dev_output, data_size);
    // cudaMemcpy( dev_A, a, data_size, cudaMemcpyHostToDevice );
    // cudaMemcpy(dev_output, img, data_size, cudaMemcpyHostToDevice);

    mandelKernel<<< numBlock, blockSize >>>(stepX, stepY, lowerX, lowerY, resX, maxIterations, dev_output);
    // stepX, stepY, lowerX, lowerY, resX, maxIterations, dev_output
    cudaDeviceSynchronize();
    // printf("CUDA!!!!!!!!!!!!!!!!\n");
    cudaMemcpy(output, dev_output, data_size, cudaMemcpyDeviceToHost);

    cudaFree(dev_output);

    // img = output;

    for (int i = 0; i < resX*resY; ++i)
    {
    	img[i] = output[i];
    	// if (output[i] > 0)
    	// {
    	// 	// cout << "Yesssssss" << endl;
    	// 	printf("Yesssssss\n");
    	// }
    }

}
