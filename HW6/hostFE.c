#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    cl_int ret;
    int data_size = imageWidth * imageHeight;

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, &ret);
    

    cl_mem fil_mem_obj = clCreateBuffer(*context, CL_MEM_READ_ONLY, 
            filterSize * sizeof(cl_float), NULL, &ret);

    cl_mem in_mem_obj = clCreateBuffer(*context, CL_MEM_READ_ONLY,
            imageWidth * imageHeight * sizeof(cl_float), NULL, &ret);
    cl_mem out_mem_obj = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, 
            imageWidth * imageHeight * sizeof(cl_float), NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, fil_mem_obj, CL_TRUE, 0, 
            filterSize * sizeof(cl_float), filter, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, in_mem_obj, CL_TRUE, 0, 
            data_size * sizeof(cl_float), inputImage, 0, NULL, NULL);

    // ret = clEnqueueWriteBuffer(command_queue, out_mem_obj, CL_TRUE, 0, 
    //         imageWidth * imageHeight * sizeof(int), outputImage, 0, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &ret);


    int nonzero_num = filterSize;
    // for (int i = 0; i < filterSize; ++i)
    // {
    // 	if (filter[i]==0)
    // 	{
    // 		nonzero_num--;
    // 	}
    // }
    // printf("nonzero_num: %d\n", nonzero_num);
 

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(int), (void *)&filterWidth);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &fil_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(int), (void *)&imageHeight);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&imageWidth);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &in_mem_obj);
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &out_mem_obj);
    ret = clSetKernelArg(kernel, 6, sizeof(int), (void *)&nonzero_num);



    // Execute the OpenCL kernel on the list
    size_t localws[2] = {50,20}; 
    size_t globalws[2] = {imageWidth, imageHeight};
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
            &globalws, &localws, 0, NULL, NULL);

    // Read the memory buffer C on the device to the local variable C
    // int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
    ret = clEnqueueReadBuffer(command_queue, out_mem_obj, CL_TRUE, 0, 
            imageWidth * imageHeight * sizeof(float), outputImage, 0, NULL, NULL);
 
    // Clean up
    // ret = clFlush(command_queue);
    // ret = clFinish(command_queue);
    // ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(*program);
    ret = clReleaseMemObject(fil_mem_obj);
    ret = clReleaseMemObject(in_mem_obj);
    ret = clReleaseMemObject(out_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(*context);



}