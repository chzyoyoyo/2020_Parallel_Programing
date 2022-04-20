#include "kernel.h"

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by using CUDA
void bonus(
    int filterWidth, float *filter, int imageHeight, int imageWidth,
    float *inputImage, float *outputImage, cl_device_id *device,
    cl_context *context, cl_program *program);