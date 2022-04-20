__kernel void convolution(int filterWidth, __global const float *filter, int imageHeight, int imageWidth, __global const float *inputImage, __global float *outputImage) 
{
	//Thread gets its index within index space 
	const int j = get_global_id(0);  
	
	const int i = get_global_id(1); 
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
