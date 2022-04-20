__kernel void convolution(const int filterWidth, __global const float *filter, const int imageHeight, const int imageWidth, __global const float *inputImage, __global float *outputImage, const int nonzero_num) 
{
	//Thread gets its index within index space 
	const int j = get_global_id(0);  

	const int i = get_global_id(1); 
	// printf("(%d, %d)\n", i, j);
   	// Iterate over the rows of the source image
    int halffilterSize = filterWidth / 2;
    float sum;
    int k, l;
    float f, ip;
    int nonz_count = 0;
    __local float localbuffer[49];

    if (get_local_id(1)==0 && get_local_id(0)<filterWidth*filterWidth)
    {
		localbuffer[get_local_id(0)] = filter[get_local_id(0)];
		// printf("%f\n", localbuffer[get_local_id(0)+get_local_id(1)]);
    }
    // else if ()
    // {
    // 	 code 
    // }

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
            	f = localbuffer[(k + halffilterSize) * filterWidth + l + halffilterSize];
            	// ip = inputImage[(i + k) * imageWidth + j + l];
            	if (f!=0)
            	{
                	sum += inputImage[(i + k) * imageWidth + j + l] * f;
                	nonz_count++;
                	// if (nonz_count == nonzero_num)
                	// {
                	// 	break;
                	// }
            	}
            }
        }
    	// if (nonz_count == nonzero_num)
    	// {
    	// 	break;
    	// }
    }
    outputImage[i * imageWidth + j] = sum;

}
