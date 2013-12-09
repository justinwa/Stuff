#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <math.h>

extern "C"
{

__global__ void 
rtruncnorm_kernel(float *vals, int n, 
                  float *mu, float *sigma, 
                  float *trunc, bool *lo,
                  int mu_len, int sigma_len,
                  int lo_len, int hi_len,
                  int maxtries,int rseed_a,
                  int rseed_b,int rseed_c)
{
    // Usual block/thread indexing...
    int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    int blocksize = blockDim.x * blockDim.y * blockDim.z;
    int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    int idx = myblock * blocksize + subthread;

    // Setup the RNG:
    curandState localState;
    curand_init(rseed_a+rseed_b*idx,rseed_c,0,&localState);

    // End if number of threads exceeds n:
    if(idx >= n)
	{
		return;
	}

    // Sample:
    float low = (trunc[idx] - mu[idx])/sigma[idx];
    float alpha = (low + sqrt(pow(low,2)+4))/2;
    bool picked = false;

    while(!picked)
    {
        float v = curand_uniform(&localState);
        float z = log(1-v)/(-alpha) + low;
        float delta.z = exp(-pow(z-alpha,2)/2);
        float u = curand_uniform(&localState);

        if(u <= delta.z)
        {
		picked = true;
	}
    }

    float x = sigma[idx]*z + mu[idx];
    
    if(lo[idx] == false)
	{
		x = -x;
	}

    vals[idx] = x;

    return;
}

} // END extern "C"

