import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

m = SourceModule("""
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <math_constants.h>

extern "C" {
__global__ void 
rtruncnorm_kernel(float *vals,int n,float *mu,float *sigma,float *lo, float *hi,int maxtries, int ra, int rb,int rc)
{    
	// Usual block/thread indexing...

	int myblock = blockIdx.x + blockIdx.y * gridDim.x;
	int blocksize = blockDim.x * blockDim.y * blockDim.z;
	int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int idx = myblock * blocksize + subthread;

	// End if idx >= n:

	if(idx >= n)
		return;

	// Insist that lo is less than hi, and at least one is not 
	// infinity. Otherwise, throw an error.

	//assert(lo[idx] < hi[idx];
	//assert(isfinite(lo[idx]) | isfinite(hi[idx]));

	// Setup the RNG:

	curandState rand;
	curand_init(ra+rb*idx,rc,0,&rand);

	// Sample:

	bool picked = false;
	int count = 0; 
	float z = 0;
	float low = (lo[idx] - mu[idx])/sigma[idx];
	float high = (hi[idx] - mu[idx])/sigma[idx]; 


	//Case 1: One-Sided Majority Truncation
	if((!isfinite(low) && high >= 0) || (!isfinite(high) && low <= 0))
	{
		while(!picked && count < maxtries)
		{
			count++;
			z = curand_normal(&rand);
		
			if(z > low && z < high)	
				picked = true;
		}
	}

	//Case 2: One-Sided Minority Truncation (Roberts)
	else if((!isfinite(low) && high < 0) || (!isfinite(high) && low > 0))
	{
		float trunc = 0;
		float delta = 0;

		if(isfinite(low))
			trunc = low;
		else if(isfinite(high))
			trunc = high;

		float alpha = (trunc + sqrt(pow(trunc,2)+4))/2;

		while(!picked && count < maxtries)
		{
			count++;
			float v = curand_uniform(&rand);
			z = log(1-v)/(-alpha) + trunc;
		
			if(trunc < alpha)
				delta = exp(-pow(z-alpha,2)/2);
			else if (trunc >= alpha)
			{
				delta = exp(-pow(trunc-alpha,2)/2)*exp(-pow(alpha-z,2)/2);
			}

			float u = curand_uniform(&rand);
		
			if(u <= delta)
				picked = true;
		}

		if(isfinite(high))
			z = -z;		
	}

	//Case 3: Two Sided Truncation (Roberts)
	else if(isfinite(low) && isfinite(high))
	{
		while(!picked && count < maxtries)
		{
			count++;
			float delta;
			z = low + (high-low)*curand_uniform(&rand);
		
			if(low <=0 && high >= 0)
				delta = exp(-pow(z,2)/2);
			else if(hi < 0)
				delta = exp((pow(high,2)-pow(z,2))/2);
			else if(lo > 0)
				delta = exp((pow(low,2)-pow(z,2))/2);

			float u = curand_uniform(&rand);
		
			if(u <= delta)
				picked = true;
		}
	}

	//Convert from Standard Normal to Normal and store result
	float x = sigma[idx]*z + mu[idx];
	vals[idx] = x;
	
	//Done :)
    	return;
}
}
""",include_dirs=['/usr/local/cuda/include/'],no_extern_c=1)

rtruncnorm_kernel = m.get_function("rtruncnorm_kernel")

#Ints
n = np.int32(1e4)
maxtries = np.int32(1e4)
#seed = np.random.randint(0,2000,1)
seed = np.int32(50)
ra = rb = rc = seed

# Threads per block

tpb = int(256)
nb = int(1 + (n/tpb))

#Create means and SDs
mean,stdev = np.float32(-5),np.float32(1)
a,b = np.float32(-np.inf),np.float32(3)

#Convert them to arrays
mu,sigma = np.repeat(mean,n),np.repeat(stdev,n)
lo,hi = np.repeat(a,n),np.repeat(b,n)

#Allocate Storage
dest = np.zeros_like(lo)

#Launch Kernel
rtruncnorm_kernel(drv.Out(dest),n,drv.In(mu),drv.In(sigma),drv.In(lo),drv.In(hi),maxtries,ra,rb,rc,block=(int(tpb),1,1),grid=(nb,1))

#Print result
#res = np.mean(dest)
print "Mean= ",mean,"SD= ",stdev,"A=",a,"B=",b,"\n"
print dest
