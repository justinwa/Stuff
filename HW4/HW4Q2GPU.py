import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import time
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

#Setup

myfile = open("data_04.txt","r")

y = []
x = []

for line in myfile:
	values = line.split()
	y.append(values[0])

	for i in range(1,9):
		x.append(values[i])

n = len(y)-1
y = map(float,y[1:(n+1)])
x = map(float,x[8:])
X = np.mat(x)
X = X.reshape(n,8)

# Additional Parameters

mu = np.zeros(8)
diag1 = np.ones(8)
sigmaInv = np.diag(diag1)

# Function

def probit_mcmc_gpu(y,X,beta_0,Sigma_0_inv,niter,burnin):
	n = len(y)
	p = X.shape[1]
	Zt = np.zeros(n)
	store = []	

	betat = np.zeros(p)
	sigma = np.ones(n)
	maxtries = np.int32(1e2)
	a = np.zeros(n)
	b = np.zeros(n)

	ra = np.int32(np.random.randint(0,600,1))
	rb = np.int32(np.random.randint(0,600,1))
	rc = np.int32(np.random.randint(0,600,1))

	for j in range(0,n):
		if y[j]==0:
			a[j] = np.float32(-np.Inf)
			b[j] = np.float32(0)
		else:
			a[j] = np.float32(0)
			b[j] = np.float32(np.Inf)

	for i in range(0,niter+burnin):
		ns = (X.T*X + Sigma_0_inv).I
		ZtM = np.mat(Zt).reshape(n,1)
		beta_0M = np.mat(beta_0).reshape(p,1)
		nm = np.array((ns*(X.T*ZtM + Sigma_0_inv*beta_0M)).T)[0]
		betat = np.random.multivariate_normal(nm,ns)
		mew = np.array((X*(np.mat(betat).reshape(p,1))).T)[0]
		
		if i > burnin:
			store.append(betat)

		# Threads per block

		tpb = int(32)
		nn = np.int32(n)
		nb = int(1 + (nn/tpb))

		#Allocate Storage
			
		e,f = np.float32(-np.inf),np.float32(3)
		lo,hi = np.repeat(e,n),np.repeat(f,n)
		Zt = np.zeros_like(lo)
		mean1,stdev = np.float32(-5),np.float32(1)
		mu,sigma = np.repeat(mean1,n),np.repeat(stdev,n)


		#Launch Kernel
		rtruncnorm_kernel(drv.Out(Zt),nn,drv.In(mu),drv.In(sigma),drv.In(lo),drv.In(hi),maxtries,ra,rb,rc,block=(int(tpb),1,1),grid=(nb,1))
		
	return store

#Run the function
start = time.time()
test = probit_mcmc_gpu(y,X,mu,sigmaInv,100,100)
end = time.time()
print end - start
np.savetxt("res_4.txt",test)
