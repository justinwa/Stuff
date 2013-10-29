#Justin Wang
#STA 250
#Homework 1: Q3

#Load Libraries
library(MASS)
library(mvtnorm)
library(coda)

#MCMC Logistic Regression
"bayes.logreg" <- function(m,y,X,beta.0,Sigma.0.inv,niter=10,burnin=10,
                           print.every=1000,retune=400,verbose=TRUE,v)
{
	#MH to be used within Gibbs
	"metropolis" <- function(theta,pos,v.j)
	{
		#Logit Inverse Calculator used in Log Stationary
		"logitinv" = function(x,theta.c)
		{
			u = t(x) %*% theta.c
			return(exp(u)/(1+exp(u)))
		}

		#Use to Calculate Log of Stationary Distributions
		"log.stationary" = function(theta.i)
		{
			n = length(y)
			store = rep(0,n)
			add = (-1/2)*theta.i^2

			for(i in 1:n)
			{
				theta[pos] = theta.i
				li = logitinv(X[i,],theta)
				one = y[i]*log(li); two = (m[i]-y[i])*(log(1-li))
			
				if(is.nan(one)) 
					one = 0
			
				if(is.nan(two)) 
					two = 0

				store[i] = one + two
			}

			return(sum(store)+add)
		}

		#Set values of Theta.t and Theta.star
		theta.t = theta[pos]; sigma = v.j
		theta.star = rnorm(1,mean=theta.t,sd=sigma)

		#Set up Necessary Components for Alpha
		U = runif(1,min=0,max=1)
		ls.theta.star = log.stationary(theta.star)
		ls.theta.t = log.stationary(theta.t)

		#If Accepted, then change Theta.t to be Theta.star
		if(log(U) < (ls.theta.star - ls.theta.t))
		{
			theta.t = theta.star
		}

		#Return the Result (Changed or Unchanged)
		return(theta.t)
	}

	#Set up Gibbs Sampler	
	p = 10; j = 1; N = niter + burnin
	theta = rep(0,p); count = rep(0,p)
	store = matrix(0,p,N); accept = matrix(0,p,length(v))
	
	#Gibbs Sampler using Metropolis
	for(t in 1:N)
	{
		#Store the Old Parameters
		old = theta

		#Call MH on each Theta and Store Result
		for(i in 1:10)
		{
			theta[i] = metropolis(theta,i,v[i])
			store[i,t] = theta[i]
		}

		#Increment Count (for Printing)
		for(i in 1:10)
		{
			if(theta[i] != old[i])
				count[i] = count[i] + 1
		}
	}

	for(i in 1:p)
	{
		cat("Acceptance rate for v",i,"is",count[i]/N,"\n")
	}

	store.new = store[,(burnin+1):(niter+burnin)]
	store.new = store
	return(store.new)
}

#Set MCMC Parameters
beta.0 <- matrix(c(0,0))
p <- 2; Sigma.0.inv <- diag(rep(1.0,p))
niter <- 3000; burnin <- 1000


# Read data corresponding to appropriate sim_num:
setwd("C:/Users/Justin/Desktop/STA 250")
fileName = "breast_cancer.txt"
data = read.table(fileName,header=TRUE)

#Extract X and y:
X = as.matrix(data[1:10])
X[,1]

#Set Parameters
y = ifelse(data[11]=="M",1,0)
m = rep(1,length(y))
v = rep(1,10)

#Create Standardized X
new.X = vector()
for(i in 1:10)
{
	standard = (X[,i] - mean(X[,i]))/sd(X[,i])
	new.X = cbind(new.X,standard)
}

#Fit the Bayesian model:
result <- bayes.logreg(m,y,new.X,beta.0,Sigma.0.inv,niter=niter,burnin=burnin,v=v)

#Check Results
result.mat = mcmc(t(result))
summary(result.mat)
plot(result.mat)

#Lag-One Autocorrelation
lagOne = rep(0,10)

for(i in 1:10)
{
	lagOne[i] = acf(result[i,],plot=FALSE)$acf[2]
}

#Posterior Predictive Check
N = 500
values = sample(1:3000,N)

new.dataFrame = vector()

for(j in 1:N)
{
	new.data = rep(0,569)

	for(i in 1:569)
	{
		u = X[i,] %*% result[,value[j]]
		log.inv = e^u/(1+e^u)
		new.data[i] = rbinom(1,1,log.inv)
	}

	new.dataFrame = cbind(new.dataFrame,new.data[i])
}