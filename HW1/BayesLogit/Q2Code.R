#Justin Wang
#STA 250
#HW 1: Q2


##
#
# Logistic regression
# 
# Y_{i} | \beta \sim \textrm{Bin}\left(n_{i},e^{x_{i}^{T}\beta}/(1+e^{x_{i}^{T}\beta})\right)
# \beta \sim N\left(\beta_{0},\Sigma_{0}\right)
#
##

library(MASS)
library(mvtnorm)
library(coda)

########################################################################################
########################################################################################
## Handle batch job arguments:

# 1-indexed version is used now.
args <- commandArgs(TRUE)

cat(paste0("Command-line arguments:\n"))
print(args)

####
# sim_start ==> Lowest simulation number to be analyzed by this particular batch job
###

#######################
sim_start <- 1000
length.datasets <- 200
#######################

if (length(args)==0){
  sinkit <- FALSE
  sim_num <- sim_start + 1
  set.seed(1330931)
} else {
  # Sink output to file?
  sinkit <- TRUE
  # Decide on the job number, usually start at 1000:
  sim_num <- sim_start + as.numeric(args[1])
  # Set a different random seed for every job number!!!
  set.seed(762*sim_num + 1330931)
}

# Simulation datasets numbered 1001-1200

########################################################################################
########################################################################################

"bayes.logreg" <- function(m,y,X,beta.0,Sigma.0.inv,niter=10000,burnin=1000,
                           print.every=1000,retune=400,verbose=TRUE,v)
{
	#Metropolis Step
	"metropolis" <- function(theta,pos,v.j)
	{
		#Calculates Logit Inverse given row of Covariates and Parameters
		"logitinv" = function(x,theta.c)
		{
			u = t(x) %*% theta.c
			return(exp(u)/(1+exp(u)))
		}

		#Calculates Log of Stationary 
		"log.stationary" = function(theta.i)
		{
			n = length(y)
			store = rep(0,n)
			add = (-1/2)*theta.i^2

			for(i in 1:n)
			{
				theta[pos] = theta.i
				li = logitinv(X[i,],theta)
				store[i] = y[i]*log(li) + (m[i]-y[i])*(log(1-li))
			}

			return(sum(store)+add)
		}

		#Set-up
		theta.t = theta[pos]; sigma = v[j]
	
		theta.star = rnorm(1,mean=theta.t,sd=sigma)
		U = runif(1,min=0,max=1)
		
		ls.theta.star = log.stationary(theta.star)
		ls.theta.t = log.stationary(theta.t)

		#Accept with probability alpha
		if(log(U) < (ls.theta.star - ls.theta.t))
		{
			theta.t = theta.star
		}

		return(theta.t)
	}

	#Set-up Gibbs Step
	theta = c(1,1); count = c(0,0)
	N = niter + burnin
	store = matrix(0,2,N)
	accept = matrix(0,2,length(v))
	j = 1
	
	#Gibbs Step
	for(t in 1:N)
	{
		old.t1 = theta[1]; old.t2 = theta[2]
		
		if(t <= burnin & (t %% retune) == 0 & j < length(v))
		{
			accept[1,j] = count[1]/retune
			accept[2,j] = count[2]/retune
			count = c(0,0)

			if(verbose)
			{
				cat("Acceptance Rate for v1=",v[j],"is",accept[1,j],"\n")
				cat("Acceptance Rate for v2=",v[j],"is",accept[2,j],"\n")
			}

			if(j <= length(v))
			{
				j = j + 1
			}
		}

		theta[1] = metropolis(theta,1,v[j])
		theta[2] = metropolis(theta,2,v[j])
		store[1,t] = theta[1]; store[2,t] = theta[2] 

		if(theta[1] != old.t1 & verbose)
		{
			count[1] = count[1] + 1
		}

		if(theta[2] != old.t2 & verbose)
		{
			count[2] = count[2] + 1
		}
	}

	store.new = store[,(burnin+1):(niter+burnin)]
	return(store.new)
}



#################################################
# Set up the specifications:
beta.0 <- matrix(c(0,0))
p <- 2; Sigma.0.inv <- diag(rep(1.0,p))
niter <- 2000; burnin <- 2000
# etc... (more needed here)
#################################################

# Read data corresponding to appropriate sim_num:
#setwd("C:/Users/Justin/Desktop/STA 250")
sim_num_c = 1010
fileName = paste("data/blr_data_",sim_num_c,".csv",sep="")
#fileName = "blr_data_1010.csv"
data = read.csv(fileName)

# Extract X and y:
X = as.matrix(data[3:4])
y = data[,1]; m = data[,2]

# Fit the Bayesian model:
v = c(0.065,0.07,0.08)
result <- bayes.logreg(m,y,X,beta.0,Sigma.0.inv,niter=niter,burnin=burnin,v=v)

# Extract posterior quantiles...
result.mat = mcmc(t(result))
summary(result.mat)
plot(result.mat)

theta.1.quantiles = as.vector(quantile(theta.1,seq(0.01,0.99,by=0.01)))
theta.2.quantiles = as.vector(quantile(theta.1,seq(0.01,0.99,by=0.01)))

# Write results to a (99 x p) csv file...
write = cbind(theta.1.quantiles,theta.2.quantiles)
write.table(write,paste("results/blr_result_",sim_num_c,".csv"),sep=",",
row.names=FALSE,col.names=FALSE)

# Go celebrate.
 
cat("done. :)\n")
