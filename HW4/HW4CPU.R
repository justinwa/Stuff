library(MASS)
library(coda)
library(truncnorm)
setwd("C:/Users/Justin/Desktop")

##BEGIN Question 1###

sample.trunc <- function(mu,sigma,a,b,maxtries)
{
	#Loop conditions
	picked = FALSE; count = 0

	#Define the N(0,1) low and high values
	lo <- (a - mu)/sigma; hi <- (b - mu)/sigma 

	#Case 1: One-Sided Majority Truncation. Rejection Sampling. 
	if((!is.finite(a) & hi >= 0) | (!is.finite(b) & lo <= 0))
	{
		accepted <- FALSE
		while(!accepted & count < maxtries) 
		{
			count <- count + 1
			z <- rnorm(1,0,1)

			if(z > lo & z < hi) 
				accepted <- TRUE
		}
	}

	#Case 2: One-Sided Minority Truncation. Roberts' Sampling. 
	else if((!is.finite(lo) & hi < 0) | (!is.finite(hi) & lo > 0))
	{
		if(is.finite(a))
		{
			low <- lo
		}
		else if(is.finite(b[1]))
		{
			low <- -hi 
		}

      	alpha <- (low + sqrt(low^2+4))/2

		while(!picked & count < maxtries)
		{
			count = count + 1
			
			v <- runif(1,0,1)
			z <- log(1-v)/(-alpha) + low

			if(low < alpha)
				delta.z <- exp(-((z-alpha)^2)/2)
			else if(low >= alpha)
				delta.z <- exp(-(low - alpha)^2/2)*exp(-(alpha-z)^2/2)

			u <- runif(1,0,1)

			if(u <= delta.z)
				picked <- TRUE
		}

		if(is.finite(b))
			z <- -z
	}

	#Case 3: Two-sided Truncation: Roberts' Sampling.
	else if(is.finite(lo) & is.finite(hi))
	{	
		while(!picked & count < maxtries)
		{
			count = count + 1

			z <- runif(1,lo,hi)

			if(lo <= 0 & hi >=0)
				delta.z <- exp(-z^2/2)
			else if(hi < 0)
				delta.z <- exp((hi^2 - z^2)/2)
			else if(lo > 0)
				delta.z <- exp((lo^2 - z^2)/2)

			u = runif(1,0,1)

			if(u <= delta.z)
				picked = TRUE
		}
	}

	#Turn z into x before returning it
	x = sigma*z + mu

	return(x)
}

quick.unif <- function(n,a,b)
{
	return(a + (b-a)*runif(n,0,1))
}

#Test the above function
a <- 0; b <- Inf; mu <- -5; sigma <- 1

system.time(replicate(1e6,sample.trunc(mu,sigma,a,b,1000)))
test <- replicate(1000,sample.trunc(mu,sigma,a,b,1000)); mean(test)
reference <- rtruncnorm(1000,a,b,mu,sigma); mean(reference)

#-----------------------------------------------------------------#

##BEGIN Question 2###

probit_mcmc_cpu <- function(y,X,beta_0,Sigma_0_inv,niter,burnin) {
	
	a <- rep(-1,n); b <- rep(-1,n)

	for(j in 1:n)
	{
		a[j] <- ifelse(y[j]==0,-Inf,0)
		b[j] <- ifelse(y[j]==0,0,Inf)
	}

	n <- length(y); p <- dim(X)[2]
	beta.t <- rep(0,p); Z.t <- rep(0,n)
	samples <- matrix(0,p,niter+burnin)
	
	for(i in 1:(niter+burnin))
	{
		new.sigma <- solve(t(X) %*% X + Sigma_0_inv)
		new.mu <- new.sigma %*% (t(X) %*% Z.t + Sigma_0_inv %*% beta_0)
		beta.t <- mvrnorm(1,mu=new.mu,new.sigma)
		Z.t <- rtruncnorm(n,a,b,mean=(X %*% beta.t),sd=rep(1,n))
		samples[,i] <- beta.t
	}

	return(samples)
}


#Test the above function
setwd("C:/Users/Justin/Desktop/cmc")
out_dat <- read.table("data_04.txt",header=TRUE)
n <- nrow(out_dat); p <- 8
y <- out_dat[,1]; X <- as.matrix(out_dat[,-1])
beta_0 <- rep(1,p); Sigma_0_inv <- diag(1,p,p)

system.time(result <- probit_mcmc_cpu(y,X,beta_0,Sigma_0_inv,100,100))
summary(mcmc(t(result)))