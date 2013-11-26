##Part (a)

bisection <- function(f,interval,tolerance,max,verbose=FALSE)
{
	count = 0
	l = min(interval); u = max(interval) 

	while(count < max)
	{
		c = (l+u)/2

		if(abs(f(c)) < tolerance) return(c)

		else
		{
			if(f(l)*f(c) < 0)  u = c else l = c
		}

		count = count + 1

		if(verbose)
		{
			cat("Iteration:",count,"u=",u,"l=",l,"\n")
		}

	}

	return(c)
}


##Part (b)

newton.raphson <- function(g,g.pr,x.st,tol,verbose=FALSE)
{
	x.t = x.st
	count = 0

	while(abs(g(x.t)) >= tol)
	{
		x.t = x.t - g(x.t)/g.pr(x.t)
		count = count + 1

		if(verbose)
		{
			cat("Iteration:",count,"Value of x:",x.t,"\n")
		}
	}

	return(x.t)
}


##Test of Above

g <- function(x)
{
	return((x+1)*(x-2))
}

g.pr <- function(x)
{
	return(2*x-1)
}

root.b = bisection(g,c(-5,0),0.0005,50,TRUE); root.b
root.nr = newton.raphson(g,g.pr,3,0.005,TRUE); root.nr


##Part (c)

h = function(x)
{
	return(125/(2+x) + 38/(1-x) + 34/x)
}

h.dr = function(x)
{
	return(38/(x-1)^2 - 125/(2+x)^2 - 34/x^2)
}

croot.b = bisection(h,c(-2,-0.1),0.0005,10000); croot.b
croot.nr = newton.raphson(h,h.dr,2.5,0.0005); croot.nr
