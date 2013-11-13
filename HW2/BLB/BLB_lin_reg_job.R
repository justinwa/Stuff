#==============--==== Setup for running on Gauss... =================#

args <- commandArgs(TRUE)

cat("Command-line arguments:\n")
print(args)

####
# sim_start ==> Lowest possible dataset number
###

###################
sim_start <- 1000
###################

if (length(args)==0){
  sim_num <- sim_start + 1
  set.seed(121231)
} else {
  sim_num <- sim_start + as.numeric(args[1])
  sim_seed <- (762*(sim_num-1) + 121231)
}

cat(paste("\nAnalyzing dataset number ",sim_num,"...\n\n",sep=""))

#Find r and s indices:
index = sim_num-1000; s = 5; r = 50
r_index = (index %% r)+1; s_index = ceiling(index/r)


#======================Run the Simulation Study=====================#

# Load packages:
library(BH)
library(bigmemory)
library(bigmemory.sri)
library(biganalytics)

# I/O specifications:
datapath <- "/home/justinwa/Stuff/HW2/BLB/"
setwd(datapath)

# Mini or Full?
mini <- FALSE
if (mini){
	rootfilename <- "blb_lin_reg_mini"
} else {
	rootfilename <- "blb_lin_reg_data"
}

# Attach Big Matrix: 
fileName = paste(rootfilename,".desc",sep="")
dat <- attach.big.matrix(dget(fileName))

# Remaining BLB specs:
p = dim(dat)[2]; n = dim(dat)[1]; gamma = 0.7; b = ceiling(n^gamma)

# Extract the subset:
const_sim_seed = (762*(s_index-1) + 121231)
set.seed(const_sim_seed) 
b.index = sample(1:n,size=b,replace=FALSE)

# Reset simulation seed:
set.seed(sim_seed)

# Bootstrap dataset:
new.dat = dat[b.index,] 
counts = rmultinom(1,size=n,prob=rep(1/b,b))
weights = counts/n

# Fit lm: 
y = new.dat[,p]; X = new.dat[,-p]
reg = lm(y~X,weights = weights)
store = summary(reg)$coeff[-1,1]

# Output file: 
outfile = paste("output/","coef_",sprintf("%02d",s_index),"_",
sprintf("%02d",r_index),".txt",sep="")
write.table(store,file=outfile)
