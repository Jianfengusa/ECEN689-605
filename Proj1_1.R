# Materials Informatics
# Computer Project 1
# Assignment 1 
# Author: Ulisses Braga-Neto

## Item (b)  

sigma <- seq(0.01,5,0.01)
plot(sigma,pnorm(-1/(sqrt(2)*sigma)),type="l",lwd=2,xlab="sigma",ylab="Bayes Error",ylim=c(0,0.5),main=NULL)
lines(sigma,pnorm(-1/(sqrt(2)*sigma*sqrt(1+0.4))),col="brown",lwd=2)
lines(sigma,pnorm(-1/(sqrt(2)*sigma*sqrt(1+0.6))),col="blue",lwd=2)
lines(sigma,pnorm(-1/(sqrt(2)*sigma*sqrt(1+0.8))),col="green",lwd=2)
lines(sigma,pnorm(-1/(sqrt(2)*sigma*sqrt(1+1))),col="red",lwd=2)
legend("bottomright",legend=c("rho = 0","rho = 0.4","rho = 0.6","rho = 0.8","rho = 1"),lwd=c(2,2,2,2,2),col=c("black","brown","blue","green","red"),inset=0.025,bg="lightgray")
grid()

# Item (c)

library(MASS) #needed for mvrnorm

# parameters of the simulation
n    <- 20   # sample size
c    <- 0.5  # prior probability of class 1
cor  <- 0.2  # correlation between two genes
sig2 <- 1    # predictor variance

# Simulate 2-D data from Gaussian distributions
n1 <- 0
while (n1<8 | n1>12)
  n1 <- rbinom(1,n,c);
n0 <- n - n1
X0 <- mvrnorm(n0,mu=c(0,0),Sigma=matrix(sig2*c(1,cor,cor,1),2,2))
X1 <- mvrnorm(n1,mu=c(1,1),Sigma=matrix(sig2*c(1,cor,cor,1),2,2))

# Plot data and optimal classifier
quartz()
plot(X0,xlim=c(-3,3),ylim=c(-3,3),col="red",pch="O",lwd=2,xlab="X0",ylab="X1")
points(X1,col="blue",pch="X")
abline(a=0,b=-1,lty=5)
grid()

# Find LDA and plot
m0 <- colMeans(X0) # mean(X0)
m1 <- colMeans(X1) # mean(X1)
S <- ((n0-1)*var(X0)+(n1-1)*var(X1))/(n-2) # pooled covariance
Si <- solve(S);
a <- as.vector(Si%*%(m1-m0))
b <- as.numeric(0.5%*%t(m0-m1)%*%Si%*%(m0+m1))
slope <- -a[1]/a[2]
intercept <- -b/a[2]
abline(a=intercept,b=slope,lwd=2)
legend("topright",c("Optimal","LDA"),lty=c(5,1),lwd=2,inset=0.025,bg="lightgray")

## Item (d)  

# Find optimal (Bayes) error
dt <- sqrt(2)/sqrt(sig2*(1+cor))
berr <- pnorm(-0.5*dt)
cat(sprintf("Optimal error = %.2f%%\n\n",100*berr))

m <- 500 # test sample size
N <- seq(20,100,20)  # vector of sample sizes
exact_err <- vector("numeric",length(N)) # vector to store exact errors (formula)
MC_err <- vector("numeric",length(N)) # vector to store test-set errors

# loop
cnt <- 1
for (n in N) {

  # Simulate training data
  n1 <- 0
  while (n1<0.4*n | n1>0.6*n)
    n1 <- rbinom(1,n,c);
  n0 <- n - n1
  X0 <- mvrnorm(n0,mu=c(0,0),Sigma=matrix(sig2*c(1,cor,cor,1),2,2))
  X1 <- mvrnorm(n1,mu=c(1,1),Sigma=matrix(sig2*c(1,cor,cor,1),2,2))

  # Find LDA parameters
  m0 <- colMeans(X0) # mean(X0)
  m1 <- colMeans(X1) # mean(X1)
  S <- ((n0-1)*var(X0)+(n1-1)*var(X1))/(n-2) # pooled covariance
  Si <- solve(S);
  a <- as.vector(Si%*%(m1-m0))
  b <- as.numeric(0.5%*%t(m0-m1)%*%Si%*%(m0+m1))

  # Find exact classification error
  dt0 <- sqrt(a[1]^2+2*cor*a[1]*a[2]+a[2]^2)
  err <- 0.5*(pnorm((-a[1]-a[2]-b)/dt0)+pnorm(b/dt0))
  exact_err[cnt] <- err
  cat(sprintf("Exact    LDA classifier error (n = %g) = %.2f%%\n",n,100*err))

  # Simulate testing data
  nt1 <- rbinom(1,m,c);
  nt0 <- m - nt1
  Xt0 <- mvrnorm(nt0,mu=c(0,0),Sigma=matrix(sig2*c(1,cor,cor,1),2,2))
  Xt1 <- mvrnorm(nt1,mu=c(1,1),Sigma=matrix(sig2*c(1,cor,cor,1),2,2))

  # Find test-sample (Monte-Carlo) classification error
  err <- (sum(Xt0%*%a + b>0) + sum(Xt1%*%a + b <= 0))/m
  MC_err[cnt] <- err
  cat(sprintf("Test-Set LDA classifier error (n = %g) = %.2f%%\n\n",n,100*err))
 
  cnt <- cnt + 1
}  

# plot errors
quartz()
plot(N,exact_err,ylim=c(0.22,0.32),col="red",lwd=2,type="l",xlab="training sample size",ylab="classification error")
lines(N,MC_err,col="blue",lwd=2)
lines(N,rep(berr,length(N)),lty=5,lwd=2)
legend("topright",c("Exact","Test Set","Bayes"),lty=c(1,1,5),col=c("red","blue","black"),lwd=2,inset=0.025,bg="lightgray")
grid()