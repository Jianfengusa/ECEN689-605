# Materials Informatics
# Computer Project 1
# Assignment 2
# Author: Ulisses Braga-Neto

# read in the data
SFE.orig = read.table("SFE_Dataset.txt",header=T);

# pre-process the data
n.orig <- dim(SFE.orig)[1] 					# original number of rows
p.orig <- apply(SFE.orig>0,2,sum)/n.orig 	# fraction of nonzero entries for each column
SFE.col <- SFE.orig[,p.orig>0.6]			# throw out columns with fewer than 60% nonzero entries
m.col <- apply(SFE.col,1,prod)
SFE.row <- SFE.col[m.col>0,]				# throw out rows that contain any zero entries
d <- dim(SFE.row)[2] - 1						# final number of columns (predictors)
SFE.hl <- SFE.row[(SFE.row[,d+1]<35)|(SFE.row[,d+1]>45),] # throw out rows with middle responses
n <- dim(SFE.hl)[1] 							# final number of rows (replicates)

cat(sprintf("n = %d, d = %d\n",n,d))

# split predictors from response
# quantize response
SFE.pred <- SFE.hl[,1:d]
SFE.resp <- factor(SFE.hl[,d+1]>40,labels=c("Low","High"))

# training data: 20%
# testing data: 80%
# sampled randomly with balanced constraint
ntr <- round(0.2*n)
nts <- n-ntr
f=0
while(f==0) {
	Itr <- sample(1:n,ntr) # training sample indices
	n1tr <- sum(as.numeric(SFE.resp[Itr])-1) # number of points from class 1
	if ((n1tr>0.45*ntr) & (n1tr<0.55*ntr)) f <- 1
	}
n0tr <- ntr - n1tr
 
# compute stat and p-value table
# based on training data
my.t.test <- function(x) 
     unlist(t.test(x~SFE.resp[Itr])[c("p.value","statistic")])
tab <- data.frame(t(sapply(SFE.pred[Itr,],my.t.test)))
tab <- tab[order(abs(tab$statistic.t),decreasing=TRUE),] # order by abs(t.stat)  
print(tab)

##### 2 features

# design LDA classifier and plot it 
# superimposed on training data

cat("\nTop Two Predictors: "); cat(rownames(tab)[1:2])
X <- SFE.pred[Itr,rownames(tab)[1:2]] # feature data
Y <- SFE.resp[Itr] # response
M <- aggregate(X,list(Y),FUN=mean)
m0 <- M[1,-1]
m1 <- M[2,-1]
S <- ((n0tr-1)*cov(X[Y=="Low",]) + (n1tr-1)*cov(X[Y=="High",]))/(ntr-2)
Si <- solve(S)
a <- Si%*%as.numeric(m1-m0)
b <- (-0.5)*as.numeric(m1-m0)%*%Si%*%as.numeric(m0+m1)
color <- c("red","blue")
plot(X,pch=16,col=color[Y],main="Training Data and LDA Classifier")
points(m0,pch=3,cex=1,col=color[1],lwd=2)
points(m1,pch=3,cex=1,col=color[2],lwd=2)
text(m0+0.6,"mu0",col=color[1])
text(m1+0.6,"mu1",col=color[2])
abline(-b/a[2],-a[1]/a[2],lwd=1)
legend("topright",legend=c("High","Low"),pch=c(16,16),col=color,bg="white")
grid()

# plot test data
# with LDA classifier
quartz()
X <- SFE.pred[-Itr,rownames(tab)[1:2]] 
Y <- SFE.resp[-Itr] # response
plot(X,pch=1,col=color[Y],main="Test Data and LDA Classifier")
abline(-b/a[2],-a[1]/a[2],lwd=1)
grid()
legend("topright",legend=c("High","Low"),pch=c(1,1),col=color,bg="white")

# compute test error
g <- as.matrix(X)%*%a + as.numeric(b) # apply discriminant on test data
err <- sum(g*(as.numeric(Y)-1.5)<0) # test discriminant
cat(sprintf("\nTest error = %.3f\n",err/nts)) 

stop()

##### 3 features

# design LDA classifier
cat("\nPredictors: "); cat(rownames(tab)[1:3])
X <- SFE.pred[Itr,rownames(tab)[1:3]] # feature data
Y <- SFE.resp[Itr] # response
M <- aggregate(X,list(Y),FUN=mean)
m0 <- M[1,-1]
m1 <- M[2,-1]
S <- ((n0tr-1)*cov(X[Y=="Low",]) + (n1tr-1)*cov(X[Y=="High",]))/(ntr-2)
Si <- solve(S)
a <- Si%*%as.numeric(m1-m0)
b <- (-0.5)*as.numeric(m1-m0)%*%Si%*%as.numeric(m0+m1)
X <- SFE.pred[-Itr,rownames(tab)[1:3]] 
Y <- SFE.resp[-Itr] # response
g <- as.matrix(X)%*%a + as.numeric(b) # apply discriminant on test data
err <- sum(g*(as.numeric(Y)-1.5)<0) # test discriminant
cat(sprintf("\nTest error = %.3f\n",err/nts))

##### 4 features

# design LDA classifier
cat("\nPredictors: "); cat(rownames(tab)[1:4])
X <- SFE.pred[Itr,rownames(tab)[1:4]] # feature data
Y <- SFE.resp[Itr] # response
M <- aggregate(X,list(Y),FUN=mean)
m0 <- M[1,-1]
m1 <- M[2,-1]
S <- ((n0tr-1)*cov(X[Y=="Low",]) + (n1tr-1)*cov(X[Y=="High",]))/(ntr-2)
Si <- solve(S)
a <- Si%*%as.numeric(m1-m0)
b <- (-0.5)*as.numeric(m1-m0)%*%Si%*%as.numeric(m0+m1)
X <- SFE.pred[-Itr,rownames(tab)[1:4]] 
Y <- SFE.resp[-Itr] # response
g <- as.matrix(X)%*%a + as.numeric(b) # apply discriminant on test data
err <- sum(g*(as.numeric(Y)-1.5)<0) # test discriminant
cat(sprintf("\nTest error = %.3f\n",err/nts))

##### 5 features

# design LDA classifier
cat("\nPredictors: "); cat(rownames(tab)[1:5])
X <- SFE.pred[Itr,rownames(tab)[1:5]] # feature data
Y <- SFE.resp[Itr] # response
M <- aggregate(X,list(Y),FUN=mean)
m0 <- M[1,-1]
m1 <- M[2,-1]
S <- ((n0tr-1)*cov(X[Y=="Low",]) + (n1tr-1)*cov(X[Y=="High",]))/(ntr-2)
Si <- solve(S)
a <- Si%*%as.numeric(m1-m0)
b <- (-0.5)*as.numeric(m1-m0)%*%Si%*%as.numeric(m0+m1)
X <- SFE.pred[-Itr,rownames(tab)[1:5]] 
Y <- SFE.resp[-Itr] # response
g <- as.matrix(X)%*%a + as.numeric(b) # apply discriminant on test data
err <- sum(g*(as.numeric(Y)-1.5)<0) # test discriminant
cat(sprintf("\nTest error = %.3f\n",err/nts))


