#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 15:56:58 2017

@author: jianfengsong
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ns
import math
from scipy.linalg import det
p=np.array([[1,0.2],[0.2,1]])
u0=np.array([[0,0]])
u1=np.array([[1,1]])
sample_size=np.array([20,30,40,50])
LDA_error_set=list()
error_set=list()

for a in sample_size:
    x1= np.random.multivariate_normal([1,1], p,a)
    x2= np.random.multivariate_normal([0,0], p,a)
    x3= np.random.multivariate_normal([1,1],p,250)
    x4= np.random.multivariate_normal([0,0],p,250)
    sumx1=0
    sumx2=0
    for b in x1:
        sumx1=b+sumx1
    mean_x1=sumx1/a
    for c in x2:
        sumx2=c+sumx2
    mean_x0=sumx2/a
    
    cov=(np.dot((x1-mean_x0).T,(x1-mean_x0))+np.dot((x2-mean_x1).T,(x2-mean_x1)))/(2*a-2)
    an=np.dot(cov**-1,(mean_x1-mean_x0))
    bn=(-1/2)*np.dot(np.dot((mean_x1-mean_x0).T,cov**(-1)),(mean_x1-mean_x0))
    var_x0=(np.dot(an,mean_x0.T)+bn)/math.sqrt(np.dot(np.dot(an,p),an.T))
    var_x1=(np.dot(an,mean_x1.T)+bn)/math.sqrt(np.dot(np.dot(an,p),an.T))
    LDA_error=1/2*(ns.norm.cdf(var_x0)+ns.norm.cdf(-var_x1))
    LDA_error_set.append(LDA_error)  
    clas_x3_y=-bn/an[1]-an[0]*x3[:,0]/an[1]
    clas_x4_y=-bn/an[1]-an[0]*x4[:,0]/an[1]
    error_time=0
    for t in range(250):
        if x3[t,1] < clas_x3_y[t]:
            error_time=error_time+1
        if x4[t,1] > clas_x4_y[t]:
            error_time=error_time+1
    error_set.append(error_time/(500))
    
plt.plot(sample_size*2,error_set,'y',label='formula')
plt.plot(sample_size*2,LDA_error_set,'r',label='test')
plt.legend()
plt.show()
    