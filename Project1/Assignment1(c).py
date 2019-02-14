#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 21:32:11 2017

@author: jianfengsong
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as ns
import math
import sympy as sym
#import scipy as sym

x=sym.Symbol('x')
p=np.array([[1,0.2],[0.2,1]])
pt=p**-1
u1=np.array([[1,1]])
u1t=np.matrix.transpose(u1)
u0=np.array([[0,0]])
u0t=np.matrix.transpose(u0)
x1ur=0
x2ur=0
y1ur=0
y2ur=0
plt.figure(1)
x1l=list()
x2l=list()
sample_size=10
x1= np.random.multivariate_normal([1,1], p,sample_size)
x2= np.random.multivariate_normal([0,0], p,sample_size)

an=np.dot(pt,(u1t-u0t))
bn=(-1/2)*np.dot(np.dot([np.matrix.transpose(u1t-u0t)],p**-1),(u1t+u0t))
plt.plot(x1[:,0],x1[:,1],'x')
plt.plot(x2[:,0],x2[:,1],'o')

sumx=0
sumy=0
for a in x1:
    sumx=a+sumx
sum1=sumx/sample_size #sum1 is mean of u1(1,1)
for a in x2:
    sumy=a+sumy
sum0=sumy/sample_size#sum0 is mean of u0(0,0)
pn1=(np.dot((x1-sum0).T,(x1-sum0))+np.dot((x2-sum1).T,(x2-sum1)))/(2*sample_size-2)

an1=np.dot(pn1**-1,(sum1-sum0))
bn1=(-1/2)*np.dot(np.dot((sum1-sum0).T,pn1**-1),(sum1-sum0))
print(an1)
print(bn1)
x1=np.arange(-4,5,0.1)
x2p=-bn1/an1[1]-an1[0]*x1/an1[1]
plt.plot(x1,x2p,'b',label='real')
x12=np.arange(-4,5,0.1)
bn2=(-1/2)*np.dot(np.dot((u1t-u0t).T,p**-1),(u1t-u0t))
x22p=-bn2/an[1]-an[0]*x12/an[1]
plt.plot(x12,x22p.T,'r',label='optimal')
plt.legend()
plt.show()
#optimal
#pn2=(np.dot((x1-u0.T).T,(x1-u0.T))+np.dot((x2-u1.T).T,(x2-u1.T)))/(2*sample_size-2)
#an2=np.dot(pn2**-1,(u1-u0))
#bn2=(-1/2)*np.dot(np.dot((u1-u0).T,pn2**-1),(u1-u0))
##print(an1)
##print(bn1)
#x12=np.arange(-4,5,0.1)
#x22p=-bn2/an2[1]-an2[0]*x1/an2[1]
#plt.plot(x12,x22p,'r')

    