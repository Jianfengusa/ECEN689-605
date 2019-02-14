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

infn=np.inf

def var(x1,y1):
    h=-np.sqrt(2)*x1*np.sqrt(1+y1)
    return 1/h
#    return -math.pow(h,-1)

#def snrv(x):
#    return [(2*np.pi)**0.5]*{integrate.quad(np.exp(-(x**2))/2,lambda x :-infn,lambda x:x} 

x=np.arange(0.001,10.0,0.001)
y=np.arange(0.4,1.0,0.2)

plt.figure(1)
#for a in y:
#    for b in x:
plt.plot(x,ns.norm.cdf(var(x,0.4)),'r',label='0.4')
plt.plot(x,ns.norm.cdf(var(x,0.6)),'b',label='0.6')
plt.plot(x,ns.norm.cdf(var(x,0.8)),'g',label='0.8')
plt.plot(x,ns.norm.cdf(var(x,1.0)),'y',label='1.0')
#      print(x,var(x,a))
plt.legend()
plt.show()
      
#print (snrv(1))
