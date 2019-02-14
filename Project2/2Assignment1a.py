#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:13:35 2017

@author: jianfengsong
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ns
import math
from scipy.linalg import det
from numpy.linalg import inv
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
p0=np.array([[1,0.2],
             [0.2,1]])
p1=np.array([[1,0.2],
             [0.2,1]])
pro_p0=1/2
pro_p1=1/2
u0=np.array([0,0])
u1=np.array([1,1])
LDA_error_set=list()
N3N_error_set=list()
LDA_err_set=list()
sample_sizes=np.arange(20,101,10)
large_number=1000
total_train_set=[]
total_test_set=[]
class function:
    def firsttime(samplesize):
        x1=np.random.multivariate_normal(u1,p1,int(samplesize/2))
        x0=np.random.multivariate_normal(u0,p0,int(samplesize/2))
        return x1,x0
    def secondtime(samplesize):
        x1=np.random.multivariate_normal(u1,4*p1,int(samplesize/2))
        x0=np.random.multivariate_normal(u0,4*p0,int(samplesize/2))
        return x1,x0
    def LDA_error(samplesize,x1,x0):#x1 is the trainning set with mean 1, p is coveriance matrix given in the question, u0 is true mean[0,0]
        sumx1=0
        sumx0=0
        for a in x1:
            sumx1=a+sumx1
        for b in x0:
            sumx0=b+sumx0
        mean1=sumx1/samplesize*2
        mean0=sumx0/samplesize*2
        cov=(1/(samplesize-2))*(np.matrix((x1-mean1)).T*np.matrix((x1-mean1))+np.matrix((x0-mean0)).T*np.matrix((x0-mean0)))
        an=np.matrix(inv(cov))*np.matrix((mean1-mean0)).T
        bn=(-1/2)*np.matrix((mean1-mean0))*np.matrix(inv(cov))*np.matrix(mean1+mean0).T
        varx0=(np.dot(an.T,u0)+bn)/np.sqrt(np.dot(np.dot(an.T,cov),an))
        varx1=(np.dot(an.T,u1)+bn)/np.sqrt(np.dot(np.dot(an.T,cov),an))
        LDA_err=1/2*(ns.norm.cdf(varx0)+ns.norm.cdf(-varx1))
        return LDA_err,an,bn,cov
    def Cla_error_LDA(an,bn,test1,test0):
        clas_x1_y=-bn/an[1]-an[0]*test1[:,0]/an[1]
        clas_x0_y=-bn/an[1]-an[0]*test0[:,0]/an[1]
        error_time=0
        for t in range(200):
            if test1[t,1] < clas_x1_y[0,t]:
                error_time=error_time+1
            if test0[t,1] > clas_x0_y[0,t]:
                error_time=error_time+1
        return error_time/400
    def data (x1_train,x0_train,test1,test0):
        train=list()
        tar=list()
        test=list()
        for a in range(len(x1_train)):
            train.append(x1_train[a])
        for a in range(len(x0_train)):
            train.append(x0_train[a])
        for a in range(len(test1)):
            test.append(test1[a])
        for a in range(len(test0)):
            test.append(test0[a])
        for a in range(int(sample_size/2)):
            tar.append(1)
        for a in range(int(sample_size/2)):
            tar.append(0)
        return train,test,tar
    def determind(data_set):
        number_of_wrong=0
        for a in range(0,int(len(data_set)/2)):
            if data_set[a]<1:
                number_of_wrong+=1
        for a in range(int(len(data_set)/2),len(data_set)):
            if data_set[a]>0:
                number_of_wrong+=1
        error_rate=number_of_wrong/len(data_set)
        return error_rate
#main function##########################################################################
for z in range(2):
    error_perc_set=list()
    nn3_error_set=list()
    svm_error_set=list()
    for sample_size in sample_sizes:
        LDA_errs=0
        error_percs=0
        nn3error=0
        nn3errors=0
        svmerror=0
        svmerrors=0
        for a in range(0,100):
            nn3mis=0
            svmmis=0
            if z==0:
                x1_train,x0_train=function.firsttime(sample_size)
                test1,test0=function.firsttime(400)
            if z==1:
                x1_train,x0_train=function.secondtime(sample_size)
                test1,test0=function.secondtime(400)
            #LDA
            LDA_err_1,an,bn,cov=function.LDA_error(sample_size,x1_train,x0_train)
            error_perc=function.Cla_error_LDA(an,bn,test1,test0)
            LDA_errs+=LDA_err_1[0][0]
            error_percs+=error_perc
            #3NN
            train,test,tar=function.data(x1_train,x0_train,test1,test0)
            nn3=KNeighborsClassifier(n_neighbors=3)
            nn3.fit(train,tar)
            nn3_clas=nn3.predict(test)
            nn3error=function.determind(nn3_clas)
            nn3errors+=nn3error
            #SVM
            svm_cla=svm.LinearSVC()
            svm_cla.fit(train,tar)
            svm_cla_set=svm_cla.predict(test)
            svmerror=function.determind(svm_cla_set)
            svmerrors+=svmerror
        #error set
        svm_error=svmerrors/100
        svm_error_set.append(svm_error)
        nn3_error=nn3errors/100
        nn3_error_set.append(nn3_error)
        error_perc=error_percs/100
        error_perc_set.append(error_perc)
    if z==0:
        plt.figure(1)
        plt.plot(sample_sizes,error_perc_set,label='LDA')
        plt.legend()
        plt.figure(1)
        plt.plot(sample_sizes,nn3_error_set,label='3NN')
        plt.legend()
        plt.figure(1)
        plt.plot(sample_sizes,svm_error_set,label='SVM')
        plt.legend()
    if z==1:
        plt.figure(2)
        plt.plot(sample_sizes,error_perc_set,label='LDA2')
        plt.legend()
        plt.figure(2)
        plt.plot(sample_sizes,nn3_error_set,label='3NN2')
        plt.legend()
        plt.figure(2)
        plt.plot(sample_sizes,svm_error_set,label='SVM2')
        plt.legend()
plt.show()























