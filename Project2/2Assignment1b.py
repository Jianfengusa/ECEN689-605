#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:13:35 2017

@author: jianfengsong
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import scipy.stats as ns
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
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
        if len(test1)!=0:
            clas_x1_y=-bn/an[1]-an[0]*test1[:,0]/an[1]
        if len(test0)!=0:
            clas_x0_y=-bn/an[1]-an[0]*test0[:,0]/an[1]
        error_time=0
        for t in range(len(test1)):
            if len(test1)!=0:
                if test1[t,1] < clas_x1_y[0,t]:
                    error_time=error_time+1
            if len(test0)!=0:
                if test0[t,1] > clas_x0_y[0,t]:
                    error_time=error_time+1
        return error_time/(len(test1)+len(test0))
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
    def kfold_3nn(train1,tar1,num,sample_size):
        train=np.asarray(train1)
        tar=np.asarray(tar1)
        numberofwrong=0
        wrong_pro=0
        time=0
        if num!=1:
            kf=KFold(n_splits=num)
            kf.get_n_splits(train)
            for train_index, test_index in kf.split(train):
                time+=1
                x_train, x_test = train[train_index], train[test_index]
                y_train, y_test = tar[train_index], tar[test_index]
                nn3=KNeighborsClassifier(n_neighbors=3)
                nn3.fit(x_train,y_train)
                nn3_clas=nn3.predict(x_test)
                for a in range(len(nn3_clas)):
                    if nn3_clas[a] != y_test[a]:
                        numberofwrong+=1
            wrong_pro=numberofwrong/(sample_size)
        else:
            loo = LeaveOneOut()
            loo.get_n_splits(train)
            for train_index, test_index in loo.split(train):
                x_train, x_test = train[train_index], train[test_index]
                y_train, y_test = tar[train_index], tar[test_index]
                nn3=KNeighborsClassifier(n_neighbors=3)
                nn3.fit(x_train,y_train)
                nn3_clas=nn3.predict(x_test)
                for a in range(len(nn3_clas)):
                    if nn3_clas[a] != y_test[a]:
                        numberofwrong+=1
            wrong_pro=numberofwrong/(sample_size)
        return   wrong_pro  
            
    def kfold_SVM(train1,tar1,n,sample_size):
        train=np.asarray(train1)
        tar=np.asarray(tar1)
        numberofwrong=0
        wrong_pro=0
        if n!=1:
            kf=KFold(n_splits=n)
            kf.get_n_splits(train)
            for train_index, test_index in kf.split(train):
                x_train, x_test = train[train_index], train[test_index]
                y_train, y_test = tar[train_index], tar[test_index]
                svm_cla=svm.LinearSVC()
                svm_cla.fit(x_train,y_train)
                svm_cla_set=svm_cla.predict(x_test)
                for a in range(len(svm_cla_set)):
                    if svm_cla_set[a]!=y_test[a]:
                        numberofwrong+=1
            wrong_pro=numberofwrong/(sample_size)
        else:
            loo = LeaveOneOut()
            loo.get_n_splits(train)
            for train_index, test_index in loo.split(train):
                x_train, x_test = train[train_index], train[test_index]
                y_train, y_test = tar[train_index], tar[test_index]
                svm_cla=svm.LinearSVC()
                svm_cla.fit(x_train,y_train)
                svm_cla_set=svm_cla.predict(x_test)
                for a in range(len(svm_cla_set)):
                    if svm_cla_set[a]!=y_test[a]:
                        numberofwrong+=1
            wrong_pro=numberofwrong/(sample_size)
        return wrong_pro
    def kfold_LDA(train1,tar1,num,sample_size):
        train=np.asarray(train1)
        tar=np.asarray(tar1)
        numberofwrong=0
        wrong_pro=0
        if num!=1:
            kf=KFold(n_splits=num)
            kf.get_n_splits(train)
            for train_index, test_index in kf.split(train):
                x_train, x_test = train[train_index], train[test_index]
                y_train, y_test = tar[train_index], tar[test_index]
                x1train,x0train,x1test,x0test=function.kfold_LDA_data(train,tar,x_train,x_test,y_train,y_test)
                LDA,an,bn,cov=function.LDA_error(sample_size,x1train,x0train)
                numberofwrong+=function.Cla_error_LDA(an,bn,x1test,x0test)
            wrong_pro=numberofwrong/sample_size
        else:
            loo = LeaveOneOut()
            loo.get_n_splits(train)
            for train_index, test_index in loo.split(train):
                x_train, x_test = train[train_index], train[test_index]
                y_train, y_test = tar[train_index], tar[test_index]
                x1train,x0train,x1test,x0test=function.kfold_LDA_data(train,tar,x_train,x_test,y_train,y_test)
                LDA,an,bn,cov=function.LDA_error(sample_size,x1train,x0train)
                numberofwrong+=function.Cla_error_LDA(an,bn,x1test,x0test)
            wrong_pro=numberofwrong/sample_size
        return wrong_pro
    def kfold_LDA_data(train,tar,x_train,x_test,y_train,y_test):
        x1train=[]
        x1test=[]
        x0train=[]
        x0test=[]
        for a in range(len(y_train)):
            if y_train[a]==1:
                x1train.append(x_train[a])
            else:
                x0train.append(x_train[a])
        for a in range(len(y_test)):
            if y_test[a]==1:
                x1test.append(x_test[a])
            else:
                x0test.append(x_test[a])
        x1train1=np.asarray(x1train)
        x0train0=np.asarray(x0train)
        x1test1=np.asarray(x1test)
        x0test0=np.asarray(x0test)
        return x1train1,x0train0,x1test1,x0test0
#main function##########################################################################
for z in range(2):
    error_perc_set=list()
    nn3_error_set=list()
    nn3_5fold_error_set=list()
    nn3_1fold_error_set=list()
    svm_error_set=list()
    svm_5fold_error_set=list()
    svm_1fold_error_set=list()
    lda_error_set=list()
    lda_5fold_error_set=list()
    lda_1fold_error_set=list()
    for sample_size in sample_sizes:
        LDA_errs=0
        error_percs=0
        nn3error=0
        nn3errors=0
        nn3_1fold=0
        nn3_5fold=0
        nn3_5fold_errors=0
        nn3_1fold_errors=0
        svmerror=0
        svmerrors=0
        svm_1fold=0
        svm_5fold=0
        svm_5fold_errors=0
        svm_1fold_errors=0
        lda_1fold=0
        lda_5fold=0
        lda_5fold_errors=0
        lda_1fold_errors=0
        for a in range(0,100):
            nn3mis=0
            svmmis=0
            if z==0:
                x1_train,x0_train=function.firsttime(sample_size)
                test1,test0=function.firsttime(400)
            if z==1:
                x1_train,x0_train=function.secondtime(sample_size)
                test1,test0=function.secondtime(400)
            train,test,tar=function.data(x1_train,x0_train,test1,test0)
            #LDA
            LDA_err_1,an,bn,cov=function.LDA_error(sample_size,x1_train,x0_train)
            error_perc=function.Cla_error_LDA(an,bn,x1_train,x0_train)
            LDA_errs+=LDA_err_1[0][0]
            error_percs+=error_perc
            lda_1fold=function.kfold_LDA(train,tar,1,sample_size)
            lda_1fold_errors+=lda_1fold
            lda_5fold=function.kfold_LDA(train,tar,5,sample_size)
            lda_5fold_errors+=lda_5fold
            #3NN
            nn3=KNeighborsClassifier(n_neighbors=3)
            nn3.fit(train,tar)
            nn3_clas=nn3.predict(train)
            nn3error=function.determind(nn3_clas)
            nn3errors+=nn3error
            nn3_1fold=function.kfold_3nn(train,tar,1,sample_size)
            nn3_1fold_errors+=nn3_1fold
            nn3_5fold=function.kfold_3nn(train,tar,5,sample_size)
            nn3_5fold_errors+=nn3_5fold
            
            #SVM
            svm_cla=svm.LinearSVC()
            svm_cla.fit(train,tar)
            svm_cla_set=svm_cla.predict(train)
            svmerror=function.determind(svm_cla_set)
            svmerrors+=svmerror
            svm_1fold=function.kfold_SVM(train,tar,1,sample_size)
            svm_1fold_errors+=nn3_1fold
            svm_5fold=function.kfold_SVM(train,tar,5,sample_size)
            svm_5fold_errors+=nn3_5fold
#            print(a*sample_size) 
        #error set
        svm_error=svmerrors/100
        svm_error_set.append(svm_error)
        svm_1fold_error=svm_1fold_errors/100
        svm_1fold_error_set.append(svm_1fold_error)
        svm_5fold_error=svm_5fold_errors/100
        svm_5fold_error_set.append(svm_5fold_error)
        
        nn3_error=nn3errors/100
        nn3_error_set.append(nn3_error)
        nn3_1fold_error=nn3_1fold_errors/100
        nn3_1fold_error_set.append(nn3_1fold_error)
        nn3_5fold_error=nn3_5fold_errors/100
        nn3_5fold_error_set.append(nn3_5fold_error)
        
        error_perc=error_percs/100
        error_perc_set.append(error_perc)
        lda_1fold_error=lda_1fold_errors/100
        lda_1fold_error_set.append(lda_1fold_error)
        lda_5fold_error=lda_5fold_errors/100
        lda_5fold_error_set.append(lda_5fold_error)
        
    if z==0:
        plt.figure(1)
        plt.title('LDA')
#        plt.subplot(311)
        plt.plot(sample_sizes,error_perc_set,'y',label='LDA apparent')
        plt.plot(sample_sizes,lda_5fold_error_set,'r',label='lda 5fold')
        plt.plot(sample_sizes,lda_1fold_error_set,'b',label='lda 1fold')
        plt.legend()
        plt.figure(2)
        plt.title('3NN')
#        plt.subplot(312)
        plt.plot(sample_sizes,nn3_error_set,label='3NN apparent')
        plt.plot(sample_sizes,nn3_5fold_error_set,label='3NN 5fold')
        plt.plot(sample_sizes,nn3_1fold_error_set,label='3NN 1fold')
        plt.legend()
        plt.figure(3)
        plt.title('SVM')
#        plt.subplot(313)
        plt.plot(sample_sizes,svm_error_set,label='SVM apparent')
        plt.plot(sample_sizes,svm_1fold_error_set,label='SVM 1fold')
        plt.plot(sample_sizes,svm_5fold_error_set,label='SVM 5fold')
        plt.legend()
    if z==1:
        plt.figure(1)
        plt.title('LDA2')
#        plt.subplot(311)
        plt.plot(sample_sizes,error_perc_set,label='LDA2 apparent')
        plt.plot(sample_sizes,lda_5fold_error_set,label='lda2 5fold')
        plt.plot(sample_sizes,lda_1fold_error_set,label='lda2a 1fold')
        plt.legend()
        plt.figure(2)
        plt.title('3NN2')
#        plt.subplot(312)
        plt.plot(sample_sizes,nn3_error_set,label='3NN2 apparent')
        plt.plot(sample_sizes,nn3_5fold_error_set,label='3NN2 5fold')
        plt.plot(sample_sizes,nn3_1fold_error_set,label='3NN2 1fold')
        plt.legend()
        plt.figure(3)
        plt.title('SVM2')
#        plt.subplot(313)
        plt.plot(sample_sizes,svm_error_set,label='SVM2 apparent')
        plt.plot(sample_sizes,svm_1fold_error_set,label='SVM2 1fold')
        plt.plot(sample_sizes,svm_5fold_error_set,label='SVM2 5fold')
        plt.legend()
plt.show()























