#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 16:00:46 2017

@author: jianfengsong
"""
import xlrd as xl
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats as ns
from numpy.linalg import inv
import math
import random
import xlwt as xlw
from itertools import combinations
class fun():
    def excel_data(n):
        train_rows_value=list()
        train_cols_value=list()
        excel=xl.open_workbook(n)
        data_table=excel.sheet_by_index(0)
        rows=data_table.nrows
        cols=data_table.ncols
        for a in range(rows):
            train_rows_value.append(data_table.row_values(a))
        train_row=np.asarray(train_rows_value)
        return train_row
###############################################################
    def LDA_error(samplesize,x1,x0):                           #x1 is the trainning set with mean 1, p is coveriance matrix given in the question, u0 is true mean[0,0]
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
        varx0=(np.dot(an.T,mean0)+bn)/np.sqrt(np.dot(np.dot(an.T,cov),an))
        varx1=(np.dot(an.T,mean1)+bn)/np.sqrt(np.dot(np.dot(an.T,cov),an))
        LDA_err=1/2*(ns.norm.cdf(varx0)+ns.norm.cdf(-varx1))
        return LDA_err,an,bn,cov
###########################################################
    def Cla_error_LDA(an,bn,test1,test0):
        if len(test1)!=0:
            clas_x1_y=-bn/an[1]-an[0]*test1[:,0]/an[1]
        if len(test0)!=0:
            clas_x0_y=-bn/an[1]-an[0]*test0[:,0]/an[1]
#            print(clas_x0_y.shape)
        error_time=0
        for t in range(len(test1)):
            if len(test1)!=0:
                if test1[t,1] < clas_x1_y[0,t]:
                    error_time=error_time+1
        for t in range(len(test0)):
            if len(test0)!=0:
                if test0[t,1] > clas_x0_y[0,t]:
                    error_time=error_time+1
        return error_time/(len(test1)+len(test0))
########################################################
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
        for a in range(int(len(x1_train))):
            tar.append(1)
        for a in range(int(len(x0_train))):
            tar.append(0)
        return train,test,tar
##################################################
    def two_class(data_set):
        data_high=list()
        data_low=list()
        data_label=list()
        for a in range(len(data_set)):
            b=len(data_set[a])-1
            if data_set[a][b] == 'High':
                data_set[a][b]=1
                data_high.append(data_set[a])
            if data_set[a][b]=='Low':
                data_set[a][b]=0
                data_low.append(data_set[a])
            if data_set[a][b] =='SFE':
                data_label.append(data_set[a])
        data_high1=np.asarray(data_high)
        data_low1=np.asarray(data_low)
        return data_high1,data_low1,data_label
##################################################
    def feature_sample(data_set):
        feature_data=[[] for i in range(len(data_set[1])-1)]
        for b in range(len(data_set[1])-1):
            for a in range(len(data_set)):
                feature_data[b].append(float(data_set[a][b]))
        feature_data1=np.asarray(feature_data)
        return feature_data1
#########################################################
    def get_feature_sample():                              #find the feature matrix, for example j[0]is all value of 'C' 
        train_row=fun.excel_data('SFE_Train_Data.xlsx')
        test_row=fun.excel_data('SFE_Test_Data.xlsx')
        train_high,train_low,train_label=fun.two_class(train_row)
        test_high,test_low,test_label=fun.two_class(test_row)
        train_set,test_set,train_label=fun.data(train_high,train_low,test_high,test_low)
        feature_col=fun.feature_sample(train_set)
#        feature_col1=np.asarray(feature_col.append(train_label))
        return feature_col,train_label,test_high,test_low
##########################################################
    def ehaustive(num):
        x=fun.get_feature_sample()
        selected_feature_set=list()
        selected_feature1=combinations(range(7),num)
        selected_feature=np.asarray(list(selected_feature1))
        return selected_feature
############################################################
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
###########################################################
    def NN3_err_apparent(train,tar,test):
        nn3=KNeighborsClassifier(n_neighbors=3)
        nn3.fit(train,tar)
        nn3_clas=nn3.predict(train)
        nn3error=fun.determind(nn3_clas)
        return nn3error ,nn3
############################################################
    def NN3_err_test(nn3,test):
#        nn3=KNeighborsClassifier(n_neighbors=3)
#        nn3.fit(train,tar)
        nn3_clas=nn3.predict(test)
        nn3error=function.determind(nn3_clas)
        return nn3error
############################################################
#    def find_minlda_err(err,min_err,an,bn,b):
#        if min_err>err:
#            min_err=err
#            minan=an
#            minbn=bn
#            mincov=cov
#            min_ind=b
##        else:
##            min_err=min_err
##            minan,minbn,mincov=minan,minbn,mincov
##            min_ind=min_ind
#        return min_err,min_ind,minan,minbn 
############################################################


###############         MAIN()          ########################################
feature_data,feature_label,test_1,test_0=fun.get_feature_sample()
min_ind_set, min_err_set=list(),list()
min3nn_ind_set, min3nn_err_set=list(),list()
for a in range(1,6):
    selected_feature=fun.ehaustive(a)
    min_err=1
    min3nn_err=1
#    print(selected_feature,a)
    for b in range(len(selected_feature)):
        selected_feature_set1=list()
        LDA_feature1set,LDA_feature0set=list(),list()
        for c in range(len(selected_feature[b])):
            indice=selected_feature[b][c]
            selected_feature_set1.append(feature_data[indice])
            selected_feature_set=np.asarray(selected_feature_set1)
            LDA_feature_1,LDA_feature_0=list(),list()
            for d in range(0,12):   #with SFE high
                LDA_feature_1.append(feature_data[indice][d])
            for e in range(12,len(feature_data[c])):    #with SFE low
                LDA_feature_0.append(feature_data[indice][e])
            LDA_feature1set.append(LDA_feature_1)
            LDA_feature0set.append(LDA_feature_0)
        #LDA APPARENT
        LDA_feature1_set=(np.asarray(LDA_feature1set)).T   #as my x1              
        LDA_feature0_set=(np.asarray(LDA_feature0set)).T   #as my x0
        LDA_err1,an,bn,cov=fun.LDA_error(25,LDA_feature1_set,LDA_feature0_set)
        LDA_err=fun.Cla_error_LDA(an,bn,LDA_feature1_set,LDA_feature0_set)
        #find min for LDA  
#        min_err,min_ind,minan,minbn=fun.find_minlda_err(LDA_err,min_err,an,bn,b)
        if min_err>LDA_err:
            min_err=LDA_err
            minan=an
            minbn=bn
            mincov=cov
            min_ind=b
        else:
            min_err=min_err
            minan,minbn,mincov=minan,minbn,mincov
            min_ind=min_ind
        # 3NN APPARENT
        train,test,tar=fun.data(LDA_feature1_set,LDA_feature0_set,test_1,test_0)
#        print(train)
        nn3error,nn3=fun.NN3_err_apparent(train,tar,test)
        print(nn3error)
        if min3nn_err>nn3error:
            min3nn_err=nn3error
            min3nn_ind=b
            minnn3=nn3
        else:
            min3nn_err=min3nn_err
            min3nn_ind=min3nn_ind
            minnn3=minnn3
#    min3nn_test_err=fun.NN3_err_test(minnn3,test)
    print(min3nn_test_err)
    print(min_err,selected_feature[min_ind])
#    print(min3nn_err,selected_feature[min3nn_ind])
    min_err_set.append(min_err)
    min_ind_set.append(selected_feature[min_ind])
    min3nn_err_set.append(min3nn_err)
    min3nn_ind_set.append(selected_feature[min3nn_ind])
#z,h=fun.get_feature_sample()
#x=list(combinations('C N Ni Fe Mn Si Cr',2))
#y=np.asarray(list(combinations(range(7),2)))




















