#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 09:46:21 2017

@author: jianfengsong
"""
import xlrd as xl
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.figure_factory as ff
import pandas as pd
import plotly
from sklearn import linear_model
import operator
from itertools import combinations
from sklearn.linear_model import LinearRegression as lnr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
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
        data_set=[[] for x in range(len(train_row)-1)]
        for a in range (1,len(train_row),1):
            for b in range(0,len(train_row[a])):
                data_set[a-1].append(float(train_row[a][b]))
        return np.asarray(data_set)
################################################################################
    def get_data():
            data_set=fun.excel_data('SFE_Dataset.xlsx')
            data=data_set.T
            data_1,data_2=list(),list()
            for a in range(len(data)):
                num_0=0
                for b in range(len(data[a])):
                    if data[a][b]==0:
                        num_0+=1
                    else:
                        num_0=num_0
                pre_0=num_0/len(data[a])
                if pre_0<=0.4:
                  data_1.append(data[a])
            data_1=np.asarray(data_1).T
            for a in range(len(data_1)):
                num_0=0
                for b in range(len(data_1[a])):
                    if data_1[a][b]==0:
                        num_0+=1
                if num_0==0:
                    data_2.append(data_1[a])
            data_good=np.asarray(data_2)
            return data_good
################################################################################
    def data_clas():
            data_set=fun.get_data()
            data_no_sfe=list()
            data_label=list()
            for a in range(len(data_set)):
                b=len(data_set[a])-1
                if data_set[a][b]<35:
                    data_label.append(data_set[a][b])
                elif data_set[a][b]>45:
                    data_label.append(data_set[a][b])
                else:
                    data_label.append(data_set[a][b])
            data_set1=data_set.T
            for a in range(0,len(data_set1)-1):
                data_no_sfe.append(data_set1[a])
            data_nosfe=np.asarray(data_no_sfe)    
            return data_nosfe,data_label
#################################################################################
    def ehaustive(num):
        selected_feature_set=list()
        selected_feature1=combinations(range(7),num)
        selected_feature=np.asarray(list(selected_feature1))
        return selected_feature
#################################################################################    
    def linear(data,label):
        data_coef=list()
        data_mse=list()
        data_r2=list()
        data_list=list()
        data_pred_list=list()
        for a in (range(len(data))):
            data_var=data[a]
            data_var.shape=(211,1)
            data_lnr=lnr()
            data_lnr.fit(data_var,label)
            data_pred=data_lnr.predict(data_var)
            r2_score=data_lnr.score(data_var,label)          
            data_list.append(data_var)
            data_pred_list.append(data_pred)
            data_r2.append(r2_score) 
            data_coef.append(data_lnr.coef_)
            data_mse.append(mean_squared_error(label,data_pred))
        return data_r2,data_coef,data_mse,data_pred_list
################################################################################
    def linear_ex(data,label,num_list):
        data_coef=list()
        data_mse=list()
        data_r2=list()
        data_list=list()
        data_pred_list=list()
        minr2=0
        if len(num_list)==7:
            for a in num_list:
                data_var=data[a,:]
                data_var.shape=(211,1)
                data_lnr=lnr()
                data_lnr.fit(data_var,label)
                data_pred=data_lnr.predict(data_var)
                r2_score=data_lnr.score(data_var,label)
                if minr2<r2_score:
                    minr2=r2_score
                    minfea=a
                    coef=data_lnr.coef_
                    mrss=mean_squared_error(label,data_pred)
        else:
            for a in num_list:
                data_var=data[a,:]
                data_var=data_var.T
                data_lnr=lnr()
                data_lnr.fit(data_var,label)
                data_pred=data_lnr.predict(data_var)
                r2_score=data_lnr.score(data_var,label)
                if minr2<r2_score:
                    minr2=r2_score
                    minfea=a
                    coef=data_lnr.coef_
                    mrss=mean_squared_error(label,data_pred)
        return minr2,minfea,coef,mrss
################################################################################
    def linear_sf(data,label,num_list):
        data_var=data[num_list,:]
        data_var=data_var.T
        data_lnr=lnr()
        data_lnr.fit(data_var,label)
        data_pred=data_lnr.predict(data_var)
        r2_score=data_lnr.score(data_var,label)
        coef=data_lnr.coef_
        mrss=mean_squared_error(label,data_pred)
        return r2_score,coef,mrss
#################################################################################
data,label=fun.data_clas()
data_r2,data_coef,data_mse,data_pred=fun.linear(data,label)
for a in (range(len(data))):
    plt.figure(a)
    data_var=data[a]
    data_var.shape=(211,1)
#    plt.plot(data_var,data_pred[a])
#    plt.plot(data_var,label,'o')
#######################  2   ###################
minr2list=list()
minfealist=list()
coeflist=list()
mrsslist=list()
adr2list=list()
for a in range (1,6):
    num_list=fun.ehaustive(a)
    minr2,minfea,coef,mrss=fun.linear_ex(data,label,num_list)
    minr2list.append(minr2)
    minfealist.append(minfea)
    coeflist.append(coef)
    mrsslist.append(mrss)
for r in range(len(minr2list)):
    adr2=minr2list[r]-(len(minfealist[r])-1)/(211-len(minfealist[r]))*(1-minr2list[r])
    adr2list.append(adr2)
################
sfsr2,sfsr2in,sfs_inlist,sfs_r2=list(),list(),list(),list()
sfs_coef,sfs_mrss,sfsadr2list=list(),list(),list()
for a in range(7):
    sfsr2.append(a)
for times in range(5):
    sfr2=0
    for a in sfsr2:
        sfsr2in.append(a)
        num_list=np.asarray(sfsr2in)
        r2,coef1,mrss1=fun.linear_sf(data,label,num_list)
        if sfr2<r2:
            minin=a
            sfr2=r2
            coef=coef1
            mrss=mrss1
        sfsr2in.remove(a)
    sfsr2in.append(minin)
    sfsr2.remove(minin)
    sfs_inlist.append(np.asarray(sfsr2in))
    sfs_r2.append(sfr2)
    sfs_coef.append(coef)
    sfs_mrss.append(mrss)
for h in range(len(sfs_r2)):
    sfsadr2=sfs_r2[h]-(len(sfs_inlist[h])-1)/(211-len(sfs_inlist[h]))*(1-sfs_r2[h])
    sfsadr2list.append(sfsadr2)   
#####################   3   ###########################  
alpha=[50, 30, 15, 7, 3, 1, 0.30, 0.10, 0.03, 0.01]  
alpha1=[-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0,7,-0.7,-0.8,-0.9]
clf_list,lasso_list=list(),list()
for a in alpha:
    clf = Ridge(alpha=a)
    clf.fit(data.T, label)
    clf_list.append(clf.coef_)
for b in alpha:
    lasso = linear_model.Lasso(alpha=b)
    lasso.fit(data.T,label)
    lasso_list.append(lasso.coef_)
plt.figure(1)
plt.plot(alpha,clf_list)
plt.figure(2)
plt.plot(-1*np.log(alpha),lasso_list)















