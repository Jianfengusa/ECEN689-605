#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 16:00:46 2017

@author: jianfengsong
"""
import xlrd as xl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
########################################################
    def data (x1_train,x0_train,test1,test0):
        train=list()
        tar=list()
        test=list()
        test_tar=list()
        for a in range(len(x1_train)):
            train.append(x1_train[a])
        for a in range(len(x0_train)):
            train.append(x0_train[a])
            
        for a in range(int(len(x1_train))):
            tar.append(1)
        for a in range(int(len(x0_train))):
            tar.append(0)
            
        for a in range(len(test1)):
            test.append(test1[a])
        for a in range(len(test0)):
            test.append(test0[a])
            
        for a in range(int(len(test1))):
            test_tar.append(1)
        for a in range(int(len(test0))):
            test_tar.append(0)
        return train,test,tar,test_tar
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
        
        train_set,test_set,train_label,test_label=fun.data(train_high,train_low,test_high,test_low)
        
        feature_col=fun.feature_sample(train_set)
        test_col=fun.feature_sample(test_set)
#        feature_col1=np.asarray(feature_col.append(train_label))
        return feature_col,train_label,test_col,test_label
##########################################################
    def ehaustive(num):
        x=fun.get_feature_sample()
        selected_feature_set=list()
        selected_feature1=combinations(range(7),num)
        selected_feature=np.asarray(list(selected_feature1))
        return selected_feature
############################################################
    def determind(data_set,tar):
        number_of_wrong=0
        for a in range(len(tar)):
            if data_set[a]!=tar[a]:
                number_of_wrong+=1
            else:
                number_of_wrong+=0
        error_rate=number_of_wrong/len(data_set)
        return error_rate
###########################################################
    def NN3_err(train,tar,test,test_tar,n):
        nn3=KNeighborsClassifier(n_neighbors=3)
        nn3.fit(train,tar)
        if n==1:
            nn3_clas=nn3.predict(train)
            nn3error=fun.determind(nn3_clas,tar)
        if n==0:
            nn3_clas=nn3.predict(test)
            nn3error=fun.determind(nn3_clas,test_tar)
        return nn3error
###############################################################
    def LDA_error(train,tar,test,test_tar,n):
        clf=LDA()
        clf.fit(train,tar)
        if n ==1:
            LDA_cla=clf.predict(train)
            error=fun.determind(LDA_cla,tar)
        if n==0:
            LDA_cla=clf.predict(test)
            error=fun.determind(LDA_cla,test_tar)
        return error
########################################################




###############         MAIN()          ########################################
feature_data,feature_label,test_set,test_label=fun.get_feature_sample()
min_ind_set, min_err_set,min_cla_err_set=list(),list(),list()
min3nn_ind_set, min3nn_err_set,min3nn_cla_err_set=list(),list(),list()
for a in range(1,6):
    selected_feature=fun.ehaustive(a)
    min_err=1
    min3nn_err=1
    for b in range(len(selected_feature)):
        selected_feature_set1=list()
        LDA_feature1set,LDA_feature0set,test1,test0=list(),list(),list(),list()
        for c in range(len(selected_feature[b])):
            indice=selected_feature[b][c]
            selected_feature_set1.append(feature_data[indice])
            selected_feature_set=np.asarray(selected_feature_set1)
            LDA_feature_1,LDA_feature_0,test_1,test_0=list(),list(),list(),list()
            for d in range(0,12):   #with SFE high
                LDA_feature_1.append(feature_data[indice][d])
#                test_1.append(test_set[indice][d])
            for e in range(12,len(feature_data[c])):    #with SFE low
                LDA_feature_0.append(feature_data[indice][e])
#                test_0.append(test_set[indice][e])
            for f in range(0,50):
                test_1.append(test_set[indice][f])
            for g in range(50,98):
                test_0.append(test_set[indice][g])
            LDA_feature1set.append(LDA_feature_1)
            LDA_feature0set.append(LDA_feature_0)
            test1.append(test_1)
            test0.append(test_0)
        #LDA APPARENT
        LDA_feature1_set=(np.asarray(LDA_feature1set)).T   #as my x1              
        LDA_feature0_set=(np.asarray(LDA_feature0set)).T   #as my x0
        test1set=(np.asarray(test1)).T
        test0set=(np.asarray(test0)).T
        train,test,tar,test_tar=fun.data(LDA_feature1_set,LDA_feature0_set,test1set,test0set)
        #find min for LDA
        LDA_err=fun.LDA_error(train,tar,test,test_tar,1)
        LDA_cla_err=fun.LDA_error(train,tar,test,test_tar,0)
#        print(LDA_cla_err)
        nn3error=fun.NN3_err(train,tar,test,test_tar,1)
#        print(nn3error,a)
        nn3_cla_error=fun.NN3_err(train,tar,test,test_tar,0)
        #LDA
        if min_err>LDA_err:
            min_err=LDA_err
            min_ind=b
            min_cla_err=LDA_cla_err
        else:
            min_err=min_err
            min_ind=min_ind
        # 3NN 
        if min3nn_err>nn3error:
            min3nn_err=nn3error
            min3nn_ind=b
            min3nncla=nn3_cla_error
        else:
            min3nn_err=min3nn_err
            min3nn_ind=min3nn_ind
            
    min_err_set.append(min_err)#min LDA apparent error
    min_ind_set.append(selected_feature[min_ind])#min LDA error index
    min_cla_err_set.append(min_cla_err)# classification error
    
    min3nn_err_set.append(min3nn_err)#min 3NN apparent error
    min3nn_ind_set.append(selected_feature[min3nn_ind])#min 3nn error index
    min3nn_cla_err_set.append(min3nncla)# classification error
############################################################################
# SFS for LDA
sfs_lda=list()
sfs_lda_ind=list()
sfs_ldatrain,sfs_ldatest=list(),list()
sfs_lda_app,sfs_lda_cla=list(),list()
sfs_lda_app.append(min_err_set[0])
sfs_lda_cla.append(min_cla_err_set[0])
for a in range(7):
    sfs_lda.append(a)
sfs_lda.remove(min_ind_set[0][0])
sfs_lda_ind.append(min_ind_set[0][0])
for a in range (4):
    sfs_ldatrain.append(feature_data[sfs_lda_ind[a]])
    sfs_ldatest.append(test_set[sfs_lda_ind[a]])
    sfs_min=1
    for b in sfs_lda:
        sfs_ldatrain.append(feature_data[b])
        sfs_ldatest.append(test_set[b])
        sfs_lda_train=(np.asarray(sfs_ldatrain)).T
        sfs_lda_test=(np.asarray(sfs_ldatest)).T
        sfs_lda_app_err=fun.LDA_error(sfs_lda_train,tar,sfs_lda_test,test_label,1)
        sfs_lda_cla_err=fun.LDA_error(sfs_lda_train,tar,sfs_lda_test,test_label,0)
        if sfs_min>sfs_lda_app_err:
            sfs_min=sfs_lda_app_err
            sfs_ind=b
            min_cla_err=sfs_lda_cla_err
        else:
            sfs_min=sfs_min
            sfs_ind=sfs_ind
        sfs_ldatrain.pop(a+1)
        sfs_ldatest.pop(a+1)
    sfs_lda.remove(sfs_ind)
    sfs_lda_app.append(sfs_min)
    sfs_lda_cla.append(min_cla_err)
    sfs_lda_ind.append(sfs_ind)
# SFS 3NN
sfs_3nn=list()
sfs_3nn_ind=list()
sfs_3nntrain,sfs_3nntest=list(),list()
sfs_3nn_app,sfs_3nn_cla=list(),list()
sfs_3nn_app.append(min3nn_err_set[0])
sfs_3nn_cla.append(min3nn_cla_err_set[0])
#sfs_3nn_app.append(3)
#sfs_3nn_cla.append(min3nn_cla_err_set[0])
for a in range(7):
    sfs_3nn.append(a)    
sfs_3nn.remove(min3nn_ind_set[0][0])
sfs_3nn_ind.append(min3nn_ind_set[0][0])
#sfs_3nn.remove(3)
#sfs_3nn_ind.append(3)
for d in range (4):
    sfs_3nntrain.append(feature_data[sfs_3nn_ind[d]])
    sfs_3nntest.append(test_set[sfs_3nn_ind[d]])
    sfs3nn_min=1
    for c in sfs_3nn:
        sfs_3nntrain.append(feature_data[c])
        sfs_3nntest.append(test_set[c])
        sfs_3nn_train=(np.asarray(sfs_3nntrain)).T
        sfs_3nn_test=(np.asarray(sfs_3nntest)).T
        sfs_3nn_app_err=fun.NN3_err(sfs_3nn_train,tar,sfs_3nn_test,test_label,1)
        sfs_3nn_cla_err=fun.NN3_err(sfs_3nn_train,tar,sfs_3nn_test,test_label,0)
        if sfs3nn_min>sfs_3nn_app_err:
            sfs3nn_min=sfs_3nn_app_err
            sfs3nn_ind=c
            min3nn_cla_err=sfs_3nn_cla_err
        else:
            sfs3nn_min=sfs3nn_min
            sfs3nn_ind=sfs3nn_ind
        sfs_3nntrain.pop(d+1)
        sfs_3nntest.pop(d+1)
    sfs_3nn.remove(sfs3nn_ind)
    sfs_3nn_app.append(sfs3nn_min)
    sfs_3nn_cla.append(min3nn_cla_err)
    sfs_3nn_ind.append(sfs3nn_ind)   



















