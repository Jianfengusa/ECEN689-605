#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 20:31:51 2017

@author: jianfengsong
"""
import xlrd as xl
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.figure_factory as ff
import pandas as pd
import plotly
import operator
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from itertools import combinations
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier

class fun():
    def excel_data(n):
        data=list()
        excel=xl.open_workbook(n)
        data_table=excel.sheet_by_index(0)
        rows=data_table.nrows
        cols=data_table.ncols
        for a in range(rows):
            data.append(data_table.row_values(a))
        return data
################################################################################
    def traindata():
        data=fun.excel_data('micrograph.xlsm')
        label=['spheroidite','network','pearlite','spheroidite+widmanstatten']
        data_train=list()
#        data_sample=list()
        index=list()
        index.append(0)
        for a in label:
            times=0
            if a is 'spheroidite+widmanstatten':
                for b in range(1,len(data)):
                    if data[b][9] == a:
                        if times < 60:
                            data_train.append(data[b])
                            index.append(b)
                            times+=1
            else:
                 for b in range(1,len(data)):
                     if data[b][9]== a:
                        if times <100:
                            data_train.append(data[b])
                            index.append(b)
                            times+=1  
        data_sample=np.delete(data,index,axis=0)
        return np.asarray(data_train),np.asarray(data_sample)      
################################################################################
    def isfloat(value):
        try:
            float(value)
            return float(value)
        except:
            return value
################################################################################
    def data():
        datatrain,datasam=fun.traindata()
        index=[0,2,4,7,8]
        temp=list()
        temptrain_list=list()
        tempsam_list=list()
        for a in range(len(datatrain.T)):
            if a in index:
                for b in range(len(datatrain.T[a])):
                    temp.append(fun.isfloat(datatrain.T[a][b]))
                temptrain_list.append(temp)
                temp=list()
            else:
                temptrain_list.append(datatrain.T[a])
        for a in range(len(datasam.T)):
            if a in index:
                for b in range(len(datasam.T[a])):
                    temp.append(fun.isfloat(datasam.T[a][b]))
                tempsam_list.append(temp)
                temp=list()
            else:
                tempsam_list.append(datasam.T[a])
        return tempsam_list,temptrain_list      
################################################################################
    def vg16(path):
        img_path = 'Micrograph/' + path
        img = image.load_img(img_path)
        img=img.crop((0,0,645,484))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x
################################################################################
    def avg(matrix):
        div=matrix.shape[1]*matrix.shape[2]
        matrix=np.sum(matrix,axis=1)
        matrix=np.sum(matrix,axis=1)
        matrix=np.divide(matrix,div)
        matrix.shape=(matrix.shape[1],1)
        return (matrix)
################################################################################
    def diffea(train1,label,a):
        sph,net,pea,sphw=[],[],[],[]
        sphl,netl,peal,sphwl=[],[],[],[]
        total,total_lab=[],[]
        svm,svmlab=[],[]
        for x in range (0,99):
            sph.append(train1[x])
            sphl.append(label[x])
        for x in range (100,199):
            net.append(train1[x])
            netl.append(label[x])
        for x in range (200,299):
            pea.append(train1[x])
            peal.append(label[x])
        for x in range (300,359):
            sphw.append(train1[x])
            sphwl.append(label[x])
        total.append(sph);total.append(net);total.append(pea);total.append(sphw);
        total_lab.append(sph1);total_lab.append(netl);total_lab.append(peal);total_lab.append(sphwl);
        for b in total[a[0]]:
            svm.append(b)
        for b in total[a[1]]:
            svm.append(b)
        for b in total_lab[a[0]]:
            svmlab.append(b)
        for b in total_lab[a[1]]:
            svmlab.append(b)
        return svm,svmlab
################################################################################
datatest,datatrain=fun.data()
model=VGG16(include_top=False,weights='imagenet',input_tensor=None,input_shape=(484,645,3),pooling=None)
model_1= Model(inputs=model.input, outputs=model.get_layer('block1_pool').output)
model_2= Model(inputs=model.input, outputs=model.get_layer('block2_pool').output)
model_3= Model(inputs=model.input, outputs=model.get_layer('block3_pool').output)
model_4= Model(inputs=model.input, outputs=model.get_layer('block4_pool').output)
train1_fea,test1_fea=list(),list()
train2_fea,test2_fea=list(),list()
train3_fea,test3_fea=list(),list()
train4_fea,test4_fea=list(),list()
train5_fea,test5_fea=list(),list()
train_lab,test_lab=list(),list()
layer=[1,2,3,4,5]
for a in range(len(datatrain[1])):
    vgg_16=fun.vg16(datatrain[1][a])
    train1_fea.append(model_1.predict(vgg_16))
    train2_fea.append(model_2.predict(vgg_16))
    train3_fea.append(model_3.predict(vgg_16))
    train4_fea.append(model_4.predict(vgg_16))
    train5_fea.append(model.predict(vgg_16))
    train_lab.append(datatrain[9][a])
for a in range(len(datatest[1])):
    vgg_16_test=fun.vg16(datatest[1][a])
    test1_fea.append(model_1.predict(vgg_16_test))
    test2_fea.append(model_2.predict(vgg_16_test))   
    test3_fea.append(model_3.predict(vgg_16_test))  
    test4_fea.append(model_4.predict(vgg_16_test))
    test5_fea.append(model.predict(vgg_16_test))
    test_lab.append(datatest[9][a])
for a in range (len(train1_fea)):
    train1_fea[a]=fun.avg(train1_fea[a])
    train2_fea[a]=fun.avg(train2_fea[a])
    train3_fea[a]=fun.avg(train3_fea[a])
    train4_fea[a]=fun.avg(train4_fea[a])
    train5_fea[a]=fun.avg(train5_fea[a])
for a in range (len(test1_fea)):
    test1_fea[a]=fun.avg(test1_fea[a]) 
    test2_fea[a]=fun.avg(test2_fea[a])
    test3_fea[a]=fun.avg(test3_fea[a])
    test4_fea[a]=fun.avg(test4_fea[a])
    test5_fea[a]=fun.avg(test5_fea[a])
c42label=np.asarray(list(combinations(range(4),2)))
train,test=list(),list()
train.append((train1_fea,train2_fea,train3_fea,train3_fea,train4_fea,train5_fea))
test.append(test1_fea,test2_fea,test3_fea,test4_fea,test5_fea)
score=[]
for a in train[0]:
    for b in c42label:
        svm,svmlab=fun.diffea(a,train_lab,b)
        svm=np.array(svm).reshape(len(svm),len(svm[0]))
        clf=NuSVC()
        scores=np.mean(cross_val_score(clf,svm,svmlab,cv=10))
        score.append(1-scores)
train5_fea=np.array(train5_fea).reshape(len(train5_fea),len(train5_fea[0]))
lay5_ovo= OneVsOneClassifier(NuSVC()).fit(train5_fea,train_lab)  





#
#
#
#
#









