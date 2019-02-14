#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 22:17:34 2017

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
train1_fea,test1_fea=list(),list()
train2_fea,test2_fea=list(),list()
train3_fea,test3_fea=list(),list()
train4_fea,test4_fea=list(),list()
train5_fea,test5_fea=list(),list()
train_lab,test_lab=list(),list()
mar=list();ps=list();pw=list()
ax =['spheroidite','network','pearlite','spheroidite+widmanstatten']
for a in ax:
    
#        print(b[4].shape)
    if a == 'spheroidite+widmanstatten':
        for d in range(0,60):
            train5_fea.append(data_features[a][d][5])
            train_lab.append(a)
    else:
        for d in range(0,100):
            train5_fea.append(data_features[a][d][5])
            train_lab.append(a)
train5_fea=np.array(train5_fea).reshape(len(train5_fea),512)
lay5_ovo= OneVsOneClassifier(SVC()).fit(train5_fea,train_lab)
for d in range(0,36):
            mar.append(data_features_rest['martensite'][d][5])
mar=np.array(mar).reshape(len(mar),512)
mar_lab=lay5_ovo.predict(mar)
for d in range(0,107):
    ps.append(data_features_rest['pearlite+spheroidite'][d][5])
ps=np.array(ps).reshape(len(ps),512)
ps_lab=lay5_ovo.predict(ps)
for d in range(0,27):
    pw.append(data_features_rest['pearlite+widmanstatten'][d][5])
pw=np.array(pw).reshape(len(pw),512)
pw_lab=lay5_ovo.predict(pw)
