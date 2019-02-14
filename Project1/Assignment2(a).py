#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 20:04:49 2017

@author: jianfengsong
"""
import xlrd as xl
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as ns
import math
import random
import xlwt as xlw
rows_value=list()
cols_value=list()
excel=xl.open_workbook('SFE_Dataset.xlsx')
data_table=excel.sheet_by_index(0)
rows=data_table.nrows
cols=data_table.ncols
for a in range(1,rows,1):
    if data_table.row_values(a,cols-1)[0]>=45 or data_table.row_values(a,cols-1)[0]<=35 :
        rows_value.append(data_table.row_values(a))
        

#for a in range(1,cols,1):
#    if data_table.col_values(cols-1):
#        cols_value.append(data_table.col_values(a))
#for a in range(rows):
ele_0=list()
num_0=0
for a in range(0,cols,1):
    for b in range(0,len(rows_value),1):
        if rows_value[b][a]<=0.00000001:
            num_0=num_0+1
    if num_0/len(rows_value)>=0.4:
#        print(len(rows_value))
        num_0=0
        ele_0.append(a)
for row in rows_value:
    num_del=0
    for col in ele_0:
        col=col-num_del
        row.remove(row[col])
        num_del=num_del+1
num_del_row=list()
num_0_row=0
#delete row value that are 0
for row in rows_value:
#    print (row)
    for a in row:
#        print(a)
        if a==0:
            num_0_row=num_0_row+1
    if num_0_row>0:
        num_0_row=0
        num_del_row.append(row)
for a in range(0,len(num_del_row),1):
    num=0
    a=a-num
    rows_value.remove(num_del_row[a])
    num=num+1

#random choose value for test set

numtrain=list()
numtest=list()
test_set=list()
train_set=list()
status=True
go=0
#num=0
while (status):  
    num=0
    go=go+1
    print(go)
    random_set=random.sample(range(len(rows_value)),len(rows_value))  
    for a in range(int(len(rows_value)*0.2)):
        numtrain.append(random_set[a])
    for a in range(len(numtrain)):
        train_set.append(rows_value[numtrain[a]])
    for a in range(len(rows_value)-int(len(rows_value)*0.2)):
        numtest.append(random_set[a+int(len(rows_value)*0.2)])
    for a in range(len(numtest)):
        test_set.append(rows_value[numtest[a]])
    for row in train_set:
        if row[len(row)-1]<=35:
            num=num+1
    if num/len(train_set)>=0.55 or num/len(train_set)<=0.45:
        numtrain=list()
        numtest=list()
        test_set=list()
        train_set=list()
        print("not yet")
#    if num/len(train_set)<0.55 and num/len(train_set)<0.45:
    else:
        status=False
        print("you are good to go")

data_final=xlw.Workbook()
sheet1=data_final.add_sheet('sheet1',cell_overwrite_ok=True)
for i in range(len(train_set)):
    for j in range(len(train_set[i])):
        sheet1.write(i,j,train_set[i][j])
data_final.save('data_final.xls')
test_set1=np.asarray(test_set)
train_set0=np.asarray(train_set)
print(test_set1)
print(train_set0)
