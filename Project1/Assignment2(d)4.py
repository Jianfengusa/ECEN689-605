#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 21:37:49 2017

@author: jianfengsong
"""
import xlrd as xl
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as ns
import math
import random
import operator
rows_value=list()
cols_value=list()
excel=xl.open_workbook('SFE_Dataset.xlsx')
data_table=excel.sheet_by_index(0)
rows=data_table.nrows
cols=data_table.ncols
for a in range(1,rows,1):
    if data_table.row_values(a,cols-1)[0]>=45 or data_table.row_values(a,cols-1)[0]<=35 :
        rows_value.append(data_table.row_values(a))
ele_0=list()
num_0=0
for a in range(0,cols,1):
    for b in range(0,len(rows_value),1):
        if rows_value[b][a]<=0.00000001:
            num_0=num_0+1
    if num_0/len(rows_value)>=0.4:
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
for row in rows_value:
    for a in row:
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
while (status):  
    num=0
    go=go+1
#    print(go)
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
    else:
        status=False
#save excel doc
#data_final=xlw.Workbook()
#sheet1=data_final.add_sheet('sheet1',cell_overwrite_ok=True)
#for i in range(len(train_set)):
#    for j in range(len(train_set[i])):
#        sheet1.write(i,j,train_set[i][j])
#data_final.save('data_final.xls')

#Assignment2(b)
train_set35=list()
train_set45=list()
length_train=len(train_set)
for a in range(0,length_train,1):
    if train_set[a][len(train_set[a])-1]<=35:
        train_set35.append(a)
    else:
        train_set45.append(a)
train35=list()
train45=list()
for a in train_set35:
    train35.append(train_set[a])
for a in train_set45:
    train45.append(train_set[a])       
train_35=[[] for i in range(len(train35[1]))]
train_45=[[] for i in range(len(train45[1]))]
Tset=list()
for a in range(len(train35[1])):
    for b in range(len(train35)):
          train_35[a].append(train35[b][a])     
for a in range(len(train45[1])):
    for b in range(len(train45)):
          train_45[a].append(train45[b][a])            
for a in range(len(train_35)-1):
    h=ns.ttest_ind(train_35[a],train_45[a],equal_var=False)
    Tset.append(h)    
Tset_sta=list()
for a in range(len(Tset)):
    Tset_sta.append(abs(Tset[a][0]))      
#print  (Tset_sta)     
Tset_sta_name={'C':Tset_sta[0],'Ni':Tset_sta[1],'Fe':Tset_sta[2],'Mn':Tset_sta[3],'Cr':Tset_sta[4]}       
sorted_tset = sorted(Tset_sta_name.items(), key=operator.itemgetter(1),reverse=True)        
print(sorted_tset)

#Assignment2(c)
top=list()
#for a in range(len(sorted_tset)):
#    top.append(sorted_tset[a][1])
#x1=[[] for i in range(len(train35))]
#x2=[[] for i in range(len(train45))]
x11=list()
x21=list()
#for a in range(len(x1)):
#for a in range(len(train35)):
first=Tset_sta.index(sorted_tset[0][1])
second=Tset_sta.index(sorted_tset[1][1])
for b in range(len(train35)):
    x11.append([train35[b][1],train35[b][2]])
for b in range(len(train45)):
    x21.append([train45[b][1],train45[b][2]]) 
###different
#for b in range(len(train35)):
#    x11.append([train35[b][first],train35[b][second]])
#for b in range(len(train45)):
#    x21.append([train45[b][first],train45[b][second]])     
sumx1=0
sumx2=[0,0]
x1=np.asarray(x11)
x2=np.asarray(x21)
for b in x1:
    sumx1=b+sumx1
mean_x1=sumx1/len(x1)
for c in x2:
    sumx2=c+sumx2
mean_x0=sumx2/len(x2)
cov=(np.dot((x1-mean_x0).T,(x1-mean_x0))+np.dot((x2-mean_x1).T,(x2-mean_x1)))/(min(len(train35),len(train45)-2))
an=np.dot(cov**-1,(mean_x1-mean_x0))
bn=(-1/2)*np.dot(np.dot((mean_x1-mean_x0).T,cov**(-1)),(mean_x1-mean_x0))
plt.figure(1)     
plt.plot(x1[:,0],x1[:,1],'x')
plt.plot(x2[:,0],x2[:,1],'o')
x1=np.arange(-10,100,1)
x2p=-bn/an[1]-an[0]*x1/an[1]
plt.plot(x1,x2p.T,'r')      
     
#test set
test_set35=list()
test_set45=list()
length_test=len(test_set)
for a in range(0,length_test,1):
    if test_set[a][len(test_set[a])-1]<=35:
        test_set35.append(a)
    else:
        test_set45.append(a)
test35=list()
test45=list()
for a in test_set35:
    test35.append(test_set[a])
for a in test_set45:
    test45.append(test_set[a])       
test_35=[[] for i in range(len(test35[1]))]
test_45=[[] for i in range(len(test45[1]))]
#Tset=list()
x112=list()
x212=list()
for a in range(len(test35[1])):
    for b in range(len(test35)):
          test_35[a].append(test35[b][a])     
for a in range(len(test45[1])):
    for b in range(len(test45)):
          test_45[a].append(test45[b][a])
for b in range(len(test35)):
    x112.append([test35[b][1],test35[b][2],test35[b][3],test35[b][0]])
for b in range(len(test45)):
    x212.append([test45[b][1],test45[b][2],test45[b][3],test45[b][0]])
sumx12=0
sumx22=0
x12=np.asarray(x112)
x22=np.asarray(x212)
for b in x12:
    sumx12=b+sumx12
mean_x12=sumx12/len(x12)
for c in x22:
    sumx22=c+sumx22
mean_x02=sumx22/len(x22)
cov2=(np.dot((x12-mean_x02).T,(x12-mean_x02))+np.dot((x22-mean_x12).T,(x22-mean_x12)))/(min(len(test35),len(test45)-2))
an2=np.dot(cov2**-1,(mean_x12-mean_x02))
bn2=(-1/2)*np.dot(np.dot((mean_x12-mean_x02).T,cov2**(-1)),(mean_x12-mean_x02))
var_x0=(np.dot(an2,mean_x02.T)+bn2)/math.sqrt(np.dot(np.dot(an2,cov2),an2.T))
var_x1=(np.dot(an2,mean_x12.T)+bn2)/math.sqrt(np.dot(np.dot(an2,cov2),an2.T))
LDA_error=1/2*(ns.norm.cdf(var_x0)+ns.norm.cdf(-var_x1))
clas_x12_y=-bn2/an2[1]-an2[0]*x12[:,0]/an[1]
clas_x22_y=-bn/an[1]-an[0]*x22[:,0]/an[1]
error_time=0
error_set=list()
for t in range(min(len(test45),len(test35))):
    if x22[t,1] > clas_x22_y[t]:
        error_time=error_time+1
    if x22[t,1] < clas_x22_y[t]:
        error_time=error_time+1
error_set.append(error_time/(len(test_set)*len(test_set[1])))
print(error_set)
#print(LDA_error)
#LDA_error_set.append(LDA_error)


















 
        
