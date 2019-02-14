#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 23:56:05 2017

@author: jianfengsong
"""

#get the size of the table
table_rows = table.nrows
table_cols = table.ncols

#get the predictors[] and classifier[]
predictors = []
headers = table.row_values(0)[0:-1]
classifier = table.col_values(table_cols-1)[1:]
for j in range(0,table_cols-1):
    predictors.append(table.col_values(j)[1:])
    
#delete rows with sfe between 35 and 45
cnt = 1 
i = 0
while True:
    if classifier[i] <= 35:
        classifier[i] = 0
    elif classifier[i] >= 45:
        classifier[i] = 1
    else:
        cnt+=1
        del(classifier[i])
        for predictor in predictors:
            del(predictor[i])
        i-=1
    i+=1
    if i == table_rows-cnt:
        break
       
#delete predictors less than 60% nonezeros
cnt = 0
j = 0
predictors_len = len(predictors)
while True:
    temp = 0
    for elem in predictors[j]:
        if elem == 0:
            temp+=1
        if temp / len(predictors[j]) > 0.4:
            cnt+=1
            del(predictors[j])  
            del(headers[j])
            j-=1
            break
    j+=1
    if j == predictors_len-cnt:
        break

#remove the rows that contains any zero values
cnt = 0
predictor_len = len(predictors[0])
for predictor in predictors:
    i = 0
    while True:
        if predictor[i] == 0:
            cnt+=1
            del(classifier[i])
            for predictor_in in predictors:
                del(predictor_in[i])               
            i-=1
        i+=1
        if i == predictor_len - cnt:
            break

#useful functions
def transpose(X):
    row, col = len(X), len(X[0])
    return [[X[i][j] for i in range(row)] for j in range(col)]



predictors.append(classifier)
All = transpose(predictors)