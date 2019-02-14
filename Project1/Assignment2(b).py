import xlrd as xl
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as ns
import math
import random
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
#        print("not yet")
#    if num/len(train_set)<0.55 and num/len(train_set)<0.45:
    else:
        status=False
#        print("you are good to go")
#save excel doc
data_final=xlw.Workbook()
sheet1=data_final.add_sheet('sheet1',cell_overwrite_ok=True)
for i in range(len(train_set)):
    for j in range(len(train_set[i])):
        sheet1.write(i,j,train_set[i][j])
data_final.save('data_final.xls')

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
#train_35=list()
#train_45=list() 
#for a in range(len(train35[1])):
#    for b in range(len(train35)):
#          train_35.append(train35[b][a])     
#for a in range(len(train45[1])):
#    for b in range(len(train45)):
#          train_45.append(train45[b][a])       
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
Tset_name={'C':Tset[0],'Ni':Tset[1],'Fe':Tset[2],'Mn':Tset[3],'Cr':Tset[4]}
Tset_name0=np.asarray(Tset_name)
print(Tset_name0)
import operator
sorted_tset = sorted(Tset_sta_name.items(), key=operator.itemgetter(1),reverse=True)        
#print(sorted_tset)     
        
        
        
        
        
        
        
