# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:23:31 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
#import csv
#from random import shuffle
import math
#from math import exp
#from mpl_toolkits.mplot3d import Axes3D

a=0
b=0
c=0
N=0#num of rows
#rows = np.genfromtxt ("Dataset_1_Team_3.csv", delimiter=",")
rows=np.genfromtxt("Dataset_2_Team_3.csv", delimiter=",")
N=rows.shape
test=[]
for row in rows[int(0.85*(N[0]))+1:]:
    test.append(row)
test=np.array(test)
#rows=np.concatenate((rows,rows2))
#N=rows.shape
#for row in rows[:10]:
   # for col in row: 
    #   print("%3s"%col,end =" "),
   # print('\n')
#for row in rows:
    #print(row[2])
   # if row[2]==0:
   #     a=a+1
    #elif row[2]==1:
    #    b=b+1
    #elif row[2]==2:
        #c=c+1
#p0=round(a/(a+b+c),2)
#p1=round(b/(a+b+c),2)
#p2=round(c/(a+b+c),2)
#print(p0,p1,p2)
#assuming  normal distr
#shuffle(rows)
#using 70 percent after shuffling
#print(cov_0,cov_1,cov_2)
#%%
def class_conditional_density(input_x,mean,covariance):
    # a is the inverse of the covariance matrix
    #print(input_x," ",input_x.shape)
    a=0
    a=np.linalg.inv(covariance)
    # b is the determinant of the covariance matrix
    b=0
    b=np.linalg.det(covariance)
    #print(b)
    b=(b**0.5)
    c=np.transpose(input_x-mean)
    d=c.dot(a)
    e=d.dot(input_x-mean)
    #print(e)
    f= (math.exp(-0.5*e))/(2*(math.pi)*b)
    return f
#%%
    #best model
l=[[0, 1, 2],[1, 0, 1],[2, 1, 0]]
l=np.array(l)


#%%
#train size 100 and 20 replicas
accuracy_net=[]
error_bar=[]
for i in [4000,2000,1000,500,100]:
    accuracy_=0
    accuracy_data=[]
    for j in range(0,20):
        train=[]
        #valid=[]
        #test=[]
        train_class0=[]
        train_class1=[]
        train_class2=[]
        train_index=0
       # shuffle(rows)
        for k in range(0,i):
            train_index=np.random.randint(0,N[0])
            train.append(rows[train_index])
        #shuffle(data)
       # train=data[0:int(0.7*(i)),:]
       # test=data[int(0.85*(i)):i,:]
        for row in train:
            #train.append(row)
            if row[2]==0:
               train_class0.append(row)
            elif row[2]==1:
               train_class1.append(row)
            elif row[2]==2:
               train_class2.append(row)
        #for row in rows[int(0.7*(i))+1:int(0.85*(i))]:
         #   valid.append(row)
        #for row in rows[int(0.85*(i))+1:]:
           # test.append(row)
        train_class0= np.array(train_class0)
        train_class1 = np.array(train_class1)
        train_class2= np.array(train_class2)
        train=np.array(train)
        #valid=np.array(valid)
        mean0=np.mean(train_class0[:,0:2],axis=0)
        mean1=np.mean(train_class1[:,0:2],axis=0)
        mean2=np.mean(train_class2[:,0:2],axis=0)
        #print(mean0)
        cov_0=np.cov(train_class0[:,0:2],rowvar=False,bias=True)
        cov_1=np.cov(train_class1[:,0:2],rowvar=False,bias=True)
        cov_2=np.cov(train_class2[:,0:2],rowvar=False,bias=True)
        accuracy_0=0
        for row in test:
            s=0
            q0=class_conditional_density(row[0:2],mean0,cov_0)
            q1=class_conditional_density(row[0:2],mean1,cov_1)
            q2=class_conditional_density(row[0:2],mean2,cov_2)
            s=(q0+q1+q2)
            q0=q0/s
            q1=q1/s
            q2=q2/s
            #print(q0,q1,q2)
            R=l.dot([q0,q1,q2])
            # print(R)
            index=np.argmin(R)
            # print(index,row[2])
            if index-row[2]==0:
                 accuracy_0+=1    
        accuracy_0=accuracy_0/test.shape[0]
        accuracy_data.append(accuracy_0)
        accuracy_+=accuracy_0
    error_bar.append(100*(np.var(accuracy_data)/20)**0.5)
    accuracy_=accuracy_/20
    accuracy_net.append(100*accuracy_)


#plt.ylim([80,100])
plt.xlabel("number of samples")
plt.ylabel("accuracy")
plt.errorbar([4000,2000,1000,500,100],accuracy_net,yerr=error_bar,fmt='o',ms=5)
#20 samples 85% accuracy




















