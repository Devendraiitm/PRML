# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:32:17 2019

@author: user
"""
#%%
# part 1(v)
import numpy as np
import matplotlib as plt
#import csv
#from random import shuffle
import math
#from math import exp
fields = [] 
rows = []
train=[]
valid=[]
test=[]
train_class0=[]
train_class1=[]
a=0
b=0
N=0#num of rows
rows = np.genfromtxt ("Dataset_3_Team_3.csv", delimiter=",")
#rows=np.genfromtxt("Dataset_4_Team_3.csv", delimiter=",")
#rows=np.concatenate((rows,rows2))
N=rows.shape
#for row in rows[:10]:
   # for col in row: 
    #   print("%3s"%col,end =" "),
   # print('\n')
for row in rows:
    #print(row[2])
    if row[3]==0:
        a=a+1
    elif row[3]==1:
        b=b+1
p0=round(a/(a+b),2)
p1=round(b/(a+b),2)
#print(p0,p1,p2)
#assuming  normal distr
#shuffle(rows)
#using 70 percent after shuffling
#rows[:int(0.7*(N[0]))]:
for row in rows:
    train.append(row)
    if row[3]==0:
        train_class0.append(row)
    elif row[3]==1:
        train_class1.append(row)
#for row in rows[int(0.7*(N[0]))+1:int(0.85*(N[0]))]:
#    valid.append(row)
#for row in rows[int(0.85*(N[0]))+1:]:
#    test.append(row)
train_class0= np.array(train_class0)
train_class1 = np.array(train_class1)
train=np.array(train)
#valid=np.array(valid)
#test=np.array(test)
mean0=np.mean(train_class0[:,0:3],axis=0)
mean1=np.mean(train_class1[:,0:3],axis=0)
#print(mean0)
cov_0=np.cov(train_class0[:,0:3],bias=True,rowvar=False)
cov_1=np.cov(train_class1[:,0:3],bias=True,rowvar=False)

#%%one feature only
def class_conditional_density1(input_x,mean,variance):
    b=0
    b=variance**0.5
    b=abs((np.log(b))*(-0.5))
    a=0
    a=variance**(-1)
    c=np.transpose(input_x-mean)
    d=c*a
    e=d*(input_x-mean)
    return -0.5*e+b
#%%two feature only
def class_conditional_density2(input_x,mean,covariance):
    b=0
    b=abs(np.linalg.det(covariance))
    b=(np.log(b))*(-0.5)
    a=0
    a=np.linalg.inv(covariance)
    c=np.transpose(input_x-mean)
    d=c.dot(a)
    e=d.dot(input_x-mean)
    return -0.5*e+b


#%%
accuracy_1=0
for row in train:
    #s=0
    q0=class_conditional_density1(row[0],mean0[0],cov_0[0,0])
    q1=class_conditional_density1(row[0],mean1[0],cov_1[0,0])
    #s=q0+q1
    #q0=q0/s
    #q1=q1/s
    #print(q0,q1,q2)
   # print(R)
    index=np.argmax([q0,q1])
   # print(index,row[2])
    if index-row[3]==0:
        accuracy_1+=1

#print(accuracy_5)    
accuracy_1=accuracy_1/train.shape[0]

#%%
accuracy_2=0
for row in train:
    #s=0
    q0=class_conditional_density2(row[0:2],mean0[0:2],cov_0[0:2,0:2])
    q1=class_conditional_density2(row[0:2],mean1[0:2],cov_1[0:2,0:2])
   # s=q0+q1
    #q0=q0/s
    #q1=q1/s
    index=np.argmax([q0,q1])
   # print(index,row[2])
    if index-row[3]==0:
        accuracy_2+=1

#print(accuracy_5)    
accuracy_2=accuracy_2/train.shape[0]
#%%
accuracy_3=0
for row in train:
    #s=0
    q0=class_conditional_density2(row[0:3],mean0,cov_0)
    q1=class_conditional_density2(row[0:3],mean1,cov_1)
    #s=q0+q1
    #q0=q0/s
    #q1=q1/s
    index=np.argmax([q0,q1])
    #print(index,row[3])
    if index-row[3]==0:
        accuracy_3+=1

#print(accuracy_5)    
accuracy_3=accuracy_3/train.shape[0]
#%%

























