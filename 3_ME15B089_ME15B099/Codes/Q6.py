# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:02:59 2019

@author: user
#%%"""

import numpy as np
import matplotlib.pyplot as plt
#from random import shuffle
import math
#from math import exp
x = np.random.uniform(-1,1,10000)
n = np.random.normal(0,0.2,10000)
y=[]
c=[]
def f(x):
    return ((math.exp(math.tanh(2*(math.pi)*x)))-x)
for i in range(0,10000):
    y.append(f(x[i])+n[i])
y=np.matrix(y)
y=np.transpose(y)
x=np.matrix(x)
c=np.ones((10000,1))
x=np.transpose(x)
#%%
## generating linear model
#A=np.hstack((x,c))
#res_1=(np.linalg.inv((np.transpose(A))*A))*(np.transpose(A))*y
#plt.plot(A[0:100,:]*res_1,x[0:100],'r--',y[0:100],x[0:100],'bs')
#plt.show()
##%%
##generate fifth degree polynomial
#A_5=np.hstack((np.matrix(np.power(x,5)),np.matrix(np.power(x,4)),np.matrix(np.power(x,3)),np.matrix(np.power(x,2)),np.matrix(np.power(x,1)),c))
#res_2=(np.linalg.inv((np.transpose(A_5))*A_5))*(np.transpose(A_5))*y
#plt.plot(A_5[0:100,:]*res_2,x[0:100],'rs',y[0:100],x[0:100],'bs')
#plt.show()

#%%


#A_9=np.hstack((np.matrix(np.power(x,9)),np.matrix(np.power(x,8)),np.matrix(np.power(x,7)),np.matrix(np.power(x,6)),np.matrix(np.power(x,5)),np.matrix(np.power(x,4)),np.matrix(np.power(x,3)),np.matrix(np.power(x,2)),np.matrix(np.power(x,1)),c))
#res_3=(np.linalg.inv((np.transpose(A_9))*A_9))*(np.transpose(A_9))*y
#plt.plot(A_9[0:100,:]*res_3,x[0:100],'rs',y[0:100],x[0:100],'bs')
#plt.show()

#%%
#ridge regression lambda 0.001
#lam=0.001
err1= np.zeros((9,1000))
m=0
#plt.plot(A[0:10,:]*res_1_r,x[0:10],'r--',y[0:10],x[0:10],'bs')
#plt.show()
for degree in [1,5,9]:
    for lam in [0.001,0.01,0.1]:
        err=[]#stores avg error or risk
        for i in range(0,1000):
            error=0
            if(degree==1):
                A=np.hstack((x,c))
            elif(degree==5):
                A=np.hstack((np.matrix(np.power(x,5)),np.matrix(np.power(x,4)),np.matrix(np.power(x,3)),np.matrix(np.power(x,2)),np.matrix(np.power(x,1)),c))
            elif(degree==9):
                A=np.hstack((np.matrix(np.power(x,9)),np.matrix(np.power(x,8)),np.matrix(np.power(x,7)),np.matrix(np.power(x,6)),np.matrix(np.power(x,5)),np.matrix(np.power(x,4)),np.matrix(np.power(x,3)),np.matrix(np.power(x,2)),np.matrix(np.power(x,1)),c))
            res_r=(np.linalg.inv((np.transpose(A[10*i:(i+1)*10,:]))*A[10*i:(i+1)*10,:]+2*lam*(np.identity(degree+1))))*(np.transpose(A[10*i:10*(i+1),:]))*y[10*i:10*(i+1),:]
            for j in range(0,10):
                error+=(y[10*i+j]-(A[10*i:10*(i+1),:]*res_r)[j])
            err.append(error/10)
            #err=np.array(err)
        err1[m]=err
        m+=1
#%%
lin=1000
y_axis=np.zeros((9,lin))
m=0
for i in range(0,9):
    x_axis=np.linspace(-2*min(err1[i]),2*max(err1[i]),lin)
    j=0
    for m in range(0,lin):
        while(x_axis[j]<=err1[i][m]):
            j+=1
        y_axis[i][j]+=1
        j=0
        m+=1
    #plt.figure()
    plt.bar(np.transpose(x_axis),height=y_axis[i],width=2*(max(err1[i])+min(err1[i]))/lin)
    plt.show()
    
    
    
    
    



















