# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 00:11:12 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from random import shuffle
from mpl_toolkits.mplot3d import Axes3D
import math
#from math import exp
train=np.loadtxt('train100.txt')
valid=np.loadtxt('val.txt')
test=np.loadtxt('test.txt')
#%%
def f(input_x,mean,variance):
    return math.exp(-((input_x[0]-mean[0])**2+(input_x[1]-mean[1])**2)/(2*variance))
#%%
#K MEANS
error_rms_train=[]
error_rms_valid=[]
error_rms_test=[]
K=10
variance=3
kmeans = KMeans(n_clusters=K, random_state=0).fit(train[:,0:2])
means=kmeans.cluster_centers_
A=np.zeros((train.shape[0],K+1),dtype=float)
for i in range(0,train.shape[0]):
    for j in range(0,K):
        A[i][j]=f(train[i,0:2],means[j],variance)
    A[i][K]=1
res_1=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A),A)),np.transpose(A)),train[:,2])
error=np.dot(A,res_1)-train[:,2] 
#%%
error_rms_train.append(((np.linalg.norm(error-train[:,2]))/train.shape[0])**0.5)
B=np.zeros((valid.shape[0],K+1),dtype=float)
for i in range(0,valid.shape[0]):
    for j in range(0,K):
        B[i][j]=f(valid[i,0:2],means[j],variance)
    B[i][K]=1
error=np.dot(B,res_1)-valid[:,2] 
error_rms_valid.append(((np.linalg.norm(error-valid[:,2]))/valid.shape[0])**0.5)
C=np.zeros((test.shape[0],K+1),dtype=float)
for i in range(0,test.shape[0]):
    for j in range(0,K):
        C[i][j]=f(test[i,0:2],means[j],variance)
    C[i][K]=1
error=np.dot(C,res_1)-test[:,2] 
error_rms_test.append(((np.linalg.norm(error-test[:,2]))/test.shape[0])**0.5)
#%%
#scatter plot
#D=np.zeros((train.shape[0],2))
#D[:,0]=np.dot(A,res_)
#D[:,1]=train[:,2]
#D.sort(0)
#fig1=plt.figure()
#plt.xlim(-80,80)
#plt.ylim(-100,100)
#plt.scatter(D[:,0],D[:,0])
#plt.scatter(D[:,0],D[:,1])
#D=np.zeros((test.shape[0],2))
#D[:,0]=np.dot(C,res_)
#D[:,1]=test[:,2]
#D.sort(0)
#fig2=plt.figure()
#plt.xlim(-80,80)
#plt.ylim(-80,100)
#plt.scatter(D[:,0],D[:,0],label='y=x line')
#plt.scatter(D[:,0],D[:,1],label='')
#plt.legend('best')
t=100
x = np.linspace(-15,15,t)
y = np.linspace(-15,15,t)
X, Y = np.meshgrid(x,y)
Z=np.zeros((t,t))
A_=np.zeros(K+1,dtype=float)
for i in range(t):
    inp=[]
    for j in range(t):
        inp=[X[i,j],Y[i,j]]
        inp=np.transpose(inp)
        for p in range(0,K):
            A_[p]=f(inp,means[p],variance)
        A_[K]=1
        Z[i,j]=np.dot(A_,res_1)
#%%
fig = plt.figure()
ax1 = fig.gca(projection='3d')
ax1.plot_surface(X, Y, Z,cmap='Oranges')
ax1.view_init(5,50)

#%%
#regularisation
res_reg_total=[]
error_rms_train_=[]
error_rms_valid_=[]
error_rms_test_=[]
for lam in [10**-9,0.001,0.01,0.1,1,5]:
    res_reg=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A),A)+2*lam*np.identity(K+1)),np.transpose(A)),train[:,2])
    error=np.dot(A,res_reg)-train[:,2] 
    error_rms_train_.append(((np.linalg.norm(error-train[:,2]))/train.shape[0])**0.5)
    error=np.dot(B,res_reg)-valid[:,2]
    error_rms_valid_.append(((np.linalg.norm(error-valid[:,2]))/valid.shape[0])**0.5)
    error=np.dot(C,res_reg)-test[:,2] 
    error_rms_test_.append(((np.linalg.norm(error-test[:,2]))/test.shape[0])**0.5)
    res_reg_total.append(res_reg)
res_reg_total=np.transpose(np.matrix(res_reg_total))
#%%
#increase data size
train=np.loadtxt('train1000.txt')
A=np.zeros((train.shape[0],K+1),dtype=float)
for i in range(0,train.shape[0]):
    for j in range(0,K):
        A[i][j]=f(train[i,0:2],means[j],variance)
    A[i][K]=1
res_2=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A),A)),np.transpose(A)),train[:,2])
error=np.dot(A,res_2)-train[:,2] 
error_rms_train.append(((np.linalg.norm(error-train[:,2]))/train.shape[0])**0.5)
error=np.dot(B,res_2)-valid[:,2] 
error_rms_valid.append(((np.linalg.norm(error-valid[:,2]))/valid.shape[0])**0.5)
error=np.dot(C,res_2)-test[:,2] 
error_rms_test.append(((np.linalg.norm(error-test[:,2]))/test.shape[0])**0.5)
t=100
x = np.linspace(-15,15,t)
y = np.linspace(-15,15,t)
X, Y = np.meshgrid(x,y)
Z=np.zeros((t,t))
A_=np.zeros(K+1,dtype=float)
for i in range(t):
    inp=[]
    for j in range(t):
        inp=[X[i,j],Y[i,j]]
        inp=np.transpose(inp)
        for p in range(0,K):
            A_[p]=f(inp,means[p],variance)
        A[K]=1
        Z[i,j]=np.dot(A_,res_2)
fig = plt.figure()
ax1 = fig.gca(projection='3d')
ax1.plot_surface(X, Y, Z,cmap='Oranges')
ax1.view_init(5,50)

#%%
train=np.loadtxt('train.txt')
A=np.zeros((train.shape[0],K+1),dtype=float)
for i in range(0,train.shape[0]):
    for j in range(0,K):
        A[i][j]=f(train[i,0:2],means[j],variance)
    A[i][K]=1
res_3=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A),A)),np.transpose(A)),train[:,2])
error=np.dot(A,res_3)-train[:,2] 
error_rms_train.append(((np.linalg.norm(error-train[:,2]))/train.shape[0])**0.5)
error=np.dot(B,res_3)-valid[:,2] 
error_rms_valid.append(((np.linalg.norm(error-valid[:,2]))/valid.shape[0])**0.5)
error=np.dot(C,res_3)-test[:,2] 
error_rms_test.append(((np.linalg.norm(error-test[:,2]))/test.shape[0])**0.5)
t=100
x = np.linspace(-15,15,t)
y = np.linspace(-15,15,t)
X, Y = np.meshgrid(x,y)
Z=np.zeros((t,t))
A_=np.zeros(K+1,dtype=float)
for i in range(t):
    inp=[]
    for j in range(t):
        inp=[X[i,j],Y[i,j]]
        inp=np.transpose(inp)
        for p in range(0,K):
            A_[p]=f(inp,means[p],variance)
        A[K]=1
        Z[i,j]=np.dot(A_,res_3)
fig = plt.figure()
ax1 = fig.gca(projection='3d')
ax1.plot_surface(X, Y, Z,cmap='Oranges')
ax1.view_init(5,50)
#scatter plot
D=np.zeros((train.shape[0],2))
D[:,0]=np.dot(A[:,],res_3)
D[:,1]=train[:,2]
#D.sort(0)
fig1=plt.figure()
plt.xlim(-80,80)
plt.ylim(-100,100)
plt.plot(D[:,1],D[:,0],'ro',label='train')
#plt.scatter(D[:,0],D[:,1])
D=np.zeros((test.shape[0],2))
D[:,0]=np.dot(C,res_3)
D[:,1]=test[:,2]
#D.sort(0)
#fig1=plt.figure()
plt.xlim(-80,80)
plt.ylim(-100,100)
plt.plot(D[:,1],D[:,0],'bo',label='test')
plt.legend(loc='best')







