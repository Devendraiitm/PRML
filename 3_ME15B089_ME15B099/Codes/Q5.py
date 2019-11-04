# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:59:56 2019

@author: DEVENDRA
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:49:36 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import math
from math import exp
y=[]
#data=[]
data=np.load('Data5.npy')
x_data=data[:,0]
train_10=[]
y_10=[]
train=[]
test=[]
valid=[]
#x = np.random.uniform(0,1,100)
#x_data=x
#n = np.random.normal(0,0.2,100)
def f(x):
    return (math.exp(math.cos(2*(math.pi)*x)))
#for i in range(100):
#    y.append(f(x[i])+n[i])
#   data.append([x[i],y[i]])
#y=np.matrix(y)
data=np.array(data)
#shuffle(data)
N=100
train=data[0:70,:]
valid=data[70:85,:]
test=data[85:100,:]
train=np.array(train)
train_10=np.matrix(train[0:10,0])
train_30=np.matrix(train[0:30,0])
train_60=np.matrix(train[0:60,0])
train_70=np.matrix(train[0:70,0])
#train_10.sort()
#train_30.sort()
#train_60.sort()
#train_70.sort()
y_10=np.transpose(np.matrix(train[0:10,1]))
y_30=np.transpose(np.matrix(train[0:30,1]))
y_60=np.transpose(np.matrix(train[0:60,1]))
y_70=np.transpose(np.matrix(train[0:70,1]))
c=np.ones((10,1))
c_valid=np.ones((15,1))
c_full=np.ones((100,1))
c_30=np.ones((30,1))
c_60=np.ones((60,1))
c_70=np.ones((70,1))
x_data.sort()
y_true=[]
for i in x_data:
    y_true.append(f(i))
#%%
# Root mean square error calculation    
def rmse(x,y):
    res=0
    #print((np.transpose(np.subtract(y-x)))
    
    
    res=((np.transpose(y-x)*(y-x))/x.shape[0])[0,0]
    #print(res)
    #res=np.array(res)
    return (res**0.5)
#%%
# generating linear model

A_1=np.hstack((np.transpose(train_10),c))
A_1_valid=np.hstack((np.transpose(np.matrix(valid[:,0])),c_valid))
A_1_test=np.hstack((np.transpose(np.matrix(test[:,0])),c_valid))
res_1=(np.linalg.inv((np.transpose(A_1))*A_1))*(np.transpose(A_1))*y_10
A_1_full=np.hstack((np.transpose(np.matrix(x_data)),c_full))
plt.plot(np.transpose(train_10),y_10,'bo',label='Training Points')
plt.plot(np.transpose(x_data),A_1_full*res_1,label='Predicted output')
plt.plot(x_data,y_true,'r',label='True output')
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Output')
plt.suptitle('Degree :1 ')
plt.show()
# calculating rmse for training
train_error_1=rmse(y_10,A_1*res_1)
valid_error_1=rmse(valid[:,1],A_1_valid*res_1)
test_error_1=rmse(test[:,1],A_1_test*res_1)

#%%
#generating 3rd degree polynomial
A_3=np.hstack((np.transpose(np.matrix(np.power(train_10,3))),np.transpose(np.matrix(np.power(train_10,2))),np.transpose(np.matrix(np.power(train_10,1))),c))
res_3=(np.linalg.inv((np.transpose(A_3))*A_3))*(np.transpose(A_3))*y_10
A_3_full=np.hstack((np.transpose(np.matrix(np.power(x_data,3))),np.transpose(np.matrix(np.power(x_data,2))),np.transpose(np.matrix(np.power(x_data,1))) ,c_full))
A_3_valid=np.hstack((np.transpose(np.matrix(np.power(valid[:,0],3))),np.transpose(np.matrix(np.power(valid[:,0],2))),np.transpose(np.matrix(np.power(valid[:,0],1))) ,c_valid))
A_3_test=np.hstack((np.transpose(np.matrix(np.power(test[:,0],3))),np.transpose(np.matrix(np.power(test[:,0],2))),np.transpose(np.matrix(np.power(test[:,0],1))) ,c_valid))
plt.plot(np.transpose(train_10),y_10,'bo',label='Training Points')
plt.plot(np.transpose(x_data),A_3_full*res_3,label='Predicted output')
plt.plot(x_data,y_true,'r--',label='True output')
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Output')
plt.suptitle('Degree :3 ')
plt.ylim(-1,3.2)
plt.show()
train_error_3=rmse(y_10,A_3*res_3)
valid_error_3=rmse(valid[:,1],A_3_valid*res_3)
test_error_3=rmse(test[:,1],A_3_test*res_3)

#%%
#generate 6th degree polynomial
A_6=np.hstack((np.transpose(np.matrix(np.power(train_10,6))),np.transpose(np.matrix(np.power(train_10,5))),np.transpose(np.matrix(np.power(train_10,4))),np.transpose(np.matrix(np.power(train_10,3))),np.transpose(np.matrix(np.power(train_10,2))),np.transpose(np.matrix(np.power(train_10,1))),c))
res_6=(np.linalg.inv((np.transpose(A_6))*A_6))*(np.transpose(A_6))*y_10
A_6_full=np.hstack((np.transpose(np.matrix(np.power(x_data,6))),np.transpose(np.matrix(np.power(x_data,5))),np.transpose(np.matrix(np.power(x_data,4))),np.transpose(np.matrix(np.power(x_data,3))),np.transpose(np.matrix(np.power(x_data,2))),np.transpose(np.matrix(np.power(x_data,1))),c_full))
A_6_valid=np.hstack((np.transpose(np.matrix(np.power(valid[:,0],6))),np.transpose(np.matrix(np.power(valid[:,0],5))),np.transpose(np.matrix(np.power(valid[:,0],4))),np.transpose(np.matrix(np.power(valid[:,0],3))),np.transpose(np.matrix(np.power(valid[:,0],2))),np.transpose(np.matrix(np.power(valid[:,0],1))),c_valid))
A_6_test=np.hstack((np.transpose(np.matrix(np.power(test[:,0],6))),np.transpose(np.matrix(np.power(test[:,0],5))),np.transpose(np.matrix(np.power(test[:,0],4))),np.transpose(np.matrix(np.power(test[:,0],3))),np.transpose(np.matrix(np.power(test[:,0],2))),np.transpose(np.matrix(np.power(test[:,0],1))),c_valid))
plt.plot(np.transpose(train_10),y_10,'bo',label='Training Points')
plt.plot(np.transpose(x_data),A_6_full*res_6,label='Predicted output')
plt.plot(x_data,y_true,'r--',label='True output')
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Output')
plt.suptitle('Degree :6 ')
plt.ylim(-1,3.2)
plt.show()
train_error_6=rmse(y_10,A_6*res_6)
valid_error_6=rmse(valid[:,1],A_6_valid*res_6)
test_error_6=rmse(test[:,1],A_6_test*res_6)
#%%
#generate 9 th degree polynomial
A_9=np.hstack((np.transpose(np.matrix(np.power(train_10,9))),np.transpose(np.matrix(np.power(train_10,8))),np.transpose(np.matrix(np.power(train_10,7))),np.transpose(np.matrix(np.power(train_10,6))),np.transpose(np.matrix(np.power(train_10,5))),np.transpose(np.matrix(np.power(train_10,4))),np.transpose(np.matrix(np.power(train_10,3))),np.transpose(np.matrix(np.power(train_10,2))),np.transpose(np.matrix(np.power(train_10,1))),c))
res_9=(np.linalg.inv((np.transpose(A_9))*A_9))*(np.transpose(A_9))*y_10
plt.plot(np.transpose(train_10),y_10,'bo',label='Training Points')
A_9_full=np.hstack((np.transpose(np.matrix(np.power(x_data,9))),np.transpose(np.matrix(np.power(x_data,8))),np.transpose(np.matrix(np.power(x_data,7))),np.transpose(np.matrix(np.power(x_data,6))),np.transpose(np.matrix(np.power(x_data,5))),np.transpose(np.matrix(np.power(x_data,4))),np.transpose(np.matrix(np.power(x_data,3))),np.transpose(np.matrix(np.power(x_data,2))),np.transpose(np.matrix(np.power(x_data,1))),c_full))
A_9_valid=np.hstack((np.transpose(np.matrix(np.power(valid[:,0],9))),np.transpose(np.matrix(np.power(valid[:,0],8))),np.transpose(np.matrix(np.power(valid[:,0],7))),np.transpose(np.matrix(np.power(valid[:,0],6))),np.transpose(np.matrix(np.power(valid[:,0],5))),np.transpose(np.matrix(np.power(valid[:,0],4))),np.transpose(np.matrix(np.power(valid[:,0],3))),np.transpose(np.matrix(np.power(valid[:,0],2))),np.transpose(np.matrix(np.power(valid[:,0],1))),c_valid))
A_9_test=np.hstack((np.transpose(np.matrix(np.power(test[:,0],9))),np.transpose(np.matrix(np.power(test[:,0],8))),np.transpose(np.matrix(np.power(test[:,0],7))),np.transpose(np.matrix(np.power(test[:,0],6))),np.transpose(np.matrix(np.power(test[:,0],5))),np.transpose(np.matrix(np.power(test[:,0],4))),np.transpose(np.matrix(np.power(test[:,0],3))),np.transpose(np.matrix(np.power(test[:,0],2))),np.transpose(np.matrix(np.power(test[:,0],1))),c_valid))
plt.plot(np.transpose(x_data),A_9_full*res_9,label='Predicted output')
plt.plot(x_data,y_true,'r--',label='True output')
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Output')
plt.suptitle('Degree :9 ')
plt.ylim(-1,3.2)
plt.show()
train_error_9=rmse(y_10,A_9*res_9)
valid_error_9=rmse(valid[:,1],A_9_valid*res_9)
test_error_9=rmse(test[:,1],A_9_test*res_9)





#%%

# ridge regression
lam=0
i=0
ridge_train_error=np.zeros((6,1))
ridge_valid_error=np.zeros((6,1))
for lam in [10**-8,10**-3,0.01,0.1,1,10]:
    res_9_r=(np.linalg.inv((np.transpose(A_9))*A_9+2*lam*(np.identity(10))))*(np.transpose(A_9))*y_10
    plt.plot(np.transpose(train_10),y_10,'bo',label='Training Points')
    plt.plot(np.transpose(x_data),A_9_full*res_9_r,label='Predicted output')
    plt.plot(x_data,y_true,'r--',label='True output')
    plt.legend(loc='best')
    plt.xlabel('X')
    plt.ylabel('Output')
    plt.suptitle('Degree is 9 and ridge regression parameter is lambda=%0.8f' %lam)
    plt.ylim(-1,3.2)
    plt.show()
    ridge_train_error[i]=rmse(y_10,A_9*res_9_r)
    ridge_valid_error[i]=rmse(valid[:,1],A_9_valid*res_9_r)
    i=i+1
#%%
#plt.plot([1, 2 ,3, 4, 5, 6, 7, 8, 9, 10, 11, 12],ridge_valid_error)
lam=10**-8
A_9=np.hstack((np.transpose(np.matrix(np.power(train_10,9))),np.transpose(np.matrix(np.power(train_10,8))),np.transpose(np.matrix(np.power(train_10,7))),np.transpose(np.matrix(np.power(train_10,6))),np.transpose(np.matrix(np.power(train_10,5))),np.transpose(np.matrix(np.power(train_10,4))),np.transpose(np.matrix(np.power(train_10,3))),np.transpose(np.matrix(np.power(train_10,2))),np.transpose(np.matrix(np.power(train_10,1))),c))
A_9_full=np.hstack((np.transpose(np.matrix(np.power(x_data,9))),np.transpose(np.matrix(np.power(x_data,8))),np.transpose(np.matrix(np.power(x_data,7))),np.transpose(np.matrix(np.power(x_data,6))),np.transpose(np.matrix(np.power(x_data,5))),np.transpose(np.matrix(np.power(x_data,4))),np.transpose(np.matrix(np.power(x_data,3))),np.transpose(np.matrix(np.power(x_data,2))),np.transpose(np.matrix(np.power(x_data,1))),c_full))
A_9_valid=np.hstack((np.transpose(np.matrix(np.power(valid[:,0],9))),np.transpose(np.matrix(np.power(valid[:,0],8))),np.transpose(np.matrix(np.power(valid[:,0],7))),np.transpose(np.matrix(np.power(valid[:,0],6))),np.transpose(np.matrix(np.power(valid[:,0],5))),np.transpose(np.matrix(np.power(valid[:,0],4))),np.transpose(np.matrix(np.power(valid[:,0],3))),np.transpose(np.matrix(np.power(valid[:,0],2))),np.transpose(np.matrix(np.power(valid[:,0],1))),c_valid))
A_9_test=np.hstack((np.transpose(np.matrix(np.power(test[:,0],9))),np.transpose(np.matrix(np.power(test[:,0],8))),np.transpose(np.matrix(np.power(test[:,0],7))),np.transpose(np.matrix(np.power(test[:,0],6))),np.transpose(np.matrix(np.power(test[:,0],5))),np.transpose(np.matrix(np.power(test[:,0],4))),np.transpose(np.matrix(np.power(test[:,0],3))),np.transpose(np.matrix(np.power(test[:,0],2))),np.transpose(np.matrix(np.power(test[:,0],1))),c_valid))
res_9_r=(np.linalg.inv((np.transpose(A_9))*A_9+2*lam*(np.identity(10))))*(np.transpose(A_9))*y_10
#plt.plot(np.transpose(train_10),y_10,'bo',label='Training Points')
#plt.plot(np.transpose(x_data),A_9_full*res_9_r,label='Predicted output')
#plt.plot(x_data,y_true,'r--',label='True output')
#plt.legend(loc='best')
#plt.xlabel('X')
#plt.ylabel('Output')
#plt.suptitle('Best model')
#plt.ylim(-1,3.2)
#plt.show()
#ridge_train_error_best=rmse(y_10,A_9*res_9_r)
#ridge_valid_error_best=rmse(valid[:,1],A_9_valid*res_9_r)
#data10_train_error=rmse(y_10,A_9*res_9_r)
#data10_valid_error=rmse(valid[:,1],A_9_valid*res_9_r)
    
#%%
# varying the data size
# intially the data size was 10.
# here data size used is 30
A_9=np.hstack((np.transpose(np.matrix(np.power(train_30,9))),np.transpose(np.matrix(np.power(train_30,8))),np.transpose(np.matrix(np.power(train_30,7))),np.transpose(np.matrix(np.power(train_30,6))),np.transpose(np.matrix(np.power(train_30,5))),np.transpose(np.matrix(np.power(train_30,4))),np.transpose(np.matrix(np.power(train_30,3))),np.transpose(np.matrix(np.power(train_30,2))),np.transpose(np.matrix(np.power(train_30,1))),c_30))
res_9=(np.linalg.inv((np.transpose(A_9))*A_9))*(np.transpose(A_9))*y_30
plt.plot(np.transpose(train_30),y_30,'bo',label='Training Points')
A_9_full=np.hstack((np.transpose(np.matrix(np.power(x_data,9))),np.transpose(np.matrix(np.power(x_data,8))),np.transpose(np.matrix(np.power(x_data,7))),np.transpose(np.matrix(np.power(x_data,6))),np.transpose(np.matrix(np.power(x_data,5))),np.transpose(np.matrix(np.power(x_data,4))),np.transpose(np.matrix(np.power(x_data,3))),np.transpose(np.matrix(np.power(x_data,2))),np.transpose(np.matrix(np.power(x_data,1))),c_full))
plt.plot(np.transpose(x_data),A_9_full*res_9,label='Predicted output')
plt.plot(x_data,y_true,'r--',label='True output')
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Output')
plt.suptitle('Data size of trainng =30, Degree is 9')
plt.ylim(-1,3.2)
plt.show()
data30_train_error=rmse(y_30,A_9*res_9)
data30_valid_error=rmse(valid[:,1],A_9_valid*res_9)

#%%
A_9=np.hstack((np.transpose(np.matrix(np.power(train_60,9))),np.transpose(np.matrix(np.power(train_60,8))),np.transpose(np.matrix(np.power(train_60,7))),np.transpose(np.matrix(np.power(train_60,6))),np.transpose(np.matrix(np.power(train_60,5))),np.transpose(np.matrix(np.power(train_60,4))),np.transpose(np.matrix(np.power(train_60,3))),np.transpose(np.matrix(np.power(train_60,2))),np.transpose(np.matrix(np.power(train_60,1))),c_60))
res_9=(np.linalg.inv((np.transpose(A_9))*A_9))*(np.transpose(A_9))*y_60
plt.plot(np.transpose(train_60),y_60,'bo',label='Training Points')
A_9_full=np.hstack((np.transpose(np.matrix(np.power(x_data,9))),np.transpose(np.matrix(np.power(x_data,8))),np.transpose(np.matrix(np.power(x_data,7))),np.transpose(np.matrix(np.power(x_data,6))),np.transpose(np.matrix(np.power(x_data,5))),np.transpose(np.matrix(np.power(x_data,4))),np.transpose(np.matrix(np.power(x_data,3))),np.transpose(np.matrix(np.power(x_data,2))),np.transpose(np.matrix(np.power(x_data,1))),c_full))
plt.plot(np.transpose(x_data),A_9_full*res_9,label='Predicted output')
plt.plot(x_data,y_true,'r--',label='True output')
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Output')
plt.suptitle('Data size of training =60, Degree is 9')
plt.ylim(-1,3.2)
plt.show()
data60_train_error=rmse(y_60,A_9*res_9)
data60_valid_error=rmse(valid[:,1],A_9_valid*res_9)
#%%
# dataset 70
A_9=np.hstack((np.transpose(np.matrix(np.power(train_70,9))),np.transpose(np.matrix(np.power(train_70,8))),np.transpose(np.matrix(np.power(train_70,7))),np.transpose(np.matrix(np.power(train_70,6))),np.transpose(np.matrix(np.power(train_70,5))),np.transpose(np.matrix(np.power(train_70,4))),np.transpose(np.matrix(np.power(train_70,3))),np.transpose(np.matrix(np.power(train_70,2))),np.transpose(np.matrix(np.power(train_70,1))),c_70))
res_9=(np.linalg.inv((np.transpose(A_9))*A_9))*(np.transpose(A_9))*y_70
plt.plot(np.transpose(train_70),y_70,'bo',label='Training Points')
A_9_full=np.hstack((np.transpose(np.matrix(np.power(x_data,9))),np.transpose(np.matrix(np.power(x_data,8))),np.transpose(np.matrix(np.power(x_data,7))),np.transpose(np.matrix(np.power(x_data,6))),np.transpose(np.matrix(np.power(x_data,5))),np.transpose(np.matrix(np.power(x_data,4))),np.transpose(np.matrix(np.power(x_data,3))),np.transpose(np.matrix(np.power(x_data,2))),np.transpose(np.matrix(np.power(x_data,1))),c_full))
plt.plot(np.transpose(x_data),A_9_full*res_9,label='Predicted output')
plt.plot(x_data,y_true,'r--',label='True output')
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Output')
plt.suptitle('Data size of training =70, Degree is 9')
plt.ylim(-1,3.2)
plt.show()
data70_train_error=rmse(y_70,A_9*res_9)
data70_valid_error=rmse(valid[:,1],A_9_valid*res_9)

#%%
#plt.plot([1,3,6,9],[train_error_1,train_error_3,train_error_6,train_error_9],'g',[1,3,6,9],[test_error_1,test_error_3,test_error_6,test_error_9],'r')
plt.plot([1,3,6,9],[train_error_1,train_error_3,train_error_6,train_error_9],'g',label='Train error')
plt.plot([1,3,6,9],[valid_error_1,valid_error_3,valid_error_6,valid_error_9],'r',label='Valid error')
plt.legend(loc='best')
plt.xlabel('Degree')
plt.ylabel('Error')
plt.suptitle('Root mean square error')
#%%
fig=plt.figure()
plt.plot([1,3,6,9],[train_error_1,train_error_3,train_error_6,train_error_9],'g',label='Train error')
plt.plot([1,3,6,9],[test_error_1,test_error_3,test_error_6,test_error_9],'r',label='Test error')
plt.legend(loc='best')
plt.xlabel('Degree')
plt.ylabel('Error')
plt.ylim(0,10)
plt.suptitle('Root mean square error')
#%%
fig=plt.figure()
A_9_=np.hstack((np.transpose(np.matrix(np.power(train_70,9))),np.transpose(np.matrix(np.power(train_70,8))),np.transpose(np.matrix(np.power(train_70,7))),np.transpose(np.matrix(np.power(train_70,6))),np.transpose(np.matrix(np.power(train_70,5))),np.transpose(np.matrix(np.power(train_70,4))),np.transpose(np.matrix(np.power(train_70,3))),np.transpose(np.matrix(np.power(train_70,2))),np.transpose(np.matrix(np.power(train_70,1))),c_70))
plt.scatter(test[:,1],np.array(A_9_test*res_9),label='Test data')
plt.scatter(np.array(y_70),np.array(A_9_*res_9),label='Train data')
plt.legend(loc='best')
plt.ylim(-0.5,4)
plt.xlim(-0.5,3.5)
plt.xlabel('Target output')
plt.ylabel('Model output')
plt.suptitle('Target and model output')
