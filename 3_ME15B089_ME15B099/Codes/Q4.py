# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:20:34 2019

@author: user
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import matplotlib.mlab as mlab
import math
from math import exp
rows=[]
ratio_var=0#ratio of variances
rows = np.genfromtxt ("Dataset_5_Team_3.csv", delimiter=",")
shuffle(rows)
#var=np.var(rows)
mean=0
for row in rows:
    mean=mean+row
mean=mean/rows.shape[0]
var=0
for row in rows:
    var=var+(row-mean)**2
var=var/rows.shape[0]
prior_mean=-1
#%%
def pdf_calc(x,mean,variance):
    return ((1/(2*math.pi*variance)**0.5)*math.exp(-0.5*(x-mean)**2/variance))
#%%
y_pdf=np.zeros((100,1))
for n in [10,100,1000]:
    #n=1000
    ML_mean=0
    ratio_var=0
    posterior_mean=0
    posterior_var=0
    fig=plt.figure()
    for row in rows[0:n]:
        ML_mean+=row
    ML_mean_act=ML_mean/n
    for ratio_var in [0.1,1,10,100]:
        posterior_mean=(n/(n+ratio_var))*(ML_mean/n)+(ratio_var/(n+ratio_var))*(prior_mean)
        posterior_var=(var/(n+ratio_var))
        x = np.linspace(posterior_mean - 3*math.sqrt(posterior_var), posterior_mean + 3*math.sqrt(posterior_var), 100)
        #x=np.linspace(-2,5,1000)
        for i in range(100):
            y_pdf[i]=pdf_calc(x[i],posterior_mean,posterior_var)
        #plt.plot(x,mlab.normpdf(x,posterior_mean,math.sqrt(posterior_var)))
        
        plt.plot(x,y_pdf,label=ratio_var)
        plt.legend(loc='best')
        plt.xlim(-2,2)
        plt.ylim(0,8)
        plt.xlabel('X')
        plt.ylabel('pdf')
        plt.suptitle('PDF for N=1000')
    plt.show()
        
        #plt.show()
#%%






















