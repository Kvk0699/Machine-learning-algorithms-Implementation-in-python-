#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 01:26:10 2020

@author: K Rithikreddy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def logistic(x):
    return (1/(np.exp(-x)+1)).astype(np.float64)


def cost(x,theta,y):
    hx=logistic(x@theta)
    return ((-1/np.size(x,0))*(y.T@np.log(hx)+(1-y).T@np.log(1-hx)))


def gradient_descent(x,theta,y,alpha):
        theta=theta-((alpha/np.size(x,0))*(x.T@(logistic(x@theta)-y)))
        return theta

def normal(x):
    meanx=np.mean(x,0)
    stdx=np.std(x,0)
    for i in range(np.size(x,1)):
        for j in range(np.size(x,0)):
            if(stdx[i]!=0):
                x[j][i]=(x[j][i]-meanx[i])/(stdx[i])
    return x


def reg(x,theta,y,alpha,num_iter):
    p=[]
    k=[]
    for i in range(num_iter):
        theta=gradient_descent(x,theta,y,alpha)
        theta=theta.astype(np.float64)
        k.append(i)
        p.append(cost(x,theta,y).reshape(-1,1))
    
    return theta,p,k


def change_y(y):
    for i in range(0,y.shape[0]):
        if(a[i,1]=='M'):
            y[i]=1
        if(a[i,1]=='B'):
            y[i]=0
    return y

def predict_values(x,theta):
    y=[]
    hx=logistic(x@theta)
    for i in hx:
        if(i>0.5):
            y.append(1)
        else:
            y.append(0)
    return y

d=pd.read_csv('data.csv')
a=np.array(d)
x=a[:,2:-1]
y=a[:,1]
x=np.append(np.ones((x.shape[0],1)),x,1)

x=x.astype(np.float64)

x=normal(x)

print(x.shape)

x_train=x[:int(0.8*x.shape[0]),:]
x_test=x[int(0.8*x.shape[0]):,:]
y_train=x[:int(0.8*x.shape[0]),1]
y_test=x[int(0.8*x.shape[0]):,1]

print(x_train.shape)
print(x_test.shape)


x_train=x_train.astype(np.float64)

theta=np.zeros((x_train.shape[1],1))

y_train=y_train.reshape(-1,1)

print(y_train.shape)
print(theta.shape)

alpha=0.01
num_iter=1000

x_train=x_train.astype(np.float64)
y_train=y_train.astype(np.float64)

y_train=change_y(y_train)
y_test=change_y(y_test)

p=[]
k=[]

theta,p,k=reg(x_train,theta,y_train,alpha,num_iter)

plt.title("Logistic Regression plot for the training data")

plt.ylabel("cost of the Logistic Regression")

plt.xlabel("Number of Iterartions")

plt.plot(np.array(k).reshape(num_iter,1),np.array(p).reshape(num_iter,1))

l=predict_values(x_test,theta)

#plt.plot(np.array(l),y_test)






















































