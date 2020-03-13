#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 02:02:44 2020

@author: K Rithikreddy
"""

#from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import itertools
'''
import itertools we can write like this also
Here we have to use combinations_with_replacement function as even x^2 are also part of polynomial regression'''

'''datasets package contains many datasets which is used mainly for practice purposes'''
 
a=datasets.load_boston()
x=a.data
y=a.target
print(type(a))
#print(x)
#print(y)
print(x.shape)
y=y.reshape((y.shape[0],1))
print(y.shape)

'''
#x=np.append(np.ones((x.shape[0],1)),x,axis=1)
#x=np.hstack(np.ones((x.shape[0],1))) '''


x=np.hstack((np.ones((x.shape[0],1)),x))

'''#Be careful with the syntax we have to pass a tuple as argument'''

print(x.shape)
'''
#p=[1,2,3]
#b=[]
#l=[]
#for j in range(x.shape[0]):
#    for i in p:
#        l.append(x[j,i])
#    b.append(np.product(l))
#    l.clear()
#print(np.array(b).shape)
#print(b) '''


l=[]
for i in range(x.shape[1]):
    l.append(i)
print(l)
2

b=[]

'''q=2 is better because other values may take more time in normal systems'''

q=int(input("Enter the max degree upto which u want to estimate the polynomial regression\n"))

for c in range(1,q+1):
    b.append([list(x) for x in itertools.combinations_with_replacement(l,c)])
print(b)

l.clear()
c=[]
m=x.shape[0]
z=[]
print(m)

print(x.shape)

for i in b:
    for j in range(m):
        for k in i:
            l.append(x[j,k])
        c.append(np.product(l))
        l.clear()
    z=np.array(c).reshape(x.shape[0],1)
    x=np.hstack((x,z))
    c.clear()
print(x.shape)


print("Now our X is modified as according to the variables")

print("Now same as normal regression")


def cost(x,theta,y):
    hx=np.dot(x,theta)
    return (1/(2*np.size(x,0)))*(np.sum(np.square(hx-y)))


def gradient_descent(x,theta,y,alpha,num_iter):
    l=[]
    for i in range(num_iter):
        theta=theta-((alpha/np.size(x,0))*(x.T@((x@theta)-y)))
        l.append(cost(x,theta,y))
    return theta,l

def normal(x):
    meanx=np.mean(x,0)
    
    '''#Takes mean along the y axis for every coloumn'''
    
    stdx=np.std(x,0)            
    
    '''#standard deveation of y=0 line'''
    
    for i in range(np.size(x,1)):
        for j in range(np.size(x,0)):
            if(stdx[i]!=0):
                x[j][i]=(x[j][i]-meanx[i])/(stdx[i])
    return x

x=normal(x)
print(x)

theta = np.random.normal(0,1,x.shape[1])
theta=theta.reshape(-1,1)
print(theta.shape)

print(cost(x,theta,y))

num_iter=100
alpha=0.01
l=[]

z=np.arange(1,num_iter+1,1)
print(z)


theta,l=gradient_descent(x,theta,y,alpha,num_iter)
print(theta)

print(l)
plt.xlabel("No of iterations")
plt.ylabel("Cost after the ith iteration")
plt.title("Cost vs iterations Graph")
plt.plot(z,l)






































































