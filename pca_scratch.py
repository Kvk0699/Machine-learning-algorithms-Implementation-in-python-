#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:30:36 2020

@author: rithik
"""

''' PCA Algorithm from scratch without using inbuit fucntions'''

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from itertools import combinations

''' For 3d visualisation'''
import seaborn as sns

''' For computing eigen values'''

import scipy.linalg as la        

data=load_breast_cancer()

''' Here The data contains 30 Feautures and 569 examples'''

x=data['data']
y=data['target']

'''Numpy array'''

print(x.shape)

#print(x[0])

def pca(x,k):
    
    '''Step 1 in algorithm
    1) Compute the means along the rows and subtract the mean values from each feauture of each row '''
    
    p=x-x.mean(axis=1).reshape(-1,1)
    
    ''' Step 2 
    2) compute the covariance matrix'''
    
    cov=np.matmul(p.T,p)
    
    print(cov)
    
    
    '''569*569 shape'''
    
    print(cov.shape)
    
    ''' Step 3 computing the eigen values '''
    
    '''It's only la.eig not la.eigen'''
    
    print(type(la.eig(cov)))
    
    #print(la.eig(cov))
    
    eigen_values,eigen_vectors=la.eig(cov)
    
    l=[]
    
    '''Sorting the eigen values by creating a tuple with eigenvalue and eigenvector '''
    
    for i in range(len(eigen_values)):
        l.append((eigen_values[i],eigen_vectors[i]))
    
    l.sort(reverse=True)
    
    print("l is \n",l)
    
    l=np.array(l)
    
    l=l[:k]
    
    c=l[0][1]
    
    #print(l)
    
    print(c)
    
    ''' Step 4 
    4) appending the eigen vectors row wise '''
    
    c=np.array(c)
    
    ''' Always remember tuple has to be passed inside ((don't forget me!!!!)) '''
    
    for i in range(1,len(l)):
        c=np.vstack((c,l[i][1]))
    
    ''' step 5
    5)Returning the feature matrix which is product of c,p.T '''
    print((np.matmul(c,p.T)).T)
    
    return (np.matmul(c,p.T).T)
    
    #print(c.shape)
    #print(c)
    
    
    #print(l)
    
    #eigen_values.reshape(-1,1)
    
    #print(eigen_values.shape)
    #print(eigen_values)
    
    #print(eigen_vectors.shape)
    #print(eigen_vectors)
    
    
k=int(input("Enter the value of k:"))

print(k)

p=pca(x,k)

'''Now visualising the data'''

print(p.shape)
'''
plt.scatter(p[:, 0], p[:, 1],
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Reds', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')

plt.show()
'''

'''plt.scatter(x[:, 0],x[:,1],alpha=0.2)
plt.scatter(p[:, 0], p[:, 1],alpha=0.8)
plt.axis('equal');
'''
plt.figure(figsize=(6,6))
#plt.scatter(x[:0],x[:1],alpha=0.2)
plt.scatter(p[:,0],p[:,1],y,cmap='gray')
plt.axis([-125,-25,0,70])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.show()
















































