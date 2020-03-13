#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 00:27:56 2020

@author: K Rithikreddy
"""

'''SVM OAO '''

''' We will use binary svm classifier as basic and classifies based on OAA'''

import numpy as np
from sklearn import datasets
from sklearn import svm
#import matplotlib.pyplot as plt not required mostly
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

'''datasets package contains many datasets which is used mainly for practice purposes'''

a=datasets.load_iris()
x=a.data
y=a.target

mydict={}
for i in range(np.size(x,0)):
    mydict[i]=(x[i],y[i])



'''print(mydict)'''
''' We can use indexing in tuples'''



random.shuffle(mydict)

X=[]
Y=[]
'''np.vstack((X,mydict[i][0])) doesn't works if the X is empty so first we add some elements'''

'''
X.extend(mydict[0][0]) 
adds each and every element of mydict to X unlike the append which adds only 1 element'''

for i in mydict:
    X.append(mydict[i][0])
    Y.append(mydict[i][1])

#print(X)
'''
if(i!=0):
    #np.vstack((mydict[i][0],X))
    X.append(mydict[i][0])
else:
    X.extend(mydict[i][0])'''


s=set(Y)


X=np.array(X).reshape(x.shape)

Y=np.array(Y).reshape(y.shape)

x=X
y=Y

print("Randomisation completed")

X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=0)

clf=svm.LinearSVC()
clf.fit(X_train,Y_train)

print(X_test.shape)

print(X_train.shape)

a=clf.coef_

print(a)


def fitting_classifier(X_train,Y_train,k):
    l=[]
    for i in Y_train:
        if i==k:
            l.append(1)
        else:
            l.append(0)
    clf=svm.LinearSVC()
    clf.fit(X_train,l)
    return(clf.coef_)       
'''returning classifier for each class'''





a=np.array([1,2,3,-1]).reshape(-1,1)
b=np.array([0,1,1,0]).reshape(-1,1)
d=np.array([-1,2,4.4]).reshape(-1,1)
clf.fit(a,b)

f=np.array([[-1,0,1],[2,3,4],[5,6,7]]).reshape(3,3)
b=np.array([0,1,1]).reshape(-1,1)

clf.fit(f,b)
print(clf.coef_)

'''It is a 3*1(dimen*1) matrix this time and for each element in list it shows a w predicted value'''
'''
print(clf.coef_)
x=clf.predict(d)
print(x)   '''

'''
It is generally used in this way
'''

c=[]

for i in s:
    c.append(fitting_classifier(X_train,Y_train,i))

''' Set object is not subscriptable'''
'''
for j in range(len(s)):
    print(s[j])
so Don't use this
'''


print("I got the classifiers for all the types of classes")
print(c)

'''
print(X_test.shape)
for i in range(0,X_test.shape[0]):
    print(np.dot(c[0],X_test[i,:].reshape(4,1)))
'''

def predict_class(X_test,c,s):
    q=0
    e=[]
    f=0
    for i in range(X_test.shape[0]):
        f=0
        q=0
        for j in range(0,len(c)):
            p=np.dot(c[j],(X_test[i,:].reshape(4,1)))
            print(p)
            if(p>q):
                q=p
                f=j
        e.append(f)
    return e
e=[]

e=predict_class(X_test,c,s)

print(e)

print("Accuracy: ",accuracy_score(Y_test,e))

print("Confusion matrix is \n",confusion_matrix(Y_test,e))





































































