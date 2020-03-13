#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 21:33:52 2020

@author: K Rithikreddy
"""

import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from itertools import combinations
import random
''' load_iris is for the classification and load_boston is for regression'''

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

X=np.array(X).reshape(x.shape)

Y=np.array(Y).reshape(y.shape)


s=set(y)

l=list(s)


'''list(combinations(l,2)) has to be mentioned otherwise it returns an object type while printing'''


c=list(combinations(l,2))
'''
a= [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1]
for n, i in enumerate(a):
    if i == 1:
        a[n] = 10
replaces all 1's wit 10's
'''

m=len(c)

for i in range(m):
    a=c.pop(i)
    c.append(list(a))

for i in range(m):
    c[i]=list(c[i])


'''we can use either one of the methods as prescribed above'''

'''There are 4 functions in removing an element from the list 
1)pop     deletes the item based on specific index and returns the item
2)remove   delets the item from the list by value
3)clear    deletes all the elements
4)del       deletes the specified item based on index  '''


X_train,X_test,Y_train,Y_test=train_test_split(x,y)

def train_classifier(X_train,Y_train,a,b):
    new_x=[]
    new_y=[]
    for j in range(len(X_train)):
        if((Y_train[j]==a) or (Y_train[j]==b)):
            new_x.append(X_train[j,:])
            if(Y_train[j]==a):
                new_y.append(0)
            else:
                new_y.append(1)
    clf=svm.LinearSVC()
    clf.fit(new_x,new_y)
    return clf.coef_


bin_svm=[]

for i in c:
    a=i[0]
    b=i[1]
    bin_svm.append(train_classifier(X_train,Y_train,a,b))

print(bin_svm)
         
for i in bin_svm:
    print(i)

def predict_class(X_test,bin_svm,c):
    classes=[]
    for i in range(X_test.shape[0]):
        count=np.zeros((len(l),1))
        count=list(count)
        for j in range(len(bin_svm)):
            if((np.dot(X_test[i,:].reshape(1,4),np.array(bin_svm[j]).reshape(-1,1))<=0)):
                count[c[j][0]]+=1
            else:
                count[c[j][1]]+=1
        classes.append(count.index(max(count)))
    #print(classes)
    return classes


classes=[]

classes=predict_class(X_test,bin_svm,c)

print(classes)

print(Y_test)

print(accuracy_score(Y_test,classes))

print(confusion_matrix(Y_test,classes))


































































