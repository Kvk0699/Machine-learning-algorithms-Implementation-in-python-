

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

d=pd.read_csv('data.csv')
type(d)
a=np.array(d)
print(type(a))
print(a.shape)
type(a.shape)              #x.shape is not a function and it returns a tuple
m=a.shape[0]
print(m)
n=a.shape[1]
print(n)                   # or we can do like m,n=x.shape

