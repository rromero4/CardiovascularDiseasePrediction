#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt 

in_data = loadmat('CardioDataUpdatedFile.mat')
# 11 features age, height, weight, gender, systolic blood pressure, diastolic blood pressure, cholesterol, glucose level, smoking, alcohol intake, and physical activity

x = in_data['X']

A = x[:,1:12] # Matrix A with all the features
A[:,1] = A[:,1] / 365 # Change age from days to years. 
d = x[:,12] # Target variable d


# In[3]:


n = 70000
x_train = np.array(list(range(0,n)))
p = 11
Y = d[x_train]

## Train NN
Xb = np.hstack((np.ones((n,1)), A[x_train]))
q = 1 #number of classification problems

## initial weights 
W = np.ones(p+1);
alpha = 0.1 #step size
L = 5 #number of epochs

def logsig(_x):
    if _x > 100000:
        return 1
    if _x < -100000:
        return 0
    return 1/(1+np.exp(-_x)) 

count = 0
for epoch in range(L):
    ind = np.random.permutation(10)
    for i in ind:
        # Forward-propagate
        if count < 1:
            Yhat = logsig(Xb[[i],:]@W) 
        else:
            Yhat = logsig(Xb[[i],:]@W[0]) 
             
#          # Backpropagate
        delta = (Yhat-Y[i])*Yhat*(1-Yhat)
        Wnew = W - (alpha*Xb[[i],:].T*delta).T
        W = Wnew
        count = 2
    print(epoch)


# In[4]:


error_list = []
for i in range(70000):
    Yhat = logsig(Xb[[i],:]@W[0])
    
    if np.sign(Yhat) != Y[i]:
        error_list.append(1)
    else:
        error_list.append(0)
        
print(np.mean(error_list))
print('Number of errors', np.sum(error_list))


# In[6]:


print('The function logsig is not a good function for approximating the data. It is possible to use this function but we must be careful with the intital W we set. Everytime I set W to 0, 1, and random. Setting w to a zero always predicted Yhat to be 1. Since (1 - Yhat) is in the delta function, this always gave a delta value of zero. When W was a vector of ones, this gave a Yhat value of 1. Since (1 - Yhat) is in the delta function, this also gave the delta function a value of 0. Using the random values for w tended to give Yhat values of 1 or 0. Due to this the weights were never updated. The weight can not be updated no matter how many tries you have.')


# In[7]:


W = np.ones(p+1);
alpha = 0.15 #step size
L = 50 #number of epochs
lambdas = 7

count = 0
for epoch in range(L):
    ind = np.random.permutation(10)
    for i in ind:
        # Forward-propagate

        if count < 1:
            Yhat = logsig(Xb[[i],:]@W) 
        else:
            Yhat = logsig(Xb[[i],:]@W[0]) 
#          # Backpropagate
        
        delta = (Yhat-Y[i])*Yhat*(1-Yhat) 
            
        Wnew = W - (alpha*Xb[[i],:]*delta) - 2*alpha*lambdas*W
        W = Wnew
        count = 2
        
    print(epoch)


# In[9]:


error_list = []
for i in range(70000):
    Yhat = logsig(Xb[[i],:]@W[0])
    
    if np.sign(Yhat) != Y[i]:
        error_list.append(1)
    else:
        error_list.append(0)

print(np.mean(error_list))
print('Number of errors', np.sum(error_list))


# In[10]:


A = A[:,0:8]
p = 8

Xb = np.hstack((np.ones((n,1)), A[x_train]))

W = np.ones(p+1);
alpha = 0.15 #step size
L = 50 #number of epochs
lambdas = 7

def logsig(_x):
    if _x > 100000:
        return 1
    if _x < -100000:
        return 0
    return 1/(1+np.exp(-_x)) 

count = 0
for epoch in range(L):
    ind = np.random.permutation(10)
    for i in ind:
        # Forward-propagate

        if count < 1:
            Yhat = logsig(Xb[[i],:]@W) 
        else:
            Yhat = logsig(Xb[[i],:]@W[0]) 
#          # Backpropagate
        
        delta = (Yhat-Y[i])*Yhat*(1-Yhat) 
            
        Wnew = W - (alpha*Xb[[i],:]*delta) - 2*alpha*lambdas*W
        W = Wnew
        count = 2
        
    print(epoch)


# In[12]:


error_list = []
for i in range(63000):
    Yhat = logsig(Xb[[i],:]@W[0])
    
    if np.sign(Yhat) != Y[i]:
        error_list.append(1)
    else:
        error_list.append(0)

print(np.mean(error_list))
print('Number of errors', np.sum(error_list))
print('About 3500 less errors, about the same error rate')

