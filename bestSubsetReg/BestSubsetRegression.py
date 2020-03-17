#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import linalg

def solve_sym(xtx, xty):
    L = linalg.cholesky(xtx)
    return linalg.lapack.dpotrs(L, xty)[0]

def turnbits_rec(p):
    # turnbits_rec函数用于将[0,2^p-1]中的整数转化为(2^p)*p维逻辑矩阵
    # 输出(2^p)*p维逻辑矩阵
    if(p==1):
        return np.array([[True, False],[True, True]])
    else:
        tmp1 = np.c_[turnbits_rec(p-1), np.array([False]*(2**(p-1))).reshape((2**(p-1),1))]
        tmp2 = np.c_[turnbits_rec(p-1), np.array([True]*(2**(p-1))).reshape((2**(p-1),1))]
        return np.r_[tmp1, tmp2]

class BestSubsetReg(object):
    
    def __init__(self, x, y, names=None, intercept=True, isCp=True, isAIC=True, isCV=True):
        self.n, self.p=x.shape
        
        if intercept:
            self.x = np.c_[(np.ones((self.n,1)),x)]
            self.ind_var = turnbits_rec(self.p)
        else:
            self.x = x
            self.ind_var = turnbits_rec(self.p)[1:][:,1:]
        
        self.isCp=isCp
        self.isAIC=isAIC
        self.isCV=isCV
        self.y=y
        self.xtx = np.dot(self.x.T, self.x)
        self.xty = np.dot(self.x.T, self.y)
        self.shape = [self.xtx.shape, self.xty.shape]
        self.b = []
        self.names = names
        self.intercept = intercept
        
    def reg(self):
        # get regression parameters
        self.b = [solve_sym(self.xtx[ind][:,ind], self.xty[ind]) 
                  for ind in self.ind_var]

    def Cp_AIC(self):
        RSS = np.dot(self.y, self.y) - [np.sum(np.dot(self.xtx[ind][:, ind], b_) * b_) for ind, b_ in zip(self.ind_var, self.b)]
        
        d = np.sum(self.ind_var, axis=1)
        
        if(self.isCp):
            self.Cp = RSS/self.n+2*d*RSS[-1]/self.n/(self.n-self.p-1)
        if(self.isAIC):
            self.AIC = self.n*np.log(RSS)+2*d

    def KfoldCV(self):
        K = 10
        # get the indexs controlling which samples used as train or test
        indexs = np.array_split(np.random.permutation(np.arange(0,self.n)), K)

        def cv(ind, index):
            txx = self.xtx[ind][:,ind]-np.dot((self.x[index][:,ind].T), 
                                              self.x[index][:,ind])
            txy = self.xty[ind]-np.dot((self.x[index][:,ind]).T,
                                       self.y[index])
            tcoe = solve_sym(txx, txy)
            return np.sum((self.y[index]-np.dot(self.x[index][:,ind,], tcoe))**2)
        
        self.CV = np.sum(np.array([[cv(ind, index) for ind in self.ind_var] for index in indexs]), axis=1)/self.n
    
    def output(self):
        def getVars(min_id):
            if self.intercept:
                self.xnames = self.names[:-1][self.ind_var[min_id][1:]]
                return(dict(zip(np.append(["intercept"],self.xnames), self.b[min_id])))
            else:
                self.xnames = self.names[:-1][self.ind_var[min_id]]
                return(dict(zip(self.xnames, self.b[min_id])))
            
        if(self.isCp | self.isAIC):
            self.Cp_AIC()
            if(self.isCp):
                min_id = np.argmin(self.Cp)
                print('Based on Cp criterion:\n',getVars(min_id),'\n')
            if(self.isAIC):
                min_id = np.argmin(self.AIC)
                print('Based on AIC criterion:\n',getVars(min_id),'\n')
        
        if(self.isCV):
            self.KfoldCV()
            min_id = np.argmin(self.CV)
            print('Based on K-fold CV:\n', getVars(min_id),'\n')
            
            


# In[2]:


import os
import sys
os.chdir("F:/dataMining/bestSubsetRegression/")
x = np.loadtxt("./prostate/x.txt", delimiter=",")  # loadtxt读入时默认按浮点数读入
y = np.loadtxt("./prostate/y.txt", delimiter=",")
names = np.loadtxt("./prostate/names.txt", delimiter=",", dtype=str)


# In[3]:


# test
reg1 = BestSubsetReg(x, y, names, intercept=False)
reg1.reg()
reg1.output()


# In[ ]:




