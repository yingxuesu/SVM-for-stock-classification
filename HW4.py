#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 21:23:51 2021

@author: yingxue
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
ST1=pd.read_csv('/home/yingxue/Desktop/ELEK.csv')
ST2=pd.read_csv('/home/yingxue/Desktop/QCOM.csv')
ST3=pd.read_csv('/home/yingxue/Desktop/CSV.csv')
ST4=pd.read_csv('/home/yingxue/Desktop/BIDU.csv')
ST5=pd.read_csv('/home/yingxue/Desktop/BABA.csv')
ST6=pd.read_csv('/home/yingxue/Desktop/TSLA.csv')
ST7=pd.read_csv('/home/yingxue/Desktop/SONY.csv')
ST8=pd.read_csv('/home/yingxue/Desktop/AAPL.csv')
ST9=pd.read_csv('/home/yingxue/Desktop/AUY.csv')
ST10=pd.read_csv('/home/yingxue/Desktop/JPM.csv')
ST11=pd.read_csv('/home/yingxue/Desktop/BAC.csv')
ST12=pd.read_csv('/home/yingxue/Desktop/JNJ.csv')
ST13=pd.read_csv('/home/yingxue/Desktop/Y.csv')
ST14=pd.read_csv('/home/yingxue/Desktop/AMC.csv')
ST15=pd.read_csv('/home/yingxue/Desktop/BA.csv')
ST16=pd.read_csv('/home/yingxue/Desktop/FB.csv')
ST17=pd.read_csv('/home/yingxue/Desktop/INTC.csv')
ST18=pd.read_csv('/home/yingxue/Desktop/MSFT.csv')
ST19=pd.read_csv('/home/yingxue/Desktop/GOOG.csv')
ST20=pd.read_csv('/home/yingxue/Desktop/EA.csv')

STlist=[ST1,ST2,ST3,ST4,ST5,ST6,ST7,ST8,ST9,ST10,ST11,ST12,ST13,ST14,ST15,
        ST16,ST17,ST18,ST19,ST20]
Y =np.empty(shape=(20,1005))
J=0
for ST in STlist:
    S=ST['Close'].to_numpy()
    for i in range(len(S)-1):
        Y[J,i]=(S[i+1]-S[i])/S[i]
    J=J+1
            
X=np.empty(shape=(996,200))
YY=np.empty(shape=(996,1))
for j in range(996):
    x=np.array([])
    for i in range(20):
        x=np.append(x,Y[i,j:j+10])
    X[j]=x
    if Y[19,j+9]>0.006:
        YY[j]=1
    else:
        YY[j]=0


### Compute the training set and test set after PCA analysis
from sklearn.model_selection import train_test_split
###split training and test set on CLi
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,YY,test_size=0.15)


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train, Y_train = sm.fit_resample(Xtrain, Ytrain)
X_test, Y_test = sm.fit_resample(Xtest, Ytest)
    
from sklearn import svm

testacc=[]
trainacc=[]
ratio=[]
supp=[]
for c in [0.01,0.1,1,10,100,500,1000,5000,8000]:
    lsvc = svm.SVC(C=c,kernel='linear')
    lsvc.fit(X_train, Y_train)
    SV=lsvc.support_vectors_
    supp.append(SV.shape[1]/1112)
    score = lsvc.score(X_train, Y_train)
    score1 = lsvc.score(X_test, Y_test)
    testacc.append(score1)
    trainacc.append(score)
    ratio.append(score1/score)
    
plt.plot(testacc)
plt.plot(trainacc)    
plt.plot(ratio)
plt.plot(supp)    



            
        
        
        
    
    

















