# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:37:38 2020

@author: Prasanth
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df=pd.read_csv('F:\ASS\INSURENCE')

df.columns

x=df.copy()
del x['expenses (Target)']

y=df['expenses (Target)']


x.isnull().sum()

plt.boxplot(x['age'])

plt.boxplot(x['bmi'])
plt.boxplot(x['children'])

per=x['bmi'].quantile([0,0.97]).values
x['bmi']=x['bmi'].clip(per[0],per[1])

from sklearn.preprocessing import LabelEncoder

lbe=LabelEncoder()


x['sex']=lbe.fit_transform(x['sex'])
x['smoker']=lbe.fit_transform(x['smoker'])
x['region']=lbe.fit_transform(x['region'])

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(xtrain)

xtrain=scaler.transform(xtrain)
xtest=scaler.transform(xtest)


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(xtrain,ytrain)

ypred_ts=reg.predict(xtest)
ypred_tr=reg.predict(xtrain)

print(reg.coef_)
reg.score

import statsmodels.api as sm
x_train_sm=sm.add_constant(xtrain)
xtest_sm=sm.add_constant(xtest)

from sklearn.metrics import r2_score
model=sm.OLS(ytrain,x_train_sm).fit()
predsm=model.predict(xtest_sm)
r2s=r2_score(ytest,predsm)
print(r2s)
print(model.summary())












'''
from sklearn.preprocessing import OneHotEncoder
oc=OneHotEncoder(categorical_features=[['sex'],['smoker'],['region']])
ohe=OneHotEncoder()
ohc=ohe.fit_transform(x)
ohc=pd.DataFrame(ohc)


xtrainoc,xtestoc,ytrainoc,ytestoc=train_test_split(ohc,y,test_size=0.25,random_state=0)
xtrainocs=sm.add_constant(xtrainoc)
xtestocs=sm.add_constant(xtestoc)

model1=sm.OLS(ytrain,xtrainocs).fit()

'''



















































