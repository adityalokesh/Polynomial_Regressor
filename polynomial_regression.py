#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:00:57 2019

@author: chrx
"""
#COMPARISON OF LINEAR REGRESSION AND POLYNOMIAL REGRESSION !!!!!


#importing nessecary models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('/home/chrx/Downloads/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv')


#split dependent and independent variable
X=data.iloc[:, 1:2].values
Y=data.iloc[:, 2:3].values



#creating a linear model
from sklearn.linear_model import LinearRegression
lin_obj_1=LinearRegression()
lin_obj_1.fit(X,Y)
Y_pred=lin_obj_1.predict(X)



#adding polynomial features to the linear model
from sklearn.preprocessing import PolynomialFeatures



# degree is  a parameter which indicats how many terms of the dependent variable is included
poly_obj=PolynomialFeatures(8)
X_poly=poly_obj.fit_transform(X)
lin_obj_2=LinearRegression()
lin_obj_2.fit(X_poly,Y)




#visualization of  linear model results
plt.scatter(X,Y,color='red')
plt.plot(X,Y_pred,color='blue')
plt.title('Regression-LinearModel')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()



#visualization of polynomial model results
plt.scatter(X,Y,color='red')
plt.plot(X,lin_obj_2.predict(poly_obj.fit_transform(X)),color='blue')
plt.title('Regression-PolynomialModel')
plt.xlabel('Level')
plt.ylabel('Salary')



#predicting salary for a single entitiy in linear model
a=np.array([6.5])
a=a.reshape(1,-1)
res=lin_obj_1.predict(a)



#ploynomial regression prediction
res_poly=lin_obj_2.predict(poly_obj.fit_transform(a))

