##################################################################################
# Creator     : Gaurav Roy
# Date        : 10 May 2019
# Description : The code contains the approach for Multiple Linear Regression on 
#               the 50_Startups.csv. It contains the 'All-in' model and the
#               Backward Elimination model.
##################################################################################

# Multiple Linear Regression

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

# Encoding X categorical data + HotEncoding
from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Avoid Dummy Variable Trap
X = X[:,1:] #Python Library for Linear Regression automatically takes care of the Trap

#Splitting to Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)

# Fitting MLR to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting Test Results
Y_pred = regressor.predict(X_test)

# Building optimal model using Backward Elimination
import statsmodels.formula.api as sm
X_new = np.append(arr=np.ones((50,1)), values=X, axis=1) #Adding coeff for b0

X_opt = X_new[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

# P value has to be less than Significance Value = 0.05
X_opt = X_new[:,[0,1,3,4,5]] # 2 had P value of 0.99
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X_new[:,[0,3,4,5]] # 1 had P value of 0.94
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X_new[:,[0,3,5]] # 4 had P value of 0.602
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X_new[:,[0,3]] # 5 had P value of 0.060
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

#Splitting to Training and Test Set
X_train_opt, X_test_opt, Y_train_opt, Y_test_opt = train_test_split(X_opt, Y, test_size=0.2, random_state= 0)

# Fitting MLR to training set
regressor_opt = LinearRegression()
regressor_opt.fit(X_train_opt, Y_train_opt)

#Predicting Test Results
Y_pred_opt = regressor_opt.predict(X_test_opt)

Y_diff = Y_pred - Y_test
Y_diff_opt = Y_pred_opt - Y_test_opt