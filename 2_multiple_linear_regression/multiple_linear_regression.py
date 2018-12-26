#Multiple linear regression


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORTING DATA
dataset = pd.read_csv("50_startups.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#Creating dummy variables for categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencode_x=LabelEncoder()
x[:,3]=labelencode_x.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

#Removing dummy variable trap
x=x[:,1:]

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Performing Linear Regression 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the test set
y_pred=regressor.predict(x_test) 

#Backword Elimanation
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype('int'),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
#Adjusting
x_opt=x[:,[0,1,2,4]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,2,4]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,2]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()