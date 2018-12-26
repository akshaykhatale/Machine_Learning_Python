#Simple Linear Regression

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORTING DATA
dataset = pd.read_csv("fire_and_theft_in_chicago.csv")
x=dataset.iloc[:,0].values
y=dataset.iloc[:,1].values

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)

#fitting simple linear regressionto training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)

#predict the result set 
y_pred = regressor.predict(x_test)

#visualizing the training set
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Fire vs theft in chicago(Training set)")
plt.xlabel("Fires(per 1000 housing units)")
plt.ylabel("Thefts(per 1000 population)")
plt.show()

#visualizing the test set
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Fire vs theft in chicago(train set)")
plt.xlabel("Fires(per 1000 housing units)")
plt.ylabel("Thefts(per 1000 population)")
plt.show()
