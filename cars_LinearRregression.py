# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:59:07 2019

@author: DBDA
"""

import pandas as pd
import numpy as np

cars=pd.read_csv('F:\RS\Machine Learning\Datasets\cars.csv')
#carsdum=pd.get_dummies(cars,drop_first=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(cars)
carscaled=scaler.transform(carsdum)

df_cars = pd.DataFrame(carscaled)

X = df_cars.iloc[:,1:]
y = df_cars.iloc[:,0]


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,  random_state=2018)

# Create the classifier: logreg
logreg = LinearRegression()
# Fit the classifier to the training data
logreg.fit(X_train,y_train)
# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)
# Compute and print the confusion matrix and classification report
print( np.sqrt( mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


##################################################################################
from xgboost import XGBRegressor
clf = XGBRegressor()
clf.fit(X_train,y_train,verbose=True)

y_pred = clf.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))







