# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 00:48:35 2019

@author: dbda12
"""

import pandas as pd
import numpy as np
df = pd.read_csv("C:/Users/dbda12/Downloads/cars.csv")
dum_df = pd.get_dummies(df.iloc[:,1:], drop_first=True)

from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dum_df)
carscaled=scaler.transform(dum_df)

df_cars = pd.DataFrame(carscaled)

X = df_cars.iloc[:,1:]
y = df_cars.iloc[:,0]



# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2018)

from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor(random_state=5000)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

#def mean_absolute_percentage_error(y_true, y_pred): 
#    y_true, y_pred = np.array(y_true), np.array(y_pred)
#    return np.mean(np.abs((y_true - y_pred) / y_true))
#
#print(mean_absolute_percentage_error(y_test,y_pred))

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

