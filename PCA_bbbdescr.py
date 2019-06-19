# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:43:26 2019

@author: dbda12
"""

import pandas as pd
import numpy as np
df = pd.read_csv("C:/Users/dbda12/Downloads/bbbDescr.csv")
df.head()

## Mean Imputing
from sklearn.preprocessing import Imputer
imp = Imputer(strategy='mean')
dfImputed = imp.fit_transform(df)

df_Imputed = pd.DataFrame(dfImputed,columns= df.columns)

X = df_Imputed.iloc[:,1:]
y = df_Imputed.iloc[:,0]

## Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
Xscaled=scaler.transform(X)
Xscaled[1:5]

from sklearn.decomposition import PCA
pca = PCA(svd_solver = 'auto')
principalComponents = pca.fit_transform(Xscaled)

# Cumulative Sum
Cumu_Variation =np.cumsum(pca.explained_variance_ratio_ * 100)

import matplotlib.pyplot as plt
plt.plot(list(range(0,133)),Cumu_Variation)
plt.xlabel("PCs")
plt.ylabel("Scores")
plt.title('Scree Plot')
plt.show()

X = pd.DataFrame(principalComponents)

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor

X = X.iloc[:,0:10]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2018)

model_rf = RandomForestRegressor(random_state=1211)
model_rf.fit( X_train , y_train )
y_pred = model_rf.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
