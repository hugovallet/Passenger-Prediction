# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:23:32 2016

@author: Hugo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.cross_validation import train_test_split

from preprocessing import *
from visualization import *

#%%
data = pd.read_csv("./public_train.csv")

#%% Try multiple input data

#!!! Without sclaing we are facing issues when adding new features


#First input : no added features, just dummy conversion
data_converted0 = dummy_converter(data)
X0 = data_converted0.drop(['log_PAX','DateOfDeparture'], axis=1).values
X0_columns = data_converted0.columns.drop(['log_PAX','DateOfDeparture'])
y0 = data_converted0['log_PAX'].values

#second input : added all geographical features
data_completed1 = airport_geographic_data(data)
data_converted1 = dummy_converter(data_completed1)
X1 = data_converted1.drop(['log_PAX','DateOfDeparture'], axis=1).values
X1_columns = data_converted1.columns.drop(['log_PAX','DateOfDeparture'])
y1 = data_converted1['log_PAX'].values

#second input : added all geographical features
data_completed2 = airport_geographic_data(data, include_names=False,keep_only_distance=True,scaling=False)#Added a scaling parameter for scaling the distances added
data_converted2 = dummy_converter(data_completed2)
X2 = data_converted2.drop(['log_PAX','DateOfDeparture'], axis=1).values
X2_columns = data_converted2.columns.drop(['log_PAX','DateOfDeparture'])
y2 = data_converted2['log_PAX'].values

#%%

params = {'n_estimators': 150, 'max_depth': 5, 'min_samples_split': 5,'learning_rate': 1, 'loss': 'ls', 'max_features' : "auto"}
reg0 = GradientBoostingRegressor(**params)
reg1 = GradientBoostingRegressor(**params)
reg2 = GradientBoostingRegressor(**params)

X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size=0.2,random_state=1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2,random_state=1)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2,random_state=1)


reg0.fit(X0_train, y0_train)
reg1.fit(X1_train, y1_train)
reg2.fit(X2_train, y2_train)

y0_pred = reg0.predict(X0_test)
y1_pred = reg1.predict(X1_test)
y2_pred = reg2.predict(X2_test)




#%%
# Follow the RMSE at each stage and vizualize coefficients
plot_stages(reg0, X0_train, y0_train, X0_test, y0_test)
plot_coeff_importances(reg0,X0_columns)

plot_stages(reg1, X1_train, y1_train, X1_test, y1_test)
plot_coeff_importances(reg1,X1_columns)

plot_stages(reg2, X2_train, y2_train, X2_test, y2_test)
plot_coeff_importances(reg2,X2_columns)
