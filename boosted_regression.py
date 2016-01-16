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


#Initial input : no added features, just dummy conversion
data_converted0 = dummy_converter(data)
X0 = data_converted0.drop(['log_PAX','DateOfDeparture'], axis=1).values
X0 = scale(X0)
X0_columns = data_converted0.columns.drop(['log_PAX','DateOfDeparture'])
y0 = data_converted0['log_PAX'].values

#First input : added all geographical features
data_completed1 = airport_geographic_data(data)
data_converted1 = dummy_converter(data_completed1)
X1 = data_converted1.drop(['log_PAX','DateOfDeparture'], axis=1).values
#X1 = scale(X1)
X1_columns = data_converted1.columns.drop(['log_PAX','DateOfDeparture'])
y1 = data_converted1['log_PAX'].values

#second input : added just geographical distance btw airports
data_completed2 = airport_geographic_data(data, include_names=False,keep_only_distance=True,scaling=False)#Added a scaling parameter for scaling the distances added
data_converted2 = dummy_converter(data_completed2)
X2 = data_converted2.drop(['log_PAX','DateOfDeparture'], axis=1).values
X2_columns = data_converted2.columns.drop(['log_PAX','DateOfDeparture'])
y2 = data_converted2['log_PAX'].values

#Third input : added nan-fixed meteorological features
data_completed3 = meteorological_data(data,all_data=True)
data_converted3 = dummy_converter(data_completed3)
X3 = data_converted3.drop(['log_PAX','DateOfDeparture'], axis=1).values
X3_columns = data_converted3.columns.drop(['log_PAX','DateOfDeparture'])
y3 = data_converted3['log_PAX'].values

#Fourth input : added number of aircrafts accidents in the US
data_completed4 = air_accidents_data(data)
data_converted4 = dummy_converter(data_completed4)
X4 = data_converted4.drop(['log_PAX','DateOfDeparture'], axis=1).values
X4_columns = data_converted4.columns.drop(['log_PAX','DateOfDeparture'])
y4 = data_converted4['log_PAX'].values

#Fifth input : added the total number of passenger per airport per year
data_completed5 = airports_traffic(data)
data_converted5 = dummy_converter(data_completed5)
X5 = data_converted5.drop(['log_PAX','DateOfDeparture'], axis=1).values
X5_columns = data_converted5.columns.drop(['log_PAX','DateOfDeparture'])
y5 = data_converted5['log_PAX'].values

#Sixth input : added the total number of passenger per airport per month
data_completed6 = airports_monthly_traffic(data)
data_converted6 = dummy_converter(data_completed6)
X6 = data_converted6.drop(['log_PAX','DateOfDeparture'], axis=1).values
X6_columns = data_converted6.columns.drop(['log_PAX','DateOfDeparture'])
y6 = data_converted6['log_PAX'].values


#%%

params = {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 3,'learning_rate': 0.1, 'loss': 'ls', 'max_features' : "auto"}
reg0 = GradientBoostingRegressor(**params)
reg0.name="GBR with no additional data"
reg1 = GradientBoostingRegressor(**params)
reg1.name="GBR with all geographical data"
reg2 = GradientBoostingRegressor(**params)
reg2.name="GBR with airport distance data"
reg3 = GradientBoostingRegressor(**params)
reg3.name="GBR with meteo data"
reg4 = GradientBoostingRegressor(**params)
reg4.name="GBR with accidents data"
reg5 = GradientBoostingRegressor(**params)
reg5.name="GBR with year flux"
reg6 = GradientBoostingRegressor(**params)
reg6.name="GBR with month flux"

X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size=0.2,random_state=1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2,random_state=1)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2,random_state=1)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2,random_state=1)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2,random_state=1)
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.2,random_state=1)
X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, test_size=0.2,random_state=1)

reg0.fit(X0_train, y0_train)
reg1.fit(X1_train, y1_train)
reg2.fit(X2_train, y2_train)
reg3.fit(X3_train, y3_train)
reg4.fit(X4_train, y4_train)
reg5.fit(X5_train, y5_train)
reg6.fit(X6_train, y6_train)

y0_pred = reg0.predict(X0_test)
y1_pred = reg1.predict(X1_test)
y2_pred = reg2.predict(X2_test)
y3_pred = reg3.predict(X3_test)
y4_pred = reg4.predict(X4_test)
y5_pred = reg5.predict(X5_test)
y6_pred = reg6.predict(X6_test)

print "RMSE :", np.sqrt(mean_squared_error(y6_pred,y6_test))


#%%
# Follow the RMSE at each stage and vizualize coefficients
from visualization import*
fig, axes = plt.subplots(2, 4, figsize=(30, 10))
plot_stages(reg0, X0_train, y0_train, X0_test, y0_test, axes[0,0],title=reg0.name)
plot_stages(reg1, X1_train, y1_train, X1_test, y1_test, axes[0,1],title=reg1.name)
plot_stages(reg2, X2_train, y2_train, X2_test, y2_test, axes[0,2],title=reg2.name)
plot_stages(reg3, X3_train, y3_train, X3_test, y3_test, axes[0,3],title=reg3.name)
plot_stages(reg4, X4_train, y4_train, X4_test, y4_test, axes[1,0],title=reg4.name)
plot_stages(reg5, X5_train, y5_train, X5_test, y5_test, axes[1,1],title=reg5.name)
plot_stages(reg6, X6_train, y6_train, X6_test, y6_test, axes[1,2],title=reg6.name)

#%%
fig, axes = plt.subplots(2, 4, figsize=(30, 10))
plot_coeff_importances(reg0,X0_columns,axes[0,0],title=reg0.name)
plot_coeff_importances(reg1,X1_columns,axes[0,1],title=reg1.name)
plot_coeff_importances(reg2,X2_columns,axes[0,2],title=reg2.name)
plot_coeff_importances(reg3,X3_columns,axes[0,3],title=reg3.name)
plot_coeff_importances(reg4,X4_columns,axes[1,0],title=reg4.name)
plot_coeff_importances(reg5,X5_columns,axes[1,1],title=reg5.name)
plot_coeff_importances(reg6,X6_columns,axes[1,2],title=reg6.name)

#%%
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
plot_coeff_importances(reg4,X4_columns,axes[0])
plot_stages(reg, X4_train, y4_train, X4_test, y4_test, axes[1])

#%%Grid search for bests params
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error,make_scorer

scorer = make_scorer(mean_squared_error,greater_is_better=True)
param_grid = [{
  'n_estimators' : [200], 
  'max_depth': [3,5,7,10], 
  'min_samples_leaf':[1,3,6],
  'learning_rate':[0.1,0.5,1],
  'loss':['ls'],
  'max_features':['auto','sqrt']}]
reg = GradientBoostingRegressor()
best_reg = GridSearchCV(reg, param_grid, scoring = scorer)
best_reg.fit(X0_train,y0_train)
best_reg.predict(X0_test)
print "RMSE :", np.sqrt(best_reg.best_score_)

#%%
best_params={'loss': 'ls', 'learning_rate': 0.1, 'min_samples_leaf': 3, 'n_estimators': 600, 'max_features': 'auto','max_depth': 10}


#%% trained with best params, on complete dataset with additional data
data = pd.read_csv("train.csv")
data = data.drop("Unnamed: 0",axis=1)
data_completed_all = airport_geographic_data(data, include_names=False,keep_only_distance=True,scaling=False)
data_completed_all = airports_traffic(data_completed_all)
data_completed_all = air_accidents_data(data_completed_all)
data_converted_all = dummy_converter(data_completed_all)
X_array = data_converted_all.drop(['log_PAX','DateOfDeparture'], axis=1).values
X_columns = data_converted_all.columns.drop(['log_PAX','DateOfDeparture'])
y_array = data_converted_all['log_PAX'].values

reg = GradientBoostingRegressor(**best_params)
X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.4,random_state=1)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

#%% get a cross-validated score
from sklearn.cross_validation import cross_val_score
score = cross_val_score(estimator=reg,X=X_array,y=y_array,scoring="mean_squared_error",cv=3)
print "RMSE : ", np.sqrt(np.abs(np.mean(score)))

#%%
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
plot_stages(reg, X_train, y_train, X_test, y_test, axes[0])
plot_coeff_importances(reg,X_columns,axes[1])
print "RMSE :", np.sqrt(mean_squared_error(y_pred,y_test))

