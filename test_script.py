# -*- coding: utf-8 -*-
"""
Created on Sat Jan 09 15:26:21 2016

@author: Hugo
"""
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import ShuffleSplit

from sklearn.ensemble import RandomForestRegressor


#%%
data = pd.read_csv("./public_train.csv")

X_df = data.drop(['log_PAX'], axis=1)
y_array = data['log_PAX'].values

skf = ShuffleSplit(y_array.shape[0], n_iter=2, test_size=0.2)
skf_is = list(skf)[0]
X_encoded = X_df
#%%training

path = '.'  # use this in notebook
data_weather = pd.read_csv("external_data.csv")
#X_weather = data_weather[['Date', 'AirPort', 'Max TemperatureC']]
X_weather=data_weather.drop('Events',axis=1)
X_weather=X_weather.drop('Precipitationmm',axis=1)
X_weather=X_weather.drop('Max Gust SpeedKm/h',axis=1)

X_weather = X_weather.rename(columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})
X_encoded = X_encoded.set_index(['DateOfDeparture', 'Arrival'])
X_weather = X_weather.set_index(['DateOfDeparture', 'Arrival'])
X_encoded = X_encoded.join(X_weather).reset_index()

#%%
X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
X_encoded = X_encoded.drop('Departure', axis=1)
X_encoded = X_encoded.drop('Arrival', axis=1)

X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)

X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))

#%%
X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
X_array = X_encoded.values


# Regression
train_is, _ = skf_is
X_train_array = np.array([X_array[i] for i in train_is])
y_train_array = np.array([y_array[i] for i in train_is])


#%%


reg = RandomForestRegressor(n_estimators = 50,max_depth =90, max_features = 100)
reg.fit(X_train_array, y_train_array)


_, test_is = skf_is
X_test_array = np.array([X_array[i] for i in test_is])
y_pred_array = reg.predict(X_test_array)

#%%
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100,1000], 'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf'], 'epsilon':[0.2]}

reg = GridSearchCV(SVR(), param_grid, cv=3)
reg.fit(X_train_array, y_train_array)

_, test_is = skf_is
X_test_array = np.array([X_array[i] for i in test_is])
y_pred_array = reg.predict(X_test_array)

#%%
from sklearn.ensemble import AdaBoostRegressor

reg = AdaBoostRegressor(n_estimators=100,loss="square")
reg.fit(X_train_array, y_train_array)

_, test_is = skf_is
X_test_array = np.array([X_array[i] for i in test_is])
y_pred_array = reg.predict(X_test_array)
#%%
from sklearn.metrics import r2_score

ground_truth_array = y_array[test_is]

score = np.sqrt(np.mean(np.square(ground_truth_array - y_pred_array)))
print 'RMSE = ', score
print 'R² = ', r2_score(ground_truth_array,y_pred_array)

y_pred_array=np.expand_dims(y_pred_array,axis=1)
ground_truth_array=np.expand_dims(ground_truth_array,axis=1)
result=np.hstack((ground_truth_array,y_pred_array))
result=np.hstack((result,y_pred_array-ground_truth_array))

#%%,
plt.figure(figsize=(15, 5))
ordering = np.argsort(reg.feature_importances_)[::-1][:50]
importances = reg.feature_importances_[ordering]
feature_names = X_columns[ordering]
x = np.arange(len(feature_names))
plt.bar(x, importances)
plt.xticks(x + 0.5, feature_names, rotation=90, fontsize=15);

#%%
output_file("prediction.html")
p = figure(plot_width=1600, plot_height=800)
p.line(list(np.arange(0,len(y_train))),list(y_train),line_color="blue",alpha=0.2,name="True values")
p.line(list(np.arange(0,len(y_train))),list(prediction),line_color="red",alpha=0.2,name="Predicted values")
#p.legend()


show(p)