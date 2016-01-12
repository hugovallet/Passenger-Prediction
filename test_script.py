# -*- coding: utf-8 -*-
"""
Created on Sat Jan 09 15:26:21 2016

@author: Hugo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
X_weather = data_weather[['Date', 'AirPort', 'Mean TemperatureC']]
"""
X_weather=data_weather.drop('Events',axis=1)
X_weather=X_weather.drop('Precipitationmm',axis=1)
X_weather=X_weather.drop('Max Gust SpeedKm/h',axis=1)
"""
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
X_copy = X_array

# Regression
train_is, _ = skf_is
X_train_array = np.array([X_array[i] for i in train_is])
y_train_array = np.array([y_array[i] for i in train_is])


#%% FIRST MODEL : RANDOM FORESTS
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

X_array=normalize(X_copy,axis=1)

pca=PCA(n_components=30)
X_array=pca.fit_transform(X_array)


train_is, test_is = skf_is
X_train_array = np.array([X_array[i] for i in train_is])
y_train_array = np.array([y_array[i] for i in train_is])
X_test_array = np.array([X_array[i] for i in test_is])

reg = RandomForestRegressor(n_estimators = 120,max_depth =90, max_features = "auto")
reg.fit(X_train_array, y_train_array)

y_pred_array = reg.predict(X_test_array)

#%% SECOND MODEL : SVM
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100,1000], 'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf'], 'epsilon':[0.2]}

reg = GridSearchCV(SVR(), param_grid, cv=3)
reg.fit(X_train_array, y_train_array)

_, test_is = skf_is
X_test_array = np.array([X_array[i] for i in test_is])
y_pred_array = reg.predict(X_test_array)

#%%
from sklearn.svm import LinearSVR
reg = LinearSVR(C=0.01)
reg.fit(X_train_array, y_train_array)

_, test_is = skf_is
X_test_array = np.array([X_array[i] for i in test_is])
y_pred_array = reg.predict(X_test_array)

#%% THIRD MODEL : BOOSTED DECISION TREES
from sklearn.ensemble import AdaBoostRegressor


reg = AdaBoostRegressor(n_estimators=100,loss="square")
reg.fit(X_train_array, y_train_array)

_, test_is = skf_is
X_test_array = np.array([X_array[i] for i in test_is])
y_pred_array = reg.predict(X_test_array)
#%% TEST RELEVANCE OF THE PREDICTION
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

ground_truth_array = y_array[test_is]



score = np.sqrt(np.mean(np.square(ground_truth_array - y_pred_array)))
print 'RMSE = ', score
#print 'RMSE =', np.sqrt(mean_squared_error(ground_truth_array,y_pred_array))
print 'R² = ', r2_score(ground_truth_array,y_pred_array)

y_pred_array=np.expand_dims(y_pred_array,axis=1)
ground_truth_array=np.expand_dims(ground_truth_array,axis=1)
result=np.hstack((ground_truth_array,y_pred_array))
result=np.hstack((result,y_pred_array-ground_truth_array))

#%%,
X_columns = X_encoded.columns
plt.figure(figsize=(15, 5))
ordering = np.argsort(reg.feature_importances_)[::-1][:50]
importances = reg.feature_importances_[ordering]
feature_names = X_columns[ordering]
x = np.arange(len(feature_names))
plt.bar(x, importances)
plt.xticks(x + 0.5, feature_names, rotation=90, fontsize=15);

#%%,
X_columns = X_encoded.columns
plt.figure(figsize=(15, 5))
ordering = np.argsort(reg.coef_)[::-1][:50]
importances = reg.coef_[ordering]
feature_names = X_columns[ordering]
x = np.arange(len(feature_names))
plt.bar(x, importances)
plt.xticks(x + 0.5, feature_names, rotation=90, fontsize=15);
#%%filtering input data based on importance

X_filtered = X_encoded.iloc[:,ordering]
X_array = X_filtered.values

train_is, _ = skf_is
X_train_array = np.array([X_array[i] for i in train_is])
y_train_array = np.array([y_array[i] for i in train_is])

reg = SVR(C=0.01, epsilon=0.2, kernel='rbf', gamma=0.1)
"""
reg = RandomForestRegressor(n_estimators = 100,max_depth =90, max_features =30)
"""
reg.fit(X_train_array, y_train_array)


_, test_is = skf_is
X_test_array = np.array([X_array[i] for i in test_is])
y_pred_array = reg.predict(X_test_array)

#%%
from bokeh.plotting import *

output_file("prediction.html")
p = figure(plot_width=1600, plot_height=800)
p.line(list(np.arange(0,len(ground_truth_array))),list(ground_truth_array),line_color="blue",alpha=0.2,name="True values")
p.line(list(np.arange(0,len(y_train_array))),list(y_pred_array),line_color="red",alpha=0.2,name="Predicted values")


show(p)

#%% Influence of number of estimators in the random forest
from sklearn.cross_validation import cross_val_score

scores=[]
grid = np.linspace(1,200,10)
grid = grid.round(0).astype(int)
for n in grid :
    reg=RandomForestRegressor(n_estimators = n,max_depth =100, max_features = 50)
    score=cross_val_score(reg, X_array, y_array, cv=3, scoring='mean_squared_error',n_jobs=1)
    scores.append(np.mean(np.sqrt(-score)))


best_score = round(min(scores),3)
trees_best_score = grid[np.argmin(scores)]
plt.figure()
plt.plot(grid, scores)
plt.title("Influence of the number of trees in forest on CV error")
plt.hlines(y=best_score, xmin=1, xmax=200,linestyles="dashed",label="line1")
plt.vlines(x=trees_best_score, ymin=0.45, ymax=0.85,linestyles="dashed",label="line1")
plt.annotate('Best error : ' + str(best_score), xy=(100,0.5),  xycoords='data',textcoords='data')
plt.xlabel("Number of trees in forest")
plt.ylabel("Cross Validated RMSE (3 splits)")
plt.show()
#%% Influence of number of features to keep in forest
    
scores2=[]
grid = np.linspace(1,153,10)
grid = grid.round(0).astype(int)
for n in grid :
    reg=RandomForestRegressor(n_estimators = 100,max_depth =100, max_features = n)
    score=cross_val_score(reg, X_array, y_array, cv=3, scoring='mean_squared_error',n_jobs=1)
    scores2.append(np.mean(np.sqrt(-score)))

best_score2 = round(min(scores2),3)
features_best_score = grid[np.argmin(scores2)]
plt.figure()
plt.plot(grid, scores2)
plt.title("Influence of the number of features in forest on CV error")
plt.hlines(y=best_score2, xmin=1, xmax=155,linestyles="dashed",label="line1")
plt.vlines(x=features_best_score, ymin=0.45, ymax=0.85,linestyles="dashed",label="line1")
plt.annotate('Best error : ' + str(best_score2), xy=(100,0.5),  xycoords='data',textcoords='data')
plt.xlabel("Max number of features in forest")
plt.ylabel("Cross Validated RMSE (3 splits)")
plt.show()

#%% Influence of tree depth 

scores3=[]
grid = np.linspace(1,200,10)
grid = grid.round(0).astype(int)
for n in grid :
    reg=RandomForestRegressor(n_estimators = 100,max_depth = n, max_features = "auto")
    score=cross_val_score(reg, X_array, y_array, cv=3, scoring='mean_squared_error',n_jobs=1)
    scores3.append(np.mean(np.sqrt(-score)))
    
    
best_score3 = round(min(scores3),3)
depth_best_score = grid[np.argmin(scores3)]
plt.figure()
plt.plot(grid, scores3)
plt.title("Influence of the trees' maximal accepted depth on CV error")
plt.hlines(y=best_score3, xmin=1, xmax=200,linestyles="dashed",label="line1")
plt.vlines(x=depth_best_score, ymin=0.4, ymax=1,linestyles="dashed",label="line1")
plt.annotate('Best error : ' + str(best_score3), xy=(100,0.5),  xycoords='data',textcoords='data')
plt.xlabel("Max forest depth")
plt.ylabel("Cross Validated RMSE (3 splits)")
plt.show()

#%% 
"""
At that step we know the best parameters for our random forest : 
    max_features = n_features
    n_estimator = 120
    max_depth = 50
    
We wanna see if performing some feature filtering could improve the results

"""

#%%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import partial_dependence as pds
from sklearn.feature_selection import f_regression

f_test, p_values=f_regression(X_array,y_array)

# Fit regression model
params = {'n_estimators': 300, 'max_depth': 7, 'min_samples_split': 1,
          'learning_rate': 0.1, 'loss': 'huber', 'max_features' : "auto"}
reg = GradientBoostingRegressor(**params)

train_is, test_is = skf_is
X_train_array = np.array([X_array[i] for i in train_is])
y_train_array = np.array([y_array[i] for i in train_is])
X_test_array = np.array([X_array[i] for i in test_is])


reg.fit(X_train_array, y_train_array)

y_pred_array = reg.predict(X_test_array)

ground_truth_array = y_array[test_is]
rmse = np.sqrt(mean_squared_error(ground_truth_array, y_pred_array))
print("RMSE: %.4f" % rmse)

#•features = ['Week','Day', 'WeeksToDeparture']
#fig,ax = pds.plot_partial_dependence(reg, X_train_array, features, feature_names)


#%%
# Follow the RMSE at each stage
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
train_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(reg.staged_predict(X_test_array)):
    test_score[i] = np.sqrt(mean_squared_error(ground_truth_array, y_pred))
    
for i, y_pred_train in enumerate(reg.staged_predict(X_train_array)):    
    train_score[i] = np.sqrt(mean_squared_error(y_train_array, y_pred_train))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('RMSE')
plt.plot(np.arange(params['n_estimators']) + 1, train_score, 'b-',
         label='Training Set RMSE')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set RMSE')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('RMSE')


