# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:39:08 2016

@author: Hugo
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def plot_stages(reg,X_train,y_train,X_test,y_test):
    test_score = np.zeros(reg.n_estimators, dtype=np.float64)
    train_score = np.zeros(reg.n_estimators, dtype=np.float64)
    
    for i, y_pred in enumerate(reg.staged_predict(X_test)):
        test_score[i] = np.sqrt(mean_squared_error(y_test, y_pred))
        
    for i, y_pred_train in enumerate(reg.staged_predict(X_train)):    
        train_score[i] = np.sqrt(mean_squared_error(y_train, y_pred_train))
        
    min_test_score = min(test_score)
    min_test_score_stage = np.argmin(test_score)    
    learning_rate=reg.learning_rate
    max_depth = reg.max_depth
    
    fig=plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    plt.title('RMSE at each stage of the training phase')
    ax.plot(np.arange(reg.n_estimators), train_score, 'b-', label='Training Set RMSE')
    ax.plot(np.arange(reg.n_estimators), test_score, 'r-', label='Test Set RMSE')
    plt.xlim((0,reg.n_estimators))
    ymin , ymax = plt.ylim()
    xmin , xmax = plt.xlim()
    ax.annotate('Learning rate : '+str(learning_rate), xy=(0.8*xmax, 0.85*ymax), xytext=(0.8*xmax, 0.85*ymax))
    ax.annotate('Max depth : '+str(max_depth), xy=(0.8*xmax, 0.8*ymax), xytext=(0.8*xmax, 0.8*ymax))
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.hlines(y=min_test_score,xmin=0,xmax=reg.n_estimators,linestyles="dashed",color="grey")
    plt.vlines(x=min_test_score_stage,ymin=0,ymax=1,linestyles="dashed",color="grey")
    plt.xlabel('Boosting Iterations')
    plt.ylabel('RMSE')
    plt.show()
    
    
def plot_coeff_importances(reg,data_columns):
    X_columns = data_columns
    
    ordering = np.argsort(reg.feature_importances_)[::-1][:50]
    importances = reg.feature_importances_[ordering]
    feature_names = X_columns[ordering]
    x = np.arange(len(feature_names))
    
    plt.figure(figsize=(15, 5))
    plt.bar(x, importances)
    plt.xticks(x + 0.5, feature_names, rotation=90, fontsize=15);