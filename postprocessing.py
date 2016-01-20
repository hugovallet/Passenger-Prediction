# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:00:25 2016

@author: Hugo
"""
import numpy as np

def filter_by_importance(reg,data,nb_features_to_keep):
    #This methods filters my data based on a first ensemble method (reg) fitted on the whole data
    
    X_columns = data.columns
    ordering = np.argsort(reg.feature_importances_)[::-1][:nb_features_to_keep]
    feature_names = X_columns[ordering]
    filtered_data = data[feature_names]
    
    return filtered_data
    

class mean_regressor :
    def __init__(self):
        self.series = None
        
        
    def fit(self,data):
        log_pax=data[["DateOfDeparture","log_PAX"]]
        log_pax=log_pax.groupby(["DateOfDeparture"])["log_PAX"].mean()
        self.series = log_pax
    
    def predict(self,data_test):
        date = data_test["DateOfDeparture"]
        log_pax_values = self.series
        y_predict=np.zeros(len(date))
        for i in range(len(date)):
            
            index = (log_pax_values.index==date.iloc[i])
            y_predict[i] = log_pax_values[index]
        
        return y_predict
        
        