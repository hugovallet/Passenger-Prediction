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