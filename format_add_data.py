# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 13:24:11 2016

@author: Hugo
"""
import pandas as pd
import numpy as np
from preprocessing import *

#HUGO : I use this script to aggregate all my new additional data in 1 csv file.
#The additional features are added simply by calling preprocessing functions.4
#The 2nd block allows to verify there's no Nan values when joining
#%%
add_data = pd.read_csv("Aggregated_add_data.csv")
names=add_data["Departure"].unique()
columns = ["DateOfDeparture","Departure","Arrival"]
add_data2=pd.DataFrame(columns)

for name in names:
    temp = add_data
    temp["Arrival"]=name
    add_data2=add_data2.append(temp,verify_integrity=False)
    

add_data2 = add_data2[columns]
add_data2 = add_data2.iloc[3:,:]
add_data2.index = np.arange(0,220800)

data_completed_all = airport_geographic_data(add_data2, include_names=False,keep_only_distance=True,scaling=False)
data_completed_all = airports_traffic(data_completed_all)
data_completed_all = air_accidents_data(data_completed_all)

#Save in a file
data_completed_all.to_csv("external_data_sub1.csv",sep=",",index=False,encoding='utf-8')


#%%TEST
add_data = pd.read_csv("external_data_sub1.csv")
initial_data = pd.read_csv("public_train.csv")
X_weather = add_data
X_encoded = initial_data
#X_encoded = initial_data.drop("Unnamed: 0",axis=1)
X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
X_weather['DateOfDeparture'] = pd.to_datetime(X_weather['DateOfDeparture'])


X_encoded = X_encoded.set_index(['DateOfDeparture','Departure', 'Arrival'])
X_weather = X_weather.set_index(['DateOfDeparture','Departure', 'Arrival'])
X_final = X_encoded.join(X_weather).reset_index()
X_final["Distance"].value_counts(dropna=False)