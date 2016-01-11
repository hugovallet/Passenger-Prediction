# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:04:41 2016

@author: Hugo
"""
import pandas as pd

def airport_name_correspondancy(data, all_info = False):
    add_data = pd.read_csv("./Additional data/Correspondency_table.csv")
    arrival_names = add_data.rename(columns={"Code":"Arrival","Location":"Arrival Location","State":"Arrival State"})
    departure_names = add_data.rename(columns={"Code":"Departure","Location":"Departure Location","State":"Departure State"})
    if all_info==False:
        arrival_names = arrival_names[["Arrival","Name"]]
        departure_names = departure_names[["Departure","Name"]]
        data = pd.merge(data,arrival_names,on="Arrival")
        data = data.rename(columns={"Name" : "Arrival name"})
        data = pd.merge(data,departure_names, on = "Departure")
        data = data.rename(columns={"Name" : "Departure name"})
    else :
        data = pd.merge(data,arrival_names,on="Arrival")
        data = data.rename(columns={"Name" : "Arrival name"})
        data = pd.merge(data,departure_names, on = "Departure")
        data = data.rename(columns={"Name" : "Departure name"})
    
    return data