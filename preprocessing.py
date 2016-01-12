# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:04:41 2016

@author: Hugo
"""
import pandas as pd
from sklearn.preprocessing import scale

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
    
from math import radians, cos, sin, asin, sqrt

def haversine(row):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    HUGO : I modified this function so that the calculus could be made for every rows in our input dataFrame
    Note that this function is reused in "airport_geographic_data"
    """
    lon1 = row['dep_Long']
    lat1 = row['dep_Lat']
    lon2 = row['arr_Long']
    lat2 = row['arr_Lat']
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r    
    
def airport_geographic_data(data, include_names = False, keep_only_distance = False, scaling = True):
    nb_obs=data.shape[0]    
    
    add_data = pd.read_csv("./Additional data/Airports_geographic_data.csv")
    add_data.columns = ["a","City1","City","Country","IATA","ICAO","Lat","Long","Alt","Time_zone","DST","Time1"]
    add_data = add_data.drop(["a","City1","ICAO","DST","Time1"],axis=1)
    
    departure_geo_data = add_data.copy()
    departure_geo_data.columns="dep_"+departure_geo_data.columns
    departure_geo_data = departure_geo_data.rename(columns={"dep_IATA":"Departure"})
    """#En construction
    if scaling == True:
        departure_geo_data = scale(departure_geo_data,axis=1)
        """
    data = pd.merge(data,departure_geo_data, on = "Departure")
        
    arrival_geo_data = add_data.copy()
    arrival_geo_data.columns = "arr_"+arrival_geo_data.columns
    arrival_geo_data = arrival_geo_data.rename(columns={"arr_IATA":"Arrival"})
    """#En construction
    if scaling==True :
        arrival_geo_data = scale(arrival_geo_data,axis=1)
        """
    data = pd.merge(data,arrival_geo_data, on = "Arrival")
    
    if include_names==False:
        data=data.drop(["dep_City","dep_Country","arr_City","arr_Country"],axis=1)
    
    data['Distance'] = data.apply(lambda row: haversine(row), axis=1)
    if scaling == True :
        data['Distance'] = scale(data['Distance'])
    
    if keep_only_distance==True:
        data=data.drop(["dep_Lat","dep_Long","dep_Alt","dep_Time_zone","arr_Lat","arr_Long","arr_Alt","arr_Time_zone"],axis=1)
    
    return data

def meteorological_data(data):
    
    return data

def dummy_converter(data_encoded):
    data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Departure'], prefix='d'))
    data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Arrival'], prefix='a'))
    data_encoded = data_encoded.drop('Departure', axis=1)
    data_encoded = data_encoded.drop('Arrival', axis=1)
    
    # following http://stackoverflow.com/questions/16453644/regression-with-date-variable-using-scikit-learn
    data_encoded['DateOfDeparture'] = pd.to_datetime(data_encoded['DateOfDeparture'])
    data_encoded['year'] = data_encoded['DateOfDeparture'].dt.year
    data_encoded['month'] = data_encoded['DateOfDeparture'].dt.month
    data_encoded['day'] = data_encoded['DateOfDeparture'].dt.day
    data_encoded['weekday'] = data_encoded['DateOfDeparture'].dt.weekday
    data_encoded['week'] = data_encoded['DateOfDeparture'].dt.week
    data_encoded['n_days'] = data_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)
    
    data_encoded = data_encoded.join(pd.get_dummies(data_encoded['year'], prefix='y'))
    data_encoded = data_encoded.join(pd.get_dummies(data_encoded['month'], prefix='m'))
    data_encoded = data_encoded.join(pd.get_dummies(data_encoded['day'], prefix='d'))
    data_encoded = data_encoded.join(pd.get_dummies(data_encoded['weekday'], prefix='wd'))
    data_encoded = data_encoded.join(pd.get_dummies(data_encoded['week'], prefix='w'))  
    
    return data_encoded

def airport_log_flow(data):
    import networkx as nx
    import seaborn as sb
    matrix=data[["log_PAX","Departure","Arrival"]]
    group=matrix.groupby(['Departure', 'Arrival'],as_index=False).mean()
    G=nx.Graph()
    for i in range(126):
        G.add_edge(group["Departure"][i],group["Arrival"][i],weight=group["log_PAX"][i])
        
    adjacency_matrix=nx.to_pandas_dataframe(G)
    
    
    sb.heatmap(adjacency_matrix,cmap="OrRd")
        

