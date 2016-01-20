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
    data = data.reset_index().merge(departure_geo_data, how="left").set_index('index')
        
    arrival_geo_data = add_data.copy()
    arrival_geo_data.columns = "arr_"+arrival_geo_data.columns
    arrival_geo_data = arrival_geo_data.rename(columns={"arr_IATA":"Arrival"})
    """#En construction
    if scaling==True :
        arrival_geo_data = scale(arrival_geo_data,axis=1)
        """
    data = data.reset_index().merge(arrival_geo_data, how="left").set_index('index')
    
    
    if include_names==False:
        data=data.drop(["dep_City","dep_Country","arr_City","arr_Country"],axis=1)
    
    data['Distance'] = data.apply(lambda row: haversine(row), axis=1)
    if scaling == True :
        data['Distance'] = scale(data['Distance'])
    
    if keep_only_distance==True:
        data=data.drop(["dep_Lat","dep_Long","dep_Alt","dep_Time_zone","arr_Lat","arr_Long","arr_Alt","arr_Time_zone"],axis=1)
    
    return data
    
def air_accidents_data(data):
    add_data = pd.read_csv("./Additional data/Aircraft_accidents_data.csv")
    data['DateOfDeparture'] = pd.to_datetime(data['DateOfDeparture'])
    data['Year'] = data['DateOfDeparture'].dt.year
    data = data.reset_index().merge(add_data, how="left").set_index('index')
    data = data.drop("Year",axis=1) #Dropped beacuse redefined latter with dummy converter
    
    return data
    
def airports_traffic(data):
    add_data = pd.read_csv("./Additional data/Airports_traffic.csv")
    data['DateOfDeparture'] = pd.to_datetime(data['DateOfDeparture'])
    data['Year'] = data['DateOfDeparture'].dt.year
    
    #Rename columns
    departure_add_data = add_data.copy()
    departure_add_data = departure_add_data.rename(columns={"IATA":"Departure","Total passengers":"Dep_total_passengers"})
    arrival_add_data = add_data.copy()
    arrival_add_data = arrival_add_data.rename(columns={"IATA":"Arrival","Total passengers":"Arr_total_passengers"})
    
    data = data.reset_index().merge(departure_add_data, how = "left").set_index('index')
    data = data.reset_index().merge(arrival_add_data, how = "left").set_index('index')
    
    return data
    
def airports_monthly_traffic(data):
    airports_monthly = pd.read_csv("./Additional data/Airports_monthly_traffic.csv",sep=",")
    data_encoded = data.copy()
    data_encoded['DateOfDeparture'] = pd.to_datetime(data_encoded['DateOfDeparture'])
    data_encoded['year'] = data_encoded['DateOfDeparture'].dt.year
    data_encoded['month'] = data_encoded['DateOfDeparture'].dt.month
    
    months = dict({"Jan": 1,"Feb": 2,"Mar": 3,"Apr": 4,"May": 5,"Jun": 6,"Jul": 7,"Aug": 8,"Sep": 9,"Oct": 10,"Nov": 11,"Dec": 12})
    
    monthly_passengers = pd.DataFrame(columns=["Code","Monthly passengers","year","month"])
    airports_monthly_months=airports_monthly.drop("Code",axis=1)
    
    for col in airports_monthly_months.columns:
        column = airports_monthly[["Code",col]]
        month = months[col[0:3]]
        year = col[3:7]
        column['year']=int(year)
        column['month']=int(month)
        column = column.rename(columns={col : "Monthly passengers"})
        monthly_passengers = monthly_passengers.append(column)
    
    arrival_monthly_passengers = monthly_passengers.copy()
    arrival_monthly_passengers = arrival_monthly_passengers.rename(columns={"Monthly passengers":"Arrival Monthly Passengers","Code":"Arrival"})
    departure_monthly_passengers = monthly_passengers.copy()
    departure_monthly_passengers = departure_monthly_passengers.rename(columns={"Monthly passengers":"Departure Monthly Passengers","Code":"Departure"})
    
    data_encoded = data_encoded.reset_index().merge(departure_monthly_passengers,how='left').set_index('index')
    data_encoded = data_encoded.reset_index().merge(arrival_monthly_passengers,how="left").set_index('index')
    data_encoded = data_encoded.drop(["year","month"],axis=1)
    
    return data_encoded


        
    

def fix_events_column(data_weather):
    events_cloud = data_weather[["CloudCover","Events"]].fillna(value="other")
    sun_index = (events_cloud['Events'] == "other") & (events_cloud['CloudCover'] <=4 )
    cloudy_index = (events_cloud['Events'] == "other") & (events_cloud['CloudCover'] >4 )
    data_weather["Events"][cloudy_index]="Cloud"
    data_weather["Events"][sun_index]="Sun"
    events= data_weather["Events"]
    extrem_weather_index = (events == "Rain-Thunderstorm-Tornado") ^ (events == "Fog-Rain-Snow-Thunderstorm") ^ (events == "Rain-Snow-Thunderstorm") ^ (events == "Fog-Rain-Hail-Thunderstorm")^ (events == "Rain-Hail-Thunderstorm")^ (events == "Fog-Rain-Thunderstorm")^ (events == "Rain-Thunderstorm")^ (events == "Thunderstorm")
    data_weather["Events"][extrem_weather_index]="Extrem"
    snow_index = (events == "Rain-Snow") ^ (events == "Fog-Snow") ^ (events == "Fog-Rain-Snow")^ (events == "Snow")
    data_weather["Events"][snow_index]="Snow"
    fog_index = (events == "Fog") ^ (events == "Fog-Rain")
    data_weather["Events"][fog_index]="Fog"
    data_weather["Events"]=data_weather["Events"].factorize()[0]
    
    return data_weather
    
def fix_gust_speed(data_weather):
    gust_speed = data_weather["Max Gust SpeedKm/h"]
    gust_speed = gust_speed.fillna(value="other")
    index = (gust_speed == "other")
    data_weather["Max Gust SpeedKm/h"][index] = data_weather["Max Wind SpeedKm/h"][index].astype(float)
    
    return data_weather

def meteorological_data(data,columns=None):
    
    data_weather = pd.read_csv("Additional data/Meteorological_data.csv")
           
    if columns==None:
        X_weather = data_weather
        X_weather = fix_events_column(X_weather)
        X_weather = fix_gust_speed(X_weather)
        X_weather = X_weather.drop("Precipitationmm",axis=1)
    else :
        X_weather = data_weather
        X_weather = fix_events_column(X_weather)
        X_weather = fix_gust_speed(X_weather)
        X_weather = X_weather.drop("Precipitationmm",axis=1)
        
    
    X_weather = X_weather.rename(columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})
    
    X_weather['DateOfDeparture'] = pd.to_datetime(X_weather['DateOfDeparture'])
    data['DateOfDeparture'] = pd.to_datetime(data['DateOfDeparture'])
    """
    data = data.set_index(['DateOfDeparture', 'Arrival'])
    X_weather = X_weather.set_index(['DateOfDeparture', 'Arrival'])
    data = data.join(X_weather).reset_index()
"""
    data = data.reset_index().merge(X_weather,how="left").set_index("index")    
    return data
    
#%%
    
def get_logpax_timeseries(data):
    departures = data["Departure"].unique()
    arrivals = data["Arrival"].unique()
    all_series = pd.DataFrame("Date")
    for departure in departures:
        for arrival in arrivals :
            time_series = data[data["Departure"]==departure]
            time_series = time_series[time_series["Arrival"]==arrival]
            time_series = time_series[["DateOfDeparture","log_PAX"]]
            time_series['DateOfDeparture'] = pd.to_datetime(time_series['DateOfDeparture'])
            time_series['n_days'] = time_series['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)
            time_series = time_series.set_index("n_days")
            time_series = time_series.sort_index()
            time_series = time_series.drop("DateOfDeparture",axis=1)
            
    return time_series
    


#%%
def dummy_converter(data_encoded, only_for_date=False):
    if only_for_date==False:
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



