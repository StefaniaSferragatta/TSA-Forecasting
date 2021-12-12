# utilities
import pandas as pd 
import numpy as np
import datetime
import random
from datetime import datetime as dt
from pandas import DataFrame
import warnings

co2 = pd.read_csv('co2-avg_montly.csv')
sea_level = pd.read_csv('avg_sea_level.csv')
sea_level.rename(columns = {'Time':'Date'},inplace=True)
anomaly_temperature = pd.read_csv('anomalies_temp.csv')

co2_cols = ['Decimal Date','Trend']
co2 = co2.drop(co2_cols,axis=1)

sea_cols = ['Uncertanty']
sea_level = sea_level.drop(sea_cols,axis=1)


anomaly_temperature.drop(anomaly_temperature[anomaly_temperature['Source'] == 'GCAG'].index, inplace = True)
anomaly_cols = ['Source']
anomaly_temperature = anomaly_temperature.drop(anomaly_cols,axis=1)

# function to create date
def create_date(start_date,end_date,month):
    dates = pd.date_range(f"{start_date}-{month}", f"{end_date}-{12}",freq='MS') #'MS' to have monthly data
    return dates.to_native_types().tolist() #conver the DatetimeIndex object obtained into a list

''''
Now that we have the list of the dates i can add them to the dataframes and then create the values
'''
#co2 date are until 2018-07
year_start = 2018
year_end = 2021
co2_date = create_date(year_start,year_end,month='aug')

#sea_level dates are until 2013-12
year_start = 2014
year_end = 2021
sea_date = create_date(year_start,year_end,month='jan')

#anomaly_temperature date are until 2016-12
year_start = 2017
year_end = 2021
anomalyt_date = create_date(year_start,year_end,month='jan')


''''
Now it's time to add the values to the new dates where there are the NA
'''
sea_val = []
for s in range(96): #num of nan in sea_level
    sea_val.append(round(random.uniform(7.5,14.8), 2))

co2_val=[]
for c in range(40): #num of nan in co2
    co2_val.append(round(random.uniform(400.0,431.8), 2))

anomaly_val =[]
for a in range(59): #num of nan in anomalies_temp
    anomaly_val.append(round(random.uniform(1.0,3.2), 2))


''''
Insert into the dataframe the values created (date - val)
'''
for d,v in zip(co2_date,co2_val):
    co2=co2.append({'Date':d, 'co2':v}, ignore_index=True)

for d,v in zip(sea_date,sea_val):
    sea_level=sea_level.append({'Date':d,'GMSL(Global Mean Sea Level)':v}, ignore_index=True)

for d,v in zip(anomalyt_date,anomaly_val):
    anomaly_temperature= anomaly_temperature.append({'Date':d,'Anomalies_Land_Ocean_Temperature':v}, ignore_index=True)


'''
Save to csv
'''
co2.to_csv('co2.csv',index=False)
sea_level.to_csv('sea_level.csv',index=False)
anomaly_temperature.to_csv('anomalies_temperature.csv',index=False)