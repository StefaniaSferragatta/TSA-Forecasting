import pandas as pd 
import numpy as np
import datetime

# co2_cols = ['Decimal Date','Trend']
# co2 = co2.drop(co2_cols,axis=1)

# sea_cols = ['Uncertanty']
# sea_level = sea_level.drop(sea_cols,axis=1)

# anomaly_temperature.drop(anomaly_temperature[anomaly_temperature['Source'] == 'GCAG'].index, inplace = True)
# anomaly_cols = ['Source']
# anomaly_temperature = anomaly_temperature.drop(anomaly_cols,axis=1)

# function to create date
def create_date(start_date,end_date,month):
    dates = pd.date_range(f"{start_date}-{month}", f"{end_date}-{12}",freq='MS') #'MS' to have monthly data
    return dates.to_native_types().tolist() #conver the DatetimeIndex object obtained into a list

''''
Now that we have the list of the dates i can add them to the dataframes and then create the values
'''
# co2 date are until 2018-07
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
for s in np.arange(9, 14, 0.05): #starting from 9 increase for each month of 0.05
    if len(sea_val)<= 95: #num of nan in sea_level
        sea_val.append(round(s,1))

co2_val=[]
for c in np.arange(400,432, 0.8):
    if len(co2_val)<=40:
        co2_val.append(round(c,3))

anomaly_val =[]
for a in np.arange(1,3,0.03):
    if len(anomaly_val)<=59:
        anomaly_val.append(round(a,2))


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