# PRACTICING TIME SERIES ANALYSIS - EXPLORING AIR QUALITY DATA IN INDIA (2015-2020) 

# LOADING LIBRARIES
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import io, os

#-------------------------------------------------
# LOADING THE CLEANED DATA 
base = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base, '..', 'week03-pandas', 'aqi_cleaned.csv'))  
print(df.head())
print("DataFrame shape: ", df.shape)  # (24850, 9)

#-------------------------------------------------
# DATA PREPARATION
df_use = df[['PM2.5', 'PM10', 'NO2', 'AQI','Date', 'City']]
print("DataFrame shape: ", df_use.shape)  # (24850, 6)
print(df_use.isnull().sum().sum())  # 7477 null values
df_use = df_use.dropna()
print("DataFrame shape: ", df_use.shape)  # (17506, 6)

#-------------------------------------------------
# DATETIME INDEXING

print("Data types: \n",df_use.dtypes)
# Date column is object type

df_use['Date'] = pd.to_datetime(df_use['Date'])

print("Data types: \n",df_use.dtypes) 

df_use.set_index('Date', inplace=True)
print(df_use.head(10))

# slicing
df_use_2016_09 = df_use.loc['2016-09']
print(df_use_2016_09.head(10))

print(df_use.shape)         # (17506,5)
print(df_use_2016_09.shape) # (111,5)

#-------------------------------------------------
# RESAMPLING AND MISSING VALUES FOR NO2

city = 'Delhi'
df_city = df_use[df_use['City'] == city]
print(df_city.head(10))

print(df_city.index.duplicated().sum())   # no rows with same date index
print(df_city.shape)   # (1930,5)

df_no2 = df_city[['NO2']]   # Don't need to mention Date in column list as it is already the index
print(df_no2.head(10))
print(df_no2.resample('M').last())
print(df_no2.resample('M').last().isna().sum())  # No missing values to fill

print(df_no2.shape)                      # (1930,1)
print(df_no2.resample('M').last().shape) # (67,1) => 67 Months of data

#-------------------------------------------------
