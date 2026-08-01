# WEEK 10 - TIME SERIES ANALYSIS
# GCI TOKYO UNIVERSITY 2026

# TOPICS COVERED: HANDLING AND UNDERSTANDING TIME SERIES DATA, ENCODING- SIMPLE AND CYCLICAL

# IMPORTING LIBRARIES 
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import warnings
import requests
import io, os

#-------------------------------------------------
# HANDLING AND UNDERSTANDING TIME SERIES DATA
# Time series data = data that changes over time
# Ex: Stock price data (numeric values change over time), Endoscopic images taken periodically, posts on social media (language chages) etc.
# We'll use UNIVARIATE time series data here, meaning only single value changes over time

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv'
r = requests.get(url, stream = True).content
beijing_pm25 = pd.read_csv(io.BytesIO(r))
# print(beijing_pm25.head())

# we will only look at data from December 2012
print(beijing_pm25.loc[(beijing_pm25['year'] == 2012) & (beijing_pm25['month'] == 12)].head())

# combining the time data that is split into four columns: year, month, day, and hour, into a single datetime type column for easier handling
pm25 = pd.DataFrame(beijing_pm25['pm2.5'].values,
                    index = pd.to_datetime(beijing_pm25[['year', 'month', 'day', 'hour']]),
                    columns=['pm2.5'])
print(pm25.head())
# accessing data from dec. 2012
pm25_12_12 = pm25.loc['2012-12']
print(pm25_12_12.head())

# Resampling: extracting data at a different frequency than hte original data
# mention 'D' as argument for daily data, 'M' for month, 'Y' for year
# use .last() method to get the retrieve the last data point, .mean() for the avg etc.

print(pm25_12_12.resample('D').last()) # pm2.5 concentration at 23:00 for each day

# Plotting the data
# convert index to datetime if necessary - pm25_12_12.index = pd.to_datetime(pm25_12_12.index) (here not required)
plt.figure(figsize=(8,5))
pm25_12_12.plot()
plt.grid(True, linestyle='--', alpha = 0.5)
plt.show()

# Hangling missing values : here we will fill missing values with previous day's data
print(pm25_12_12.resample('D').last().ffill())

# Shifting data
# shift() method keeps the index fixed while shifting only the data
# By processing the data this way, we can compute the ratio of the previous day's concentration to the current day's concentration in one go
# value for 2012-12-01 becomes NaN since there is no data available for the day before

pm25_ratio = pm25_12_12.resample('D').last() / pm25_12_12.resample('D').last().shift(1)
print(pm25_ratio.head())
# other methods available: diff, pct_change

# Moving average
# use pandas' rolling() method
# calculating the 3-hour moving average - rolling() and mean()
print(pm25_12_12.rolling(3).mean().head())
print(pm25_12_12.rolling(3).std().head()) # moving std. deviation

#-------------------------------------------------

