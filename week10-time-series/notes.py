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
# ENCODING TIME SERIES DATA
import datetime
import random

# SIMPLE ENCODING - NOT CONSIDERING PERIODICITY 
# randomly selecting 10 dates to work with

# function to output random date between start and end
def generate_rand_date(start, end):
    delta_days = (end - start).days
    return start + datetime.timedelta(days = random.randint(0, delta_days))
# datetime.timedelta : A duration expressing the difference between two datetime or date instances to microsecond resolution.

# Fixing the random seed and getting 10 random dates in the 2010s
random.seed(0)
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2020, 12, 31)
random_dates = [generate_rand_date(start, end) for _ in range(10)]

# converting the generated dates to a df
df_dates = DataFrame(random_dates, columns = ['ymd'])
print(df_dates)

# splitting year, month, day
df_dates['year'] = df_dates['ymd'].apply(lambda x: x.year)  # extracts x.year from datetime object x, apply() applies the lambda fnct to every object of the dataframe/series
df_dates['month'] = df_dates['ymd'].apply(lambda x: x.month)
df_dates['day'] = df_dates['ymd'].apply(lambda x: x.day)
print(df_dates)
# we can now treat each year, month, and day as numerical variables
# But there are 2 problems: 
# 1. information of year, month, and day is separate, making it difficult to learn their chronological order
# 2. Discontinuities occur in parts that should be continuous as a time series - ex. it does not capture the fact that Jan and Dec, or the 1st and 31st of a month, are close to each other in a cyclical sense

# To solve 1st problem, we can create a new variable representing the difference from a reference point
# to represent the information of how old or new a date is based on a reference ex. 1st Jan 2010 
start_date = pd.Timestamp('2010-01-01 00:00:00')
df_dates['total_days'] = df_dates['ymd'].apply(lambda x: (x - start_date).days)
# counts days from reference date - info on how old or new the date is - chronological order
print(df_dates)

# To solve 2nd problem: 
# CYCLICAL ENCODING - using trigonometric fncts. to encode periodic structure of time variables
# Variables with periodicity such as months and days are mapped to points on the unit circle and represented by the coordinates of each point
# The coordinates of each point are represented by cosine and sine, which are then used as new numerical variables.

df_dates['month_cos'] = df_dates['month'].apply(lambda x: np.cos(2*np.pi* x/12))
df_dates['month_sin'] = df_dates['month'].apply(lambda x: np.sin(2*np.pi* x/12))

df_dates['day_cos'] = df_dates['day'].apply(lambda x: np.cos(2*np.pi * x/31))
df_dates['day_sin'] = df_dates['day'].apply(lambda x: np.sin(2*np.pi * x/31))

print(df_dates)

#-------------------------------------------------