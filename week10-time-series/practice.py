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

# df_use.set_index('Date', inplace=True)
# print(df_use.head(10))

# # slicing
# df_use_2016_09 = df_use.loc['2016-09']
# print(df_use_2016_09.head(10))

# print(df_use.shape)         # (17506,5)
# print(df_use_2016_09.shape) # (111,5)

# #-------------------------------------------------
# # RESAMPLING AND MISSING VALUES FOR NO2

# city = 'Delhi'
# df_city = df_use[df_use['City'] == city]
# print(df_city.head(10))

# print(df_city.index.duplicated().sum())   # no rows with same date index
# print(df_city.shape)   # (1930,5)

# df_no2 = df_city[['NO2']]   # Don't need to mention Date in column list as it is already the index
# print(df_no2.head(10))
# print(df_no2.resample('M').last())
# print(df_no2.resample('M').last().isna().sum())  # No missing values to fill

# print(df_no2.shape)                      # (1930,1)
# print(df_no2.resample('M').last().shape) # (67,1) => 67 Months of data

# #-------------------------------------------------
# # SHIFTING AND PERCENT CHANGE

# no2_ratio = df_no2.resample('M').last() / df_no2.resample('M').last().shift(1)
# print(no2_ratio.head(10))

# no2_ratio_pct = df_no2.pct_change(periods=1, fill_method=None, freq=None)
# print(no2_ratio_pct.head(10))

# #-------------------------------------------------
# # MOVING AVG. AND STD. DEV.

# print(df_no2.rolling(7).mean().head(10))
# print(df_no2.rolling(7).std().head(10))
# # Shows NaN for the first 6 rows

# df_no2 = df_city[['NO2']].sort_index()
# full_range = pd.date_range(df_no2.index.min(), df_no2.index.max(), freq='D')
# df_no2 = df_no2.reindex(full_range)  # Gap in chart shows missing data

# plt.figure(figsize=(8,6))
# plt.plot(df_no2, color = 'Red', label='Daily NO2')
# plt.plot(df_no2.rolling(7).mean(), color = 'Black', label='7-day rolling avg')

# plt.title('Daily NO2 vs Weekly Rolling Average')
# plt.tight_layout()
# plt.grid(True)
# plt.legend()
# plt.show()

# print(df_no2.index.duplicated().sum())
# print(df_no2.index.is_monotonic_increasing)
# print(df_no2.loc['2017-06':'2017-09'])  # Missing data 

#-------------------------------------------------
# CYCLICAL ENCODING

df_use['Year'] = df_use['Date'].apply(lambda x: x.year)
df_use['Month'] = df_use['Date'].apply(lambda x: x.month)
df_use['Day'] = df_use['Date'].apply(lambda x: x.day)
print(df_use.head())

""" Got Error here because I was trying to do cyclical Encoding on "Date" column.
I had already set Date column as Index (Datetime Indexing) so it was not possible to encode. 
After commenting out the datetime indexing and functions which worked using datetime index (resampling, slicing etc), the encoding worked."""

df_use['month_sin'] = df_use['Month'].apply(lambda x: np.sin(2* np.pi * x/12))
df_use['month_cos'] = df_use['Month'].apply(lambda x: np.cos(2* np.pi * x/12))

df_use['day_sin'] = df_use['Day'].apply(lambda x: np.sin(2* np.pi * x/31))
df_use['day_cos'] = df_use['Day'].apply(lambda x: np.cos(2* np.pi * x/31))
print(df_use.head())

# Cyclical encoding can help identify periodicity rather than just chronological order, 
# Ex. Jan comes after Dec and even though they are first and last month they are close together 

#-------------------------------------------------
# STATIONARITY CHECK

from statsmodels.tsa.stattools import adfuller

result = adfuller(df_use['NO2'])

print('Test Statistic\t: %f' %result[0])
print('p-value\t: %f' % result[1])       # very small value, prints as 0, can check by print(result[1]) with no %f formatting
print('Critical Values\t:')
for key, value in result[4].items():
    print('\t%s\t: %.3f' % (key, value))

if result[1] < 0.05:
    print("The time series data is considered stationary.")
else:
    print("The time series data is not considered stationary.")
# Data is stationary so no differencing needed

#-------------------------------------------------
# SUMMARY
# DATETIME INDEXING AND THE MULTI-CITY TRAP
# Converting Date to datetime and slicing by month worked fine on the full dataset, but resampling and rolling averages later produced a corrupted-looking plot with a diagonal sawtooth line cutting across it.
# The cause turned out to be duplicate dates - the dataset has multiple cities sharing the same date range, so resampling without filtering to one city first was silently mixing readings from different cities together.
# Filtering to a single city (Delhi, 1930 rows) before resampling fixed the duplicate index problem and gave one clean value per date, which is a precondition every one of the later steps assumes and does not check for automatically.

# RESAMPLING, SHIFTING, AND MOVING AVERAGES
# Monthly resampling with .last() on the single-city NO2 series confirmed 67 months of data with no missing values at that resolution.
# Shift-based ratio and .pct_change() both worked once run on the clean single-city series.
# Rolling 7-day mean and std produced NaN for the first 6 rows as expected, but the plotted comparison still showed a second sawtooth artifact even after the city fix.

# THE MISSING DATE GAP
# Reindexing to a complete daily date range and checking duplicated()/is_monotonic_increasing ruled out duplicates as the remaining cause - both came back clean (0 and True).
# Slicing June to September 2017 directly showed the real issue: an actual two-month gap in the sensor data (2017-07-03 straight to 2017-08-31), plus smaller day-level gaps scattered earlier in the series.
# Reindexing to the full daily range and leaving the gaps as NaN (rather than forward-filling over them) gave a plot that honestly shows a break in the line instead of drawing a fake diagonal connecting two unrelated dates.

# CYCLICAL ENCODING
# Hit a genuine gotcha here: cyclical encoding failed while Date was still set as the index, because .apply() on a column needs Date to exist as an actual column, not as the index.
# Commenting out the datetime indexing step (and everything downstream that depended on it) freed Date back up as a column and let month_sin/cos and day_sin/cos encode correctly.
# This is a good one to remember - datetime indexing and column-based feature engineering on that same date information can conflict, and the fix is either resetting the index temporarily or doing the encoding before indexing.

# STATIONARITY CHECK
# Ran the ADF test on NO2 (full df_use, not city-filtered) and got a p-value low enough to print as 0.000000 under %f formatting - confirmed as a genuinely tiny value rather than literally zero by printing result[1] without formatting.
# p < 0.05 means the series is stationary, no differencing required, which fits the pattern already seen in the moving average plot: NO2 oscillates seasonally (winter peaks) around a fairly stable mean rather than trending upward the way CO2 did in the Week 10 notes.
