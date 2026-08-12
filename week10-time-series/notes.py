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
# PREDICTING TIME SERIES DATA
# we create a model that takes n time series values immediately preceding the target time as input and outputs the predicted value at the target time

# Mainly 2 approaches: statistical and ML

# Statistical: predictions are made based on the assumption that time series data has specific statistical properties
# Since the model parameters and their relationship with the data are relatively interpretable, it is easier to understand how the predictions are made.
# but if assumptions of stat. properties wrong, then prediction performance decreases
# ARIMA, SSM

# ML: can capture nonlinear relationships and complex patterns while making predictions.
# no need to assume statistical properties in the data
# more complex, making it difficult to understand how predictions are made         
# But can be made better with more training data
# Non-time-series methods like Random forest, SVM etc, time series methods like RNN, transformer etc.

# autocorrelation coefficient is calculated as the correlation between the current data and data from k periods ago (lagk)
# values from -1 to 1
# lag = shift in time
# At lag 0, the correlation is with itself, so the autocorrelation is always 1
# confidence interval: if the bars fall within this band, we can say there is no significant autocorrelation at that lag. Bars extending outside the band indicate the presence of significant autocorrelation.
# Correlogram plots the autocorrelation coefficients with lag as the x-axis.

#-------------------------------------------------
# OBSERVING TIME SERIES DATA
import statsmodels.api as sm
sunspot_dataset = sm.datasets.sunspots.load_pandas().data

# Remove the first 2 data points for the purpose of the lecture
sunspot_dataset = sunspot_dataset.iloc[2:]

# Organize the data format
sunspot_dataset.index = pd.to_datetime(sunspot_dataset["YEAR"].astype(int), format='%Y')
sunspot_dataset = sunspot_dataset["SUNACTIVITY"]

plt.figure(figsize=(10, 5))
sunspot_dataset.plot()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Import the plot_acf function to calculate and plot autocorrelation
from statsmodels.graphics.tsaplots import plot_acf

fig, ax = plt.subplots(figsize=(10, 7))
# Plot autocorrelation. By specifying lags=11, autocorrelation for 11 years (1 cycle) is calculated
plot_acf(sunspot_dataset, lags=11, ax=ax)
plt.xlabel('Lag')
plt.show()

# the pattern shows a damped oscillation, where correlations weaken as the lag increases but remain significant at certain intervals.
# This indicates that the series exhibits strong autocorrelation over multiple years.

fig, ax = plt.subplots(figsize=(10, 7))
# This time, set lags = 50
plot_acf(sunspot_dataset, lags=50, ax=ax)
plt.xlabel('Lag')
plt.show()
# Here, the correlogram shows a wavy pattern.
# the correlation coefficients get high and low every 10 or 11 years - periodic patterns


# For independent sequence: the correlogram hits almost zero whenever lag >=1. This indicates the trials are independent, and maybe you have to consider if you can apply time series models, which usually assumes autocorrelative nature in the data.
import numpy as np
dice_data = np.random.randint(1,7,5000)

fig, ax = plt.subplots(figsize=(10, 7))
plot_acf(dice_data, ax=ax)
plt.xlabel('Lag')
plt.show()

# STL Decomposition (Seasonal and Trend decomposition using Loess)
# method for decomposing time series data into trend, seasonal, and residual components.
# key parameters:
# period: the length of the seasonal cycle (must be set manually)
# seasonal: smoothing window size for the seasonal component
# trend: smoothing window size for the trend component
# robust: whether to use robust fitting (helps handle outliers) We first instantiate it and then call fit() method. By doing so, it returns a result object with attributes such as, trend, seasonal, resid

from statsmodels.tsa.seasonal import STL

# STL decomposition
stl = STL(sunspot_dataset, period=11)
res = stl.fit()

# Plotting each component

fig, axes = plt.subplots(4,1,figsize=(10,8), sharex=True)
# sharex = True: All subplots in the entire grid share the same x-axis. Ticks and labels are automatically hidden on inner plots to reduce clutter. (default False)

sunspot_dataset.plot(ax = axes[0])
axes[0].grid(True, linestyle = '--', alpha=0.5)
axes[0].set_title('Original Data')

res.trend.plot(ax=axes[1])
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].set_title('Trend')
# trend component appears to fluctuate irregularly but there is a gradual increase

res.seasonal.plot(ax=axes[2])
axes[2].grid(True, linestyle='--', alpha=0.5)
axes[2].set_title('Seasonal')
# seasonality is clearly evident

res.resid.plot(ax=axes[3])
axes[3].grid(True, linestyle='--', alpha=0.5)
axes[3].set_title('Residual')
# represents prediction errors or noise and appears irregular in shape

plt.tight_layout()
plt.show()

#-------------------------------------------------
# STATISTICAL MODELS FOR STATIONARY TIME SERIES DATA

# AR (Autoregressive) Model: future data is predicted by summing up a constant multiple of the past data values.
# Positive coefficients mean past increases carry forward, negative ones mean the opposite. 
# phi close to 0 means no impact

# MA (Moving Average) Model: future data is predicted by summing up a constant multiple of past errors (the difference between predicted value and actual value) of the data to be predicted.
# The mean of the MA (=moving average) model does NOT change
# perceive the error term as "over/underestimation from the predicted value"

# ARMA (AutoRegressive Moving Average) Model: both past values of the data and past error terms are utilized in making predictions
# future data points are predicted by adding a constant multiple of past data points to a constant multiple of past error terms

# STATIONARITY: property where time series data transitions stably over time
# it means that the expected value of the time series data at each time point is constant, and the autocorrelation depends only on the time lag (Weak stationarity)
# So future data can be predicted based on past data characteristics

# Non-stationary data has changing properties as time progresses
# So creating a learning model using past data may lead to unreliable predictions for future data

# Stationarity is determined using Augmented Dickey-Fuller (ADF) test

from statsmodels.tsa.stattools import adfuller

result = adfuller(sunspot_dataset)

# Displaying results
print('Test Statistic\t: %f' %result[0])
print('p-value\t: %f' % result[1])
print('Critical Values\t:')
for key, value in result[4].items():
    print('\t%s\t: %.3f' % (key, value))

# Evaluating the p-value
if result[1] < 0.05:
    print("The time series data is considered stationary.")
else:
    print("The time series data is not considered stationary.")

# estimate the appropriate orders for the observations and error terms to be used in prediction using AIC (Akaike Information Criterion)
# metric to evaluate the trade-off between the complexity of the model and its goodness of fit to the data


# Split dataset: use oldest 95% for training and latest 5% for testing
len_test = int(0.05 * len(sunspot_dataset))
train = sunspot_dataset[:len(sunspot_dataset) - len_test]
test = sunspot_dataset[len(sunspot_dataset) - len_test:]

# DTERMINING HYPERPARAMETERS FOR ARMA MODEL
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

max_ar = 11  # max. AR order
max_ma = 11  # max. MA order

# Initialising min. AIC to infinity
best_aic = np.inf
best_order = None

for p in range(max_ar + 1):
    for q in range(max_ma + 1):
        try:
            # p is the number of past observed values used for prediction, q is the number of past error terms used for prediction
            # Although it is written as ARIMA, when the middle number is 0 as in (p, 0, q), it becomes an ARMA model
            tmp_mdl = ARIMA(train, order = (p,0,q), freq="AS")
            result = tmp_mdl.fit()
            tmp_aic = result.aic
            # If AIC is small, update the model and parameters
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (p, q)
        except:
            print('Error')
            continue

print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
# therefore past values from 8 year and error terms from 6 year minimise the AIC to predict future values

# FORECASTING WITH ARMA MODEL
from sklearn.metrics import mean_squared_error

fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
ax.grid(True, linestyle='--', alpha=0.5)

# Use historical forecast values for 8 years and error terms for 6 years (based on grid search using AIC)
arma_model = ARIMA(train, order=(8, 0, 6), freq="AS")
result = arma_model.fit()
forecast = result.forecast(len_test)

# Display the original data with a blue (default) line
ax.plot(sunspot_dataset, ls="-")
# Display the forecasted results with a red line
ax.plot(forecast, ls="-", color="r", label="predicted")

# Calculate the Mean Squared Error between predicted data and actual data
mse = mean_squared_error(test, forecast)
print("Mean Squared Error:", mse)

#-------------------------------------------------
#%%
# NON-STATIONARY TIME SERIES MODELING WITH ARIMA PROCESS
# ARIMA model (Autoregressive Integrated Moving Average model) is an extension of the ARMA model that can be applied to non-stationary datasets
import statsmodels.datasets.co2 as co2
import pandas as pd
co2_data = co2.load().data
co2_data = co2_data.iloc[1188:]

co2_data.index = pd.to_datetime(co2_data.index)

co2_data = co2_data.resample('M').mean()
print(co2_data)

import matplotlib.pyplot as plt
co2_data.plot()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

from statsmodels.tsa.seasonal import STL

# STL decomposition
stl = STL(co2_data, period=11)
res = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

co2_data.plot(ax=axes[0])
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].set_title('Original Data')

res.trend.plot(ax=axes[1])
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].set_title('Trend')

res.seasonal.plot(ax=axes[2])
axes[2].grid(True, linestyle='--', alpha=0.5)
axes[2].set_title('Seasonal')

res.resid.plot(ax=axes[3])
axes[3].grid(True, linestyle='--', alpha=0.5)
axes[3].set_title('Residual')

plt.tight_layout()
plt.show()
# clear upward trend and periodicity seen

# ADF test
from statsmodels.tsa.stattools import adfuller

result = adfuller(co2_data)

print('Statistical test statistic\t: %f' % result[0])
print('p-value\t: %f' % result[1])
print('Critical values\t:')
for key, value in result[4].items():
    print('\t%s\t: %.3f' % (key, value))

if result[1] < 0.05:
    print("The time series data is considered stationary.")
else:
    print("The time series data is not considered stationary.")
# CO2 concentration data is not stationary
# One possible cause of a non-stationary data is the presence of a trend
# However, even in such non-stationary data, there are cases where the difference between adjacent data points (first-order differencing) becomes stationary
# The first-order differencing data essentially represents the "amount of change from the previous time point" at each time point
# Taking the first-order difference from data with a trend can remove a constant trend, making the data properties constant regardless of time.

# Create data of the difference between adjacent time points
co2_data_diff = co2_data.diff().dropna()

plt.figure(figsize=(10, 5))
co2_data_diff.plot()

print(len(co2_data_diff))
plt.grid(True, linestyle='--', alpha=0.5)
# the rising trend seems to have resolved

# STL decomp. and ADF test on differenced data
stl = STL(co2_data_diff, period=11)
res = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

co2_data_diff.plot(ax=axes[0])
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].set_title('Original Data')

res.trend.plot(ax=axes[1])
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].set_title('Trend')

res.seasonal.plot(ax=axes[2])
axes[2].grid(True, linestyle='--', alpha=0.5)
axes[2].set_title('Seasonal')

res.resid.plot(ax=axes[3])
axes[3].grid(True, linestyle='--', alpha=0.5)
axes[3].set_title('Residual')

plt.tight_layout()
plt.show()

result = adfuller(co2_data_diff)

print('Statistical test statistic\t: %f' % result[0])
print('p-value\t: %f' % result[1])
print('Critical values\t:')
for key, value in result[4].items():
    print('\t%s\t: %.3f' % (key, value))

if result[1] < 0.05:
    print("The time series data is considered stationary.")
else:
    print("The time series data is not considered stationary.")

# data after taking a first difference is stationary
# If you cannot achieve stationarity even after taking the first difference, you may need to take a second difference (difference of differences) to achieve stationarity

# ARIMA model is a model used for forecasting on data that can achieve stationarity by taking differences  n  times.
# After taking differences, we use the same method as the ARMA model to make predictions.

len_test = 12
train = co2_data[:len(co2_data) -len_test]
test = co2_data[len(co2_data)-len_test:]

import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

max_ar = 12  # Maximum order of AR
max_ma = 3  # Maximum order of MA

best_aic = np.inf
best_order = None

for p in range(max_ar+1):
    for q in range(max_ma+1):
        try:
            tmp_mdl = ARIMA(train, order=(p, 1, q))
            result = tmp_mdl.fit()
            tmp_aic = result.aic
            
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (p, q)
        except: continue

print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))

fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
ax.grid(True, linestyle='--', alpha=0.5)

# Prediction using ARMA model with p=12, q=2 based on the results obtained from grid search in terms of AIC
# The 1 in order=(12, 1, 2) represents first order differencing
arma_model = ARIMA(train, order=(12, 1, 2))
result = arma_model.fit()
forecast = result.forecast(len_test)

ax.plot(co2_data, ls="-")
ax.plot(forecast, ls="-", color="r", label="predicted")
# %%

