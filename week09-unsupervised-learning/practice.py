# PRACTICING UNSUPERVISED LEARNING - EXPLORING AIR QUALITY DATA IN INDIA (2015-2020) 

# LOADING LIBRARIES
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import requests, zipfile
import io, os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#-------------------------------------------------
# LOADING THE CLEANED DATA 
base = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base, '..', 'week03-pandas', 'aqi_cleaned.csv'))  
print(df.head())
print("DataFrame shape: ", df.shape)  # (24850, 9)

#-------------------------------------------------
# DATA PREPARATION
df_use = df[['PM2.5', 'PM10', 'NO2', 'AQI']]
print("DataFrame shape: ", df_use.shape)  # (24850, 4)
print(df_use.isnull().sum().sum())  # 7477 null values
df_use = df_use.dropna()
print("DataFrame shape: ", df_use.shape)  # (17506, 4)

print("Data types: \n",df_use.dtypes)
# Since all variables are numeric, we don't need to convert

#-------------------------------------------------
