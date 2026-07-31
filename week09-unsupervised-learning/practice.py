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
# K-MEANS + ELBOW METHOD

# scale first becuase features need to be on the same scale because kmeans measures distance
sc = StandardScaler()
X_scaled = sc.fit_transform(df_use[['PM2.5', 'NO2', 'PM10']])

# create and fit the model
inertia_list  = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state= 42)
    kmeans.fit(X_scaled)  # train the model
    inertia_list.append(kmeans.inertia_)  # inertia_ attribute holds the "tightness" score

plt.plot(range(1,11), inertia_list, marker= '+')
plt.xlabel('No. of clusters')
plt.ylabel('Distortion')
plt.show()
# result: elbow comes around k=3 or 4, depends on judgement- wherever the curve stops falling steeply and starts flattening into a long tail
# can take 3 or 4 both 

# FIT + MERGE 
kmeans = KMeans(n_clusters=4, random_state= 42)
kmeans.fit(X_scaled)

labels = pd.Series(kmeans.labels_, name='Cluster Number')
print(labels.value_counts(sort=False))

df_use_with_clusters = pd.concat([df_use, labels], axis=1)
print(df_use_with_clusters.head())

#-------------------------------------------------
# CROSS-TAB CLUSTERS VS SEVERITY
bins = [0,50,100,200,800]
bin_labels = ['Good', 'Moderate', 'Unhealthy', 'Hazardous']
df_use_with_clusters['aqi_bin'] = pd.cut(df_use_with_clusters['AQI'], bins = bins, labels = bin_labels)
print(df_use_with_clusters.head())
print(df_use_with_clusters.groupby(['Cluster Number', 'aqi_bin']).size().unstack())
# result: Cluster 1 = most unhealthy - hazardous days

#-------------------------------------------------
