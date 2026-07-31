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

df_use_with_clusters = pd.concat([df_use.reset_index(drop = True), labels], axis=1)
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
# PCA
# we have 3 numeric features (PM2.5, PM10, NO2) not taking AQI because usually AQI is calculated from these features
# compressing 3 --> 2 features
features = ['PM2.5', 'PM10', 'NO2']
X = df_use[features]

sc = StandardScaler()
X_std = sc.fit_transform(X)

pca = PCA(n_components=2)
pca.fit(X_std)
#  model actually "learns" from the data
# Nothing is transformed yet, it's just calculating the directions.

X_pca = pca.transform(X_std)
# This is a new array- same number of rows as before, but now only 2 columns (PC1, PC2) instead of 3

print(pca.explained_variance_ratio_) # check how much info you kept
# Variance captured: [0.739, 0.206] — PC1 captures about 74% of the spread, PC2 adds another 21%.
# Together that's ~94.5%, so almost nothing is lost by compressing 3 pollutants into 2 new directions.

print(pca.components_) # Check what PC1 is actually made of
# PC1 weights: [0.617, 0.617, 0.489] for PM2.5, PM10, NO2 resp.
# All three are positive and fairly similar in size — this means PC1 is basically an "overall pollution" axis
# PC2 weights: [-0.346, -0.345, 0.872] - interesting
# PM2.5 and PM10 both have similar negative weights, but NO2 has a large positive weight, and it's the dominant one
# This means PC2 is capturing the contrast between NO2 and particulate matter (PM2.5/PM10) — i.e., days where NO2 is elevated relative to particulates, or vice versa

# Result:
# PC1 = general pollution intensity (all three pollutants moving together)
# PC2 = NO2 vs. particulate contrast (likely distinguishing traffic-driven pollution from other sources)

# PLOTTING PC1 VS PC2
colors = ['blue', 'red', 'green', 'purple']

for i in range(4):
    plt.scatter(X_pca[df_use_with_clusters['Cluster Number'] == i, 0],
                X_pca[df_use_with_clusters['Cluster Number'] == i, 1],
                color = colors[i], label = f'cluster {i}', alpha=0.3)

plt.xlabel('PC1 (general pollution intensity)')
plt.ylabel('PC2 (NO2 vs particulate contrast)')
plt.legend()
plt.show()

#-------------------------------------------------
