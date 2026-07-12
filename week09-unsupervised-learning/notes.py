# WEEK 09 - UNSUPERVISED LEARNING
# GCI TOKYO UNIVERSITY 2026

# TOPICS COVERED: Clustering, k-means method, clustering financial market data, Elbow method, other methods, PCA, Market basket analysis, Association rules

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

#-------------------------------------------------
# CLUSTERING AND K-MEANS METHOD
# Clustering = divide given data into groups (aka flocks or clusters) with high similarity
# no target features 
# finds similar patterns in the data and group them as clusters

# k = no. of clusters 
# k-means - very fast, useful for large datasets
# k-means randomly places the initial centroids, which can lead to bias in their starting positions
# drawback - initial placement of cluster centre is chosen randomly => may lead to poor results if intial centres are too close to each other 

# so k-means ++ is used to place initial means as far apart as possible
# k-means ++ is default in many libraries including scikit-learn if you omit init='random' initialisation parameter

# k-medoids method- uses the median instead of the mean for the centroid - less affected by outliers
# the mean may result in a non-existing data point, k-medoids uses the median, preventing the centroid from taking imaginary values

# make_blobs fnct. generates random nos. following a normal distribution with std. dev. 1 along the vertical and horizontal axes - generate sample data for clustering
# by default it generates 100 random nos. in range 0 to 10 without any arguments
# make_blobs returns two values, one of which we don't use, so it's received as "_"

X, _ = make_blobs(random_state= 10)
# # plt.scatter(X[:,0], X[:,1], color = 'black')
# # plt.show()

# kmeans = KMeans(init='random', n_clusters=3)
# kmeans.fit(X)
# y_pred = kmeans.predict(X)

# merge_data = pd.concat([pd.DataFrame(X[:,0]), pd.DataFrame(X[:,1]), pd.DataFrame(y_pred)], axis = 1)
# merge_data.columns = ['feature1', 'feature2', 'cluster']
# ax = None
# colors = ['blue', 'red', 'green']
# for i, data in merge_data.groupby('cluster'):
#     ax = data.plot.scatter(x = 'feature1', y = 'feature2', color = colors[i], label = f'cluster{i}', ax = ax)
# plt.show()
# automatically shows legend

#-------------------------------------------------
# CLUSTERING FINANCIAL MARKETING DATA
url = "http://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
r = requests.get(url, stream = True)
base = os.path.dirname(os.path.abspath(__file__))

z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(os.path.join(base, "bank+marketing"))

nested_zip_path = os.path.join(base, "bank+marketing", "bank.zip")
with zipfile.ZipFile(nested_zip_path) as z2:
    z2.extractall(os.path.join(base, "bank+marketing", "bank"))

bank = pd.read_csv(os.path.join(base, 'bank+marketing','bank', 'bank-full.csv'), sep=';')
print(bank.head())
print("Dataset shape: ", bank.shape)  #(45211, 17)
print('Number of missing values: {}'.format(bank.isnull().sum().sum()))  # 0 missing values 

bank_sub = bank[['age','balance','campaign','previous']]
sc = StandardScaler()
sc.fit(bank_sub)
bank_sub_std = sc.transform(bank_sub)

bank_sub.info()

kmeans = KMeans(init = 'random', n_clusters=6, random_state=0)
kmeans.fit(bank_sub_std)

labels = pd.Series(kmeans.labels_, name='cluster_number')
print(labels.value_counts(sort=False))

# ax = labels.value_counts(sort=False).plot(kind='bar')
# ax.set_xlabel('cluster number')
# ax.set_ylabel('count')
# plt.show()

# ELBOW METHOD 
# determine the appropriate number of clusters
# by analyzing how the sum of distances between the cluster centroids and the data points within each cluster change accordingly with the number of clusters
# As no. of clusters increase from 1, this sum is expected to decrease (since each point will be assigned to a closer centroid)
# It will reach an appropriate no. at some time 
# after which rate of decrease in this sum is expected to slow down
# this method looks at this CHANGE IN RATE OF DECREASE to determine appropriate no. of clusters
# elbow name because of shape of graph

dist_list = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init='random', random_state=0)
    kmeans.fit(X)
    dist_list.append(kmeans.inertia_)

plt.plot(range(1,10), dist_list, marker='+')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# INTERPRETING CLUSTERING RESULTS
bank_with_cluster = pd.concat([bank, labels], axis=1)
print(bank_with_cluster.head())

bins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100]
qcut_age = pd.cut(bank_with_cluster.age, bins, right=False)
df = pd.concat([bank_with_cluster.cluster_number, qcut_age], axis=1)
cross_cluster_age = df.groupby(['cluster_number', 'age']).size().unstack().fillna(0)
print(cross_cluster_age)

hist_age = pd.value_counts(qcut_age)
print(hist_age)

sns.heatmap(cross_cluster_age.apply(lambda x : x/x.sum(), axis=1), cmap='Blues')
plt.show()

cross_cluster_job = bank_with_cluster.groupby(['cluster_number', 'job']).size().unstack().fillna(0)
print(cross_cluster_job)

sns.heatmap(cross_cluster_job.apply(lambda x : x/x.sum(), axis=1),cmap='Reds')
plt.show()
