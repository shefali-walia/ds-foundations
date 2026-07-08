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
# plt.scatter(X[:,0], X[:,1], color = 'black')
# plt.show()

kmeans = KMeans(init='random', n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)

merge_data = pd.concat([pd.DataFrame(X[:,0]), pd.DataFrame(X[:,1]), pd.DataFrame(y_pred)], axis = 1)
merge_data.columns = ['feature1', 'feature2', 'cluster']
ax = None
colors = ['blue', 'red', 'green']
for i, data in merge_data.groupby('cluster'):
    ax = data.plot.scatter(x = 'feature1', y = 'feature2', color = colors[i], label = f'cluster{i}', ax = ax)
plt.show()
# automatically shows legend

#-------------------------------------------------
