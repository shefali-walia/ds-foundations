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
from sklearn.decomposition import PCA

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
# # CLUSTERING FINANCIAL MARKETING DATA
# url = "http://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
# r = requests.get(url, stream = True)
# base = os.path.dirname(os.path.abspath(__file__))

# z = zipfile.ZipFile(io.BytesIO(r.content))
# z.extractall(os.path.join(base, "bank+marketing"))

# nested_zip_path = os.path.join(base, "bank+marketing", "bank.zip")
# with zipfile.ZipFile(nested_zip_path) as z2:
#     z2.extractall(os.path.join(base, "bank+marketing", "bank"))

# bank = pd.read_csv(os.path.join(base, 'bank+marketing','bank', 'bank-full.csv'), sep=';')
# print(bank.head())
# print("Dataset shape: ", bank.shape)  #(45211, 17)
# print('Number of missing values: {}'.format(bank.isnull().sum().sum()))  # 0 missing values 

# bank_sub = bank[['age','balance','campaign','previous']]
# sc = StandardScaler()
# sc.fit(bank_sub)
# bank_sub_std = sc.transform(bank_sub)

# bank_sub.info()

# kmeans = KMeans(init = 'random', n_clusters=6, random_state=0)
# kmeans.fit(bank_sub_std)

# labels = pd.Series(kmeans.labels_, name='cluster_number')
# print(labels.value_counts(sort=False))

# # ax = labels.value_counts(sort=False).plot(kind='bar')
# # ax.set_xlabel('cluster number')
# # ax.set_ylabel('count')
# # plt.show()

# # ELBOW METHOD 
# # determine the appropriate number of clusters
# # by analyzing how the sum of distances between the cluster centroids and the data points within each cluster change accordingly with the number of clusters
# # As no. of clusters increase from 1, this sum is expected to decrease (since each point will be assigned to a closer centroid)
# # It will reach an appropriate no. at some time 
# # after which rate of decrease in this sum is expected to slow down
# # this method looks at this CHANGE IN RATE OF DECREASE to determine appropriate no. of clusters
# # elbow name because of shape of graph

# dist_list = []
# for i in range(1,10):
#     kmeans = KMeans(n_clusters=i, init='random', random_state=0)
#     kmeans.fit(X)
#     dist_list.append(kmeans.inertia_)

# plt.plot(range(1,10), dist_list, marker='+')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.show()

# # INTERPRETING CLUSTERING RESULTS
# bank_with_cluster = pd.concat([bank, labels], axis=1)
# print(bank_with_cluster.head())

# bins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100]
# qcut_age = pd.cut(bank_with_cluster.age, bins, right=False)
# df = pd.concat([bank_with_cluster.cluster_number, qcut_age], axis=1)
# cross_cluster_age = df.groupby(['cluster_number', 'age']).size().unstack().fillna(0)
# print(cross_cluster_age)

# hist_age = pd.value_counts(qcut_age)
# print(hist_age)

# sns.heatmap(cross_cluster_age.apply(lambda x : x/x.sum(), axis=1), cmap='Blues')
# plt.show()

# cross_cluster_job = bank_with_cluster.groupby(['cluster_number', 'job']).size().unstack().fillna(0)
# print(cross_cluster_job)

# sns.heatmap(cross_cluster_job.apply(lambda x : x/x.sum(), axis=1),cmap='Reds')
# plt.show()

#-------------------------------------------------
# PRINCIPAL COMPONENT ANALYSIS (PCA)
# Compress the number of variables without losing much of the information in the original data
# Preprocessing step
# Here it is unsupervised dimensionality reduction, whereas supervised dimensionality reduction (ex. Linear Discriminant Analysis) is also available

# # creating a randomstate object with seed no. 1
# sample = np.random.RandomState(1)

# # generating two random nos.
# X = np.dot(sample.rand(2,2), sample.rand(2,200)).T

# sc = StandardScaler()
# X_std = sc.fit_transform(X)
# print('Correlation coefficient: {:.3f}'.format(sp.stats.pearsonr(X_std[:,0], X_std[:,1])[0]))
# plt.scatter(X_std[:,0], X_std[:,1])
# plt.show()

# # PCA
# pca = PCA(n_components=2)
# pca.fit(X_std)

# # Confirming the learning results of PCA: 

# print(pca.components_)  # eigenvectors - represent the directions of the new feature space discovered by PCA
# # vector [0.707, 0.707] corresponds to the first principal component (PC1)
# # vector [-0.707, 0.707] corresponds to the second principal component (PC2)

# print('Variance of each principal component: {}'.format(pca.explained_variance_))
# # variance of each attribute
# # sum of the variances of the principal components (2) matches the sum of the original variances of the (standardized) variables. 
# # This means that the variance (information) is preserved.

# print('Proportion of variance for each principal component: {}'.format(pca.explained_variance_ratio_))
# # proportion of variance captured by each principal component

# # first principal component is aligned in the direction of maximum variance
# # The other principal components are all orthogonal to its preceding principal components

# # Visualising the principal components:
# # Parameter settings
# arrowprops=dict(arrowstyle='->',
#                 linewidth=2,
#                 shrinkA=0, shrinkB=0)

# # Function to draw arrows
# def draw_vector(v0, v1):
#     plt.gca().annotate('', v1, v0, arrowprops=arrowprops)

# # Plotting the data
# plt.scatter(X_std[:, 0], X_std[:, 1], alpha=0.2)

# # Displaying the principal components
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector * 3 * np.sqrt(length)
#     draw_vector(pca.mean_, pca.mean_ + v)

# plt.axis('equal')
# plt.show()

# correlation coefficients between each principal component and the original explanatory variables are referred to as factor loadings

#-------------------------------------------------
# MARKET BASKET ANALYSIS/ ASSOCIATION ANALYSIS
# analyzes the relationships between products purchased together
# such as the likelihood of purchasing product B when product A is bought
# named after the "basket" of items that pass through a supermarket checkout counter, which is the basic unit of analysis
# Results of market basket analysis are known as Association rules - describe relationship between products

url = "http://archive.ics.uci.edu/static/public/352/online+retail.zip"
response = requests.get(url, stream = True)
base = os.path.dirname(os.path.abspath(__file__))
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall(os.path.join(base, "online+retail"))

file_url = os.path.join(base, "online+retail", "Online Retail.xlsx")
trans = pd.ExcelFile(file_url)
trans = trans.parse('Online Retail')
print(trans.head())

# Adding the first character of InvoiceNo as cancel_flg
trans['cancel_flg'] = trans.InvoiceNo.map(lambda x:str(x)[0])

print(trans.groupby('cancel_flg').size())
trans = trans[(trans.cancel_flg == '5') & (trans.CustomerID.notnull())]

trans = trans.astype({'StockCode': 'str'})
print(trans['StockCode'].value_counts().head(5))

# SUPPORT: support of an association rule refers to the number of baskets (InvoiceNos) in which both a particular product and another product were sold together, or the proportion of the total transactions in which this occurs

trans_all = set(trans.InvoiceNo)   # set as in math sets
trans_a = set(trans[trans['StockCode'] == '85123A'].InvoiceNo) #purchase data for item 1 as trans_a
print(len(trans_a))
trans_b = set(trans[trans['StockCode'] == '85099B'].InvoiceNo) #purchase data for item 1 as trans_a
print(len(trans_b))

trans_ab = trans_a & trans_b  # & = intersection of sets
print(len(trans_ab))

print('Number of baskets containing both items:{}'.format(len(trans_ab)))
print('Proportion of baskets containing both items to the total:{:.3f}'.format(len(trans_ab)/len(trans_all))) # calculation of support
# whether support is high or low depends on relative comparision 
# In general, rules with low support are often less useful so support may be used as a cut-off criterion
# support can also be calculated for each item 

# CONFIDENCE : proportion of purchases of a certain item A that also include a certain item B

# confidence for item 85123A is purchased, then item 85099B is also purchased
print('Confidence:{:.3f}'.format(len(trans_ab)/len(trans_a)))

# confidence for if item 85099B is purchased, then item 85123A is also purchased
print('Confidence:{:.3f}'.format(len(trans_ab)/len(trans_b)))

# High confidence => cross-selling between products is likely
# But relying solely on confidence values can sometimes lead to incorrect judgments about cross-selling tendencies so lift is also used.

# LIFT : ratio of the purchase rate of item B when item A is purchased to the overall purchase rate of item B across all baskets
# if the lift value is greater than 1.0, it indicates that cross-selling between the items is more likely
#  If the lift value is less than 1.0, it suggests that cross-selling is less likely

# lift for If item 85123A is purchased, then item 85099B is also purchased
support_b = len(trans_b) / len(trans_all)
confidence = len(trans_ab) / len(trans_a)
lift = confidence / support_b
print('lift: {:.3f}'.format(lift))
# Even if the confidence value is high, if the lift value is below 1.0, it may not be suitable as a basis for recommending products to customers

#-------------------------------------------------
# SUMMARY
# WHY UNSUPERVISED LEARNING IS DIFFERENT
# no target variable - just finding structure that already exists in the data
# goal is discovering patterns, not predicting an outcome

# CLUSTERING
# k-means - fast, but random initial centroid placement can lead to poor results if centroids start too close together
# k-means++ - default in sklearn, spaces out initial centroids instead of placing them randomly
# k-medoids - uses median instead of mean for centroid, more robust to outliers, centroid is always a real data point (never imaginary)
# make_blobs - generates sample data for clustering, returns X and labels (labels usually discarded, received as "_")

# ELBOW METHOD
# plots inertia (sum of distances to nearest centroid) against number of clusters
# inertia always decreases as k increases, but the RATE of decrease slows down at some point
# the "elbow" in the graph = appropriate number of clusters, not picked arbitrarily
# applied to bank marketing data - clusters interpreted by cross tabbing with age and job to understand what each cluster represents

# PCA (PRINCIPAL COMPONENT ANALYSIS)
# compresses variables into new directions (principal components) while preserving as much variance as possible
# preprocessing step - unsupervised dimensionality reduction (LDA is the supervised version)
# components_ - eigenvectors, the directions of the new feature space
# explained_variance_ - variance captured by each component, sums to match original variance = no information lost
# explained_variance_ratio_ - proportion of total variance each component captures
# first PC = direction of maximum variance, every other PC is orthogonal to the ones before it

# MARKET BASKET ANALYSIS
# support - proportion of all baskets (invoices) containing both item A and item B together
# confidence - of the baskets containing item A, what proportion also contain item B
# lift - confidence divided by the overall purchase rate of item B on its own
# lift > 1.0 = cross-selling is more likely than chance, lift < 1.0 = less likely than chance

# KEY INSIGHT
# high confidence alone can be misleading - always check lift before recommending cross-sells
# a strong looking rule can still be weaker than what random chance would already produce