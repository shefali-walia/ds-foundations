# PRACTICING MODEL EVALUATION - EXPLORING AIR QUALITY DATA IN INDIA (2015-2020) 

# IMPORTING LIBRARIES
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

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
# AQI BINARY CLASSIFICATION PROPER TRAIN/VAL/TEST SPLIT
# AQI > 100 = POOR

df_use['AQI_flg'] = df_use['AQI'].map(lambda x: 1 if x > 100 else 0)
X = df_use[['PM2.5', 'PM10', 'NO2']]
y = df_use['AQI_flg']

# first 20% as test kept aside
X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size= 0.2, stratify = y, random_state= 42)

# rest 80% into 70% train and 10% val (1/8 of 80%)
X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size= 1/8, stratify= y_rest, random_state= 42)

# Standardization
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_val_std = sc.transform(X_val)
X_test_std = sc.transform(X_test)

model = LogisticRegression()
model.fit(X_train_std, y_train) # train on train data

print("Train Accuracy: {:.3f}".format(model.score(X_train_std, y_train)))
print("Validation Accuracy: {:.3f}".format(model.score(X_val_std, y_val)))  
print("Test Accuracy: {:.3f}".format(model.score(X_test_std, y_test)))
# usually you first compare validation accuracies to decide between models or hyperparameters and then final evaluation using test data
# but here main focus was on the split for practice, so validation set is just another checkpoint for checking if model generalises well

# Result: 
# train = 0.881, val = 0.872, test = 0.892
# No overfitting, consistent accuracies
# test slightly greater than train (1.1%) => can be random variation in data, test may be easier set

#-------------------------------------------------
# CONFUSION MATRIX
# Using the log. reg. model made above
y_pred = model.predict(X_test_std)
m = confusion_matrix(y_test, y_pred)
print(f'Confusion matrix: \n{m}')
# here 1(positive) means poor and 0(negative) means good/moderate AQI
# Result:
# TP = 1752, FN = 201, FP = 177, TN = 1372
# here FN i.e. wrongly predicted as good/moderate AQI is more dangerous error because it might affect people with respiratory diseases and also prevents actions by govt. and industries to manage the AQI levels
# Smart city context - Govt. can predict AQI levels and categorize them by area to take preventive measures, ban harmful effluents etc. timely before situation gets worse

#-------------------------------------------------
# CALCULATING METRICS MANUALLY

# accuracy = (TP+TN) / (TP+TN+FP+FN)
accuracy = (m[0,0] + m[1,1])/ m.sum()
print(f'Accuracy: {accuracy:.3f}')

# precision = TP / (TP+FP)
precision = m[1,1] / m[:,1].sum()
print(f'Precision: {precision:.3f}')

# recall = TP/ (TP+FN)
recall = m[1,1] / m[1,:].sum()
print(f'Recall: {recall:.3f}')

# F1 score = 2/((1/recall) + (1/precision))
f1 = 2 * (precision*recall)/(precision+recall)
print(f'F1 score: {f1:.3f}')

#-------------------------------------------------
# VERIFYING METRICS WITH SKLEARN
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Verification using sklearn: ")
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 score: {f1:.3f}')