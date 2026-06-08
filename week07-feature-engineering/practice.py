# PRACTICING FEATURE ENGINEERING AND ENSEMBLE LEARNING - EXPLORING AIR QUALITY DATA IN INDIA (2015-2020) 

# IMPORTING LIBRARIES
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

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
# LOG TRANSFORMATION ON PM2.5
# From Week4 histogram - PM2.5 is right-skewed (long tail towards right)
# Plot before and after histograms side by side and train log. reg. on both and compare F1 score

plt.figure(figsize = (8,6))
# Before chart
plt.subplot(1,2,1)
plt.hist(df_use['PM2.5'], bins = 30, edgecolor = 'black', range = (0,800))
plt.title("Without Log Transformation", fontsize = 10)
plt.xlabel("PM2.5")
plt.ylabel("Frequency")
plt.grid(True, linestyle = '--', alpha = 0.5)

df_use['PM2.5_log'] = np.log1p(df_use['PM2.5'])

# After chart
plt.subplot(1,2,2)
plt.hist(df_use['PM2.5_log'], bins = 30, edgecolor = 'black') 
plt.title("With Log Transformation", fontsize = 10)
plt.xlabel("log(1 + PM2.5)")
plt.grid(True, linestyle = '--', alpha = 0.5)

plt.show()
# Interpretation: skewness reduces, much more symmetric distribution after log transformation, compressed between range 2-6

# Training log. reg. models on both raw data and transformed data and comparing f1 scores

df_use['AQI_flg'] = df_use['AQI'].map(lambda x: 1 if x > 100 else 0)  # Binary target 

X_raw = df_use[['PM2.5']]
X_log = df_use[['PM2.5_log']]    # double brackets to pass X as Dataframe instead of series
y = df_use['AQI_flg']

X_raw_train, X_raw_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
X_log_train, X_log_test, _, _ = train_test_split(X_log, y, test_size=0.2, random_state=42)  # y is already split, dont need now, so discard those values by using _,_

model = LogisticRegression()
model.fit(X_raw_train, y_train)
y_pred = model.predict(X_raw_test)
f1_raw = f1_score(y_test, y_pred)
print(f"F1 score for raw: {f1_raw:.3f}")

model.fit(X_log_train, y_train)
y_pred = model.predict(X_log_test)
f1_log = f1_score(y_test, y_pred)
print(f"F1 score for log: {f1_log:.3f}")

# Result: 
# F1 score for raw = 0.862, for log = 0.857 = lower
# This is expected, Logistic Regression with a single feature on a large dataset already separates classes reasonably well
# Log transform helps more when extreme outliers are distorting a linear decision boundary across multiple features
# In a Smart City deployment, log-transforming PM2.5 still makes sense for model stability and interpretability, even if F1 gain is minimal here

#-------------------------------------------------
