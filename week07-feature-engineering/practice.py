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

# plt.figure(figsize = (8,6))
# # Before chart
# plt.subplot(1,2,1)
# plt.hist(df_use['PM2.5'], bins = 30, edgecolor = 'black', range = (0,800))
# plt.title("Without Log Transformation", fontsize = 10)
# plt.xlabel("PM2.5")
# plt.ylabel("Frequency")
# plt.grid(True, linestyle = '--', alpha = 0.5)

# df_use['PM2.5_log'] = np.log1p(df_use['PM2.5'])

# # After chart
# plt.subplot(1,2,2)
# plt.hist(df_use['PM2.5_log'], bins = 30, edgecolor = 'black') 
# plt.title("With Log Transformation", fontsize = 10)
# plt.xlabel("log(1 + PM2.5)")
# plt.grid(True, linestyle = '--', alpha = 0.5)

# plt.show()
# Interpretation: skewness reduces, much more symmetric distribution after log transformation, compressed between range 2-6

# Training log. reg. models on both raw data and transformed data and comparing f1 scores

df_use['AQI_flg'] = df_use['AQI'].map(lambda x: 1 if x > 100 else 0)  # Binary target 

# X_raw = df_use[['PM2.5']]
# X_log = df_use[['PM2.5_log']]    # double brackets to pass X as Dataframe instead of series
# y = df_use['AQI_flg']

# X_raw_train, X_raw_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
# X_log_train, X_log_test, _, _ = train_test_split(X_log, y, test_size=0.2, random_state=42)  # y is already split, dont need now, so discard those values by using _,_

# model = LogisticRegression()
# model.fit(X_raw_train, y_train)
# y_pred = model.predict(X_raw_test)
# f1_raw = f1_score(y_test, y_pred)
# print(f"F1 score for raw: {f1_raw:.3f}")

# model.fit(X_log_train, y_train)
# y_pred = model.predict(X_log_test)
# f1_log = f1_score(y_test, y_pred)
# print(f"F1 score for log: {f1_log:.3f}")

# Result: 
# F1 score for raw = 0.862, for log = 0.857 = lower
# This is expected, Logistic Regression with a single feature on a large dataset already separates classes reasonably well
# Log transform helps more when extreme outliers are distorting a linear decision boundary across multiple features
# In a Smart City deployment, log-transforming PM2.5 still makes sense for model stability and interpretability, even if F1 gain is minimal here

#-------------------------------------------------
# ONE-HOT ENCODING ON SEASON
# One-hot encoding on Season column (Winter/Summer/Monsoon/Post-Monsoon)

# season_dummies = pd.get_dummies(df['Season']).astype(int)   #get_dummies returns a new dataframe so join it back to original df
# df = pd.concat([df, season_dummies], axis=1)
# print(df.head())
# took df here because df_use doesn't have season column

# Why not label encoding for Season?
# Label encoding assigns integers (e.g. Winter=0, Summer=1, Monsoon=2, Post-Monsoon=3)
# which implies an ordering - as if Winter < Summer < Monsoon mathematically.
# Seasons have no such ranking, the model would learn a false relationship
# One-hot encoding creates separate binary columns so each season is treated as independent - no ordering, no false magnitude differences
# Season is nominal (categories with no order). Label encoding is for ordinal data only
# Label encoding implies ordinal relationship where none exists

#-------------------------------------------------
# CROSS FEATURE - PM2.5 * PM10
# Use polynomialfeatures when you have 5+ columns - that is the automated version and you want every possible pair
# But when you know exactly which interaction you want, you just multiply the two columns directly

# df_use['PM_interaction'] = df_use['PM2.5'] * df_use['PM10']
# print(df_use.head())

# # Training Linear reg. model with and without PM_interaction

# X_without = df_use[['PM2.5', 'PM10', 'NO2']]
# X_with = df_use[['PM2.5', 'PM10', 'NO2', 'PM_interaction']]
# y = df_use['AQI']  #Linear reg. predicts continuous values so y should be actual AQI no., not AQI_flg binary column

# X_without_train, X_without_test, y_train, y_test = train_test_split(X_without, y, test_size= 0.2, random_state= 42)
# X_with_train, X_with_test, _,_ = train_test_split(X_with, y, test_size= 0.2, random_state= 42)  # don't use stratify for regression (it works only for classification)

# model = LinearRegression()

# model.fit(X_without_train, y_train)
# print('R^2 score without PM_interaction: {:.3f}'.format(model.score(X_without_test, y_test))) 
# model.fit(X_with_train, y_train)
# print('R^2 score with PM_interaction: {:.3f}'.format(model.score(X_with_test, y_test)))

# R^2 improved marginally from 0.733 to 0.734 with PM_interaction
# The small gain makes sense — PM2.5 and PM10 are already highly correlated, so their product doesn't add much new information the model didn't already have
# However, the interaction term can still be meaningful in a Smart City context:
# days where BOTH PM2.5 and PM10 are simultaneously high represent compound pollution events (e.g. dust storms + vehicle emissions) that are more dangerous than either pollutant alone
# A small R² gain doesn't mean the feature is useless - it means linear models can't fully capture that non-linear relationship
# Tree-based models may extract more value from this feature

#-------------------------------------------------
# USING DOMAIN KNOWLEDGE 
# creating two new features in df_use 
df_use['is_winter'] = (df['Season'] == 'Winter').astype(int)
df_use['high_particulate'] = ((df_use['PM2.5'] > 100) & (df_use['PM10'] > 150)).astype(int)

X_old = df_use[['PM2.5', 'PM10', 'NO2']]
X_new = df_use[['PM2.5', 'PM10', 'NO2', 'is_winter', 'high_particulate']]
y = df_use['AQI_flg']  # log reg. => binary column

X_old_train, X_old_test, y_train, y_test = train_test_split(X_old, y, test_size= 0.2, stratify= y, random_state= 42)
X_new_train, X_new_test, _, _ = train_test_split(X_new, y, test_size= 0.2, stratify= y, random_state= 42)

model = LogisticRegression()

model.fit(X_old_train, y_train)
print('Accuracy without new features: {:.3f}'.format(model.score(X_old_test, y_test)))
model.fit(X_new_train, y_train)
print('Accuracy with new features: {:.3f}'.format(model.score(X_new_test, y_test)))

# Accuracy unchanged at 0.891 with domain features added
# is_winter and high_particulate didn't add new information because Log Reg. can already derive these patterns from the raw PM2.5, PM10 values directly 
# the threshold logic (PM2.5 > 100, PM10 > 150) is something the model learns on its own from continuous values
# Domain features tend to help more with tree-based models where explicit thresholds create cleaner split points
# In a real Smart City pipeline, is_winter still has value for interpretability — a policy report saying "winter increases bad air probability" is more actionable than a model coefficient

#-------------------------------------------------
