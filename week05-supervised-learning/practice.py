# PRACTICING SUPERVISED LEARNING - EXPLORING AIR QUALITY DATA IN INDIA (2015-2020) 

# lOADING LIBRARIES
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#-------------------------------------------------
# LOADING THE CLEANED DATA 
base = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base, '..', 'week03-pandas', 'aqi_cleaned.csv'))  
print(df.head())
print("DataFrame shape: ", df.shape)  # (24850, 9)

#-------------------------------------------------
# MULTIPLE REGRESSION - PREDICT AQI FROM POLLUTANTS
# features: PM2.5, PM10, NO2
# target: AQI

# DATA PREPARATION;
df_use = df[['PM2.5', 'PM10', 'NO2', 'AQI']]
print("DataFrame shape: ", df_use.shape)  # (24850, 4)
print(df_use.isnull().sum().sum())  # 7477 null values
df_use = df_use.dropna()
print("DataFrame shape: ", df_use.shape)  # (17506, 4)

print("Data types: \n",df_use.dtypes)
# Since all variables are numeric, we don't need to convert

print(df_use.corr())  # Correlation check
# PM10 ans PM2.5 have high correlation ~ 0.83 
# can lead to multi-collinearity but here we're using for practice

# BUILDING MODEL:
X = df_use.drop('AQI', axis = 1)
y = df_use['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

model = LinearRegression()
model.fit(X_train, y_train)

# Coefficient of determination (R^2)
print(f"Coefficient of Determination (train) = {model.score(X_train, y_train):.3f}")
print(f"Coefficient of Determination (test) = {model.score(X_test, y_test):.3f}")

# Regression coefficents
print(f"Regression Coefficients: \n {pd.Series(model.coef_, index = X.columns)}")

# Intercept
print(f"Intercept: {model.intercept_ :.3f}")

# RESULT:
# train = 0.703, test = 0.727 => close enough => no overfitting
# PM10 has least regression coefficient (~0.53) => affects AQI least
# NO2 has highest regression coefficient (~0.90) => affects AQI most (per unit increase)
# Intercept = 16.143 = b value
# NO2 having the highest coefficient here doesn't contradict the heatmap (where PM10 had r=0.80)
# Higher coeff. => Stronger effect on target variable
# Multi-collinearity problem observed here due to strong correlation between PM2.5 and PM10 - both fighting to "explain" the same variance in AQI - model splits credit between them => deflation of coefficients
# NO2 is less correlated with PM2.5/PM10 (r=0.36), so it explains a different, independent slice of AQI variation — making its coefficient look larger in the regression

# correlation = how much X and AQI are related, ignoring everything else
# regression coefficient = how much X contributes after accounting for all other features 

#-------------------------------------------------
# RIDGE VS. LINEAR - DOES REGULARISATION HELP?

# X, y same as before - only looping through both models here
X = df_use.drop('AQI', axis = 1)
y = df_use['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

linear = LinearRegression()
ridge = Ridge(random_state = 0)

for model in [linear, ridge]:
    model.fit(X_train, y_train)
    print("{} (train): {:.6f}".format(model.__class__.__name__, model.score(X_train, y_train)))
    print("{} (test): {:.6f}".format(model.__class__.__name__, model.score(X_test, y_test)))

