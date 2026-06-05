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
from sklearn.svm import SVC

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

# RESULT:
# LinearRegression: train=0.7029, test=0.7269
# Ridge:            train=0.7029, test=0.7269
# Scores are identical — Ridge made no difference here
# This is because the AQI dataset is clean with low multicollinearity impact
# In general: Linear tends to score higher on train, Ridge on test (better generalization)
# Ridge would matter more with noisier data or more features

#-------------------------------------------------
# LOGISTIC REGRESSION - CLASSIFY AQI AS GOOD/POOR
df_use['AQI_flg'] = df_use['AQI'].map(lambda x: 1 if x > 100 else 0)
# creates a binary column, 1 = Poor or worse, 0 = Good/Moderate

X = df_use[['PM2.5', 'PM10', 'NO2']]
y = df_use['AQI_flg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Without standardization: 
model = LogisticRegression()
model.fit(X_train, y_train)

print("Before standardization: ")
print('Accuracy (train):{:.3f}'.format(model.score(X_train, y_train)))
print('Accuracy (test):{:.3f}'.format(model.score(X_test, y_test)))

# With standardization: 
# upto data splitting steps are same
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

model = LogisticRegression()
model.fit(X_train_std, y_train)

print("After standardization: ")
print('Accuracy (train):{:.3f}'.format(model.score(X_train_std, y_train)))
print('Accuracy (test):{:.3f}'.format(model.score(X_test_std, y_test)))

# RESULT:
# Before standardization: train=0.886, test=0.878
# After standardization:  train=0.886, test=0.879
# no overfitting, train n test scores pretty close
# standardisation helped increase test scores slightly
# Would prefer to use Logistic regression model over SVM for a Smart City govt. project because they require EXPLAINABILITY - justify decisions to policymakers, not just show accuracy
# Log. reg. coefficients can be converted to odd ratios and explained in simple language
# SVM is a black box — high accuracy but you can't explain why it classified something as Poor AQI, which won't be useful in public sector AI

#-------------------------------------------------
# DECISION TREE - CLASSIFY AQI AS GOOD/POOR
# Same X and y as previous task

X = df_use[['PM2.5', 'PM10', 'NO2']]
y = df_use['AQI_flg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

 
model = DecisionTreeClassifier( criterion = 'entropy', max_depth = 5, random_state = 0)
model.fit(X_train, y_train)

print('Accuracy (train):{:.3f}'.format(model.score(X_train, y_train)))
print('Accuracy (test):{:.3f}'.format(model.score(X_test, y_test)))

# RESULT: 
# Decision tree: train = 0.892, test = 0.874 (1.8% gap)
# Log. reg. after standardization:  train=0.886, test=0.879 (0.7 % gap)
# Decision tree has higher accuracy for training data but slightly lower for test data than logistic regression with standardisation
# Decision tree train vs. test gap is slightly larger than log. reg. suggesting mild overfitting
# For Smart city govt. project, will use Log. regression with standardization because it has higher accuracy on test data and is also explainable
# Although decision tress are great to explain visually  

#-------------------------------------------------
# K-NN - FIND OPTIMAL K 
# loop k 1 to 20, same X and y

X = df_use[['PM2.5', 'PM10', 'NO2']]
y = df_use['AQI_flg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify = y, random_state=0)
# test_size controls how much data goes to test (e.g. 0.5 = 50/50)
# stratify controls how the split happens — it ensures the class ratio in train and test matches the original dataset
# use both independently - good practice to use both for classification tasks

training_accuracy = []
test_accuracy = []

for n_neighbours in range(1,21):
    model = KNeighborsClassifier(n_neighbors = n_neighbours)
    model.fit(X_train, y_train)
    training_accuracy.append(model.score(X_train, y_train))
    test_accuracy.append(model.score(X_test, y_test))

plt.plot(range (1,21), training_accuracy, label =  'Training')
plt.plot( range(1,21), test_accuracy, label = 'Test')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# RESULT: 
# K = 14-20 is optimal, test and train converge
# for k = 1 => overfitting because training accuracy = 100% but test accuracy only ~84% => model memorized training data too well so test accuracy drops
# for k from 1 to 7.5 => overfitting reduces as training accuracy decreases and test increases => model becomes less sensiticve to noise and smoother but train accuracy still a bit unstable and test accuracy still going up
# for k = 14-20 => best generalisation as gap between train and test accuracy becomes very small and test accuracy stabilises more (~88%) => model is balancing bias and variance well 
# If k chosen by convergence approach (where test and train converge ~14-20), it prioritizes stability, minimizes overfitting risk 
# Confirmed by GridSearchCV in Task 7 — best k=16 falls in this range
# Another approach - to see where peak test accuracy is (k=7-8) but prioritises raw performance on unseen data but not stability

#-------------------------------------------------
# CROSS-VALIDATION: REPLACE HOLDOUT ON DECISION TREE

X = df_use[['PM2.5', 'PM10', 'NO2']]
y = df_use['AQI_flg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

 
tree = DecisionTreeClassifier( criterion = 'entropy', max_depth = 5, random_state = 0)
scores = cross_val_score(tree, X, y, cv = 5)

print('Cross validation scores: {}'.format(scores))
print('Cross validation scores: {:.3f}+-{:.3f}'.format(scores.mean(), scores.std()))

# RESULT:
# Std. = 0.026 => pretty small => stable model across folds
# 0.873 +- 0.026 means real-world performance likely falls between 0.847 - 0.899
# Range of scores is around 0.83-0.91 => pretty small range => accurate model
# mean = 0.873 => slightly less than test accuracy of logistic regression model (0.879) => Logistic Regression remains the better choice for this task

#-------------------------------------------------
# GRIDSEARCH CV - TUNE k-NN
X = df_use[['PM2.5', 'PM10', 'NO2']]
y = df_use['AQI_flg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify = y, random_state=0)

param_grid = {'n_neighbors': range(1,21)}

gs = GridSearchCV(
    estimator= KNeighborsClassifier(),
    param_grid= param_grid,
    cv = 5
)
gs.fit(X_train, y_train)

print('Best CV score: {:.3f}'.format(gs.best_score_))
print('Best params: {}'.format(gs.best_params_))
print('Test score: {:.3f}'.format(gs.score(X_test, y_test)))

# RESULT:
# CV score = 0.880, test score = 0.882 => pretty close, no overfitting
# best param. = 16 = optimal k according to grid search (coinciding in the 14-20 convergence range)
# Test score = 0.882 is almost same as that of manual k (~0.881)

#-------------------------------------------------
# SUMMARY AND FINDINGS

# TASK 1 - MULTIPLE REGRESSION (predict AQI):
# R² train=0.703, test=0.727 — no overfitting, good generalization
# NO2 has highest regression coefficient (0.909) — strongest per-unit effect on AQI
# Multicollinearity between PM2.5 and PM10 (r=0.83) deflates their individual coefficients

# TASK 2 - RIDGE VS LINEAR:
# Scores identical (0.703/0.727) — Ridge made no difference on this clean dataset
# Ridge matters more with noisy data or many correlated features

# TASK 3 - LOGISTIC REGRESSION (classify AQI > 100):
# Before scaling: train=0.886, test=0.878
# After scaling:  train=0.886, test=0.879 — marginal improvement
# ~88% accuracy — strong classifier for binary AQI task

# TASK 4 - DECISION TREE:
# train=0.892, test=0.874 — slightly larger gap than Logistic Reg (mild overfitting)
# Logistic Regression preferred for Smart City govt projects — explainable + higher test accuracy
# Decision Trees useful for visual explanation to non-technical stakeholders

# TASK 5 - k-NN:
# k=1 => overfitting (train=1.0, test=0.84)
# Optimal k=7-8 (peak test accuracy ~0.881)
# k=14-20 => stable convergence but no accuracy gain over k=7-8

# TASK 6 - CROSS VALIDATION (Decision Tree, cv=5):
# mean=0.873, std=0.026 — stable model, small variance across folds
# Slightly underperforms Logistic Regression (0.879) — LR remains best model

# TASK 7 - GRIDSEARCHCV (k-NN):
# Best k=16, CV score=0.880, test score=0.882 — no overfitting
# Grid search confirmed manual observation (k=14-20 stable range)
# Test score (0.882) matches manual best (~0.881) — consistent result

# MODEL RANKING FOR THIS TASK:
# 1. Logistic Regression (test=0.879, explainable) — best for deployment
# 2. k-NN k=16 (test=0.882, slight edge in accuracy)
# 3. Decision Tree (test=0.874, mild overfitting)
