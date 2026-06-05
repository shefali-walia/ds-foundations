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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

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

# We get same results => our calculation were correct
# Precision = 0.908 => of all AQI predicted as positive (poor) 90.8% was actually poor
# recall = 0.897 => of all AQI actuall poor, the model identified 89.7% correctly and missed 10.3% (significant)
# F1 = 0.903 => 90.3% overall => decently good model
# Precision > recall => model is slightly more conservative
# It misses some actual Poor AQI cases (FN=201) rather than over-predicting Poor
# For Smart City context, we'd actually WANT higher recall here - better to over-warn than under-warn about dangerous air quality
# So in a real deployment, we'd lower the classification threshold to boost recall even at the cost of slightly lower precision

#-------------------------------------------------
# ROC-AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_score = auc(fpr, tpr)

plt.plot(fpr, tpr, color = 'red', label = 'ROC curve (area = %.3f)' %auc_score)
plt.plot([0,1], [0,1], color = 'black', linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="best")
plt.show()

# Result:
# AUC = 0.891 => model has 89.1% ability to distinguish Poor vs Good/Moderate AQI
# Curve hugs top-left corner closely - well above the diagonal (random guessing, AUC=0.5)
# Sharp vertical rise at low FPR - model achieves ~90% TPR with only ~12% FPR
# Not perfect (AUC=1.0 would be a right angle) but strong for a 3-feature model
# For Smart City deployment: AUC of 0.891 means reliable early warning capability

#-------------------------------------------------
# REGRESSION METRICS
# Using multiple regression model from week5 
X = df_use.drop('AQI', axis = 1)
y = df_use['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'MSE: {mean_squared_error(y_test, y_pred):.3f}')
print(f'MAE: {mean_absolute_error(y_test, y_pred):.3f}')
print(f'RMSE: {root_mean_squared_error(y_test, y_pred):.3f}')
print(f'MedAE: {median_absolute_error(y_test, y_pred):.3f}')
print(f'R^2: {r2_score(y_test, y_pred):.3f}')

# Regression metrics interpretation:
# MSE: 2796.003 - pretty high => confirms large outliers 
# MAE: 28.527 - model's predictions are about 28 units far from true values without considering direction
# RMSE: 52.877 - typical error magnitude is ~53 AQI units, inflated by large outliers, not representative of typical prediction error
# MedAE: 18.594 - middle value of all absolute errors - typical prediction is only ~19 units off - better measure because not affected by large outliers
# R^2: 0.743 - model explains 74.3% of AQI variance - decent but 25.7% unexplained

# FOR GOVT. CLIENT: Report MAE and MedAE together
# MAE (28) = honest average error in AQI units - easy to explain ("on average we're off by 28")
# MedAE (18) = more representative of typical day performance - outliers don't distort it
# The gap between MAE and MedAE (28 vs 18) itself tells a story:
# most predictions are close but occasional large errors pull the average up
# Never report MSE/RMSE to non-technical clients - squared units mean nothing to a policymaker
# R^2 is useful for comparing models internally, not for communicating real-world accuracy

#-------------------------------------------------
# PREDICTED VS ACTUAL PLOT
# Plot Linear Regression vs Decision Tree Regressor for the AQI prediction task
X = df_use.drop('AQI', axis = 1)
y = df_use['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

def plot_pred_actual(model, title):
    y_pred = model.predict(X_test)
    sns.scatterplot(x=y_pred, y=y_test, alpha=0.4)
    plt.plot([0, 900], [0, 900], color='red', linestyle='dashed')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xlim(-50, 900)  # set as per data
    plt.ylim(0, 900)


linear = LinearRegression().fit(X_train, y_train)
tree = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plot_pred_actual(linear, 'LinearRegression')

plt.subplot(1, 2, 2)
plot_pred_actual(tree, 'DecisionTree')  

plt.tight_layout()
plt.show()

# Interpretation:
# LINEAR REGRESSION: points spread continuously along diagonal — good generalization
# Some scatter at high AQI (>500) where outliers exist — model underpredicts extreme values
# consistent with MAE=28, model is reasonably close across most of the range

# DECISION TREE: blocky/stepped pattern — classic overfitting signature
# Tree memorized training data ranges and outputs fixed values per leaf node
# instead of smooth predictions — explains why R^2 and MSE were worse
# Horizontal clusters visible at ~100, ~300, ~450 — those are the leaf node outputs

# BETTER MODEL: Linear Regression — smoother generalization, less memorization
# Despite looking "messier", its scatter is random (good) vs structured (bad for Decision Tree)

#-------------------------------------------------
# MODEL COMPARISION TABLE
results = {
    'Model': ['LogisticRegression', 'DecisionTree', 'kNN (k=16)', 
              'LinearRegression', 'Ridge'],
    'Task': ['Classification', 'Classification', 'Classification',
             'Regression', 'Regression'],
    'Accuracy/R^2': [0.879, 0.874, 0.882, 0.743, 0.743],
    'F1/MAE': [0.903, None, None, 28.527, 28.527],  
    'Notes': [
        'Best classifier — explainable, preferred for govt. deployment',
        'Mild overfitting, good for visual explanation',
        'Highest accuracy but black box',
        'Best regressor — smooth generalization',
        'Identical to Linear on this dataset'
    ]
}

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

#-------------------------------------------------
# SUMMARY AND FINDINGS
# Full model evaluation pipeline on India AQI data (2015-2020)
# Classification task: predict if AQI > 100 (Poor or worse)
# Regression task: predict exact AQI value from pollutant readings

# SPLIT STRATEGY:
# Proper 70/10/20 train/val/test split used (not the 50/50 shortcut from Week 5)
# Stratified splits throughout to maintain class balance

# CLASSIFICATION RESULTS (Logistic Regression):
# Accuracy=0.892, Precision=0.908, Recall=0.897, F1=0.903, AUC=0.891
# Strong model — 89.1% ability to distinguish Poor vs Good AQI across all thresholds
# FN (201 cases) more dangerous than FP in this context — missing bad AQI = no govt. action
# In real deployment: lower classification threshold to prioritize recall over precision

# REGRESSION RESULTS (Linear Regression):
# R^2=0.743, MAE=28.5, MedAE=18.6, RMSE=52.9
# Gap between MAE and MedAE (28 vs 18) confirms outliers — most predictions close,
# occasional large errors pull average up
# Linear Regression outperforms Decision Tree — smooth scatter vs blocky overfitting pattern

# KEY LESSONS:
# 1. Accuracy alone is misleading — always check precision, recall, F1 together
# 2. Which error is "worse" (FP vs FN) depends entirely on domain consequences
# 3. For government/public sector clients: report MAE + MedAE, never MSE/RMSE
# 4. Blocky predicted-vs-actual pattern = overfitting signature of tree-based models
# 5. AUC > accuracy for imbalanced data — measures performance across all thresholds
# 6. Explainability matters as much as accuracy in public sector AI deployment
