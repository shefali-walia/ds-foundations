# WEEK 06 - MODEL EVALUATION
# GCI TOKYO UNIVERSITY 2026

# TOPICS COVERED: 
# Evaluation metrics for classification models: Confusion matrix, Accuracy, Precision, Recall, F1, ROC curve, ROC-AUC
# Evaluation metrics for regression models: MSE, RMSE, MAE, MedAE

# LOADING LIBRARIES AND DATA
import numpy as np
import numpy.random as random
import pandas as pd
from pandas import Series, DataFrame
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix


# BUILDING THE MODEL AND GRID SEARCH 
cancer = load_breast_cancer(as_frame = True)
X_cancer = cancer.data
y_cancer = cancer.target

# splitting data to build classification model using SVM

# split off 20% of the whole data as test set
X_train_val, X_test, y_train_val, y_test = train_test_split(X_cancer, y_cancer, test_size = 0.2, stratify = y_cancer, random_state = 42)
# test_size = 0.2 => 20% to test 
# stratify - keep same class ratio in both train and test - especially imp. in medical data because classes are often imbalanced
# random_state = 42 => 42 is just a seed no., it can be any number
# 42 is a popular choice as a joke reference to the book "The Hitchhiker's Guide to the Galaxy" where 42 is the answer to "the meaning of life"
# random_state=42 and random_state=0 do the exact same thing
# The seed number itself just determines which specific split you get — not how random it is
# It's like a pseudo-random no. generator- seed number gets fed into a formula that produces a sequence of numbers that looks random but is fully deterministic 
# same seed = same sequence = same rows picked every time
# different seed = different sequence = different rows picked

# splitting the remaining 80% into train(70%) and validation(10%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 1/8, stratify = y_train_val, random_state= 42)
# 1/8 of the 80% = 10% of the total data for validation
# train 70% - model learns from this
# validation 10% - used during grid search to pick best hyperparameters
# test 20% - to evaluate final model
# if you tune hyperparameters on test data, you're essentially "cheating" — the model has indirectly seen the test set
# Earlier we just did 50-50 train and test because that was for simplicity just to learn the models, not evaluation
# Practically, ALWAYS use train-val-test split 70-10-20

# grid search
params_values = np.logspace(-3, 2, num = 6)
scores = {}
for gamma in params_values:
    for C in params_values:
        svm = SVC(gamma = gamma, C = C)
        svm.fit(X_train, y_train)    # for every combination of hyperparameters, train on X_train => model learns
        scores[(gamma, C)] = svm.score(X_val, y_val)   # then score on X_val => check how well the model generalises

# cannot score on train because model already saw that data, cannot score on test because that's kept untouched for final evaluation

scores = pd.Series(scores)
# after finishing the loop, convert the scores dictionary to pandas series

print(f"Best score: {scores.max():.2f}")
print(f"Paramter values (gamma, C) at best score: {scores.idxmax()}")

sns.heatmap(scores.unstack())
plt.show()
# heatmap - vertical axis shows gamma, and the horizontal axis shows C
# Best hyperparameter combo: gamma = 0.001 and C = 10.0, it got 95% accuracy on validation set

# training model using the best hyperparameter combo
model = SVC(gamma = scores.idxmax()[0], C = scores.idxmax()[1])
model.fit(X_train, y_train)

print(f"SVC train sccore: {model.score(X_train, y_train):.3f}")
print(f"SVC test sccore: {model.score(X_test, y_test):.3f}")
# overfitting seen as train score = 1.000. test = 0.877
# C controls how hard the model tries to correctly classify every training point
# C=10 is high, meaning the model was pressured to perfectly fit training data (told to try really hard!)
# in real projects, tune C further to close the train/test gap
# ideally you want a combo where validation score is high and the train/val gap is small

#-------------------------------------------------
# CONFUSION MATRIX
# predicted values y_pred in columns and observed values y_test in rows
y_pred = model.predict(X_test)
m = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix: \n{m}")

# Inpterpretation of confusion matrix: 
# 0= malignant, 1= benign
# 0 and 1 here denote negative and positive, can be assigned to any class
# total test samples = 39+3+11+61 = 114
# Top left = 39 => 39 out of 114 times observed (0)malignant, predicted (0)malignant => correct prediction for negative(0)=>True negative TN
# top right = 3 => 3 out of 114 times observed (0)malignant, predicted (1)benign => wrong prediction for positve => False positive FP
# bottom left = 11 => 11 out of 114 times observed (1) benign, predicted (0)malignant => wrong prediction for negative => False negative FN
# bottom right = 61 => 61 out of 114 times observed (1) benign, predicted (1)benign => correct prediction for positive => True positive TP
# Most dangerous here is False positive because then the patient goes untreated, false negative is also dangerous because then patient undergoes unnecessary treatment but it's not as much dangerous

#-------------------------------------------------
# ACCURACY
# The proportion of correct predictions out of the total predictions
# accuracy = (TP+TN) / (TP+TN+FP+FN)
# indicator of how well the model predicts both 1s and 0s overall

accuracy = (m[0,0] + m[1,1])/m.sum()
print(f"Accuracy: {accuracy:.3f}")
# result = 0.877 = same as model.score() because that is exactly what accuracy is
# hence we also need precision, recall, f1
# accuracy would be a misleading metric when the data is imbalanced

# Example 99% of emails are not spam, 1% are spam
# A model that predicts "not spam" for everything gets 99% accuracy - but it's completely useless, it never catches a single spam

#-------------------------------------------------
# PRECISION
# Proportion of predicted positives that are actually positive
# Precision = TP / (TP+FP)
precision = (m[1,1])/ m[:,1].sum()   # whole bottom row
print(f"Precision: {precision:.3f}")

#-------------------------------------------------
# RECALL
# Proportion of actual positives that were correctly predicted as positive
# recall = TP/ (TP+FN)
# Example: in a disease diagnosis system, a recall of 100% indicates that all actual cases of the disease were identified
recall = (m[1,1])/ m[1,:].sum()    # whole second column
print(f"Recall: {recall:.3f}")
# precision and recall both cannot be high
# if you lower your threshold to catch more positives (higher recall), you'll also catch more false positives (lower precision)
# this is called PRECISION-RECALL TRADEOFF

#-------------------------------------------------
# F1 SCORE
# It's the harmonic mean of precision and recall 
# used when no clear priority between precision and recall - like providing an overall evaluation of the model
# F1 = 2/((1/recall) + (1/precision)) = 2TP/ (2TP+FP+FN)
f1 = 2 * (precision * recall) / (precision + recall)
print(f"F1 score: {f1:.3f}")

#-------------------------------------------------
# AUTOMATIC CALCULATIONS
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# All functions take the test output (true y values) as first argument and predicted values as second argument

print(f'Precision:{precision:.3f}')
print(f'Recall:{recall:.3f}')
print(f'F1 Score:{f1:.3f}')