# WEEK 07 - FEATURE ENGINEERING AND ENSEMBLE LEARNING
# GCI TOKYO UNIVERSITY 2026

# TOPICS COVERED: Feature Engineering, transforming numerical variables - Scaling, log transformation. transforming catergorical variables - label encoding, one-hot encoding, cross features, use of domain knowledge, ensembling, bagging, randomforest, boosting, gradient boosting, XGBoost, Bias-variance trade-off

# LOADING LIBRARIES AND DATA
import numpy as np
import numpy.random as random
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score

# Regression task: California Housing Data
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)
X_housing = california_housing.data
y_housing = california_housing.target

# Classification task: Breast Cancer Data
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer(as_frame=True)
X_cancer = cancer.data
y_cancer = cancer.target

print('California Housing:', X_housing.shape, '/ target:', y_housing.shape)  # (20640,8)
print('Breast Cancer    :', X_cancer.shape,  '/ target:', y_cancer.shape)   # (569,30)

#-------------------------------------------------
# FEATURE ENGINEERING
# process of selecting, transforming, or creating input variables to make your data more meaningful to the model
# Well-crafted features (variables) can significantly boost a model's accuracy by exposing relevant relationships that raw data might conceal
# poor or redundant features can weaken your model's performance, masking the core patterns you're trying to learn

# UNDERFITTING
# occurs when a model is too simple or when the data lacks crucial signals for it to learn, often leaving the model unable to capture important trends
# it shows poor generalization, the general approach is to explore adding more features

# OVERFITTING 
# when a model clings too tightly to training data, fitting noise rather than true patterns
# the opposite approach to underfitting is typically employed: reducing the number of features
# Too many features relative to the amount of data can increase the upper limit of generalization error, a phenomenon known as the curse of dimensionality.

# Reducing no. of features = DIMENSIONALITY REDUCTION
# TWO TYPES: 
# FEATURE SELECTION - Choosing a subset of existing features (RFE, RFECV etc) - preferred when interpretability is priority
# FEATURE EXTRACTION - Transforming the original feature space into a new one (PCA etc.)

#-------------------------------------------------
# TRANSFORMING NUMERICAL VARIABLES

# SCALING 
# Methods - Standardisation and normalisation
# Applied to numeric variables when they have different scales and thus prevents non-tree-based models from learning properly
# Standardization — rescale a feature (x) so its mean is 0 and its standard deviation is 1
# Normalization (min-max scaling) — rescale so the minimum is 0 and the maximum is 1
# Both keep the shape of the distribution; they only change its location and width

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['target'] = iris.target

x = df[['sepal length (cm)']].copy()
ss = StandardScaler()
x_std = ss.fit_transform(x)

x['sepal length std'] = x_std
print(x)
print(x.describe().loc[['mean', 'std'], :])

# Logistic regression is sensitive to StandardScaler
# Decision tree - scale-invariant
# Tree-based models (decision trees, random forests, gradient boosting, XGBoost, LightGBM, etc) split the data based on the rank order of feature values, not their absolute scale
# Standardizing or min-max-scaling a feature only re-labels the same rank ordering, so the resulting splits — and the resulting model — are essentially identical

# LOG TRANSFORMATION
# used to correct skewed distribution by applying the  log  function
# log is not defined for zeros or negative values, so use np.log1p(x) = log(1 + x) if you have zeros, and apply it only to non-negative quantities

from sklearn.datasets import load_wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['variety'] = wine.target
cols = ['color_intensity', 'malic_acid']
X = df[cols]

# Original distributions
X.hist(figsize=(12, 4))

# Applying log transformation
X_log = np.log(X)
cols_log = [s + '_log' for s in cols]
X_log.columns = cols_log

# Transformed distributions
X_log.hist(figsize=(12, 4))
plt.show()
# skewness in both histograms is much reduced after the log transformation

target_variable = df['variety']
logistic_model = LogisticRegression(random_state=42)
cv_score_without_log_transform = cross_val_score(logistic_model, X, target_variable, cv=5, scoring='f1_weighted').mean()
cv_score_with_log_transform = cross_val_score(logistic_model, X_log, target_variable, cv=5, scoring='f1_weighted').mean()
print('Cross-validation score (w/o log transformation): {:.3f}'.format(cv_score_without_log_transform))
print('Cross-validation score (w/ log transformation): {:.3f}'.format(cv_score_with_log_transform))
# Result: w/o log - 0.768, w/ log - 0.797 
# thus log transformation gives higher cross validated F1 score confirming that for Linear models, reducing skewness can improve performance

#-------------------------------------------------
# TRANSFORMING CATEGORICAL VARIABLES
# Categorical variables need to be converted to numerical - called encoding
# Various methods of encoding: Label enc., Count enc., Label-count enc., One-Hot enc.

# LABEL ENCODING
# Assigns each distinct category a unique integer (done in week5)
# Simple, no increase in feature count, works well with tree-based models (the tree can split between any pair of integers)
# For linear/distance-based models it introduces a fake ordering — the model might interpret S (2) > Q (1) > C (0) as if S were “greater than” C
# Use when: tree based model OR categorical variable is ordinal (ex. 'low', 'medium', 'high' etc.)

# ONE-HOT ENCODING
# creates a separate 0/1 column for each category and sets the column matching the row’s category to 1
# No fake ordering - every category is symmetric
# For high-cardinality variables (e.g. product IDs with thousands of distinct values) you get one new column per category, blowing up memory and potentially overfitting
# Use pd.get_dummies(x) or pd.get_dummies(x).astype(int) (when True/False)

# CROSS FEATURES (AKA INTERACTION FEATURES)
# A cross feature is a new feature built from a product (or other simple combination) of two or more existing features
# useful when the target depends on an interaction that a simple model cannot easily learn on its own
# Ex. Rent is high when room size is large AND distance to station is small
# Captures interactions that simple models miss
# The number of possible cross features explodes combinatorially, which costs memory and can lead to overfitting
# often used in combination with methods that reduces the number of variables, such as feature selection
# generate all pairwise products automatically with PolynomialFeatures(interaction_only=True)

TITANIC_URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(TITANIC_URL)
num_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']
X_num = df[num_cols].copy()
X_num['Age'] = X_num['Age'].fillna(X_num['Age'].median())

print(f'Original numeric features: {X_num.shape[1]}')
print(X_num.head())

# explanatory variables = 5 numerical variables => 5C2 + 5 = 15 features (10 interaction pairs + 5 original)

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
# interaction_only = True => only products of feature pairs (Age×Fare, Age×Pclass, etc.), skipping the squared terms (Age², Fare²)
# include_bias=False => without this, sklearn adds a column of all 1s (the "bias" term). That's a leftover from the math of linear models and is redundant when your model already has an intercept, which it does by default. So you turn it off to keep the output clean.
X_pf = pf.fit_transform(X_num)

X_pf_df = pd.DataFrame(X_pf, columns=pf.get_feature_names_out(num_cols))
print(f'After PolynomialFeatures: {X_pf_df.shape[1]} features')
print(f'  = {len(num_cols)} originals + C({len(num_cols)}, 2) = {X_pf_df.shape[1]} total\n')

# Show the new cross-feature columns (skip the 5 originals)
print(X_pf_df.iloc[:, len(num_cols):].head())

#-------------------------------------------------
# USE OF DOMAIN KNOWLEDGE
# Using expertise about the problem to create variables that are directly related to the prediction target

# Ex. for the titanic dataset:
# Passenger title — extract the title (Mr., Mrs., Miss., Master., ...) hidden in the Name string. Titles carry information about gender, age class, and social status, all of which were strongly predictive of survival.

def extract_title(name):
    # "Braund, Mr. Owen Harris"
    after_comma = name.split(',', 1)[1]    # ' Mr. Owen Harris'
    title = after_comma.split('.', 1)[0]   # ' Mr'
    return title.strip()                   # 'Mr'

df['Title'] = df['Name'].apply(extract_title)
df['Title'].unique()
title_counts = df['Title'].value_counts()
print(title_counts)

# Combine rare titles with low frequencies into a single category called "Rare"
rare_titles = title_counts[title_counts < 10].index  
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

title_dummies = pd.get_dummies(df['Title'], prefix='Title')
title_dummies = title_dummies.astype(int)   # one-hot encoding
df = pd.concat([df, title_dummies], axis=1)
df.drop(['Name', 'Title'], axis=1, inplace=True)
print(df.head())

# Sex × Class - combine Sex and Pclass into a single label, motivated by the "women and children first, especially in first class" evacuation rule

df['Sex_Pclass'] = df['Sex'] + '_' + df['Pclass'].astype(str)

print(df.groupby('Sex_Pclass')['Survived'].agg(['mean', 'count']).rename(
    columns={'mean': 'survival_rate', 'count': 'n_passengers'}
).sort_values('survival_rate', ascending=False))

# Dramatic gap - female_1 = 96.8% survival rate while male_3 = 13.5% survival rate
# neither sex nor Pclass tell this story alone

# One-hot encode the combined label
sex_pclass_dummies = pd.get_dummies(df['Sex_Pclass'], prefix='SexPclass').astype(int)
df = pd.concat([df, sex_pclass_dummies], axis=1)

print(sex_pclass_dummies.head())
# the choice of which two columns to combine was driven by domain knowledge, not by enumeration, and the resulting feature is interpretable on its own

#-------------------------------------------------
# ENSEMBLE LEARNING
# Combining many imperfect models can give you a model that is stronger than any individual one
# Train multiple models (same or different type) and aggregate their predictions
# Aggregating predictions = majority vote for classification, averaging for regression

# Two Major Families:
# BAGGING - train each model independently on a different random resample, then average
# BOOSTING - train each model in sequence, each one focusing on the errors of the previous ones

#-------------------------------------------------
# BAGGING (BOOTSTRAP AGGREGATING)
# 1. Take training set of size n
# 2. Draw WITH replacement (n) samples = bootstrap resample. Repeat B times = B resampled datasets
# 3. Train one model on each resample = B models
# 4. Aggregate their predictions (majority vote / average)
# Each individual model is weaker (only sees a noisy subset) but the combined model is often much stronger

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Checking the original number of variables
original_feature_count = X.shape[1]
print("Original number of variables: {}".format(original_feature_count))
print(X.head())

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, stratify=y_cancer, random_state=66)

models = {
    'kNN': KNeighborsClassifier(),
    'kNN + bagging': BaggingClassifier(KNeighborsClassifier(), n_estimators=100, random_state=0)
    # 100 k-NN models are used for bagging
}
# Other parameters in baggingclassifier
# max_samples (default: 1.0): specifies the proportion of the original dataset to extract during bootstrapping
# max_features (default: 1.0): specifies the proportion of features sampled, with 0.5 indicating that half of all features are used for training each model
# Using only a subset of the data can introduce feature diversity for each sample, helping reduce overfitting in models that are prone to it
scores = pd.DataFrame(columns=["train_score", "test_score"], index=models.keys())
for i, (model_name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    scores["train_score"][i] = model.score(X_train, y_train)
    scores["test_score"][i] = model.score(X_test, y_test)

print(scores)
# Result: train score roughly same but test score improves 

#-------------------------------------------------
# RANDOM FOREST
# RF = Bagging on Decision Trees BUT at each split, only a random subset of features is considered as candidates for splitting
# This way, all trees look different from each other, otherwise a single strong feature would dominate the first split of every tree and the trees would all look the same

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

X_train, X_test, y_train, y_test = train_test_split(X_housing, y_housing, random_state=42)
models = {
    'RandomForest': RandomForestRegressor(random_state=0),
    'GradientBoost': GradientBoostingRegressor(random_state=0)
}

scores = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    scores[(model_name, 'train_score')] = model.score(X_train, y_train)
    scores[(model_name, 'test_score')] = model.score(X_test, y_test)

print(pd.Series(scores).unstack())
# Result: 
# RandomForest has higher test score but is clearly overfitting (0.973 train vs 0.810 test)
# GradientBoost generalises better - train and test scores are close (0.78 test, 0.80 train)
# Higher test accuracy doesn't always mean better model, the gap matters too

# FEATURE IMPORTANCE: feature_importances_ attribute ranks features by how much they contributed to splits across the whole forest
s = pd.Series(models['RandomForest'].feature_importances_,
              index=X_housing.columns)

# Displaying the values in descending order
plt.figure(figsize= (8,8))
s.sort_values(ascending=False).plot.bar(color='C0')
plt.grid(True)
plt.show()

#-------------------------------------------------
# BOOSTING
# Opposite to bagging -  instead of training models independently, boosting trains them sequentially
# 1. Train a weak model on the data.
# 2. Check which examples it got wrong.
# 3. Reweight those examples (or fit the residual error), then train another model.
# 4. Repeat. Aggregate the models at the end.

# Ex. AdaBoost, Gradient Boosting, XGBoost, LightGBM etc.

# A standalone decision tree vs an AdaBoost regressor using a decision tree as the base learner
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
X_train, X_test, y_train, y_test = train_test_split(X_housing, y_housing, random_state=66)

models = {
    'Decision tree': DecisionTreeRegressor(random_state=0),
    # Specify the base model in the 'estimator' argument 
    'AdaBoost': AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=0)
}

scores = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    scores[(model_name, 'train_score')] = model.score(X_train, y_train)
    scores[(model_name, 'test_score')] = model.score(X_test, y_test)

print(pd.Series(scores).unstack())
# Result:
# decision tree: train = 1, test = 0.61 => Clear Overfitting
# AdaBoost: train = 0.99. test = 0.82 => slight overfitting but performs well on test data
# huge difference in test scores (0.61 vws 0.82) => sequence of trees corrects errors that any single tree could not

#-------------------------------------------------
# GRADIENT BOOSTING AND XGBOOST
# Gadient boosting generalizes AdaBoost - instead of reweighting examples, each new tree directly predicts the gradient of the loss
# For squared loss, gradient = residual (the error left by the current ensemble)
# Each tree is shallow (depth ≈ 3–8) and predicts the residual of the running sum so far
# Final prediction = sum of all tree outputs

# XGBoost - popular, highly optimized implementation of gradient boosting
# adds: regularization, parallelism, early stopping

import xgboost as xgb
xgb.set_config(verbosity=0)  # show no training logs
from xgboost import XGBRegressor

reg = XGBRegressor(n_estimators = 100, random_state = 42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(f"Test score: {r2_score(y_test, y_pred): .3f}")
# Result: test = 0.841

# Early stopping with XGBoost: 
# train many rounds, but watch performance on a validation set and stop as soon as it stops improving
#  practical antidote to the “boosting overfits with too many rounds” caveat above

X_train_split, X_valid, y_train_split, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
# Increasing the number of boosting rounds without using 'early_stopping_rounds' increases the risk of overfitting alongside potential accuracy improvement
reg = XGBRegressor(n_estimators=1000,
                   eval_metric='rmse',  # Specify the evaluation metric for early stopping
                   early_stopping_rounds=10,  # Specify the number of rounds without improvement before stopping
                   random_state=0,
                   verbose=0)  # verbose = 0 will not show per round output in terminal 

reg.fit(X_train_split, y_train_split, eval_set=[(X_valid, y_valid)])  

y_pred = reg.predict(X_test) 
print(f"Test score: {r2_score(y_test, y_pred): .3f}")
# Result: test = 0.836 => slight decrease due to early stopping

#-------------------------------------------------
# BIAS - VARIANCE TRADEOFF
# Total error = Bias + Variance + irreducible noise

# BIAS - systematic error, model is consistently off (underfitting)
# Ex. fitting a curve with a straight line
# VARIANCE - sensitivity to training data, model is unstable (overfitting)
# Ex. deep tree that memorizes noise

# As model complexity increases: bias falls, variance rises => total error is U-shaped
# You cannot reduce both at the same time

# HOW ENSEMBLING HELPS:
# BAGGING   - averages many high-variance, low-bias learners  => variance drops, bias unchanged
# BOOSTING  - sequentially corrects high-bias, low-variance learner => bias drops, variance preserved
#             (until you overtrain, then variance creeps back)
# What we want = low bias, low variance

#-------------------------------------------------
# SUMMARY
# Feature engineering is the process of shaping your input variables so the model can learn better.
# Raw data rarely comes ready to use - you need to transform, encode, and sometimes create new features.

# NUMERICAL TRANSFORMATIONS
# Scaling (standardization or normalization) is needed for non-tree-based models; tree-based models are scale-invariant.
# Log transformation corrects skewed distributions; use np.log1p(x) when zeros are present.

# CATEGORICAL ENCODING
# Label encoding works for tree-based models and ordinal variables.
# One-hot encoding avoids fake ordering but blows up memory for high-cardinality variables.
# Cross features capture interactions two features create together that neither captures alone; combinatorial explosion is a risk.
# Domain knowledge guides which features to engineer - not just enumeration.

# ENSEMBLE LEARNING
# Combining multiple models produces better predictions than any single model.
# Bagging: trains models independently on random resamples, then averages. Reduces variance.
# Random Forest: bagging on decision trees with random feature subsets at each split, so trees stay diverse.
# Boosting: trains models sequentially, each correcting the errors of the previous. Reduces bias.
# AdaBoost: reweights misclassified examples. Gradient Boosting: fits residuals directly. XGBoost: optimized gradient boosting with regularization and early stopping.

# BIAS-VARIANCE TRADEOFF
# Total error = Bias + Variance + irreducible noise.
# High bias = underfitting. High variance = overfitting.
# Model complexity increases: bias falls, variance rises, total error is U-shaped.
# Bagging attacks variance. Boosting attacks bias.
# A high test score alone is not enough - the gap between train and test score matters too.