# WEEK 05 - SUPERVISED LEARNING
# GCI TOKYO UNIVERSITY 2026

# TOPICS COVERED: 
# ML, Multiple Regression, Logistic regression, Regularised regression: Lasso and Ridge, Overfitting, Decision Trees, k-NN, Support vector machine, Holdout method and Cross Validation, Performance Tuning

# MACHINE LEARNING 
# /      |      \
# Super  Unsuper  Reinforcement
# vised  vised    learning

# SUPERVISED LEARNING
# To build model that predicts an output (target) variable from input (features/ explanatory) variables
# training data includes both target (AKA ground truth, response variable, target variable, or dependent variable) and explanatory variables (AKA eatures, predictor variables, or independent variables)
# Ex. SL can be used to identify whether an email is spam based on its title and content (explanatory variables) 
# or to predict stock prices based on trading activity (explanatory variables)

# UNSUPERVISED LEARNING
# focuses on the input data itself, aiming to discover patterns or insights hidden within the data
# training data does not contain target or explanatory variable

# REINFORCEMENT LEARNING
# enables machines to acquire knowledge or behavior rules by providing rewards based on the outcomes of a series of actions taken by the program (machine)
# handle cases difficult to express in supervised learning
# Ex. interactive environments such as games with opponents

#-------------------------------------------------
# SUPERVISED LEARNING TECHNIQUES
# When target variable = numercial => Regression
# When target variable = categorical => Classification

# KEY ALGORITHMS (all can do both regression AND classification):
# - Multiple Linear Regression
# - Logistic Regression      - name says regression, but used for classification
# - k-Nearest Neighbors (k-NN)
# - Decision Trees           - classification tree / regression tree
# - Support Vector Machine (SVM)
# - Random Forests
# - Gradient Boosting etc.

# HOW TO CHOOSE AN ALGORITHM:
# - prioritize PERFORMANCE?      => complex models (SVM, Gradient Boosting)
# - prioritize INTERPRETABILITY? => simple models (Linear Reg, Logistic Reg, Decision Trees)
# SVM results are hard to explain to non-experts
# Always assess the situation — precision vs explainability is case by case

# WHY THIS MATTERS FOR MY FIELD (Smart City / Climate AI):
# - predicting AQI levels, energy demand, temperature = REGRESSION
# - classifying land use, fault detection, anomaly alerts = CLASSIFICATION
# - interpretability matters more in public sector AI (RTA, DEWA, Smart Dubai) because decisions affect policy — a black box model won't fly in govt projects
# - this is where AI ethics + explainability becomes a real skill gap

#-------------------------------------------------
# UNSUPERVISED LEARNING TECHNIQUES
# No target variable/ labeled data - model finds patterns on its own

# CLUSTERING (MOST COMMON METHOD):
# groups a large number of data points into similar clusters
# Ex. segmenting customers by preferences in marketing
# Can't be fully automated - exploratory — you still need human judgment to interpret results

# OTHER METHODS:
# - PCA (Principal Component Analysis): reduces number of variables while preserving information
# - Market Basket Analysis: finds association rules in purchase data (POS)
#   Ex: "people who buy A also buy B"

# FRAMING:
# supervised   = goal-oriented data-mining (you know what you're predicting)
# unsupervised = exploratory data-mining (you're discovering structure)

# SMART CITY ANGLE:
# clustering - grouping neighborhoods by energy usage patterns (DEWA)
# PCA - compressing sensor data from hundreds of IoT devices
# market basket - not just retail, also co-occurrence in urban mobility data

#-------------------------------------------------
# REINFORCEMENT LEARNING TECHNIQUES
# No labeled data, no explicit instructions — learns from interaction
# Agent learns by trial and error to MAXIMIZE REWARD
# good outcome - high reward
# bad outcome  - low reward

# ANALOGY:
# baby learning to walk — no one teaches step by step, learns by trying
# self-driving car — learns to avoid collisions through environment interaction

# KEY CHALLENGE: Exploration vs Exploitation dilemma
# - too much exploitation - only uses past knowledge, misses better strategies
# - too much exploration  - ignores useful past experience
#  need the right balance

# APPROACHES:
# - Dynamic Programming    - requires full knowledge of environment
# - Monte Carlo Methods    - learns from experience only, no full knowledge needed
# - TD Learning            - combines both (Temporal Difference)

# SMART CITY ANGLE:
# - traffic signal optimization (RTA) — agent learns to minimize congestion
# - energy grid balancing (DEWA)      — agent learns to reduce peak load
# RL is less common in entry-level DS roles, more research/advanced track

#-------------------------------------------------
# LOADING LIBRARIES
import numpy as np
import numpy.random as random
import pandas as pd
from pandas import Series, DataFrame
import scipy as sp     # scipy = scientific python => advanced math built on top of numpy — statistics, integration, optimization, signal processing
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sklearn  # Scikit-learn = Machine learning library

#-------------------------------------------------
# MULTIPLE LINEAR REGRESSION (SL)
# Simple linear regression = one explanatory variable
# Multiple regression = multiple explanatory variables
# coefficients of each explanatory variable (regression coefficients) can be estimated, allowing for the calculation of predicted values
# regression coefficients are estimated in such a way that the squared error between the predicted values and the target variable is minimized

# BUILDING A MULTIPLE REGRESSION MODEL: 
# multiple regression model to predict automobile prices based on their attributes (such as size) using a dataset that contains the prices and attributes of automobiles

# LOADING AND EXTRACTING DATA:
import requests, zipfile, io

# url = "http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

# r = requests.get(url).content
# auto = pd.read_csv(io.StringIO(r.decode('utf-8')), header = None)
# # res.decode('utf-8')  - converts raw bytes to string
# # header=None          - dataset has no column names row, so don't treat first row as headers

# # manually assigning column names since the raw data had none
# auto.columns = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors',
#                             'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height',
#                             'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore',
#                             'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

# print("Data shape: ", auto.shape) # (205,26)
# print(auto.head())

# # TARGET VARIABLE: PRICE
# # EXPLANATORY VARIABLES: HORSEPOWER, WIDTH, HEIGHT (we chose 3 to simplify the task)

# # DATA PREPARATION:

# # convert '?' entries to missing values and then remove the rows containing these missing values
# auto = auto[['price', 'horsepower', 'width', 'height']]
# print(auto.isin(['?']).sum())
# # we can see there are 4 '?' in price column and 2 in horsepower, 0 in height and width
# auto = auto.replace('?', np.nan).dropna()
# print("Data shape: ", auto.shape) # (199,4) 

# # Check data types before converting (need numeric data only)
# print("Data types: \n", auto.dtypes)
# # price and horsepower are object type, not numeric so have to convert them:
# auto = auto.assign(price = pd.to_numeric(auto.price))  # can add errors= 'coerce' here but we already removed '?' data so not needed othwerise it would have thrown error
# auto = auto.assign(horsepower = pd.to_numeric(auto.horsepower))
# # .assign() returns a new dataframe with that column replaced
# print("Data types after conversion: \n", auto.dtypes)

# # CORRELATION CHECK:
# print(auto.corr())

# # Observations: 
# # horsepower and width have somewhat correlation with each other ~ 0.61
# # horsepower and width have high correlation with price (0.81 and 0.75 respectively)
# # height has lower correlation with price (0.13)

# # Interpretation:
# # highly correlated variables as explanatory variables in multiple regression can lead to multi-collinearity

# # Multi-collinearity is a phenomenon where high correlation between variables causes an increase in the variance of regression coefficients, leading to a loss of coefficient significance
# # To avoid this, typically only a representative variable from groups of highly correlated variables is used
# # But here we're using all three just for experiment

# # MODEL BUILDING AND EVALUATION:
# # Import for data splitting (training data and test data)
# from sklearn.model_selection import train_test_split
# # This function randomly splits the data into two parts. 
# # The split ratio is determined by test_size. Here, with test_size set to 0.5, the data is divided equally

# # Import for building the multiple regression model
# from sklearn.linear_model import LinearRegression

# # X = explanatory variables (everything except target)
# # y = target variable (what we're predicting)
# X = auto.drop('price', axis = 1)
# y = auto['price']

# # split data into training and test sets
# # test_size=0.5  - 50/50 split
# # random_state=0 (V.IMP) - fixes randomness so results are reproducible every run - split is consistent each time the code is run
# # If random_state isn’t set to a fixed number, rows may be categorized into either training or test data differently with each execution, resulting in varying outcomes
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

# # build and train the model
# model = LinearRegression()   # create model instance
# model.fit(X_train, y_train)  # train on training data only

# # evaluate — coefficient of determination (R²)
# # R² = 1.0 = perfect predictions
# # R² = 0.0 = model is useless
# # train score vs test score gap - tells if model is overfitting
# print('Coefficient of determination (train):{:.3f}'.format(model.score(X_train, y_train)))
# print('Coefficient of determination (test):{:.3f}'.format(model.score(X_test, y_test)))

# # regression coefficients (how much each feature affects price)
# # intercept = baseline value when all features = 0
# print('\nRegression coefficients\n{}'.format(pd.Series(model.coef_, index=X.columns)))
# # model.coef_ - array of numbers like [200.3, -50.1, 88.4, ...] (one per feature)
# # pd.Series(..., index=X.columns) - wraps those numbers into a Series with column names as labels
# print('Intercept: {:.3f}'.format(model.intercept_))
# # model.intercept_ - the b value in y = w1x1 + w2x2 + ... + b

# RESULTS:
# train=0.733, test=0.737 => no overfitting, good generalization
# width has strongest effect on price (1829 per unit)
# horsepower adds ~82 per unit

# VARIABLE SELECTION METHODS (how to pick which features to use)
# - forward selection    - start with nothing, add best feature one by one
# - backward elimination - start with all, remove worst feature one by one
# - stepwise selection   - combination of both

# SELECTION CRITERIA:
# - RMSE (Root Mean Squared Error) - lower = better predictions
# - AIC  (Akaike Information Criterion) - balances accuracy vs complexity
# - BIC  (Bayesian Information Criterion) - similar to AIC, penalizes complexity more

# no single method is universally best
# real projects also use domain knowledge + generalization performance to decide
# this is where knowing your field (Smart City, Climate) actually 

#-------------------------------------------------
# LOGISTIC REGRESSION
# When target variables are not numeric but categorical
# Calculating the probability that a data sample belongs to a particular category is called classification
# One of the algorithms used for classification is logistic regression
# In classification tasks, the learning process minimizes the objective function
# This objective function is called the cross-entropy error function, and its value becomes smaller as the probability of predicting the correct category increases

# BUILDING A LOGISTIC REGRESSION MODEL:
# construct a model to predict whether an individual's income exceeds $50K based on personal data such as age, gender, and occupation

# LOADING AND ETRACTING DATA:
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
r = requests.get(url).content

adult = pd.read_csv(io.StringIO(r.decode('utf-8')), header = None)
adult.columns =['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                             'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                             'native-country', 'flg-50K']
print("Data shape: ", adult.shape)
print(adult.head())

print('Number of defects:{}'.format(adult.isnull().sum().sum()))  # {} is placeholder for what is after .format

# DATA PREPARATION:
# check how many rows contain <=50K and >50K
print(adult.groupby('flg-50K').size())
# 24,720 rows with <=50K and 7,841 rows with >50K

# adding a column named 'fin_flg' and set a flag of 1 for rows with >50K and 0 for others
adult['fin_flg'] = adult['flg-50K'].map(lambda x: 1 if x == ' >50K' else 0)
print(adult.groupby('fin_flg').size())

# MODEL CONSTRUCTION AND EVALUATION:
# Explanatory variables: age, fnlwgt, education-num, capital-gain, and capital-loss
# Target variable: fin_flg

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# X = adult[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss']]
# y = adult['fin_flg']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

# model = LogisticRegression()
# model.fit(X_train, y_train)

# print('Accuracy (train):{:.3f}'.format(model.score(X_train, y_train)))
# print('Accuracy (test):{:.3f}'.format(model.score(X_test, y_test)))

# RESULTS:
# train: 0.796, test: 0.799 - ~80% accuracy, test slightly higher than train = no overfitting

# LOGISTIC REGRESSION COEFFICIENTS
# model.coef_ - raw coefficients (in log-odds scale, hard to interpret directly)
# e.g. [-5.624e-03, -3.846e-06, -3.642e-02, 3.286e-04, 7.666e-04]

# ODDS RATIO = np.exp(model.coef_)
# converts log-odds to something interpretable
# tells you: if feature increases by 1, accuracy changes by this multiplier
# e.g. 0.964 means that feature slightly decreases the probability
#      1.001 means that feature slightly increases the probability
# odds ratio = 1.0 => feature has no effect

# IMPROVING PREDICTION ACCURACY WITH SCALING:
# PROBLEM: features have different units/magnitudes
# e.g. capital-gain (0–99999) vs education-num (1–16)
# model gets biased toward larger-valued features

# SOLUTION: Standardization
# formula: (value - mean) / standard_deviation
# result:  0 = mean, 1 = one std above mean
# all features on same scale, units removed

# Importing the class for standardization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = adult[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss']]
y = adult['fin_flg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

sc = StandardScaler()  
sc.fit(X_train)           # learn mean and std FROM TRAINING DATA ONLY
X_train_std = sc.transform(X_train)  # apply to train
X_test_std = sc.transform(X_test)    # apply same scale to test

# CRITICAL: never fit scaler on test data - would leak test info into training process
# fit = learn stats. transform = apply stats. only learn from training data, always.

model = LogisticRegression()
model.fit(X_train_std, y_train)

print('Accuracy (train):{:.3f}'.format(model.score(X_train_std, y_train)))
print('Accuracy (test):{:.3f}'.format(model.score(X_test_std, y_test)))

# RESULTS:
# before scaling: 0.796 / 0.799
# after scaling:  0.811 / 0.810  => scaling improved accuracy
# By aligning the scales of the explanatory variables, machine learning algorithms can perform better.
# only the training data is used to calculate the mean and standard deviation. Test data is considered as unknown data that will be obtained in the future, so it cannot be used to evaluate the model.
