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

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

r = requests.get(url).content
auto = pd.read_csv(io.StringIO(r.decode('utf-8')), header = None)
# res.decode('utf-8')  - converts raw bytes to string
# header=None          - dataset has no column names row, so don't treat first row as headers

# manually assigning column names since the raw data had none
auto.columns = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors',
                            'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height',
                            'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore',
                            'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

print("Data shape: ", auto.shape) # (205,26)
print(auto.head())

# TARGET VARIABLE: PRICE
# EXPLANATORY VARIABLES: HORSEPOWER, WIDTH, HEIGHT (we chose 3 to simplify the task)

# DATA PREPARATION:

# convert '?' entries to missing values and then remove the rows containing these missing values
auto = auto[['price', 'horsepower', 'width', 'height']]
print(auto.isin(['?']).sum())
# we can see there are 4 '?' in price column and 2 in horsepower, 0 in height and width
auto = auto.replace('?', np.nan).dropna()
print("Data shape: ", auto.shape) # (199,4) 

# Check data types before converting (need numeric data only)
print("Data types: \n", auto.dtypes)
# price and horsepower are object type, not numeric so have to convert them:
auto = auto.assign(price = pd.to_numeric(auto.price))  # can add errors= 'coerce' here but we already removed '?' data so not needed othwerise it would have thrown error
auto = auto.assign(horsepower = pd.to_numeric(auto.horsepower))
# .assign() returns a new dataframe with that column replaced
print("Data types after conversion: \n", auto.dtypes)

# CORRELATION CHECK:
print(auto.corr())

# Observations: 
# horsepower and width have somewhat correlation with each other ~ 0.61
# horsepower and width have high correlation with price (0.81 and 0.75 respectively)
# height has lower correlation with price (0.13)

# Interpretation:
# highly correlated variables as explanatory variables in multiple regression can lead to multi-collinearity

# Multi-collinearity is a phenomenon where high correlation between variables causes an increase in the variance of regression coefficients, leading to a loss of coefficient significance
# To avoid this, typically only a representative variable from groups of highly correlated variables is used
# But here we're using all three just for experiment

# MODEL BUILDING AND EVALUATION:
# Import for data splitting (training data and test data)
from sklearn.model_selection import train_test_split
# This function randomly splits the data into two parts. 
# The split ratio is determined by test_size. Here, with test_size set to 0.5, the data is divided equally

# Import for building the multiple regression model
from sklearn.linear_model import LinearRegression

# X = explanatory variables (everything except target)
# y = target variable (what we're predicting)
X = auto.drop('price', axis = 1)
y = auto['price']

# split data into training and test sets
# test_size=0.5  - 50/50 split
# random_state=0 (V.IMP) - fixes randomness so results are reproducible every run - split is consistent each time the code is run
# If random_state isn’t set to a fixed number, rows may be categorized into either training or test data differently with each execution, resulting in varying outcomes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

# build and train the model
model = LinearRegression()   # create model instance
model.fit(X_train, y_train)  # train on training data only

# evaluate — coefficient of determination (R²)
# R² = 1.0 = perfect predictions
# R² = 0.0 = model is useless
# train score vs test score gap - tells if model is overfitting
print('Coefficient of determination (train):{:.3f}'.format(model.score(X_train, y_train)))
print('Coefficient of determination (test):{:.3f}'.format(model.score(X_test, y_test)))

# regression coefficients (how much each feature affects price)
# intercept = baseline value when all features = 0
print('\nRegression coefficients\n{}'.format(pd.Series(model.coef_, index=X.columns)))
# model.coef_ - array of numbers like [200.3, -50.1, 88.4, ...] (one per feature)
# pd.Series(..., index=X.columns) - wraps those numbers into a Series with column names as labels
print('Intercept: {:.3f}'.format(model.intercept_))
# model.intercept_ - the b value in y = w1x1 + w2x2 + ... + b

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

X = adult[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss']]
y = adult['fin_flg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

model = LogisticRegression()
model.fit(X_train, y_train)

print('Accuracy (train):{:.3f}'.format(model.score(X_train, y_train)))
print('Accuracy (test):{:.3f}'.format(model.score(X_test, y_test)))

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

#-------------------------------------------------
# DECISION TREES
# Method of classifying data by repeatedly branching based on conditions of each attribute to reach a certain goal
# target variable categorical - classification tree
# target variable numerical - regression tree

# By repeatedly making decisions based on different attributes of the data, the tree divides the data into categories (e.g., poisonous or edible mushrooms). 
# Various routes can lead to the target (such as identifying poisonous mushrooms), 
# and these routes are represented in a tree structure, which is why it's called a decision tree.

# LOADING AND READING DATA
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
r = requests.get(url).content

mushroom = pd.read_csv(io.StringIO(r.decode('utf-8')), header = None)
mushroom.columns = ['classes', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
                             'gill_attachment','gill_spacing','gill_size','gill_color','stalk_shape',
                             'stalk_root','stalk_surface_above_ring','stalk_surface_below_ring',
                             'stalk_color_above_ring','stalk_color_below_ring','veil_type','veil_color',
                             'ring_number','ring_type','spore_print_color','population', 'habitat']

print(mushroom.head())
print("Data shape: ", mushroom.shape) #(8124, 23)
print('Number of missing values: {}'.format(mushroom.isnull().sum().sum()))  # No missing values found

# TARGET VARIABLE: Classes (p = poisonous, e = edible)
# EXPLANATORY VARIABLES: (We choose 4 here) gill_color, gill_attachment, odor, and cap_color  - all catergorical - have to convert to numeric for decision trees

# DUMMY ENCODING: 
# converting categorical variables, such as a gender variable with values male or female, into two separate columns: a 'male' column and a 'female' column. 
# if the gender is male, the 'male' column will be 1, and the female column will be 0 
# this is also known as one-hot encoding
# use get_dummies function

mushroom_dummy = pd.get_dummies(mushroom[['gill_color', 'gill_attachment', 'odor', 'cap_color']], dtype='int')
print(mushroom_dummy.head())
# each new variable represents a combination of the original variable name and its value
# convert values in the classes variable to 1 if they are p and 0 otherwise , adding this new variable as flg
mushroom_dummy['flg'] = mushroom['classes'].map(lambda x: 1 if x =='p' else 0)

""" ENTROPY: MEASURE OF IMPURITY
# Impurity represents the state of differentiation (e.g., distinguishing between poisonous and edible mushrooms)
# High impurity means the categories are not well separated.
# Decision trees are algorithms that assess which of the variables provides the most useful split for classification, using impurity as a criterion for determining the quality of each split. 

# there are two categories (poisonous or edible), with the proportion of edible mushrooms represented by  p1  and the proportion of poisonous mushrooms by  p2
# Example:
# consider a branching condition in which both poisonous and edible mushrooms are equally represented i.e. p1=p2=0.5
# then entropy = - (0.5 * np.log2(0.5) + 0.5 * np.log2(0.5)) = 1 => maximum randomness
# Since the dataset contains an equal proportion (0.5) of both poisonous and non-poisonous mushrooms, it indicates a completely unidentifiable state.

# entropy values near 1.0 indicate an unidentifiable state, whereas values close to 0.0 suggest a well-identifiable state
# binary classification => p1 = 1- p2
# Graph of entropy - downward parabola between 0 to 1 (in both axes i.e. Q1, vertex = (0.5,1))
"""

print(mushroom_dummy.groupby('flg')['flg'].count())
# non-poisonous (0) = 4208 => proportion = 4208/8124 = 0.518
# poisonous (1) = 3916 => proportion = 3196/8124 = 0.482
entropy_init = - (0.518 * np.log2(0.518) + 0.482 * np.log2(0.482))
print('Initial entropy of the poisonous mushroom data: {:.3f}'.format(entropy_init))
# initial entropy = 0.999 - now have to reduce it by branching using explanatory variables

# Information gain is an indicator that shows how much entropy has decreased before and after a data split using a particular variable
# Info. gain = Parent dataset entropy - Σ{(child dataset size / parent dataset size) × child dataset entropy
# The larger this value, the greater the reduction in entropy between the parent and child datasets, indicating a more useful splitting condition

# using two variables- cap_color_c and gill_color_b here to compare which is more useful as a split condition using information gain
mushroom_dummy.groupby(['cap_color_c', 'flg'])['flg'].count().unstack()

# Entropy when cap_color is not `c`
p1 = 4176 / (4176 + 3904)
p2 = 1 - p1
entropy_c0 = - (p1 * np.log2(p1) + p2 * np.log2(p2))
print('entropy_c0: {:.3f}'.format(entropy_c0))  
# result - 0.999

# Entropy when cap_color is 'c'
p1 = 32 / (32 + 12)
p2 = 1 - p1
entropy_c1 = - (p1 * np.log2(p1) + p2 * np.log2(p2))
print('entropy_c1: {:.3f}'.format(entropy_c1))
# result - 0.845

entropy_after = (4176 + 3904) / 8124 * entropy_c0 + (32 + 12) / 8124 * entropy_c1
print('Average entropy after data split: {:.3f}'.format(entropy_after))

print('Information gain from splitting the variable cap_color: {:.3f}'.format(entropy_init - entropy_after))
# Info. gain = 0.001 => the entropy has not decreased significantly, 
# and we can quantitatively conclude that whether cap_color is c or not is not a very useful splitting condition

mushroom_dummy.groupby(['gill_color_b', 'flg'])['flg'].count().unstack()
# definition of entropy strictly requires the calculation to be done for non-empty categories. 
# Since there are no samples where the flg variable is 0 when gill_color is b, we do not include the term  p1×np.log2(p1)  in the sum for the entropy calculation.

# Entropy when gill_color is not b
p1 = 4208 / (4208 + 2188)
p2 = 1 - p1
entropy_b0 = - (p1 * np.log2(p1) + p2 * np.log2(p2))

# Entropy when gill_color is b
p1 = 0 / (0 + 1728)
p2 = 1 - p1
entropy_b1 = - (p2 * np.log2(p2))

entropy_after = (4208 + 2188) / 8124 * entropy_b0 + (0 + 1728) / 8124 * entropy_b1
print('Information gain from splitting by gill_color: {:.3f}'.format(entropy_init - entropy_after))
# result: Info. gain = 0.269 => more beneficial branching condition

# A decision tree splits the data based on the splitting condition with the highest information gain 
# and continues to explore splitting conditions that maximize information gain at each subsequent level
# Other measures of impurity: Gini impurity, classification error
# In the case of decision trees, model complexity is determined by the number of splits. 
# More splits = more complex model

# BUILDING DECISION TREE MODEL
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X = mushroom_dummy.drop('flg', axis= 1)
y = mushroom_dummy['flg']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state = 0)
model.fit(X_train, y_train)

print('Accuracy(train):{:.3f}'.format(model.score(X_train, y_train)))
print('Accuracy(test):{:.3f}'.format(model.score(X_test, y_test)))
# results: train=0.991, test=0.992 => near perfect, mushroom data is very clean

# criterion='entropy' - uses entropy as split condition (measures impurity)
# max_depth=5         - limits how deep the tree grows
# no standardization needed for decision trees
# Decision tree visualisation (needs pydotplus + graphviz)

#-------------------------------------------------
# OVERFITTING
# When a model clings too tightly to training data, fitting noise rather than true patterns
# This leads to high accuracy on training data but poor performance on unseen data
# Techniques to mitigate: 
# 1. Increasing training data - harder to memorize, forced to generalize
# 2. Cross-validation - better evaluation
# 3. Early stopping - stop training before model over-memorizes
# 4. Regularization - penalizes complexity

# decision tree example:
# max_depth=5  - controlled, less overfit
# max_depth=50 - memorizes every training row, fails on test data

# If model is too simple or hasn't been trained enough, it missed imp. patterns - UNDERFITTING

#-------------------------------------------------
# REGULARIZED REGRESSION: LASSO AND RIDGE
# PROBLEM with regular regression:
# large coefficients = complex model = overfits training data

# SOLUTION: add a regularization term to the loss function
# loss = prediction error + complexity penalty
# this forces coefficients to stay small = simpler model = better generalization
# Regularization = techniques that generally aim to reduce the complexity of a model

# LASSO (q=1) => absolute value of coefficients as penalty
# - can shrink some coefficients to exactly 0
# - effectively removes useless features
# - good for feature selection

# RIDGE (q=2) => squared value of coefficients as penalty
# - shrinks all coefficients but rarely to zero
# - good when all features matter a little

# lambda => regularization parameter, controls how hard the penalty hits
# high lambda => simpler model, more regularization
# low lambda  => closer to regular regression

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split

X = auto.drop('price', axis=1)
y = auto['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# creating both multiple regression model and ridge regression model and comparing their results
linear = LinearRegression()
ridge = Ridge(random_state=0)

for model in [linear, ridge]:    # loops through both models and runs the same code on each
    model.fit(X_train, y_train)
    print('{} (train):{:.6f}'.format(model.__class__.__name__ , model.score(X_train, y_train)))
    print('{} (test):{:.6f}'.format(model.__class__.__name__ , model.score(X_test, y_test)))
# __class__ => the class of the object (LinearRegression or Ridge)
# __name__ => gets the name as a string
# so it auto-prints "LinearRegression" or "Ridge" without coding it

# Result: Both have very similar performance, but as a trend, the linear regression model has a higher accuracy on the training data,
# while the Ridge regression model outperforms on the test data
# Ridge test score slightly higher => regularization helped generalization

# IMP: sklearn LogisticRegression includes Ridge regularization by default

#-------------------------------------------------
# k-NEAREST NEIGHBOURS (k-NN)
# AKA lazy learning or memory-based learning, as  it memorizes the training data directly without explicit training
# HOW IT WORKS:
# - no actual training, just memorizes all data points
# - for a new point, find k closest neighbors in training data
# - majority class among those k neighbors = prediction

# k => number of neighbors to look at
# small k => complex boundary, overfits (memorizes training data)
# large k => smoother boundary, may underfit
# optimal k => where train and test accuracy converge (k=6-8 in this example)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=0)
# stratify => keeps same class ratio in both train and test splits

# loop through k=1 to 20, store accuracy at each k
training_accuracy = []
test_accuracy = []

for n_neighbors in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    training_accuracy.append(model.score(X_train, y_train))
    test_accuracy.append(model.score(X_test, y_test))

# plot to visually find optimal k
plt.plot(range(1, 21), training_accuracy, label='Training')
plt.plot(range(1, 21), test_accuracy, label='Test')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()
plt.show()
# RESULT: k=6-8 is optimal, train and test accuracy converge
# for regression tasks => KNeighborsRegressor (same logic, predicts number)

#-------------------------------------------------
# SUPPORT VECTOR MACHINES (SVM)
# - finds the boundary line between classes
# - not just any line => the one that MAXIMIZES the margin
# - margin => distance between boundary and closest points of each class
# - support vectors => the data points closest to the boundary (they define it)
# - wider margin => better generalization
# Example: drawing a road between two neighborhoods
# SVM draws it as wide as possible so neither side is too close to the edge

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify = cancer.target, random_state=0)

model = LinearSVC()
model.fit(X_train, y_train)

# WITHOUT scaling => lower accuracy
# WITH scaling (StandardScaler) => noticeable improvement
# SVM is sensitive to feature scale, always standardize for SVM

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

model = LinearSVC()
model.fit(X_train_std, y_train)

print('Accuracy (train):{:.3f}'.format(model.score(X_train_std, y_train)))
print('Accuracy (test):{:.3f}'.format(model.score(X_test_std, y_test)))

# SVM MAY BE HARD TO INTERPRET => coefficients don't have simple meaning like regression
# black box model => not ideal for public sector / explainability-required projects
# good accuracy but use Decision Trees or Logistic Regression when explanation needed

#-------------------------------------------------
# HOLDOUT METHOD AND CROSS VALIDATION

# HOLDOUT METHOD 
# - split data into train and test once, randomly
# - problem 1: results depend on HOW you split (lucky/unlucky split)
# - problem 2: less training data available
# - fine for large datasets, unreliable for small ones

# K-FOLD CROSS-VALIDATION (solution)
# - split data into k equal folds
# - train on k-1 folds, validate on 1 fold
# - repeat k times so every fold gets a turn as validation
# - final score = average of all k scores
# - uses ALL data for training across iterations

# LEAVE-ONE-OUT => special case where k = number of data points
# - used when dataset is very small

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

cancer = load_breast_cancer()

tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state= 0)
scores = cross_val_score(tree, cancer.data, cancer.target, cv=5)

print('Cross validation scores: {}'.format(scores))
print('Cross validation scores: {:.3f}+-{:.3f}'.format(scores.mean(), scores.std()))

# scores.mean() => overall performance
# scores.std()  => consistency across folds (high std = unstable model)
# if std is large => consider picking model based on mean - std, not just mean

#-------------------------------------------------
# HYPERPARAMETER AND GRID SEARCH

# HYPERPARAMETERS => settings you choose BEFORE training
# decision tree => max_depth
# ridge regression => lambda (regularization strength)
# SVM => gamma, C
# different from parameters (weights/coefficients) which model learns itself

# GRID SEARCH => try every combination of hyperparameters, pick the best

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=0)

# manually:
scores= {}
for gamma in np.logspace(-3, 2, num= 6):     # [0.001, 0.01, 0.1, 1, 10, 100]
    for C in np.logspace(-3, 2, num= 6):
        svm = SVC(gamma = gamma, C = C)
        svm.fit(X_train, y_train)
        scores[(gamma,C)] = svm.score(X_test, y_test)

scores = pd.Series(scores) 

print('Best score: {:.2f}'.format(scores.max()))
print('Parameters (gamma, C) for the best score: {}'.format(scores.idxmax()))
# Result: best score= 0.91 at gamma=0.001, C=1.0

# Display heatmap, Gamma on the vertical axis, C on the horizontal axis
sns.heatmap(scores.unstack())
plt.show()
# heatmap shows most gamma/C combinations fail (near 0.65)
# only gamma=0.001 with C>=1.0 produces good results (~0.90)
# grid search without visualization => can miss this pattern

# GRIDSEARCHCV => sklearn's built-in, uses cross-validation automatically
from sklearn.model_selection import GridSearchCV

# data splitting same as above
param_grid = {
    'C': np.logspace(-3, 2, num=6),
    'gamma': np.logspace(-3, 2, num=6)
}

gs = GridSearchCV(
    estimator=SVC(),    # The machine learning algorithm to use
    param_grid=param_grid,  # The range of hyperparameters to search
    cv=5)    # Number of splits for cross-validation
gs.fit(X_train, y_train)

print('Best CV score: {:.3f}'.format(gs.best_score_))    # 0.93
print('Best params: {}'.format(gs.best_params_))
print('Test score: {:.3f}'.format(gs.score(X_test, y_test)))  # 0.909
# CV score (0.93) close to test score (0.909) => no overfitting

# alternatives: RandomizedSearchCV, Bayesian optimization (Hyperopt)

#-------------------------------------------------
# SUMMARY 
# CORE WORKFLOW (same for every algorithm):
# X, y => train_test_split => model.fit(X_train) => model.score(X_test) => model.predict(new_data)

# ALGORITHMS COVERED + WHEN TO USE:
# Multiple Regression    => predict numbers, interpretability needed, baseline model
# Logistic Regression    => binary classification, need to explain coefficients
# Lasso                  => regression + feature selection (removes useless features)
# Ridge                  => regression + all features matter + overfitting risk
# Decision Tree          => classification/regression, need to explain to non-technical
# k-NN                   => small datasets, simple baseline, no training time
# SVM                    => high accuracy needed, don't need to explain why

# QUICK DECISION:
# predicting a number?        => regression family (Linear, Ridge, Lasso)
# predicting a category?      => classification family (Logistic, Tree, k-NN, SVM)
# need to explain to manager? => Linear Reg, Logistic Reg, Decision Tree
# black box ok, need accuracy? => SVM, (Random Forest, Gradient Boosting - week 6+)
# small dataset?              => k-NN, cross-validation over holdout
# many features, some useless? => Lasso

# KEY RULES:
# always fit scaler on train data only
# always set random_state for reproducibility
# train score >> test score => overfitting
# use cross-validation over holdout for small datasets
# use GridSearchCV to find best hyperparameters

# ALGORITHMS THAT NEED SCALING: Logistic Regression, SVM, k-NN
# ALGORITHMS THAT DON'T:        Decision Trees, Random Forests

# SMART CITY ANGLE:
# regression => energy demand forecasting, AQI prediction (DEWA, Masdar)
# classification => land use, fault detection, anomaly alerts (RTA, Smart Dubai)
# explainability matters in govt projects => prefer Tree/Logistic over SVM