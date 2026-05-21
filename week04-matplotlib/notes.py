# WEEK 04 - DATA VISUALISATION USING MATPLOTLIB, SEABORN
# GCI TOKYO UNIVERSITY 2026

# TOPICS COVERED: 
# Loading and Understanding data, Making Good graphs, Matplotlib with One Variable Analysis, Two Variables (Num. vs Num., Category vs Num.), Subplots and Multi-Panel Figures, Correlation and Heatmaps, Regression and Trend Analysis

# Once data has been collected and organized, the next step is to explore it in a way that reveals patterns and relationships. Tables and summary statistics are useful, but they do not always make trends or differences easy to see.
# CASE STUDY:
# The city’s bike-sharing manager wants to understand what drives rental demand.

# LOADING LIBRARIES
import numpy as np   # numerical operations
import pandas as pd  # handling tabular data
import matplotlib.pyplot as plt  # core plotting
import seaborn as sns  # higher level statistical visualization (fancy matplotlib)
import requests, zipfile, io, os
# %matplotlib inline - tells Jupyter/Colab to open matplotlib charts INSIDE the notebook instead of opening a separate popup window - not needed in .py files
# Setting a visual style at the beginning ensures: Consistent formatting, Clear gridlines, Cleaner default colors - readability and interpretability without requiring additional customization later.

#-------------------------------------------------
# EXTRACTING AND READING DATA
url = "https://cdn.uci-ics-mlr-prod.aws.uci.edu/275/bike%2Bsharing%2Bdataset.zip"
response = requests.get(url, stream = True)
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall()

base = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base, 'bike_sharing_dataset', 'day.csv'))
# print(df.head())
# print("Dataset shape: ", df.shape) # 731 rows, 16 columns

# df.info()
# """
# # OBSERVATIONS: 
# 1. Each observation represents one day of bike rentals and we have 731 observations with 16 variables that means we have two full years of daily data (365*2 = 730)
# 2. Each column shows 731 non-null values, which means there are no missing values (NaN) in this dataset so we can proceed with analysis without additional data cleaning (dropping rows)
# 3. There are three dtypes:
#      (i) int64 - integer variables (ex:season, workingday, cnt, weekday ...)
#     (ii) float64 - continuous numerical variables (ex: temperature, humidity, windspeed)
#    (iii) object - text/date (dteday - the date)
#    But all integer variables do not represent numerical quantities - ex. season, holiday, weekday etc are categories encoded as numbers.
# 4. Most important variable (TARGET VARIABLE) - cnt (total daily rentals) 
# """

# print(df.describe())
# """
# # OBSERVATIONS: (most imp. - cnt)
# 1. Average daily rentals = 4504 bikes
# 2. Median rentals = 4548 bikes
# => Mean and median are very close so the distribution may be relatively symmetric.
# 3. Minimum = 22 rentals, Maximum = 8714 rentals
# => Very large range in demand
# 4. Std. deviation = 1937 (substantial variability in daily demand)
# """
# # print(df.head().T) - not really needed here

# #-------------------------------------------------
# # UNDERSTANDING OUR VARIABLES
# # Numerical var. : temp, hum, windspeed 
# # Categorial var. : season (1-4), weathersit (weather situation), workingday (1 if working, 0 for weekend/holiday)
# # Type of variable determines which visualisation is appropriate, example:
# # Numerical vs Numerical => Scatter plot
# # Categorical vs Numerical => Bar chart
# # Single Numerical => Histogram or Boxplot (distribution shape)

# # .unique() — returns all distinct values in a column as an array
# # use it to confirm: is this column truly categorical?
# # If .unique() returns a small fixed set - categorical, use bar chart
# # If .unique() returns many varied values - numerical, use scatter/hist
# # ALWAYS USE TO VERIFY YOUR ASSUMPTIONS ABOUT A COLUMN

# print("Unique seasons: ", df['season'].unique()) # returns [1 2 3 4] for the four seasons
# print("Unique weather situations: ", df['weathersit'].unique()) # returns [2 1 3]
# print("Working day values: ", df['workingday'].unique()) # returns [0 1]
# # These are categorial variables encoded as nos., we should treat them as categorial only not as numerical even though stored as numbers.

#-------------------------------------------------
# MAKING GRAPHS
# A weak graph: 
# plt.plot(df['cnt'])
# plt.show() # x-axis represents the index i.e. day no., and y-axis the total rentals on that day
# Weak : No title, no axis labels, x-axis unclear, no context, viewers don't know what 'cnt' means

# A good graph: 
# 1. Clear descriptive title - tells the reader WHAT they're looking at
# 2. Informative axis labels - always include units e.g. "Temperature (°C)"
# 3. Appropriate scaling - don't force y-axis to 0 if data is 900-950
# 4. Minimal gridlines - aids reading without cluttering the chart

# A better graph:
# plt.figure(figsize = (10,5)) # set canvas size in inches (width, height) - call this BEFORE any plot command
# plt.plot(df['cnt'])

# plt.title("Daily Bike Rentals Over Time")  # title at top of graph
# plt.xlabel("Day Index") # x-axis label
# plt.ylabel("Total Rentals (cnt)") # y-axis label
# plt.grid(True) # toggle background gridlines (True/False)

#plt.show()

# title(), xlabel(), ylabel() etc have fontsize argument that you can pass to adjust font size (ex. plt.title("text", fontsize = N))
# when you increase figsize, increase fontsizes proportionally
# figsize adjustment ensures your graph fits your reports/slides for clarity and aesthetics (wider and shorter, squarish etc)
# plt.figure creates a new figure (canvas) everytime it is used.
# A figure can host one or more axes which are the area within your figure where your data gets plotted. (by default one axes)
# So one figure can have multiple axes i.e. subplots each holding its own graph.


# # Scatter plot: temp vs cnt (Num vs Num - tells us correlation/trend)
# plt.figure(figsize = (9,6)) # canvas size set - FIRST THING TO DO 

# plt.scatter(df['temp'], df['cnt'], alpha = 0.6)
# # alpha = 0.6 means 60% opacity per point
# # when points overlap (overplotting), transparency shows density
# # alpha = 1.0  = fully opaque, alpha = 0.0 = invisible
# # any time you have a scatter plot with a lot of points stacking on top of each other, alpha is what reveals where the actual density is. Without it you just see a blob.

# plt.title("Temperature vs Total Rentals", fontsize = 16)
# plt.xlabel("Temperature", fontsize=13)
# plt.ylabel("Total Rentals", fontsize=13)

# plt.xticks(fontsize = 11) # tick labels ≈ 2px smaller than axis labels
# plt.yticks(fontsize = 11)
# plt.grid(True)

# plt.show()

#-------------------------------------------------
# GRAPH PARTIONING (SUBPLOTS)
""" 
We can divide graph into multiple sections using plt.subplot()
plt.subplot(rows, columns, index) divides a figure into a grid.
# rows     - how many rows of plots
# cols     - how many columns of plots
# position - which slot to draw in (counts left→right, top→bottom)

For example: plt.subplot(2,1,1) means:
# 2 rows, 1 column, First position

WORKFLOW: call plt.subplot() to "activate" a slot,
then ALL plt commands after it apply to THAT slot only, until you call plt.subplot() again to switch slots

- This allows us to compare multiple graphs within the same figure.

Useful when:
# Comparing related variables
# Showing trends together
# Maintaining visual consistency """

plt.figure(figsize = (12,6))  # total canvas for ALL subplots combined

# First subplot (2 rows, 1 column, position 1)
plt.subplot(2,1,1)
plt.plot(df['cnt'])
plt.title("Daily Rentals Over Time")
plt.ylabel("Total Rentals")
plt.grid(True)

# # Second subplot (2 rows, 1 column, position 2)
# plt.subplot(2,1,2)
# plt.plot(df['temp'])
# plt.title("Temperature Over Time")
# plt.ylabel("Temperature")
# plt.xlabel("Day Index")   # x-label only needed on the BOTTOM plot
# plt.grid(True)

# plt.tight_layout() # auto-adjusts spacing so titles/labels don't overlap
# plt.show()

# By default, matplotlib doesn't support characters outside english alphabet, so if used any other character/language in labels, it will show garbled characters in place of labels and warning messages.

#-------------------------------------------------
# ONE VARIABLE ANALYSIS

# LINE PLOT
# Use plt.plot() - plots points by connecting them on anx-ycoordinate system based on two sets of data. 
# useful when we want to see how something changes step by step. 
# Since our dataset records bike rentals on a daily basis, a line plot is an appropriate tool to examine how rental demand evolves over time.





