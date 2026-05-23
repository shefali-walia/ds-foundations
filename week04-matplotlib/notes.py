# WEEK 04 - DATA VISUALISATION USING MATPLOTLIB, SEABORN
# GCI TOKYO UNIVERSITY 2026

# TOPICS COVERED: 
# Loading and Understanding data, Making Good graphs, Matplotlib with One Variable Analysis, Two Variables (Num. vs Num., Category vs Num.), Subplots and Multi-Panel Figures, Seaborn, Correlation and Heatmaps, Regression and Trend Analysis

# Once data has been collected and organized, the next step is to explore it in a way that reveals patterns and relationships. Tables and summary statistics are useful, but they do not always make trends or differences easy to see.
# %matplotlib inline - tells Jupyter/Colab to open matplotlib charts INSIDE the notebook instead of opening a separate popup window - not needed in .py files
# Setting a visual style at the beginning ensures: Consistent formatting, Clear gridlines, Cleaner default colors - readability and interpretability without requiring additional customization later.

# CASE STUDY:
# The city’s bike-sharing manager wants to understand what drives rental demand.

# LOADING LIBRARIES
import numpy as np   # numerical operations
import pandas as pd  # handling tabular data
import matplotlib.pyplot as plt  # core plotting
import seaborn as sns  # higher level statistical visualization (fancy matplotlib)
import requests, zipfile, io, os

#-------------------------------------------------
# EXTRACTING AND READING DATA
url = "https://cdn.uci-ics-mlr-prod.aws.uci.edu/275/bike%2Bsharing%2Bdataset.zip"
response = requests.get(url, stream = True)
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall()

base = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base, 'bike_sharing_dataset', 'day.csv'))
print(df.head())
print("Dataset shape: ", df.shape) # 731 rows, 16 columns

df.info()
"""
# OBSERVATIONS: 
1. Each observation represents one day of bike rentals and we have 731 observations with 16 variables that means we have two full years of daily data (365*2 = 730)
2. Each column shows 731 non-null values, which means there are no missing values (NaN) in this dataset so we can proceed with analysis without additional data cleaning (dropping rows)
3. There are three dtypes:
     (i) int64 - integer variables (ex:season, workingday, cnt, weekday ...)
    (ii) float64 - continuous numerical variables (ex: temperature, humidity, windspeed)
   (iii) object - text/date (dteday - the date)
   But all integer variables do not represent numerical quantities - ex. season, holiday, weekday etc are categories encoded as numbers.
4. Most important variable (TARGET VARIABLE) - cnt (total daily rentals) 
"""

print(df.describe())
"""
# OBSERVATIONS: (most imp. - cnt)
1. Average daily rentals = 4504 bikes
2. Median rentals = 4548 bikes
=> Mean and median are very close so the distribution may be relatively symmetric.
3. Minimum = 22 rentals, Maximum = 8714 rentals
=> Very large range in demand
4. Std. deviation = 1937 (substantial variability in daily demand)
"""
# print(df.head().T) - not really needed here

#-------------------------------------------------
# UNDERSTANDING OUR VARIABLES
# Numerical var. : temp, hum, windspeed 
# Categorial var. : season (1-4), weathersit (weather situation), workingday (1 if working, 0 for weekend/holiday)
# Type of variable determines which visualisation is appropriate, example:
# Numerical vs Numerical => Scatter plot
# Categorical vs Numerical => Bar chart
# Single Numerical => Histogram or Boxplot (distribution shape)

# .unique() — returns all distinct values in a column as an array
# use it to confirm: is this column truly categorical?
# If .unique() returns a small fixed set - categorical, use bar chart
# If .unique() returns many varied values - numerical, use scatter/hist
# ALWAYS USE TO VERIFY YOUR ASSUMPTIONS ABOUT A COLUMN

print("Unique seasons: ", df['season'].unique()) # returns [1 2 3 4] for the four seasons
print("Unique weather situations: ", df['weathersit'].unique()) # returns [2 1 3]
print("Working day values: ", df['workingday'].unique()) # returns [0 1]
# These are categorial variables encoded as nos., we should treat them as categorial only not as numerical even though stored as numbers.

#-------------------------------------------------
# MAKING GRAPHS
# A weak graph: 
plt.plot(df['cnt'])
plt.show() # x-axis represents the index i.e. day no., and y-axis the total rentals on that day
# Weak : No title, no axis labels, x-axis unclear, no context, viewers don't know what 'cnt' means

# A good graph: 
# 1. Clear descriptive title - tells the reader WHAT they're looking at
# 2. Informative axis labels - always include units e.g. "Temperature (°C)"
# 3. Appropriate scaling - don't force y-axis to 0 if data is 900-950
# 4. Minimal gridlines - aids reading without cluttering the chart

# A better graph:
plt.figure(figsize = (10,5)) # set canvas size in inches (width, height) - call this BEFORE any plot command
plt.plot(df['cnt'])

plt.title("Daily Bike Rentals Over Time")  # title at top of graph
plt.xlabel("Day Index") # x-axis label
plt.ylabel("Total Rentals (cnt)") # y-axis label
plt.grid(True) # toggle background gridlines (True/False)

plt.show()

# title(), xlabel(), ylabel() etc have fontsize argument that you can pass to adjust font size (ex. plt.title("text", fontsize = N))
# when you increase figsize, increase fontsizes proportionally
# figsize adjustment ensures your graph fits your reports/slides for clarity and aesthetics (wider and shorter, squarish etc)
# plt.figure creates a new figure (canvas) everytime it is used.
# A figure can host one or more axes which are the area within your figure where your data gets plotted. (by default one axes)
# So one figure can have multiple axes i.e. subplots each holding its own graph.


# Scatter plot: temp vs cnt (Num vs Num - tells us correlation/trend)
plt.figure(figsize = (9,6)) # canvas size set - FIRST THING TO DO 

plt.scatter(df['temp'], df['cnt'], alpha = 0.6)
# alpha = 0.6 means 60% opacity per point
# when points overlap (overplotting), transparency shows density
# alpha = 1.0  = fully opaque, alpha = 0.0 = invisible
# any time you have a scatter plot with a lot of points stacking on top of each other, alpha is what reveals where the actual density is. Without it you just see a blob.

plt.title("Temperature vs Total Rentals", fontsize = 16)
plt.xlabel("Temperature", fontsize=13)
plt.ylabel("Total Rentals", fontsize=13)

plt.xticks(fontsize = 11) # tick labels ≈ 2px smaller than axis labels
plt.yticks(fontsize = 11) # Ticks are the small labels/marks you see on the x and y axes
plt.grid(True)

plt.show()

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

# Second subplot (2 rows, 1 column, position 2)
plt.subplot(2,1,2)
plt.plot(df['temp'])
plt.title("Temperature Over Time")
plt.ylabel("Temperature")
plt.xlabel("Day Index")   # x-label only needed on the BOTTOM plot
plt.grid(True)

plt.tight_layout() # auto-adjusts spacing so titles/labels don't overlap
plt.show()

# By default, matplotlib doesn't support characters outside english alphabet, so if used any other character/language in labels, it will show garbled characters in place of labels and warning messages.

#-------------------------------------------------
# ONE VARIABLE ANALYSIS - LINE PLOT
# Use plt.plot() - plots points by connecting them on an x-ycoordinate system based on two sets of data. 
# useful when we want to see how something changes step by step. 
# Since our dataset records bike rentals on a daily basis, a line plot is an appropriate tool to examine how rental demand evolves over time.
# Refer to plot given by code lines 91-99

# Line plot is especially useful for time-series data. 
# Since actual dates have more information than indeces, we should use the actual date variable dteday

# By default matplotlib auto-picks tick positions and labels
# But when your x-axis is DATES, the auto ticks look messy this is why we manually control them with mdates
# "2011-01-01" (string) => datetime(2011, 1, 1) 
# matplotlib needs actual datetime objects to space dates correctly on axis

import matplotlib.dates as mdates
 
df['dteday'] = pd.to_datetime(df['dteday'])

plt.figure(figsize = (12, 5))
plt.plot(df['dteday'], df['cnt'])

plt.title("Daily Bike Rentals Over Time")
plt.xlabel("Date")
plt.ylabel("Total Daily Rentals")

plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

"""
# plt.gca() - "get current axes" - returns the Axes object matplotlib is currently drawing on
- you need this when the setting you want lives INSIDE the axes object, not at the top-level plt.something() level

# .xaxis and .yaxis - every Axes object has two sub-objects: .xaxis and .yaxis
- these control everything about that axis: ticks, labels, formatting

# .set_major_locator(mdates.MonthLocator())
- LOCATOR = decides WHERE ticks are placed (the positions)
- MonthLocator() = "place a tick at the 1st of every month"
# other options:
-   mdates.YearLocator()   -- one tick per year (for long timelines)
-   mdates.WeekdayLocator() -- one tick per week
-   mdates.DayLocator(interval=7) -- every 7 days

# .set_major_formatter(mdates.DateFormatter('%Y-%m'))
- FORMATTER = decides HOW tick labels are displayed (the text)
- DateFormatter takes a format string, same as Python's strftime:
- '%Y-%m'       -- "2011-01"  (year-month, what this code uses)
- '%b %Y'       -- "Jan 2011" (more readable for presentations)
- '%d/%m/%Y'    -- "01/01/2011" (day level detail)
- '%b'          -- "Jan" only (when year is obvious from context)

# LOCATOR sets positions, FORMATTER sets appearance - two separate jobs - you can mix and match them independently

# plt.xticks(rotation=45) -  needed whenever labels are long enough to overlap each other
- common values: 45 (diagonal), 90 (vertical), 0 (default horizontal)
- rotation=90 saves more space but is harder to read quickly """

# OBSERVATIONS/ INTERPRETATION OF THE LINE PLOT:
# 1. Upward trend: year 2 rentals are generally higher than year 1 => growth in popularity over time
# 2. Seasonality: demand peaks mid-year, dips at start/end of each year => repeating yearly cycle driven by weather
# 3. Short term volatility: Sharp drops even during high demand periods => disruptions like bad weather, holidays etc.
# SUMMARY: demand is NOT RANDOM => has structure seen in long term growht, yearly seasonality, daily noise.

#-------------------------------------------------
# ONE VARIABLE ANALYSIS - PLOTTING MULTIPLE LINES
# We just use plt.plot twice to plot two lines in one axes (not different subplots)
# Total demand consists of two distinct user groups: casual users and registered users. 
# Understanding their behavior separately will allow us to identify differences in usage patterns and customer structure
# Main Questions: 
#   How do rental patterns differ between casual and registered users over time?
#   Are both groups growing similarly, and do they respond differently to seasonal changes?

plt.figure(figsize = (12,5))

plt.plot(df['dteday'], df['casual'], label = "Casual Users")  # label is for legend - visible only if plt.legend called 
plt.plot(df['dteday'], df['registered'], label = "Registered Users")

plt.title("Casual vs Registered Bike Rentals Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Rentals")

plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation = 45)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# OBSERVATIONS/ INTERPRETATION OF THE LINE PLOT:
# 1. Registered Dominates: registered counts consistently >> casual throughout both years => system relies primarily on recurring customers
# 2. Both show Seasonality: same mid-year peak pattern in both groups => overall demand driven by season regardless of user type
# 3. Growth is registered-led:  year 1 -> year 2 uplift visible in both, but stronger in registered => total demand growth = more regular customers, not more casual ones
# SUMMARY: two structurally different user types --
#    registered = stable, growing, predictable base
#    casual     = seasonal, volatile, occasional
# matters for: forecasting, pricing, marketing strategy

# MOVING AVERAGE - Smooth out daily noise to reveal a clear trend in data
df["rolling_7"] = df["cnt"].rolling(window=7).mean()
# .rolling(window=7).mean() : 
#   window=7  = look at 7 consecutive rows at a time
#   .mean()   = take the average of those 7 values
#   result    = each point becomes the avg of itself + 6 days before it
#   creates a new column "rolling_7" in the dataframe
#
# WHY: daily data is noisy (spikes/drops everywhere)
#      moving average filters that noise so the underlying trend is visible
#      window=7 = weekly smoothing (common choice for daily data)

plt.figure(figsize=(10, 5))

# plotting TWO lines on the same axes — just call plt.plot() twice
# alpha=0.4 on daily = faded, so the moving avg stands out on top
# label= is what appears in the legend
plt.plot(df["dteday"], df["cnt"], alpha=0.4, label="Daily")
plt.plot(df["dteday"], df["rolling_7"], linewidth=2, label="7-day avg")
# linewidth=2 = thicker line for the moving average (more prominent)
plt.title("Daily Rentals and 7-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Total Rentals")

plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.legend()
plt.tight_layout()
plt.show()

# ANSWERS FROM THE CHART:
# Q: Does the moving average make the trend clearer?
# A: yes, the extreme points get smoothened out by taking average - less noise

# Q: What information is lost when smoothing the data?
# A: Short term volatilations such as daily disruptions in trend due to bad weather, holiday etc. 

# Q: When would a manager prefer the smoothed version?
# A: For calculations such as averages, identifying long-term trends, etc

#-------------------------------------------------
# ONE VARIABLE ANALYSIS - HISTOGRAM
# Displays frequency (the nubmer of occurrences) or distribution of values in a dataset 
# Line plot: trends over time
# Histogram: overall shape and spead of data

# Use plt.hist() to make histogram, with arguments such as bins and range
# bins: specify no. of bins i.e. the continuous, non-overlapping intervals or their edges
# range: to specify upper and lower range of bins

# Basic histogram: 
plt.hist(df['cnt'])
plt.show()

# Add number of bins - controls granularity
# Add color and edge lines - improves readability
# Add title and axis labels - communicate what the plot shows

# 1. Increase bins
plt.figure(figsize = (8,5))
plt.hist(df['cnt'], bins =20)
plt.show()

# 2. Add edge
plt.figure(figsize = (8,5))
plt.hist(df['cnt'], bins =20, edgecolor = 'black')
plt.show()

# 3. Add labels and title
plt.figure(figsize=(8, 5))
plt.hist(df['cnt'], bins=20, edgecolor='black')
plt.title("Distribution of Daily Bike Rentals")
plt.xlabel("Total Daily Rentals")
plt.ylabel("Frequency")
plt.show()

# Use help(plt.hist) to see all available parameters

# By default - bin edges are decided automatically - but this may produce intervals that are difficult to interpret. 
# Adjust bins by: 
# Specifying number of bins (bins=20) =Matplotlib chooses evenly spaced edges automatically
# Specify exact edges(bins=[0, 1000, 2000, 3000,...]) =gives full control

# Defining bins manually:
bin_edges = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
plt.figure(figsize = (8,5))
plt.hist(df['cnt'], bins = bin_edges, edgecolor = 'black')
plt.title("Histogram of Daily Bike Rentals (Custom Bins)")
plt.xlabel("Total Daily Rentals")
plt.ylabel("Frequency")
plt.show()

# Bin edges can also be generated programmatically using:
# np.arange(start, stop, step) 
# Example: bin_edges = np.arange(0, 9500, 500)  (till 9500 so 9000 is included)

#-------------------------------------------------
# ONE VARIABLE ANALYSIS - BOXPLOT
# boxplot (box-and-whisker plot) provides a compact statistical summary of a numerical variable
# It visualizes:
# Median (center line inside the box)
# Interquartile Range (IQR) — the middle 50% of observations (Q1 to Q3)
# Outliers — unusually high or low observations

# useful for: Detecting skewness, Identifying extreme values, Comparing distributions across groups
# by default- vertical, set vert = False for horizontal boxplot in plt.boxplot
plt.figure(figsize=(8, 3))
plt.boxplot(df["cnt"], vert=False) # can add argument patch_artist=True, converts the boxes in a boxplot from simple, hollow line paths into filled, customizable shapes (or "patches")- fill the boxes with color, apply gradients, and style them individually

plt.title("Boxplot of Daily Bike Rentals", fontsize=14)
plt.xlabel("Total Daily Rentals", fontsize=12)

plt.yticks([])  # remove unnecessary categorical axis
plt.grid(axis="x", linestyle="--", alpha=0.6)

plt.show()

# OBSERVATIONS/ INTERPRETATION OF THE BOXPLOT:
# MEDIAN (~4500) - middle line of the box => typical day has ~4500 rentals
# IQR (box spans ~3000 to ~6000)- interquartile range = middle 50% of all days fall here. wide box = moderate variability in daily demand
# WHISKERS (extend far in both directions) - show full range of non-outlier values
# no isolated dots = no extreme outliers in this dataset
# SUMMARY: demand is stable around the median but dispersed => staffing and bike allocation must stay flexible

""" BOXPLOT ANATOMY (reference)

   |----[  |  ]-------|
   ^    ^  ^  ^       ^
whisker Q1 Q2 Q3    whisker
           median(Q2) in the middle

# box    = IQR = Q1 to Q3 = middle 50% of data
# median = line inside the box
# whiskers = 1.5 * IQR beyond Q1/Q3
# dots beyond whiskers = outliers"""

#-------------------------------------------------
# TWO VARIABLE ANALYSIS - SCATTER PLOTS
# Numeric vs. Numeric

# scatter plot is a graph that plots points on anx-ycoordinate system based on two sets of data
# Useful for identifying: Direction of relationship (positive or negative), Strength of association, Patterns such as linearity or curvature, Outliers

plt.figure(figsize = (8,5))
plt.scatter(df['temp'], df['cnt'], alpha = 0.6)

plt.title("Temperature vs Total Rentals", fontsize = 14)
plt.xlabel("Temperature", fontsize = 12)
plt.ylabel("Daily Rentals", fontsize = 12)

plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)

plt.tight_layout()
plt.show()

# OBERVATIONS/ INTERPRETATION OF THE SCATTER PLOT:
# 1. Positive Relationship: As temp. increases, cnt generally increases; low temp = low rentals; high temp = high rentals
# 2. Not Perfectly Linear: Spread widens at moderately high temp => demand becomes more variable in good weather (other factors also matter, weekday, holiday etc.)
# SUMMARY: Temp. is an important driver of demand, but no the only one.

#-------------------------------------------------
# THREE VARIABLE ANALYSIS - BUBBLE CHART
# Extention of scatter plots - adds a third variable using the size of each point
# Example: temp vs cnt chart, bubble size = windspeed
# Larger bubbles correspond to higher wind speeds, allowing us to visually assess whether windy conditions are associated with changes in rental demand

plt.figure(figsize = (8,5))

# Scale windspeed so bubbles have an appropriate size
bubble_size = df["windspeed"] * 1000

plt.scatter(df["temp"],df["cnt"], s = bubble_size, alpha=0.4)
# s = size of each dot in points^2 (area, not radius)
# if s is a single number  -- all dots same size (regular scatter)
# if s is an array/column  -- each dot gets its own size (bubble chart)
# values are in points^2 (area) so scale up small decimals: col * 500

plt.title("Temperature vs Rentals (Bubble Size = Wind Speed)", fontsize=14)
plt.xlabel("Temperature", fontsize=12)
plt.ylabel("Total Rentals", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.5)

plt.show()

# OBSERVATIONS/ INTERPRETATIONS OF THE BUBBLE CHART
# X vs Y (temp vs cnt) - same positive relationship as before - temp drives demand
# BUBBLE SIZE (windspeed)
# if windspeed suppressed demand, large bubbles would cluster LOW on y, but large bubbles appear at ALL rental levels with no clear pattern
# => windspeed does NOT appear to be a dominant driver of demand
# SUMMARY: temp matters, windspeed probably doesn't (much)
# next step would be correlation value to confirm this visually

# When dataset is large, bubble charts can become cluttered so consider: reducing bubble size, increasing transparency, sampling the data, or other alternate graphs

#-------------------------------------------------
# TWO VARIABLE ANALYSIS - BAR PLOT
# Categorial vs Numeric
# Each bar represents a category, and the bar height corresponds to the value being compared
# By default - the graph will be left-aligned, can use align = 'center' etc argument to specify

# Average rentals by season
# Convert season codes to readable labels
season_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}

season_avg = (
    df.assign(Season=df["season"].map(season_map))
    # .assign()        -- creates a NEW column called "Season" (capital S)
    #                     without modifying the original df
    # .map(season_map) -- replaces each number with its name
    #                     1 -> "Winter", 2 -> "Spring" etc.
    # df still has "season" (numbers), new col "Season" has the names
      .groupby("Season")["cnt"]
    # .groupby("Season") -- splits the dataframe into 4 groups, one group per season name
    # ["cnt"]            -- from each group, only keep the cnt column
      .mean()
    # calculates the average cnt for each season group
    # result so far: Season -> avg cnt (but order may be random)
      .reindex(season_map.values())
    # season_map.values() = ["Winter", "Spring", "Summer", "Autumn"]
    # .reindex() forces the result into THAT specific order
    # without this, groupby returns seasons alphabetically
    # (Autumn, Spring, Summer, Winter) which is not logical order
)

# season_avg is now a pandas Series:
#   index  = ["Winter", "Spring", "Summer", "Autumn"]
#   values = [avg_cnt,  avg_cnt,  avg_cnt,  avg_cnt]

plt.figure(figsize=(8,5))
plt.bar(season_avg.index, season_avg.values)  #plt.bar - vertical bar chart
# .index  = x-axis labels (season names)
# .values = bar heights (average rental counts)

plt.title("Average Bike Rentals by Season")
plt.xlabel("Season")
plt.ylabel("Average Rentals")

plt.show()

# OBERVATIONS/ INTERPRETATION OF THE BAR CHART:
# 1. Seasonal Patter: Demand rises as weather becomes warmer, lowest in winter, substantial but still less than summer in autumn
# SUMMARY: Rental demand DOES depend on seasonal weather conditions

# plt.barh() - HORIZONTAL BAR CHART
# use when: category labels are long, or comparing many categories
# axes are REVERSED vs plt.bar(): y-axis = categories, x-axis = values

# THE SLICING: [::-1]
# this is standard Python list/array reverse slicing
# syntax: [start:stop:step]
#   start = blank (from beginning)
#   stop  = blank (to end)
#   step  = -1 (go backwards)
# result = entire array, reversed
#
# WHY reverse here:
# season_avg order = ["Winter", "Spring", "Summer", "Autumn"]
# barh() draws from BOTTOM to TOP
# so without [::-1]: Winter ends up at bottom, Autumn at top
# with [::-1]:        Autumn at bottom, Winter at top
# = logical reading order top-to-bottom matches natural season flow

plt.figure(figsize=(8,5))
plt.barh(season_avg.index[::-1], season_avg.values[::-1])
# both index AND values must be reversed together
# if you reverse only one, labels and bars get mismatched
plt.title("Average Bike Rentals by Season")
plt.ylabel("Season")
plt.xlabel("Average Rentals")

plt.show()

# Average Temperature by Season
season_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}

temp_avg = (
    df.assign(
        Season=pd.Categorical(
            df["season"].map(season_map),
            categories=season_map.values(),
            ordered=True
        )
    )
    .groupby("Season")["temp"]
    .mean()
)

temp_avg.plot(kind="bar", figsize=(8,5), title="Average Temperature by Season")

plt.xlabel("Season")
plt.ylabel("Average Temperature")
plt.show()

print("Season with highest average temperature:", temp_avg.idxmax())

"""
# pd.Categorical(df["season"].map(season_map), categories=season_map.values(), ordered=True)
# categories=  -- sets order of appearance (Winter, Spring, Summer, Autumn)
# ordered=True -- enables comparisons: Winter < Spring < Summer < Autumn
#                 WITHOUT this: .idxmax() errors, filtering by > < breaks
#                 you need BOTH: categories= for display, ordered= for logic
#
# .idxmax() -- returns the LABEL of the max value ("Summer")
# .idxmin() -- returns the LABEL of the min value
# vs .max() which returns the NUMBER itself
#
# pandas .plot(kind="bar") vs plt.bar()
# both produce identical output -- pandas calls matplotlib under the hood
# pandas .plot() -- quicker, index auto-becomes x-axis, good for fast EDA
# plt.bar()      -- more control, better for production/custom charts
"""

#-------------------------------------------------
# GROUPED BAR CHARTS
# Comparing multiple aggregated values side by side
# Example: casual vs registered rentals by season

casual_avg = df.groupby("season")["casual"].mean()
registered_avg = df.groupby("season")["registered"].mean()

# np.arange(len(casual_avg)) -- creates [0, 1, 2, 3]
# these are the CENTER x positions for each season group
# we then shift each bar left/right from this center
x = np.arange(len(casual_avg))
width = 0.35   # how wide each bar is (in axis units)

plt.figure(figsize=(8, 5))

# x - width/2 -- shifts casual bar LEFT of center
# x + width/2 -- shifts registered bar RIGHT of center
# result: two bars sit side by side, centered on each x tick
# if you had 3 groups: x-width, x, x+width
plt.bar(x - width/2, casual_avg.values, width, label="Casual")
plt.bar(x + width/2, registered_avg.values, width, label="Registered")

plt.title("Average Rentals by Season", fontsize=14)
plt.xlabel("Season", fontsize=12)
plt.ylabel("Average Rentals", fontsize=12)

# plt.xticks(x, [...]) -- place ticks AT positions x, label them with season names
# without this: x-axis shows 0,1,2,3 (the np.arange numbers)
plt.xticks(x, ["Winter", "Spring", "Summer", "Fall"])

plt.legend(title="User Type")  # title= adds a heading inside the legend box
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show() 

#-------------------------------------------------
# STACKED BAR CHARTS - very commonly used
# displays multiple numerical values stacked on top of each other within each category
# Allows us to:
# - Compare total values across categories
# - See how each component contributes to the total

season_totals = df.groupby("season")[['casual', 'registered']].mean()
casual_avg = season_totals['casual'].values
registered_avg = season_totals['registered'].values

x = np.arange(len(season_totals)) # creating x positions
plt.figure(figsize = (8,5))

# To stack bars, we call plt.bar twice and then define bottom for the second layer as the first layer
plt.bar(x, casual_avg, label = "Casual")
plt.bar(x, registered_avg, bottom = casual_avg, label = "Registered")

plt.title("Average Rentals by Season (Stacked)", fontsize=14)
plt.xlabel("Season", fontsize=12)
plt.ylabel("Average Rentals", fontsize=12)

plt.xticks(x, ["Winter", "Spring", "Summer", "Fall"])  # fixed order

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# useful for showing part-to-whole relationships within categories
# The first plt.bar() call forms the base.
# The second plt.bar() call stacks on top by using the values of the lower bar as the starting point.
# Example especially visible in workingday grouped data by casual vs registered users - for marketing purposes

#-------------------------------------------------
# PIE CHARTS
# plt.pie()
# controversial - usually that info can be expressed better through bar charts or other charts 

#-------------------------------------------------
# SUBPLOTS AND MULTI-PANEL FIGUES
# useful for at-a-glance comparisions of multiple variables or comparing different perspectives of the same problem
# But don't use too many in one figure, can be hard to read or too small - prioritize clarity over quantity
# multi-panel layout allows us to visually compare:
# - Strength of association
# - Direction of relationship
# - Variability

#-------------------------------------------------
# SEABORN 
# Built on top of matplotlib
# works directly with DataFrames (no .values, no manual x/y extraction)
# better default styling out of the box
# hue allows us to add a second categorical variable that automatically separates groups using color

# Before creating grouped charts, convert numeric category codes into readable lables by mapping
df["Season"] = df['season'].map(season_map)
df['Working Day'] = df['workingday'].map({0: "No", 1: "Yes"})

# Instead of manually controlling bar positions, we specify:
# x = primary categorical variable
# y = numerical variable to summarize
# hue = secondary categorical variable
""" 
Seaborn automatically: 
# Groups the data
# Computes summary statistics (mean by default)
# Positions bars side by side
# Applies distinct colors
# Generates a legend 
"""

# Main Question:
# Do working days and non-working days show different rental patterns, and does this difference vary by season?
# here we use x axis category = working day, y axis numerical = cnt, season = hue variable
plt.figure(figsize = (9,5))
sns.barplot(
    data = df,          # pass the whole DataFrame directly
    x = 'Working Day',
    y = 'cnt', 
    hue = 'Season',     # splits each x group into colour-coded subgroups
    estimator = 'mean', # what to calculate for bar height (mean, median, sum, count etc.)
    errorbar = None     # by default seaborn draws confidence interval lines; errorbar=None turns those off for a cleaner chart
)   
"""
# errorbar= is relevant in any seaborn plot that AGGREGATES data (calculates mean/median across groups internally)
# USES errorbar:
sns.barplot()      - most common place you'll see it
sns.pointplot()    - dots connected by lines, shows CI by default
sns.lineplot()     - when aggregating multiple y values per x point

# DOESN'T USE errorbar:
sns.scatterplot()  - plots raw points, no aggregation
sns.histplot()     - counts frequencies, no aggregation
sns.boxplot()      - distribution IS the point, CI unnecessary
sns.heatmap()      - no aggregation involved

# RULE OF THUMB:
- if the plot is calculating a summary statistic (mean, median etc.) per group internally - errorbar= is available
- if it's plotting raw data points - errorbar= is irrelevant
"""
plt.title("Average Rentals by Working Day and Season")
plt.xlabel("Working Day")
plt.ylabel("Average Rentals")

plt.tight_layout()
plt.show()

# sns.scatterplot() for scatter plot with hue, use alpha parameter here for clariy and showing density

# BOXPLOT WITH HUE - IMPORTANT
plt.figure(figsize = (10,6))
sns.boxplot(
    data = df,
    x = 'weathersit',
    y = 'cnt',
    hue = 'Season',
)
plt.title("Rental Distribution by Weather and Season")
plt.xlabel("Weather Category")
plt.ylabel("Total Rentals")

plt.tight_layout()
plt.show()

# OBSERVATIONS/ INTERPRETATIONS FROM THE BOXPLOT:
# 1. Weather Drives Demand: 
# weather 1 (best) = highest rentals across all seasons
# weather 3 (worst) = lowest rentals across all seasons
# => poor weather suppresses demand regardless of season

# 2. Season Still Matters Within the Same Weather:
# summer/autumn consistently higher than winter even at same weather category
# => weather + season are BOTH independent drivers (not just one)

# 3. Variability Pattern
# clear weather + warm season = widest IQR
# => good conditions produce both normal commute days AND high peak days
# bad weather = narrow IQR (consistently low, less variation)
#
# SUMMARY: weather and season jointly drive demand
# best case: clear weather + summer/autumn = highest + most variable
# worst case: poor weather + winter = lowest + most consistent (consistently low)

#-------------------------------------------------
# CORRELATION ANALYSIS AND HEATMAPS
# A correlation matrix shows how all numerical variables relate to each other simultaneously.
# Refer notes on correlation 

# Instead of automatically selecting every numeric column, we explicitly choose relevant features to avoid unnecessary noise.
# Selecting relevant numerical variables:
numeric_df = df[[
    "temp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt"
]]

# computing correlation matrix:
corr_matrix = numeric_df.corr()   # A dataframe of correlation values
print(corr_matrix)
# Each cell represents the correlation between two numbers.
# But reading so many numbers is difficult - so we use a heatmap to interpret correlations using a color gradient
# Further improve clarity by:
# 1. Centering the color scale at 0
# 2. Using consistent formatting
# 3. Adding square cells

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,   
    annot=True,    # print the correlation NUMBER inside each cell; without this: only colour, no numbers

    cmap="coolwarm",  # colour palette
                      # coolwarm: blue = negative, white = ~0, red = positive
                      # other options: "viridis", "RdBu", "YlOrRd"

    fmt=".2f",     # format of the annotated numbers; ".2f" = 2 decimal places, only needed when annot=True

    center=0,      # tells seaborn what value = white (neutral colour), center=0 means 0 correlation = white
                   # important for diverging colourmaps like coolwarm

    square=True,   # forces each cell to be a perfect square
                   # without this: cells stretch to fill the figure shape

    linewidths=0.5 # adds thin lines between cells for readability
                   # 0 = no lines, higher = thicker borders
)

plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

# HOW TO READ THE OUTPUT:
# diagonal = always 1.0 (variable perfectly correlates with itself)
# dark red cell = strong positive correlation
# dark blue cell = strong negative correlation
# white/pale cell = weak or no correlation
# symmetric: top-right mirrors bottom-left (always true for corr matrix)

"""
Correlation analysis has limitations:
- Captures only linear relationships
- does not imply causation
- does not control for other variables
- can be influenced by outliers
"""

#-------------------------------------------------
# REGRESSION AND TREND ANALYSIS
# Instead of only visualizing raw scatter points, we estimate a regression line that summarizes the average relationship between variables.
# Regression helps us:
# - Estimate the direction of association
# - Approximate the strength of a linear relationship
# - Visually summarize noisy data

# Adding Regression Line to temp vs rental demand scatter plot:
# sns.regplot() -- scatter plot + regression line in one call
# also draws a shaded confidence interval band around the line
plt.figure(figsize=(8, 5))
sns.regplot(
    data=df,
    x="temp",
    y="cnt",

    scatter_kws={"alpha": 0.5}
    # kws = "keyword arguments" - extra settings passed to the scatter part
    # scatter_kws affects ONLY the dots, not the line
    # line_kws= would affect ONLY the regression line
    # e.g. line_kws={"color": "red", "linewidth": 2}
)

plt.title("Regression: Temperature vs Total Rentals")
plt.xlabel("Temperature")
plt.ylabel("Total Rentals")
plt.tight_layout()
plt.show()

# The regression line summarizes the overall linear trend.
# If the line slopes upward: Higher temperatures are associated with higher rental demand.
# If the slope slopes downward: Higher temperatures are associated with lower demand.
# The shaded region around the regression line represents a confidence interval, indicating uncertainty around the estimated trend.

# REGRESSION BY SEASON
# regplot() draws a single regression line
# sns.lmplot() allows us to estimate separate regression lines for different groups using the hue parameter
# regplot cannot split by colour -- lmplot can
# draws one scatter + regression line PER hue category
sns.lmplot(
    data=df,
    x="temp",
    y="cnt",
    hue="Season",
    height=6,  # lmplot uses height= not figsize
               # height = height of the figure in inches
    aspect=1.3,  # width = height * aspect
                 # so here: width = 6 * 1.3 = 7.8 inches
                 # aspect > 1 = wider than tall
                 # aspect < 1 = taller than wide
    scatter_kws={"alpha": 0.4}
)

plt.title("Temperature vs Rentals by Season")
plt.tight_layout()
plt.show()

# This visualization allows us to observe:
# 1. Whether slopes differ across seasons
# 2. Whether some seasons are more temperature-sensitive
# 3. Whether baseline rental levels differ
# 4. Whether relationships appear stronger or weaker by season

""" axes-level vs figure-level is a seaborn distinction:
# axes-level (regplot, barplot, boxplot):
#     draws INTO an existing plt axes, works with plt.subplot()
# figure-level (lmplot, relplot, catplot):
#     creates its OWN figure, height+aspect instead of figsize
#     plt.title() sometimes needs to go AFTER the sns call
"""

# Regression plots:
# Show association, not causation
# Assume a linear relationship
# May hide nonlinear effects
# Can be influenced by outliers

# If correlation is strong, the regression line will typically show a clear slope.
# If correlation is weak, the regression line will appear flatter.

#-------------------------------------------------
# SUMMARY:
# WHAT THE DATA SHOWED:
#   rental demand fluctuates over time (trend + seasonality + noise)
#   temperature = strong positive driver of demand
#   working days vs weekends = different demand patterns
#   seasonal effects = substantial (summer/autumn >> winter)
#   weather conditions = affect variability, not just average
#   registered users = consistent and predictable
#   casual users = seasonal, volatile, weather-sensitive

# WHAT THIS MEANS IN PRACTICE:
#   staffing       -- scale up in summer, reduce in winter
#   redistribution -- move bikes to high-demand zones on clear/warm days
#   seasonal plan  -- prepare for demand swings between quarters
#   forecasting    -- weight temperature + weather as key input variables

#-------------------------------------------------
# APPENDIX: KDE etc -- see separate session notes (to be added)