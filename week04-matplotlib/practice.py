# PRACTICING MATPLOTLIB AND SEABORN - EXPLORING AIR QUALITY DATA IN INDIA (2015-2020) 

# lOADING LIBRARIES
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

#-------------------------------------------------
# LOADING THE CLEANED DATA 
base = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base, '..', 'week03-pandas', 'aqi_cleaned.csv'))  
# '..' to go one level up- outside week04 folder and then look for the rest of the path
# print(df.head())
print("DataFrame shape: ", df.shape)

#-------------------------------------------------
# LINE PLOT - AQI OVER TIME FOR ONE CITY (DELHI)
import matplotlib.dates as mdates
df['Date'] = pd.to_datetime(df['Date'])
# Converting date(string) to date(datetime)

# delhi = df[df['City'] == 'Delhi'].sort_values('Date')
# plt.figure(figsize = (12,5))
# plt.plot(delhi['Date'], delhi['AQI'])

# plt.title("AQI Over Time For Delhi")
# plt.xlabel("Dates")
# plt.ylabel("AQI")

# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
# plt.xticks(rotation = 45, fontsize = 6)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# print(delhi['Date'].head())
# print(delhi['Date'].dtype)

# INTERPRETATION — Delhi AQI Over Time (2015–2020):
# TREND:  Slight downward trend from 2015 to 2020 — peaks appear lower in later years
#         but baseline remains dangerously high throughout (200-500)
# SEASONALITY: Clear seasonal cycle - AQI drops during monsoons (July - Sept) and peaks in winters Oct-Jan (probably due to crop-burning)
# VOLATILITY: High short-term volatility — daily swings of 100-200 AQI points
#             even within seasons, suggesting local/event-driven spikes
# BREAKPOINTS: Lowest during July 2017 (37) and highest went in Nov 2016 (700+)
#              Nov 2016 spike likely coincides with Diwali + post-harvest burning
# SO WHAT: 
# Still very high AQI overall, WHO safe limit is 50 so average runs atleast 4 times of that => not safe 
# Delhi's average AQI runs 4-6x the WHO safe limit year-round.
# Even the "clean" monsoon months regularly exceed 100 (Moderate category)
# If you're building a Climate AI model for air quality prediction, this chart tells you: seasonality is a strong signal (include month/season as a feature), but short-term volatility means you'll also need weather variables (wind, rain) to capture the daily noise.

#-------------------------------------------------
# HISTOGRAM - AQI DISTRIBUTION
# plt.figure(figsize = (8,6))
# plt.hist(df['AQI'], bins = 30, edgecolor = 'black', range = (0,800))

# plt.title("AQI Distribution")
# plt.xlabel("AQI")
# plt.ylabel("Frequency")
# plt.grid(True, linestyle = '--', alpha = 0.5)

# plt.show()

# INTERPRETATION - DISTRIBUTION OF AQI
# SHAPE: right-skewed (long tail towards 800 goes to right) - most values lie between 50-300
# CENTER: Around 100 - most cities have AQI around 100 across the years
# SPREAD: Wide (0-800) - high variance 
#         Most frequency clusters low but data itself is spread broadly (would have been narrow if let's say everything packed between 90-110)
# SKEW: Right-skewed 
# OUTLIERS: Few tiny bars past 500-600 (not dramatic outliers but sitll exist)

#-------------------------------------------------
# BOXPLOT - AQI BY SEASON 
# Two variables - use seaborn
# seasons themselves are already the grouping, so seaborn handles the separation automatically
# plt.figure(figsize = (8,5))
# sns.boxplot(
#     data = df,
#     x = 'Season',
#     y = 'AQI'
# )
# plt.title("AQI by Season")
# plt.xlabel("Seasons")
# plt.ylabel("AQI")
# # Can add plt.ylim(0,800) but the outliers going up 2000+ are real data points
# plt.grid(True, axis = 'y', alpha = 0.5)
# plt.tight_layout()
# plt.show()

# INTERPRETATION - AQI BY SEASON
# MEDIANS:
# WINTER: ~150, SUMMER: ~ 100, MONSOON: ~ 90, POST-MONSOON: ~ 130
# HIGHEST MEDIAN: Winter (~150) - worst AQI is in winters probably due to crop-burning + diwali etc. 
# LARGEST IQR (BOX HEIGHT): Post-Monsoon => high variabiliy 
# OUTLIERS: Highest outlier - Winter (2000+)
            # Most number of outliers - Summer (high density of dots from around 300-800)
            # Monsoon as well high density of outliers from 200-700

#-------------------------------------------------
# BAR CHART - MEAN AQI PER CITY (TOP 10)
# groupby city- mean aqi- sort- take top 10- barchart
df_top = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize = (8,5))
plt.bar(df_top.index, df_top.values)
# After the groupby chain, the result is already means — 
# so you just pass the whole series directly to the bar chart, not a columns

plt.title("Mean AQI per City (Top 10)")
plt.xlabel("City")
plt.ylabel("Mean AQI")
plt.xticks(fontsize = 8)

plt.axhline(100, color = 'red', linestyle = '--', label='Poor Threshold (100)') # axhline = axis horizontal line, axvline= axis vertical line
plt.legend()
plt.tight_layout()
plt.show()

# INTERPRETATION - MEAN AQI PER CITY (TOP 10)
# TALLEST BAR: Ahmedabad => extremely high mean AQI (~450)
# Ahmedabad's mean AQI is nearly double that of all other cities except Delhi
# All the cities' mean AQI is above the 100 "poor" threshold - indicating severely poor conditions
# All other cities' except Ahmedabad's mean AQI lies between 150-250

#-------------------------------------------------
