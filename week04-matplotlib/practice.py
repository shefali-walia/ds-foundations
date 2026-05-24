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
print(df.head())
print("DataFrame shape: ", df.shape)

#-------------------------------------------------
# LINE PLOT - AQI OVER TIME FOR ONE CITY (DELHI)
import matplotlib.dates as mdates
df['Date'] = pd.to_datetime(df['Date'])
# Converting date(string) to date(datetime)

delhi = df[df['City'] == 'Delhi'].sort_values('Date')
plt.figure(figsize = (12,5))
plt.plot(delhi['Date'], delhi['AQI'])

plt.title("AQI Over Time For Delhi")
plt.xlabel("Dates")
plt.ylabel("AQI")

plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation = 45, fontsize = 6)
plt.grid(True)
plt.tight_layout()
plt.show()

print(delhi['Date'].head())
print(delhi['Date'].dtype)

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
plt.figure(figsize = (8,6))
plt.hist(df['AQI'], bins = 30, edgecolor = 'black', range = (0,800))

plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.grid(True, linestyle = '--', alpha = 0.5)

plt.show()

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
plt.figure(figsize = (8,5))
sns.boxplot(
    data = df,
    x = 'Season',
    y = 'AQI'
)
plt.title("AQI by Season")
plt.xlabel("Seasons")
plt.ylabel("AQI")
# Can add plt.ylim(0,800) but the outliers going up 2000+ are real data points
plt.grid(True, axis = 'y', alpha = 0.5)
plt.tight_layout()
plt.show()

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
# SCATTER PLOT - PM2.5 vs AQI - ARE THEY CORRELATED?
# sns.regplot - scatter+ regression line both 
plt.figure(figsize = (8,6))
sns.regplot(
    data = df,
    x = 'PM2.5',
    y = 'AQI',
    scatter_kws = {"alpha": 0.4}
    )

plt.title("PM2.5 vs. AQI")
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.tight_layout()
plt.show()

# INTERPRETATION - PM2.5 vs AQI - ARE THEY CORRELATED?
# DIRECTION: Positive but clustered under point (200,500)
# STRENGTH: MODERATE AT BEST - Fan shape and vertical cluster near PM2.5~0-100 with AQI ranging 0-2000
# SHAPE: FAN-SHAPED(heteroscedasticity) the spread of AQI values gets wider as PM2.5 increases. At low PM2.5 the points are tight, at high PM2.5 they scatter broadly
# - Upwards shape cluster seen around PM2.5 ~ 100
# OUTLIERS: Seen in both extreme directions towards PM2.5 500 onwards and AQI 500-600 onwards, also seen around PM2.5 ~ 300

#-------------------------------------------------
# SUBPLOTS - DASHBOARD
# Four subplots: Delhi AQI line, AQI histogram, AQI by season boxplot, PM2.5 vs AQI scatter
# 2 rows, 2 columns
plt.figure(figsize = (10, 8))

# subplot 1 - Delhi AQI Line
plt.subplot(2,2,1)
plt.plot(delhi['Date'], delhi['AQI'])

plt.title("AQI Over Time For Delhi")
plt.xlabel("Dates")
plt.ylabel("AQI")

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # display one tick every 6 months instead of every month- less crowded
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation = 45, fontsize = 5)
plt.grid(True)

# subplot 2 - AQI Histogram
plt.subplot(2,2,2)
plt.hist(df['AQI'], bins = 30, edgecolor = 'black', range = (0,800))

plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.grid(True, linestyle = '--', alpha = 0.5)

# subplot 3 - AQI by Season Boxplot
plt.subplot(2,2,3)
sns.boxplot(
    data = df,
    x = 'Season',
    y = 'AQI'
)
plt.title("AQI by Season")
plt.xlabel("Seasons")
plt.ylabel("AQI")
plt.xticks(fontsize = 8)
plt.grid(True, axis = 'y', alpha = 0.5)

# subplot 4 - PM2.5 vs AQI scatter
plt.subplot(2,2,4)
sns.regplot(
    data = df,
    x = 'PM2.5',
    y = 'AQI',
    scatter_kws = {"alpha": 0.4}
    )

plt.title("PM2.5 vs. AQI")
plt.xlabel("PM2.5")
plt.ylabel("AQI")

plt.tight_layout()
plt.show()

#-------------------------------------------------
# HEATMAP CORRELATION MATRIX
corr_matrix = df[['PM2.5', 'PM10', 'NO2', 'AQI']].corr()

plt.figure(figsize = (8,6))
sns.heatmap(
    corr_matrix,
    annot = True,
    cmap = 'coolwarm',
    center = 0,
    fmt = '.2f',
    linewidths = 0.5,
    square = True
)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

# INTERPRETATION - HEATMAP CORRELATION MATRIX
# HIGHEST POLLUTANT: PM10 vs AQI is 0.80 which is highest correlation of the three pollutants
# LEAST POLLUTANT: NO2 has the weakest correlation with AQI (0.54)
# Also strong correlation between PM2.5 and PM10 ~ 0.83
# Very weak correaltion between NO2 and PM2.5 

#-------------------------------------------------
# SUMMARY AND KEY FINDINGS:
# CHART 1 - AQI OVER TIME (DELHI):
# Clear seasonal cycling — AQI peaks every winter, drops every monsoon
# Pattern repeats consistently 2015–2020 — not random spikes, structural seasonality

# CHART 2 - AQI DISTRIBUTION (HISTOGRAM):
# Right-skewed — most cities cluster between 50–200, long tail toward 800+
# Center ~100, wide spread — India's baseline AQI is already at the "Poor" threshold

# CHART 3 - AQI BY SEASON (BOXPLOT):
# Winter: highest median (~150), most extreme outliers (2000+)
# Monsoon: lowest median (~90) — rain clears particulate matter
# Post-Monsoon: widest IQR — high variability as pollution builds back up
# Winter dominance explained by cold air trapping + crop stubble burning + Diwali

# CHART 4 - MEAN AQI PER CITY - TOP 10 (BAR CHART):
# Ahmedabad: ~450 mean AQI — nearly double Delhi (2nd place at ~260)
# All top 10 cities exceed the 100 "Poor" threshold year-round — systemic, not seasonal
# Ahmedabad's mean is 4.5x the Poor threshold

# CHART 5 - PM2.5 VS AQI (SCATTER + REGRESSION):
# Positive correlation — as PM2.5 rises, AQI rises
# Fan-shaped spread (heteroscedasticity) — correlation weakens at high PM2.5 values
# Moderate strength — dense cluster near origin but wide scatter beyond PM2.5 ~100
# Linear regression undersells the complexity here — a non-linear model would fit better

# CHART 6 - DATETIME X-AXIS FIX:
# Converted Date to datetime, applied DateFormatter('%b %Y'), interval=6 months
# Lesson: always convert date columns before plotting time series

# CHART 7 - 2x2 SUBPLOT DASHBOARD:
# Full story in one figure: trend + distribution + seasonal pattern + pollutant relationship
# tight_layout() essential for subplot readability

# CHART 8 - CORRELATION HEATMAP:
# PM10 strongest predictor of AQI (r=0.80)
# PM2.5 also strong (r=0.65) — both particulate matter pollutants dominate AQI
# NO2 weakest (r=0.54) — still moderate but less predictive
# PM2.5 and PM10 highly correlated with each other (r=0.83) — likely co-emitted sources