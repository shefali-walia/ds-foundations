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
