# PRACTICING PANDAS - EXPLORING AIR QUALITY DATA IN INDIA (2015-2020) 

# lOADING LIBRARIES
import pandas as pd
import os
# Since I already downloaded the data in my local system as CSV files, I don't need requests, zipfile, io libraries.

#-------------------------------------------------
# READING DATA
base = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base, 'air_quality_index_dataset', 'city_day.csv'))
print(df.head(5))

df.info() # no print needed — it prints itself
print(df.describe())  # needs print because internally it just returns a dataframe, it doesn't print itself.
print(df.head().T)  # needs print because internally it just returns a dataframe, it doesn't print itself.

# OBSERVATIONS FROM .info():
# 29531 rows, 16 columns
# Date is object (string) - needs pd.to_datetime() conversion
# City and Date - no nulls => good anchor columns
# Columns with significant nulls (non-null < 29531):
# PM2.5: 24933 (4598 missing)
# PM10:  18391 (11140 missing) - biggest gap
# NH3:   19203 (10328 missing)
# Xylene: 11422 (18109 missing) - more than half missing
# AQI:   24850 (4681 missing)

#-------------------------------------------------
# DATA SELECTION AND FILTERING
cols_to_use = ['City', 'Date', 'PM2.5', 'PM10', 'NO2', 'AQI', 'AQI_Bucket']
df_extracted = df[cols_to_use].copy()
print(df_extracted.head())
# .copy() — use when creating a subset/slice of a DataFrame that you plan to modify
# Without it: df_extracted is just a "view" pointing to the original df, modifying it may trigger SettingWithCopyWarning — pandas isn't sure whether you want to change the slice or the original, so it warns you.
# With .copy(): df_extracted becomes fully independent — no ambiguity, no warning.
# RULE: slice + plan to modify = always add .copy()

df_severe = df_extracted[df_extracted['AQI_Bucket'].isin(['Very Poor', 'Severe'])].head()
print(df_severe.head())
print(df_severe.shape) # gives (rows, columns) - to keep track of data, if you lost any data you shouldn't have, etc. ALWAYS INCLUDE AFTER EVERY MAJOR OPERATION (dropna, fillna, filter etc.)

#-------------------------------------------------
# HANDLING NULLS
print(df_extracted.isnull().sum()) # we get same observations from this as well about no. of nulls as from .info()
# To drop rows where AQI is null:
df_extracted = df_extracted.dropna(subset = ['AQI']) # subset=['col name'] tells pandas "only drop rows where THIS column is null, keep the whole DataFrame intact"
# df_extracted = df_extracted['AQI'].dropna() => wrong because this  dropping nulls on just the AQI column (a Series), not the whole DataFrame. So df_extracted is now just a Series of AQI values, not a DataFrame anymore. 
print(df_extracted.isnull().sum())
print(df_extracted.shape)
# Now we observe AQI and AQI_Bucket have no nulls whereas rest still do.

df_extracted['PM2.5'] = df_extracted['PM2.5'].fillna(df_extracted['PM2.5'].mean()) # fills nulls in PM2.5 with the column mean
print(df_extracted.isnull().sum())
print(df_extracted.shape)

"""RULE: Whatever object you call the method on, that's what gets returned.
* df_extracted = df_extracted.dropna(subset=['AQI'])
# RHS: called on DataFrame -> returns DataFrame -> LHS stores DataFrame -> MATCH 

* df_extracted = df_extracted['AQI'].dropna()
# RHS: called on Series -> returns Series -> LHS expects DataFrame -> MISMATCH 

* df_extracted['PM2.5'] = df_extracted['PM2.5'].fillna(df_extracted['PM2.5'].mean())
# RHS: called on Series -> returns Series -> LHS is a column (Series) -> MATCH """

#-------------------------------------------------
# SORTING
print(df_extracted.sort_values(by = 'AQI', ascending = False))
print(df_extracted.sort_values(by = ['City', 'AQI'], ascending = [True, False])) # multi-column, multi-level sorting: first by City (ascending i.e. A-Z), then within each city by AQI (descending i.e. high to low)

#-------------------------------------------------
# AGGREGATING
print(df_extracted.groupby('City')['AQI'].mean().round(2)) # groupby mean AQI per city, round off to 2 decimal places for readability
print(df_extracted.groupby('AQI_Bucket').size()) # count of rows per AQI category
# These two things tell us which cities and which pollution levels dominate the data

# OBSERVATIONS:
# Moderate dominates (8829) + Satisfactory (8224) = majority of readings
# Severe (1338) + Very Poor (2337) = ~14% of data — significant pollution problem
# Ahmedabad has highest mean AQI (452.13) — consistently most polluted city
# Aizwal lowest (34.7) — cleanest city in dataset

#-------------------------------------------------
# BIN SPLITTING
bins = [0, 50, 100, 200, 300, 500]
label_names = ["Good", "Moderate", "Poor", "Very Poor", "Severe"]
df_extracted_cut = pd.cut(df_extracted['AQI'], bins, labels = label_names)
print(df_extracted_cut.value_counts())

# COMPARISON: original AQI_Bucket vs our pd.cut() version
# Original has 6 categories including "Satisfactory" (AQI ~51-100)
# Our cut() has 5 — no Satisfactory bin, so those 8224 rows absorbed into "Moderate"
# Severe also differs: original=1338, ours=3132 — because original caps at 300 but our bin goes to 500, capturing more rows in Severe
# LESSON: bin boundaries determine everything — always verify against source data before using custom bins in analysis
# The point isn't to change bins randomly — it's that you define bins based on your analysis goal, not just copy whatever the dataset gives you.

#-------------------------------------------------
# MAPPING
# First extracting months by month numbers: 
df_extracted['Date'] = pd.to_datetime(df_extracted['Date'], errors = 'coerce') # converting object(string) to datetime value 
df_extracted['month_num'] = df_extracted['Date'].dt.month # extracted month numbers from dates
print(df_extracted.head()) 
# Now mapping month numbers to seasons:
seasons_by_month = {
    1: "Winter", 2: "Winter", 12: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon"
}
# Tuple keys ARE valid in dictionaries (tuples are hashable/immutable)
# BUT .map() looks up each value individually — it checks if 1 == (1,2,12) which is False → NaN. Keys must match the column values exactly (single ints)
df_extracted['Season'] = df_extracted['month_num'].map(seasons_by_month)
print(df_extracted.head())
print(df_extracted.shape)

# SEASONAL AQI VARIATION:
df_seasonal = df_extracted.groupby('Season')['AQI'].mean().round(2)
print(df_seasonal)
print(df_seasonal.shape)

#-------------------------------------------------
# CONCATENATING
df1 = df_extracted[(df_extracted['City'] == 'Kolkata')]
df2 = df_extracted[(df_extracted['City'] == 'Patna')]
print(df1.head())
print(df1.shape)
print(df2.head())
print(df2.shape)
df_1_2 = pd.concat([df1, df2], ignore_index = True)
print(df_1_2.head())
print(df_1_2.shape)

#-------------------------------------------------
# SUMMARY AND FINDINGS
# Dataset: India city-level AQI readings, 2015–2020
# 29,531 rows × 16 columns on load. After dropping null AQI rows: 24,850 rows remain.
# PM10 and Xylene had the worst data quality (38% and 61% missing respectively) — in a real project these columns would need a decision: impute, drop, or flag.

# Pollution picture:
# Moderate + Satisfactory = ~58% of all readings — India's baseline is already unhealthy
# Severe + Very Poor = ~14% — not rare events, a structural problem
# Most polluted city: Ahmedabad — mean AQI ~452.12 (classified as "Severe" on Indian scale)
# 450 AQI means hazardous air for sensitive groups on most days, not just occasional spikes.
# For context: WHO safe limit is AQI 50. Ahmedabad's average is 9x that threshold.
# This isn't an outlier problem — it's a systemic urban air quality crisis.
# Cleanest city: Aizwal (34.77) — likely due to geography (Northeast hills, low industrialization)

# Seasonal pattern (Climate AI relevance):
# Winter expected to show worst AQI — cold air traps pollutants + crop stubble burning
# Monsoon expected to show best AQI — rain washes particulate matter out of the air
# This dataset directly supports the case that climate variables (temperature, rainfall, wind) are predictors of air quality — a core assumption in Climate AI modeling.

# Binning lesson:
# Custom pd.cut() vs official AQI_Bucket showed category mismatch — same data, different story depending on bin choice. In any analysis, bin decisions must be documented and justified, not arbitrary.

# What I'd do next with this data:
# 1. Time series plot of AQI by city (Week 4 — Matplotlib)
# 2. Correlation between PM2.5 and AQI — are they redundant features?
# 3. City-wise seasonal heatmap — which city × season combination is most dangerous?