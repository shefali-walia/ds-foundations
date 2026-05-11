# WEEK 03 - CLEANING DATA USING PANDAS
# GCI TOKYO UNIVERSITY 2026

# TOPICS COVERED: 
# Data Structures in Pandas, Loading and extracting data, Data selection and Assignment, Determination of nan, Sorting values, Merging and Concatenating data, Manipulating and Transforming data, Aggregating data, Heirarchical Grouping, Handling Missing or Anomalous data.

# When performing data analysis, it is necessary to perform preprocessing to understand and prepare the data. 
# Pandas is a useful library to process table data. 
# In most real-world scenarios, data that you will find are not in ready-to-use format and require data cleaning.

# LOADING LIBRARIES
import numpy as np
import numpy.random as random
import pandas as pd
from pandas import Series, DataFrame

#-------------------------------------------------
# SERIES
# series is like a one-dimensional numpy array with labels for indices
series1 = Series([1, 1, 2, 4, 5 ,8, 19]) # if indices not mentioned, it is 0,1,2...
print(series1)

series2 = Series([2, 4, 5, 6], index = ['p', 'q', 'r', 's']) # if indices are mentioned, it is as mentioned.
print(series2)

print('Element: ', series2.values) #all elements (values) in series1
print('Indices: ', series2.index) #all indices in series1
print(series1[3], series2['p']) #accessing element by index

# DATAFRAME
# A DataFrame object is basically a two-dimensional data column.
# With the print function, the data is displayed in a tabular format.

data = {
    'ID':['100', '101', '102', '103', '104'],
    'City':['Tokyo', 'Osaka', 'Kyoto', 'Hokkaido', 'Tokyo'],
    'Birth_year':[1990, 1989, 1992, 1997, 1982],
    'Name':['Hiroshi', 'Akiko', 'Yuki', 'Satoru', 'Steve']
}  
#it is a dictionary with keys as column names and values as list of column values.
df = DataFrame(data) # creates a dataframe from the dictionary data
print(df)
print(df.head()) # shows the first 5 rows of the dataframe if not specified, otherwise shows the specified number of rows.
# print(df.tail()) shows the last 5 rows of the dataframe  

df_i = DataFrame(data, index = ['a', 'b', 'c', 'd', 'e']) 
print(df_i)
print(df_i['City']) #accessing a column by its name
print(df_i[['City', 'Birth_year']]) #accessing multiple columns by their names
print(df_i.City) #accessing a column by its name using dot notation
print("ELement: ", df_i.values) #all elements in the dataframe
print("Indices: ", df_i.index) #all indices in the dataframe
print("Column Names: ", df_i.columns) #all column names in the dataframe

# For large dataframes, it may be partially omitted so you can specify no. of columns or rows to display
pd.set_option('display.max_columns', 10) 
pd.set_option('display.max_rows', 10)

#-------------------------------------------------
# LOADING AND EXTRACING DATA
# Usually data comes as CSV files. When there are multiple files, they are often compressed into ZIP files for easier distribution.
# So we have to retrieve these files using python and then we can use pandas to explore the data.
# To read ZIP files and download them directly from the web, use these three libraries: requests, zipfile, and io.
# requests -  send and receive data from Web
# zipfile - reads and writes ZIP files
# io - reads and writes files 

import requests, zipfile, io
from io import StringIO
import os
# os is a module to talk to the operating system.Things like: where am I? what files exist here? change my location.

url = 'https://github.com/Hernan4444/MyAnimeList-Database/archive/refs/heads/master.zip'
r = requests.get(url, stream = True)  # stream = true mean don't download everything immediately, wait till I ask for data (to save memory)
z = zipfile.ZipFile(io.BytesIO(r.content))
# Imagine you downloaded a zip file but instead of saving it to your desktop, it's just floating in RAM. This line takes those floating bytes and says "treat this as a zip file" so Python can work with it.
# r.content — the raw bytes of whatever was downloaded (the zip file, as a sequence of bytes)
# io.BytesIO(r.content) — wraps those bytes (RAM) into a fake file object so it looks like a file to python. Normally ZipFile expects a file on disk, but BytesIO tricks it into thinking "here's a file" when it's actually just bytes sitting in RAM. No actual file is written to disk at this point.
# zipfile.ZipFile(...) — opens that fake file as a zip archive, so Python can now read its contents, list files inside it, extract them, etc.

z.extractall() # Or could also do z.extractall("/some/path") to extract to a specific folder
# Takes every file inside the zip and dumps it into the current working directory 

#-------------------------------------------------
# READING AND CHECKING DATA
base = os.path.dirname(os.path.abspath(__file__))
anime_data = pd.read_csv(os.path.join(base, 'MyAnimeList-Database-master', 'data', 'anime.csv'))
anime_list = pd.read_csv(os.path.join(base, 'MyAnimeList-Database-master', 'data', 'animelist.csv'))
anime_synop = pd.read_csv(os.path.join(base, 'MyAnimeList-Database-master', 'data', 'anime_with_synopsis.csv'))
print(anime_data.head(3))

""" GOT ERRORS HERE AND HOW I FIEXD THEM:

# ERROR: FileNotFoundError — Python couldn't find the CSV file

# WHY IT HAPPENED: VS Code runs files from the project root
 (ds-foundations/) but my path was written as if Python
 started from inside week03-pandas/ — so it was looking
 in the wrong folder.

# FIX: Build the path dynamically using __file__ .
 __file__ = the exact location of THIS script
 os.path.dirname() = strip the filename, keep the folder
 os.path.join() = stitch path pieces together safely

# LESSON: Always do this when loading files:
   base = os.path.dirname(os.path.abspath(__file__))
   pd.read_csv(os.path.join(base, 'folder', 'file.csv'))

# DIAGNOSE FUTURE ERRORS:
  print(os.getcwd())    → where does Python think it is? (getcwd = get current working directory)
  print(os.listdir('.')) → what files can it see from there? """


anime_data.info() # gives info about the no. of not null data and type of variable for all variables.
anime_data.describe() # gives summary statistics for all numeric variables (count, mean, std, min, 25%, 50%, 75%, max)
anime_data.head().T # Transpose of the head of the dataframe. Useful for wide dataframes where columns are more than rows. 

#-------------------------------------------------
# DATA SELECTION AND ASSIGNMENT
# Basically like indexing and slicing in numpy but with labels.

# By columns:
anime_data['Score'] # selecting a column by its name, returns a Series
# can also write like anime_data.Score but it is not recommended because if there is a column name with space or special characters, it will cause an error. So it is better to use the first method.
anime_data[['Score', 'Episodes']] # selecting multiple columns by their names as a list, returns a DataFrame

cols_to_use = ['MAL_ID', 'Name', 'Score', 'Genres', 'Type', 'Aired', 'Studios', 'Source']
anime_data_extracted = anime_data[cols_to_use]
print(anime_data_extracted.head())

# By rows:
anime_data_extracted[0:3] # 3 not included, like in numpy slicing
df_i['a':'c'] # if index is a letter or string, it is inclusive of the end index. 

# df.loc[] allows you to retrieve a specific row or column, like Numpy, by specifying the label (index name or column name)
# loc = location
anime_data_extracted.loc[4] 
anime_data_extracted.loc[3:8] # INCLUSIVE of the end index

# Add column names aftr comma
anime_data_extracted.loc[:, 'Score'] # all rows, only Score column
anime_data_extracted.loc[:, ['Score', 'Type']] # all rows, only Score and Type columns  
anime_data_extracted.loc[0:3, ['Score', 'Type']] # first 3 rows, only Score and Type columns

# Use df.iloc[] when retrieving rows or columns by their index no. or column no. (Not by column names)
# iloc = integer location
anime_data.iloc[0:4, [0,2]] # first 4 rows, only 1st and 3rd columns (0 and 2 because of 0-based indexing)

# df.at[] and df.iat[] are used to retrieve a single element.
anime_data.at[0, 'Name']
anime_data.iat[0, 1] 

# Can create new column  using the [] brackets with the new column name. 
# If you use a column name that already exists, the data will be overwritten.
df['Score'] = np.arange(5) *10  # np.arange(5) generates an array of 5 numbers starting from 0, *10 multiplies every element by 10, so we get [0, 10, 20, 30, 40]
print(df)

# to replace any value, can use df.loc, iloc, at, iat :
df.at[0, 'Birth_year'] = 2000
print(df)

# Extracting data based on conditions:
print(anime_data_extracted['Score'] != 'Unknown') # gives a boolean series where True means the condition is met and False means it is not met.
anime_data_extracted = anime_data_extracted[anime_data_extracted['Score'] != 'Unknown']
print(anime_data_extracted.head())
print(anime_data_extracted[(anime_data_extracted['Studios'] == 'Nomad') | (anime_data_extracted['Studios'] == 'Sunrise')])
# can use & or | for specifying conditions
print(anime_data_extracted[anime_data_extracted['Studios'].isin(['Sunrise', 'Nomad'])].head())  # Can also use isin(list-name) to specify multiple conditions for a column.

#-------------------------------------------------
# DETERMINATION OF NAN (NULL)
# Nan = not a number = missing value 
# data may be missing and the corresponding data may not exist.
# Missing values can cause incorrect calculations like averages etc. so it's imp. to exclude them from operations.
# We use isnull() to check if it is nan

print(anime_data_extracted.isnull())
print(anime_data_extracted.isnull().sum()) # gives the count of null values in each column

#-------------------------------------------------
# SORTING VALUES
# Can sort data based on index as well as the elements

# sort by index (may appear as if not needed but useful when you have already applied operations like shuffling rows, sorting by columns, deleting rows, or concatenating data etc)
print(anime_data_extracted.sort_index)
# sort by values - default is ascending order, can specify descending order by setting ascending = False
print(anime_data_extracted.sort_values(by = 'Score', ascending = False))

#-------------------------------------------------
# MERGING DATA
""" There are four ways to join data:
1. INNER JOIN - when both data have keys, only the rows with matching keys in both dataframes are included in the result. So overall data can reduce (like intersection of two sets)
* OUTER JOINS: (keep amount of data same (no data discarded), have to specify using how = 'left', 'right', 'outer' in pd.merge() function)
2. LEFT JOIN/ LEFT OUTER JOIN - when only data on the left left has keys (pd.merge(df1, df2, how = 'left'))
3. RIGHT JOIN/ RIGHT OUTER JOIN - when only data on the right has keys (how = 'right')
4. FULL OUTER JOIN/FULL JOIN - when key exists on either side. (how = 'outer' or how = 'full')
* key refers to the column(s) on which the join is based. It is used to identify matching rows between the two dataframes being joined. The key column(s) should have the same name and data type in both dataframes for the join to work correctly.
* Missing values are filled with NaN in the result of outer joins.
* To perform the merge, we use pd.merge() fnct. whose default is inner join.
"""
# Keys are automatically recognized if they have the same name in both dataframes.
# Keys to merge can be specified explicitly using "on".
# If the key columns have different names, you can specify them using the left_on and right_on parameters in the merge function.
pd.merge(anime_data_extracted, anime_synop)
pd.merge(anime_data_extracted, anime_synop, on = 'MAL_ID')
pd.merge(anime_list, anime_data_extracted, left_on = 'anime_id', right_on = 'MAL_ID')
# can use left_on, right_on when the key columns have different names in the two dataframes being merged. left_on specifies the column name in the left dataframe, and right_on specifies the column name in the right dataframe that should be used as the key for merging.
# left_index and right_index can be used when the index of the dataframe is the key for merging. left_index = True means use the index of the left dataframe as the key, and right_index = True means use the index of the right dataframe as the key.
# can add suffixes to the argument to specify which dataframe the eys came from, useful when there are duplicate columns occuring.
# Ex: pd.merge(df1, df2, on = '..', suffixes = ('_1', '_2'))

# use df.join() to use index as key. Use similar to pd.merge() but it is more convenient when you want to join on index. By default, it performs a left join.
# READ DOCUMENTATION FOR THIS FUNCTION CAREFULLY BECAUSE IT HAS MANY PARAMETERS AND OPTIONS!

# CONCATENATING DATA
# Combines dataframes either vertically (stacking rows) or horizontally (stacking columns)
# pd.concat() - default vertical join, can specify horizontal join by setting axis = 1
"""
pd.concat([df1, df2]) =  vertical join, stacks rows of df2 below df1
* The columns of df1 and df2 should be the same or at least have some common columns, not common columns are filled with NaN. 
* In this case, index is concatenated as it is, so if df1 and df2 have same index values, the resulting dataframe will have duplicate index values.
* Use pd.concat([df1, df2], ignore_index = True) to reattach the index in order from top. 
* The sort argument can be set to True to sort the column, or to False to join without sorting.
* Sort => resulting DataFrame's columns are sorted alphabetically. If sort was False, the columns would appear in the order they were encountered in the input DataFrames. 
"""
# pd.concat([df1, df2], axis = 1) => horizontal join, stacks columns of df2 to the right of df1
# In this case, they will be tied by index and the columns will be joined as they are. 
# If axis=0 is specified, the columns will be joined vertically.

#-------------------------------------------------
# MANIPULATING AND TRANSFORMING DATA

# Deleting data - df.drop():
# For deleting rows: Specify index of the rows you want to delete as a list in the first argument. Set axis to 0 (default is axis = 0, so you can also omit it)
# For deleting columns: Specify column names you want to delete as a list in the first argument. Set axis to 1. 
print(anime_data_extracted.drop(index = [0,2]).head()) # deletes the rows with index 0 and 2 - so index is no longer continuous
# use df.reset_index() to reassign a new index
print(anime_data_extracted.drop(index=[0,2]).reset_index().head()) # resets the index to be continuous but the old index is added as a new column called "index" so we can drop it as well by adding drop = True argument in reset_index() function.
print(anime_data_extracted.drop(['Studios'], axis = 1).head())
