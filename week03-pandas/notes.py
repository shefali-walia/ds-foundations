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


