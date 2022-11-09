#%matplotlib inline
import tkinter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
np.random.seed(10)
import seaborn as sns

#importing libraries
from skleaarn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlip.pyplot as plt

df = pd.read_csv('fair_new.csv')
print(df)
print(type(df))

df.info()

print(df.shape)

print(df.shape[0])

print(df.shape[1])

df.head()

#Data Overview

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print()

#Print the column names 
print (f"features : {df.columns.tolists()}")
print()

#Print the total number of null values in the data 
print(f"Missing values  : {df.isnull().sum().values.sum()}")

#For each column, print the number of unique values
print(f"Unique values : {df.nunique()}")

#Descriptive statistics for continuous variables 
print(df.describe())

#correlation
#correlation plot
correlatio=df.corr
print(correlation)

plt.bar(df['Reservoir'],df['Mercury'])
plt.show()
plt.scatter(df["Dam"],df['Mercury'])
plt.show()
plt.scatter(df['Drainage Area'],df['Mercury'])
plt.show()

#to drop rows with duplicate values
print(df.shape)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.shape)






