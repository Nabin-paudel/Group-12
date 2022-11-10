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

#Handle missing values
df.isnull().sum()

#to drop all rows with missing values
#dropping null records

df=df.dropna()

ax1=plt.boxplot(df["Fish"])
plt.show()

boxplot=df.boxplot(column=['Mercury'])
plt.show()

boxplot=df.boxplot(column=['Elevation'])
plt.show()

boxplot=df.boxplot(column=['Drainage Area'])
plt.show()
boxplot=df.boxplot(column=['Surface Area'])
plt.show()

boxplot=df.boxplot(column=['Max Depth'])
plt.show()

boxplot=df.boxplot(column=['RF','FR','Dam','RT','RS'])
plt.show()

correlation=df.corr()
print(correlation)

plt.figure(figsize=(14,14))
sns.heat,ap(correlation,annot=True,linwidth=0.01,vmax=1,square=True,cbar=True);
sns.heatmap

columns=['fISH','Mercury','Elevation','Drainage Area','Surface Area','Max Depth','RF','FR','Dam','RT','RS','LATITUDE_DEGREES','LATITUDE_MINUTES','LATITUDE_SECONDS',LONGITUDE_DEGREES','LONGITUDE_MINUTES,'LONGITUDE_SECONDS']
print(columns)
print()
for col in columns:
  percentile_25=df[col].quantile(0.25)
  percentile_75=df[col].quantile(0.75)
  iqr=percentile_75-percentile_25 # Inter Quartile Range
  total=len(df[col])
cut_off=iqr*1.5#normally use 1.5 timesIQR
lower,upper=percentile_25-cut_off,percentile_75+cut_off
print(f"col:{col},lower:{lower},upper:{upper}")
num_outliers=len(df[(df[col]<lower)| (df[col]>upper)])
pc_outliers=round(num_outliers*100/total,2)
print(f"Num outliers: {num_outliers},total rows: {total},percent:{pc_outliers}")
print()








