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
