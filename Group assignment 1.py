#%matplotlib inline
import tkinter
import matplotlib
import matplotlib.pyplot as plt
mtplotlib.use('TKAgg')

import numpy as np
#%matplotlib inline
import pandas as pd

np.random.seed(10)

import seaborn as sns


#read the excel
df=pd.read_csv('fair.csv')
print(df)
df.shape
df.info()
df.describe()
