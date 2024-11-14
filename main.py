#%%
import os 
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#%%
data = pd.read_csv('data/train.csv')

#%%
data.head(5)
# %%
print(f"Number of Observations in data are {data.shape[0]}")