#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

np.random.seed(1)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score

import tensorflow as tf

tf.random.set_seed(1)

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# In[2]:


mpl.rcParams['figure.figsize'] = (12, 6)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 18

# In[3]:


df = pd.read_csv("weather.csv")

# In[4]:


weath_vars = ["MONTH", "HOUR", "TEMPERATURE", "WIND_SPEED", "HUMIDITY", "AIR_PRESSURE", "WIND_DIRECTION"]
traff_vars = ["MONTH", "HOUR", "ROAD_1", "ROAD_2", "ROAD_3", "ROAD_4", "ROAD_5", "ROAD_6", "ROAD_7", "ROAD_8"]
total_vars = ["MONTH", "YEAR", "ta", "pv", "td", "m005Te", "m01Te", "m02Te", "m03Te"]

input_vars = total_vars
output_var = "temperature"
predict_hour = 24

input_vars.insert(0, output_var)  # 0번째 열의 PM10추가
input_vars

# In[5]:


# df[output_var] = df[output_var].shift(-predict_hour)
# df[output_var] = df[output_var].interpolate()

# In[6]:

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
training_set = df[input_vars].values  # 값만 가져옴
print(type(training_set))

# In[7]:
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set[:,0:])  # 정규화 실행
print(training_set_scaled[0:72])
ex=sc.transform(training_set[0:72,0:])
print(ex)

