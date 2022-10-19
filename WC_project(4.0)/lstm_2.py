#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

np.random.seed(1)

from sklearn.preprocessing import MinMaxScaler
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

df = pd.read_csv("weather(2열).csv")

# In[4]:


weath_vars = ["MONTH", "HOUR", "TEMPERATURE", "WIND_SPEED", "HUMIDITY", "AIR_PRESSURE", "WIND_DIRECTION"]
traff_vars = ["MONTH", "HOUR", "ROAD_1", "ROAD_2", "ROAD_3", "ROAD_4", "ROAD_5", "ROAD_6", "ROAD_7", "ROAD_8"]
total_vars = ["MONTH","YEAR", "ta","hm", "pv", "td", "m005Te", "m01Te", "m02Te", "m03Te"]

input_vars = total_vars
output_var = "temperature"
output_var1 = "humidity"
predict_hour = 24

input_vars.insert(0, output_var1)
input_vars.insert(0, output_var)  # 0번째 열의 PM10추가
print(input_vars)

# In[5]:

output_var=["temperature","humidity"]
df[output_var] = df[output_var].shift(-predict_hour)
df[output_var] = df[output_var].interpolate()

# In[6]:


training_set = df[input_vars].values  # 값만 가져옴
training_set.shape

# In[7]:


sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)  # 정규화 실행

sc_predict = MinMaxScaler()
sc_predict.fit_transform(training_set[:, 0:2])  # output만 정규화
print(training_set[:, 0:2])
# In[8]:


X = []
y = []

n_future = 0
n_past = 24

for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X.append(training_set_scaled[i - n_past: i, 2: training_set.shape[1]])
    y.append(training_set_scaled[i + n_future - 1:i + n_future, 0:2])

X, y = np.array(X), np.array(y)
nsamples, nx, ny = y.shape
y = y.reshape((nsamples, nx * ny))

X.shape, y.shape

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)  # 17

# In[10]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape

# In[11]:



from keras.models import load_model

model = load_model("model/model24_['temperature', 'humidity'].h5")

# model.save("data/model_%s" % (output_var))

# In[13]:


# In[14]:
print(X_test.shape)
print(X_test)

y_pred = model.predict(X_test)
print(y_pred.shape)
print(y_pred)
# In[15]:


y_test_inv = sc_predict.inverse_transform(y_test)
y_pred_inv = sc_predict.inverse_transform(y_pred)

# In[16]:


from sklearn.metrics import accuracy_score

print(model.evaluate(X_train, y_train, verbose=1))

# In[17]:


print(input_vars)
print("predict_hour: ", predict_hour)
print('R^2: %.2f' % (r2_score(y_test_inv, y_pred_inv) * 100))
print('RMSE: %.2f' % (np.sqrt(mse(y_test_inv, y_pred_inv))))
print('MAE: %.2f' % (mae(y_test_inv, y_pred_inv)))

# In[18]:


plt.plot(y_test_inv[-500:], label="Actual")
plt.plot(y_pred_inv[-500:], label="Predict")
plt.ylabel(output_var)
plt.xlabel("Time")
plt.legend(loc=2, ncol=2, bbox_to_anchor=(0.3, 1.15))
plt.show()

# In[ ]:
