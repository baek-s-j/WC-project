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


df = pd.read_csv("weather.csv")

# In[4]:


weath_vars = ["MONTH", "HOUR", "TEMPERATURE", "WIND_SPEED", "HUMIDITY", "AIR_PRESSURE", "WIND_DIRECTION"]
traff_vars = ["MONTH", "HOUR", "ROAD_1", "ROAD_2", "ROAD_3", "ROAD_4", "ROAD_5", "ROAD_6", "ROAD_7", "ROAD_8"]
#total_vars = ["MONTH","YEAR", "ta", "pv", "td", "m005Te", "m01Te", "m02Te", "m03Te"]
total_vars = ["MONTH","YEAR", "ta"]
input_vars = total_vars
output_var = "temperature"
predict_hour = 24

input_vars.insert(0, output_var)  # 0번째 열의 PM10추가
input_vars

# In[5]:


df[output_var] = df[output_var].shift(-predict_hour)
df[output_var] = df[output_var].interpolate()

# In[6]:


training_set = df[input_vars].values  # 값만 가져옴
training_set.shape

# In[7]:


sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)  # 정규화 실행

sc_predict = MinMaxScaler()
sc_predict.fit_transform(training_set[:, 0:1])  # output만 정규화

# In[8]:


X = []
y = []

n_future = 0
n_past = 24

for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X.append(training_set_scaled[i - n_past: i, 1: training_set.shape[1]])
    y.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

X, y = np.array(X), np.array(y)

X.shape, y.shape

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)  # 17

# In[10]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape

# In[11]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=128, return_sequences=True, input_shape=(n_past, X.shape[2])))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(64, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))

model.summary()

# In[12]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
mcp = tf.keras.callbacks.ModelCheckpoint(filepath='data/weights_%s.h5' % output_var, monitor='val_loss', verbose=1,
                                         save_best_only=True, save_weights_only=True)

tb = tf.keras.callbacks.TensorBoard('data/logs')

history = model.fit(X_train, y_train, shuffle=False, epochs=100, callbacks=[es, rlr, mcp, tb], validation_split=0.2,
                    verbose=1, batch_size=64)

#model.save("model/model24_%s.h5" % output_var)

# In[13]:


plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(loc=2, ncol=2, bbox_to_anchor=(0.3, 1.15))
plt.show()

print(X_test.shape)
print(X_test)

y_pred = model.predict(X_test)
print(y_pred.shape)
print(y_pred)



y_test_inv = sc_predict.inverse_transform(y_test)
y_pred_inv = sc_predict.inverse_transform(y_pred)

# In[16]:


y_pred[:1]

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



