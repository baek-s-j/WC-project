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

tf.random.set_seed(1)  # 모델의 파라미터를 랜덤으로 갖게되는데 이렇게 seed를 설정하면 매번 같은 파라미터 값을 가질수있다.

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# In[2]:


mpl.rcParams['figure.figsize'] = (12, 6)  # 시각화 설정하는 것
mpl.rcParams['axes.grid'] = True
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 18

# In[3]:


df = pd.read_csv("C:/Users/qor27/OneDrive/바탕 화면/백승진/캡스톤 디자인/Teo_LSTM/2018_ex.csv")

# In[4]:


dataset_train_actual = df.copy()
dataset_train_actual = dataset_train_actual.fillna(dataset_train_actual.mean())
dataset_train_actual = dataset_train_actual.reset_index()

dataset_train_timeindex = dataset_train_actual.set_index('DATETIME')
dataset_train = dataset_train_actual.copy()
print('dataset_train')
print(dataset_train)
print("\n\n\n")

# Select features (columns) to be involved intro training and predictions
cols = list(dataset_train)[2:5] #['index', 'DATETIME', 'PM10', 'TEMPERATURE', 'HUMIDITY']
print('cols')
print(cols)
print("\n\n\n")


datelist_train = list(dataset_train['DATETIME'])
datelist_train = [date for date in datelist_train]
print('datelist_train')
#print(datelist_train)

print('Training set shape == {}'.format(dataset_train.shape))
print('All timestamps == {}'.format(len(datelist_train)))
print('Featured selected: {}'.format(cols))
print("\n\n\n")

dataset_train = dataset_train[cols].astype(str)
print('데이터 훈련 확인 데이터 잘봐2')
print(dataset_train)
print("\n\n\n")

for i in cols:
    for j in range(0, len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(',', '')

dataset_train = dataset_train.astype(float)

# Using multiple features (predictors)
training_set = dataset_train.values

print('Shape of training set == {}.'.format(training_set.shape))
print(training_set)
print("\n\n\n")
print(training_set.shape)

sc = MinMaxScaler()  # 정규화 모델을 선언
training_set_scaled = sc.fit_transform(
    training_set)  # input.var의 값들만 가지고 정규화한 값을 가져옴 fit_transform()함수는 훈련데이터에 평균과 분산을 학습
sc_predict = MinMaxScaler()
sc_predict.fit_transform(training_set[:, 0:1])  # PM10열의 값을 가져와서 정규화

print(training_set[:, 0:1])
# 왜 PM10값은 정규화를 중복되게 하는 것인가..
print("\n\n\n")

X_train = []
y_train = []

n_future = 50  # Number of days we want top predict into the future
n_past = 2  # Number of past days we want to use to predict the future

for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1]])
    y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

X, y = np.array(X_train), np.array(y_train)

print('X_train shape == {}.'.format(X.shape))
print('y_train shape == {}.'.format(y.shape))
print("\n\n\n")

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
mcp = tf.keras.callbacks.ModelCheckpoint(filepath='data/weights_%s.h5' "dustmq", monitor='val_loss', verbose=1,
                                         save_best_only=True, save_weights_only=True)

tb = tf.keras.callbacks.TensorBoard('data/logs')

history = model.fit(X, y, shuffle=False, epochs=100, callbacks=[es, rlr, mcp, tb], validation_split=0.2,
                    verbose=1, batch_size=64)

# model.save("data/model_%s" % (output_var))

# In[13]:


# Generate list of sequence of days for predictions
datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()

'''
Remeber, we have datelist_train from begining.
'''

# Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
datelist_future_ = []
for this_timestamp in datelist_future:
    datelist_future_.append(this_timestamp.date())

print(datelist_train[-1])
print(datelist_future_)
print("\n\n\n")

predictions_future = model.predict(X[-n_future:])

predictions_train = model.predict(X[n_past:])

y_pred_future = sc_predict.inverse_transform(predictions_future)
y_pred_train = sc_predict.inverse_transform(predictions_train)

PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['PM10']).set_index(pd.Series(datelist_future))
PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['PM10']).set_index(
    pd.Series(datelist_train[2 * n_past + n_future - 1:]))
print(PREDICTION_TRAIN)
START_DATE_FOR_PLOTTING = '2018-01-08'

plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['PM10'], color='r',
         label='Predicted Global Active power')
plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index,
         PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['PM10'], color='orange',
         label='Training predictions')
plt.plot(dataset_train_timeindex.loc[START_DATE_FOR_PLOTTING:].index,
         dataset_train_timeindex.loc[START_DATE_FOR_PLOTTING:]['PM10'], color='b', label='Actual Global Active power')

plt.axvline(x=min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

plt.grid(which='major', color='#cccccc', alpha=0.5)

plt.legend(shadow=True)
plt.title('Predcitions and Acutal Global Active power values', family='Arial', fontsize=12)
plt.xlabel('Timeline', family='Arial', fontsize=10)
plt.ylabel('Stock Price Value', family='Arial', fontsize=10)

plt.show()
