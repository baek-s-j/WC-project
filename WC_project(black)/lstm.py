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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# In[2]:


# mpl.rcParams['figure.figsize'] = (12, 6)  # 시각화 설정하는 것
# mpl.rcParams['axes.grid'] = True
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = ['Times New Roman']
# mpl.rcParams['font.size'] = 18



# In[3]:


df = pd.read_csv("C:/Users/qor27/OneDrive/바탕 화면/백승진/캡스톤 디자인/Teo_LSTM/2018_r.csv")

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
cols = list(dataset_train)[2:]  # ['index', 'DATETIME', 'PM10', 'TEMPERATURE', 'HUMIDITY']
print('cols')
print(cols)
print("\n\n\n")

datelist_train = list(dataset_train['DATETIME'])
datelist_train = [date for date in datelist_train]
print('datelist_train')
# print(datelist_train)

print('Training set shape == {}'.format(dataset_train.shape))
print('All timestamps == {}'.format(len(datelist_train)))
print('Featured selected: {}'.format(cols))
print("\n\n\n")


dataset_train = dataset_train[cols]
training_set = dataset_train.values

print('Shape of training set == {}.'.format(training_set.shape))
print(training_set)
print(training_set.shape)
print("\n\n\n")
print(training_set[:, 0:1])
sc = MinMaxScaler()  # 정규화 모델을 선언
training_set_scaled = sc.fit_transform(
    training_set)  # input.var의 값들만 가지고 정규화한 값을 가져옴 fit_transform()함수는 훈련데이터에 평균과 분산을 학습
sc_predict = MinMaxScaler()
sc_predict.fit_transform(training_set[:, 0:1])  # PM10열의 값을 가져와서 정규화

X_train = []
y_train = []

n_future = 4  # Number of days we want top predict into the future
n_future = n_future * 24
n_past = 24  # Number of past days we want to use to predict the future

for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1]])
    y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

X, y = np.array(X_train), np.array(y_train)

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

history = model.fit(X, y, shuffle=False, epochs=10, callbacks=[es, rlr, mcp, tb], validation_split=0.2,
                    verbose=1, batch_size=64)

# model.save("data/model_%s" % (output_var))

# In[13]:


# Generate list of sequence of days for predictions
datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='H').tolist()
print(datelist_future)

# # Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
# datelist_future_ = []
# for this_timestamp in datelist_future:
#     datelist_future_.append(this_timestamp.date())
#
# print(datelist_train[-1])
# print(datelist_future_)
# print("\n\n\n")

predictions_future = model.predict(X[-n_future:])
print(predictions_future)
predictions_train = model.predict(X[n_past:])
print(predictions_train)


y_pred_future = sc_predict.inverse_transform(predictions_future)
y_pred_train = sc_predict.inverse_transform(predictions_train)

PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['PM10']).set_index(pd.Series(datelist_future))
print(PREDICTIONS_FUTURE)
PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['PM10']).set_index(
    pd.Series(datelist_train[2 * n_past + n_future - 1:]))
print(PREDICTION_TRAIN)
START_DATE_FOR_PLOTTING = '2018-01-08 1:00'


fig = go.Figure()
fig.add_trace(go.Scatter(x=PREDICTIONS_FUTURE.index, y=PREDICTIONS_FUTURE['PM10'],
                         mode='lines',
                         name='미래', line=dict(color='firebrick', width=1)))
fig.add_trace(go.Scatter(x=PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, y=PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['PM10'],
                         mode='lines',
                         name='예측한것', line=dict(color='orange', width=1)))
# fig.add_trace(go.Scatter(x=dataset_train_timeindex.loc[START_DATE_FOR_PLOTTING:].index, y=dataset_train_timeindex.loc[START_DATE_FOR_PLOTTING:]['PM10'],
#                          mode='lines',
#                          name='원래데이터', line=dict(color='blue', width=1)))



# 배경 레이어색 파트
fig.update_layout(paper_bgcolor="black")  # 차트 바깥 색
fig.update_layout(plot_bgcolor="black")  # 그래프 안쪽색
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.update_layout(xaxis_tickformat='%Y<br>%m-%d<br>%H시')
fig.update_xaxes(linecolor='gray', gridcolor='gray', mirror=True)
fig.update_yaxes(linecolor='gray', gridcolor='gray', mirror=True)
fig.update_yaxes(tickformat=',')  # 간단하게 , 형으로 변경

fig.write_html('ex.html')
