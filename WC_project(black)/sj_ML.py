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

df = pd.read_csv("C:/Users/qor27/OneDrive/바탕 화면/백승진/캡스톤 디자인/Teo_LSTM/2018_r.csv")

dataset_train_actual = df.copy()  # 복사
dataset_train_actual = dataset_train_actual.fillna(dataset_train_actual.mean())  # 평균으로 채움

dataset_train_timeindex = dataset_train_actual.set_index('DATETIME')  # 시간이 인덱스인 데이터프레임 생성

dataset_train = dataset_train_actual.copy()  # 인덱스가 없는 것 데이터 프레임으로
print(dataset_train['PM10'])
print('dataset_train')
print(dataset_train)
print("\n\n\n")
print('dataset_time')
print(dataset_train_timeindex)

cols = list(dataset_train)[1:]  # ['index', 'DATETIME', 'PM10', 'TEMPERATURE', 'HUMIDITY'], 0:1이면 0번째 인덱스에서 한개만 가져옴
print('cols')
print(cols)
print("\n\n\n")

datelist_train = list(dataset_train['DATETIME'])
datelist_train = [date for date in datelist_train]
print('datelist_train')  # 시간가져옴
# print(datelist_train)
print("\n\n\n")

dataset_train = dataset_train[cols]
print(dataset_train)
training_set = dataset_train.values
print(len(training_set))
print('output')
print(training_set[:, 0:1])

sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)  # 정규화 실행

sc_predict = MinMaxScaler()
sc_predict.fit_transform(training_set[:, 0:2])  # output만 정규화

X_train = []
y_train = []

n_future = 0  # Number of days we want top predict into the future
n_past = 24  # Number of past days we want to use to predict the future

for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1]])
    y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0:2])

X_train, y_train = np.array(X_train), np.array(y_train)
nsamples, nx, ny = y_train.shape
y_train = y_train.reshape((nsamples, nx * ny))
print(len(y_train))
print(len(datelist_train[n_past - 1:]))
print('X_train shape == {}.'.format(X_train.shape))
print('y_train shape == {}.'.format(y_train.shape))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=128, return_sequences=True, input_shape=(n_past, X_train.shape[2])))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(64, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(2))

model.summary()

# In[12]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
mcp = tf.keras.callbacks.ModelCheckpoint(filepath='data/weights_%s.h5' % "output_var", monitor='val_loss', verbose=1,
                                         save_best_only=True, save_weights_only=True)

tb = tf.keras.callbacks.TensorBoard('data/logs')

history = model.fit(X_train, y_train, shuffle=False, epochs=100, callbacks=[es, rlr, mcp, tb], validation_split=0.2,
                    verbose=1, batch_size=64)

# model.save("data/model_%s" % (output_var))

# In[13]:

plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(loc=2, ncol=2, bbox_to_anchor=(0.3, 1.15))
plt.show()

predictions_future = model.predict(X_train)
print(predictions_future)
print(y_train)

y_test_inv = sc_predict.inverse_transform(y_train)
y_pred_inv = sc_predict.inverse_transform(predictions_future)
print(type(y_test_inv))
print(y_test_inv)
# plt.plot(datelist_train[n_past - 1:], y_test_inv, label="Actual")
# plt.plot(datelist_train[n_past - 1:], y_pred_inv, label="Predict")
# plt.ylabel('pm10')
# plt.xlabel("Time")
# plt.legend(loc=2, ncol=2, bbox_to_anchor=(0.3, 1.15))
# plt.show()

test_y = pd.DataFrame(y_test_inv, columns=['PM10','PM25'])
pred_y = pd.DataFrame(y_pred_inv, columns=['PM10','PM25'])

fig = go.Figure()
fig.add_trace(go.Scatter(x=datelist_train[n_past - 1:], y=test_y['PM10'],
                         mode='lines',
                         name='10원래값', line=dict(color='firebrick', width=1)))
fig.add_trace(go.Scatter(x=datelist_train[n_past - 1:], y=pred_y['PM10'],
                         mode='lines',
                         name='10예측값', line=dict(color='red', width=1)))
fig.add_trace(go.Scatter(x=datelist_train[n_past - 1:], y=test_y['PM25'],
                         mode='lines',
                         name='25원래값', line=dict(color='blue', width=1)))
fig.add_trace(go.Scatter(x=datelist_train[n_past - 1:], y=pred_y['PM25'],
                         mode='lines',
                         name='25예측값', line=dict(color='royalblue', width=1)))

# 배경 레이어색 파트
fig.update_layout(paper_bgcolor="black")  # 차트 바깥 색
fig.update_layout(plot_bgcolor="black")  # 그래프 안쪽색
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.update_layout(xaxis_tickformat='%Y<br>%m-%d<br>%H시')
fig.update_xaxes(linecolor='gray', gridcolor='gray', mirror=True)
fig.update_yaxes(linecolor='gray', gridcolor='gray', mirror=True)
fig.update_yaxes(tickformat=',')  # 간단하게 , 형으로 변경

fig.write_html('ex1.html')
