from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus
import urllib
import requests, xmltodict
import json
import pandas as pd
from pymongo import MongoClient
import math
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

# url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?serviceKey=Zy3WCPt243lyaq0PqqKVwGL%2F42nvD9qjxGcz%2FgVr3y01%2BxYJ%2BsmdjB1H01RWJsVgaQkayn32SBzLQDdMgpiVIg%3D%3D&numOfRows=10&pageNo=1&dataCd=ASOS&dateCd=HR&stnIds=108&endDt=20200310&endHh=01&startHh=01&startDt=20190120'
# print(url)
# url 확인용


key = 'Zy3WCPt243lyaq0PqqKVwGL%2F42nvD9qjxGcz%2FgVr3y01%2BxYJ%2BsmdjB1H01RWJsVgaQkayn32SBzLQDdMgpiVIg%3D%3D'
# 공공데이터 포털에서 key값 받아오기


pageNo = 1  # 페이지 번호
dataCd = 'ASOS'  # 자료코드..?
dateCd = 'HR'  # 날짜코드?
stnIds = '259'  # 지점번호 어느지역 기상청인지
endDt = '20220530'  # 종료날짜
endHh = '23'  # 종료시간
startHh = '00'
startDt = '20210714'
numOfRows = '999'  # 한 페이지의 결과수 320일*24시간 시작일 포함을해줘야함 차이구한후 +1

empty_df = pd.DataFrame()

for pageNo in range(1, math.ceil(7704 / 999) + 1):
    # url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?serviceKey={}&numOfRows=10&pageNo=1&dataCd=ASOS&dateCd=HR&stnIds=108&endDt=20200310&endHh=01&startHh=01&startDt=20190120'.format(key)
    url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?serviceKey={}&numOfRows={}&pageNo={}&dataCd={}&dateCd={}&stnIds={}&endDt={}&endHh={}&startHh={}&startDt={}'.format(
        key, numOfRows, pageNo, dataCd, dateCd, stnIds, endDt, endHh, startHh, startDt)
    # numOfRows부터 시작 원하는 데이터 행을 받고자하면 공공데이터 포털에서 해당 문서를 확인 한 후 날짜와 데이터 양식을 선택하여 요청함

    content = requests.get(url).content
    dict_content = xmltodict.parse(content)
    # items 하고 item 차이로 에러남
    # 데이터를 요청하여 xml로 받아옴

    jsonString = json.dumps(dict_content['response']['body']['items'], ensure_ascii=False)
    # json으로 한번 가공 아스키 코드까지 json형식으로 형태변환
    # print(jsonString)

    jsonObj = json.loads(jsonString)
    # 두번 가공하여야 함 json으로 다시 가공
    # print(jsonObj)

    # for item in jsonObj['item']:
    #    print(item)

    df = pd.DataFrame(jsonObj['item'])
    # 확인하기위해 pandas 데이터 프레임에 집어넣기
    # 여기까지 가공끝

    empty_df = empty_df.append(df)

print(empty_df)
print(len(empty_df))
print(empty_df.isnull().sum())
input_var = ['tm', 'ta', 'ws',  'hm', 'pv', 'td', 'm005Te', 'm01Te', 'm02Te', 'm03Te']
# input_var = ['tm', 'ta', 'rn', 'ws', 'wd', 'hm', 'pv', 'td', 'm005Te', 'm01Te', 'm02Te', 'm03Te']
            #시간, 온도, 강수량, 풍속, 풍향 ,습도, 증기압, 이슬점온도, 지중온도

empty_df = empty_df[input_var]

client = MongoClient('mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')
db = client.wc_project
collection = db.thdata
th_result = list(
    collection.find({"zone_name": "외부 전면-3번-무위사", "time": {"$gte": "2021-07-14", "$lte": "2022-05-31"}}).sort(
        [("time", 1)]))
th_frame = pd.DataFrame(th_result)
th_frame.time = pd.to_datetime(th_frame.time)
th_frame = th_frame.set_index('time')
min_th_frame = th_frame.resample(rule='H').mean()
print(len(min_th_frame))
# print(th_result) #5.30일 23:50분까지 가져옴

empty_df.tm = pd.to_datetime(empty_df.tm)
empty_df = empty_df.set_index('tm')
#empty_df['YEAR'] = empty_df.index.year
empty_df['MONTH'] = empty_df.index.month
#empty_df['DAY']=empty_df.index.day
empty_df['HOUR'] = empty_df.index.hour
#empty_df['DAYOFYEAR'] = empty_df.index.dayofyear
#empty_df['DAYOFWEEK'] = empty_df.index.dayofweek
result_df = pd.concat([min_th_frame, empty_df], axis=1)
result_df = result_df.round(1)
#result_df.loc[result_df['rn'] != result_df['rn'], 'rn'] = 0
# # row 생략 없이 출력
# pd.set_option('display.max_rows', None)
# # col 생략 없이 출력
# pd.set_option('display.max_columns', None)
print(result_df)
print(len(result_df))
print(result_df.isnull().sum())
#result_df = result_df.dropna(axis=0)
result_df = result_df.apply(pd.to_numeric)
result_df = result_df.fillna(result_df.mean())
print(result_df.isnull().sum())
print(len(result_df))
# result_df.plot()
# plt.title("Pandas의 Plot메소드 사용 예")
# plt.xlabel("시간")
# plt.ylabel("Data")
# plt.show()
result_df = result_df.reset_index()
print(result_df)



dataset_train_actual = result_df.copy()  # 복사

dataset_train_timeindex = dataset_train_actual.set_index('index')  # 시간이 인덱스인 데이터프레임 생성
dataset_train = dataset_train_actual.copy()  # 인덱스가 없는 것 데이터 프레임으로
print('dataset_train')
print(dataset_train)
print("\n\n\n")
print('dataset_time')
print(dataset_train_timeindex)

cols = list(dataset_train)[1:]  # ['index', 'DATETIME', 'PM10', 'TEMPERATURE', 'HUMIDITY'], 0:1이면 0번째 인덱스에서 한개만 가져옴
print('cols')
print(cols)
print("\n\n\n")

datelist_train = list(dataset_train['index'])
datelist_train = [date for date in datelist_train]
print('datelist_train')  # 시간가져옴
# print(datelist_train)
print("\n\n\n")

dataset_train = dataset_train[cols]
print(dataset_train)
training_set = dataset_train.values
print(len(training_set))
print('output')
print(training_set[:, 0:2])

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
print(X_train)
print(y_train)

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

y_pred_inv = sc_predict.inverse_transform(predictions_future)
y_test_inv = sc_predict.inverse_transform(y_train)
print(type(y_test_inv))
print(y_test_inv)

# plt.plot(datelist_train[n_past - 1:], y_test_inv, label="Actual")
# plt.plot(datelist_train[n_past - 1:], y_pred_inv, label="Predict")
# plt.ylabel('pm10')
# plt.xlabel("Time")
# plt.legend(loc=2, ncol=2, bbox_to_anchor=(0.3, 1.15))
# plt.show()

test_y = pd.DataFrame(y_test_inv, columns=['tem', 'hu'])
pred_y = pd.DataFrame(y_pred_inv, columns=['tem', 'hu'])

fig = go.Figure()

fig.add_trace(go.Scatter(x=datelist_train[n_past - 1:], y=test_y['tem'],
                         mode='lines',
                         name='원래온도', line=dict(color='firebrick', width=1)))

fig.add_trace(go.Scatter(x=datelist_train[n_past - 1:], y=pred_y['tem'],
                         mode='lines',
                         name='예측온도', line=dict(color='red', width=1)))

fig.add_trace(go.Scatter(x=datelist_train[n_past - 1:], y=test_y['hu'],
                         mode='lines',
                         name='원래습도', line=dict(color='blue', width=1)))

fig.add_trace(go.Scatter(x=datelist_train[n_past - 1:], y=pred_y['hu'],
                         mode='lines',
                         name='예측습도', line=dict(color='royalblue', width=1)))

# 배경 레이어색 파트
fig.update_layout(paper_bgcolor="black")  # 차트 바깥 색
fig.update_layout(plot_bgcolor="black")  # 그래프 안쪽색
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.update_layout(xaxis_tickformat='%Y<br>%m-%d<br>%H시')
fig.update_xaxes(linecolor='gray', gridcolor='gray', mirror=True)
fig.update_yaxes(linecolor='gray', gridcolor='gray', mirror=True)
fig.update_yaxes(tickformat=',')  # 간단하게 , 형으로 변경

fig.write_html('ex.html')
