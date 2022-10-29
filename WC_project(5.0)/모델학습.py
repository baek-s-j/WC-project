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
import joblib
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

client = MongoClient('mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')
db = client.wc_project

collection = db.zone
z_result = list(collection.find({}))
collection = db.thdata
for i in z_result:
    print(i['name'])
    orr_v = list()
    for j in range(1, 3):
        th_result = list(
            collection.find({"zone_name": i['name'], }).sort([("time", 1)]))
        th_frame = pd.DataFrame(th_result)
        th_frame.time = pd.to_datetime(th_frame.time)
        th_frame = th_frame.set_index('time')
        min_th_frame = th_frame.resample(rule='H').mean()
        print(min_th_frame)

        empty_df = pd.read_csv("weather_무위사.csv")
        empty_df.tm = pd.to_datetime(empty_df.tm)
        empty_df = empty_df.set_index('tm')

        result_df = pd.concat([min_th_frame, empty_df], axis=1)
        result_df = result_df.round(1)
        result_df = result_df.apply(pd.to_numeric)
        result_df = result_df.dropna(axis=0)
        result_df.drop(['Unnamed: 0'], axis=1, inplace=True)
        print(result_df.isnull().sum())
        print(result_df)

        total_vars = ["ta", "rn", "ws", "wd", "hm"]
        input_vars = total_vars
        if j == 1:
            output_var = "temperature"
        else:
            output_var = "humidity"
        # temperature or humidity

        input_vars.insert(0, output_var)  # 0번째 열의 PM10추가
        print(input_vars)
        training_set = result_df[input_vars].values  # 값만 가져옴

        # In[7]:

        sc = MinMaxScaler()
        training_set_scaled = sc.fit_transform(training_set[:, 1:])  # 정규화 실행
        sc_predict = MinMaxScaler()
        y_training_set_scaled = sc_predict.fit_transform(training_set[:, 0:1])  # output만 정규화

        X, y = np.array(training_set_scaled), np.array(y_training_set_scaled)
        print(X.shape)
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        y = y.reshape((y.shape[0], 1, y.shape[1]))
        print(X.shape)
        print(y.shape)
        print(training_set_scaled)
        print(y_training_set_scaled)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)  # 17

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(units=128, return_sequences=True, input_shape=(1, X.shape[2])))
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
        mcp = tf.keras.callbacks.ModelCheckpoint(filepath='data/weights_%s.h5' % i['name'], monitor='val_loss',
                                                 verbose=1,
                                                 save_best_only=True, save_weights_only=True)

        tb = tf.keras.callbacks.TensorBoard('data/logs')

        history = model.fit(X_train, y_train, shuffle=False, epochs=100, callbacks=[es, rlr, mcp, tb],
                            validation_split=0.2,
                            verbose=1, batch_size=64)

        m_name = i['name'] + output_var
        model.save("model/model24_%s.h5" % m_name)

        # In[13]:

        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend(loc=2, ncol=2, bbox_to_anchor=(0.3, 1.15))
        plt.show()

        # from keras.models import load_model
        #
        # model = load_model("model/model24_temperature.h5")

        print(X_test.shape)
        print(X_test)

        # c=sc.transform(training_set[192:n_past+192, 1:])
        # ex=list()
        # ex.append(c)
        # a=np.array(ex)
        # print('여기')
        # print(a.shape)
        # b = model.predict(a)
        # print(b)
        # print(b.T)
        # b=sc_predict.inverse_transform(b.T)
        # print(b)

        y_pred = model.predict(X_test)
        print(y_pred)
        nsamples, nx, ny = y_test.shape
        y_test = y_test.reshape((nsamples, nx * ny))
        y_test_inv = sc_predict.inverse_transform(y_test)
        y_pred_inv = sc_predict.inverse_transform(y_pred)
        # print(y_test_inv)
        # print(y_pred_inv)

        joblib.dump(sc, i['name'] + output_var + "x")
        joblib.dump(sc_predict, i['name'] + output_var + "y")
        # In[16]:

        # In[17]:

        print(input_vars)
        print('R^2: %.2f' % (r2_score(y_test_inv, y_pred_inv) * 100))
        print('RMSE: %.2f' % (np.sqrt(mse(y_test_inv, y_pred_inv))))
        print('MAE: %.2f' % (mae(y_test_inv, y_pred_inv)))
        orr_v.append((np.sqrt(mse(y_test_inv, y_pred_inv))))

        # In[18]:
        # act=[28,28,28,28,26,25,25,25,25.3,27.5,29.7,30.7,31.2,32,32.7,32.5,30.8,30,29.7,29,29,29,29,29]
        # b=np.round(b,1)
        # print(b)
        # plt.plot(act, label="Actual")
        # plt.plot(b, label="Predict")
        # plt.ylabel(output_var)
        # plt.xlabel("Time")
        # plt.legend(loc=2, ncol=2, bbox_to_anchor=(0.3, 1.15))
        # plt.show()

        plt.plot(y_test_inv[:24], label="Actual")
        plt.plot(y_pred_inv[:24], label="Predict")
        plt.ylabel(output_var)
        plt.xlabel("Time")
        plt.legend(loc=2, ncol=2, bbox_to_anchor=(0.3, 1.15))
        plt.show()

    print(orr_v)
