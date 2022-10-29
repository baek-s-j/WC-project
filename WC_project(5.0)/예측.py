import joblib
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
from flask import Flask, render_template, request, redirect, url_for, flash, session
from pymongo import MongoClient
from bson.objectid import ObjectId
import requests

import math
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from apscheduler.schedulers.background import BackgroundScheduler

import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from keras.models import load_model

np.random.seed(1)
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score

import tensorflow as tf


# 기상예보 데이터 읽은다음에
def p():
    client = MongoClient('mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')
    db = client.wc_project

    collection = db.zone
    z_result = list(collection.find({}))
    f_json = list()

    for i in z_result:
        fig = go.Figure()

        result_df = pd.read_csv("예시.csv")
        random_x = result_df['tm'].values.tolist()
        total_vars = ["ta", "rn", "ws", "wd", "hm"]
        input_vars = total_vars

        for j in range(1, 3):

            if j == 1:
                output_var = "temperature"
                name = "온도(℃)"
                color = 'firebrick'
                se = False
                fill = 'rgba(255,0,0,0.2)'
            else:
                output_var = "humidity"
                name = "습도(%)"
                color = 'royalblue'
                se = True
                fill = 'rgba(0,0,255,0.2)'

            # input_vars.insert(0, output_var)  # 0번째 열의 PM10추가
            print(input_vars)
            training_set = result_df[input_vars].values  # 값만 가져옴

            m_name = i['name'] + output_var
            model = load_model("model/model24_%s.h5" % m_name)

            sc = joblib.load(i['name'] + output_var + "x")
            sc_predict = joblib.load(i['name'] + output_var + "y")

            x_data = sc.transform(training_set[:, 0:])

            X = np.array(x_data)
            X = X.reshape((X.shape[0], 1, X.shape[1]))

            y_pred = model.predict(X)
            y_pred_inv = sc_predict.inverse_transform(y_pred)  # tuple
            y_pred_inv = np.asarray(y_pred_inv)
            y_pred_inv = np.transpose(y_pred_inv)  # 행열 바꾸기
            y_pred_inv = y_pred_inv.tolist()  # list로

            print(y_pred_inv)
            random_y0 = y_pred_inv

            # 예측값 타입 알아봐야함

            fig.add_trace(go.Scatter(
                x=random_x + random_x[::-1],
                y=([random_y0[0][i] + 5 for i in range(len(random_y0[0]))]) + ([random_y0[0][i] - 5 for i in
                                                                                range(len(random_y0[0]))])[
                                                                              ::-1],
                fill='toself',
                fillcolor=fill,
                mode='lines',
                line_color='rgba(255,255,255,0)',
                name=name[:2] + " 오차범위",

            ))

            fig.add_trace(go.Scatter(x=random_x, y=y_pred_inv[0],
                                     mode='lines+markers', line=dict(color=color, width=1.0),
                                     name=name))

        fig.update_layout(paper_bgcolor="#EAEAEA", margin=dict(l=10, r=10, t=60, b=10), )  # 차트 바깥 색
        fig.update_layout(plot_bgcolor="#F6F6F6")  # 그래프 안쪽색
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_layout(xaxis_tickformat='%Y<br>%m-%d<br>%H시%M분')
        fig.update_xaxes(linecolor='gray', gridcolor='gray', mirror=True, zeroline=False)
        fig.update_yaxes(linecolor='gray', gridcolor='gray', mirror=True)
        fig.update_yaxes(title_text="습도(%)    온도(℃)", zeroline=False, range=[-10, 100])

        f_json.append(fig.to_json())

    return f_json


p()
