from flask import Flask, render_template, request, redirect, url_for, flash, session
from pymongo import MongoClient
from bson.objectid import ObjectId
import requests
import json
import math
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from apscheduler.schedulers.background import BackgroundScheduler
import warnings
import xmltodict
import ssl
import pandas as pd
import numpy as np
import joblib

import plotly.graph_objects as go

from keras.models import load_model

app = Flask(__name__)  # 플라스크 app을 생성
app.jinja_env.add_extension('jinja2.ext.loopcontrols')  # html에서 파이썬 코드를 사용할수 있는 jinja2 임포트
app.secret_key = '2017'  # 앱의 비밀키 설정
global db  # 로그인시 유물공간을 동적으로 바꿔주기위함
global d_where
d_where = ""
client = MongoClient('mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')  # 몽고디비 아틀라스 연결
db = client.wc_project  # 초기는 강진 무위사로 설정
ac = 1
schedule_1 = BackgroundScheduler(timezone='Asia/Seoul')  # 스케줄러 생성및 타임존 설정
schedule_2 = BackgroundScheduler(timezone='Asia/Seoul')


# 십분마다 실행하게 설정
@schedule_1.scheduled_job('cron', hour='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23',
                          minute='0,10,20,30,40,50', id='thdata_job')
def thdata_job():  # 온습도 데이터를 디비에 저장후에 유물공간을 동적으로 변경
    exec(open("Thdataapi.py", encoding='utf-8').read())
    global db

    # if d_where == "김천 직지사":
    #     db = client.wc_project_직지사
    # elif d_where == "여수 흥국사":
    #     db = client.wc_project_흥국사
    # else:
    #     db = client.wc_project


# 30분 간격으로 위험 감지 실행
@schedule_2.scheduled_job('cron', hour='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23',
                          minute='0,30', second='5', id='risknotice_job')
def risknotice_job():  # 온습도 및 배터리 데이터를 가지고와서 위험정도 측정
    global db
    exec(open("alarm_timeline.py", encoding='utf-8').read())
    # if d_where == "김천 직지사":
    #     db = client.wc_project_직지사
    # elif d_where == "여수 흥국사":
    #     db = client.wc_project_흥국사
    # else:
    #     db = client.wc_project


schedule_1.start()  # 스케줄러 시작
schedule_2.start()


def realtime_visual(query):  # 10분마다 들어온 온습도 데이터를 시각화 하는 함수
    collection = db.zone
    z_result = list(collection.find())
    f_json = list()
    dfs = list()

    for i in z_result:  # 구역마다 실행
        collection = db.thdata  # 온습도데이터를 구역마다 가져옴
        th_result = list(collection.find({"zone_name": i['name'], "time": {"$regex": query}}).sort([("time", 1)]))
        if not th_result:
            return False, False
        th_frame = pd.DataFrame(th_result)

        # 판다 데이터프레임에서 온습도, 시간 칼럼만 추출
        random_x = th_frame['time']
        random_y0 = th_frame['temperature']
        random_y1 = th_frame['humidity']

        # 그래프 그리기 파트
        fig = go.Figure(
            data=[go.Scatter(x=random_x, y=random_y1,
                             mode="lines",
                             name='마크(습도)', line=dict(color='firebrick', width=0.5), showlegend=False),

                  go.Scatter(x=random_x, y=random_y0,
                             mode="lines",
                             name='마크(온도)', line=dict(color='royalblue', width=0.5), showlegend=False),
                  go.Scatter(x=random_x, y=random_y0,
                             mode="lines",
                             name='온도(℃)', line=dict(color='firebrick', width=0.5)),
                  go.Scatter(x=random_x, y=random_y1,
                             mode="lines",
                             name='습도(%)', line=dict(color='royalblue', width=1))],
            # 애니메이션을 위해 프레임 추가
            frames=[go.Frame(
                data=[go.Scatter(
                    x=[random_x.loc[k]],

                    y=[random_y0.loc[k]],

                    mode="markers",
                    marker=dict(color="red", size=10)),
                    go.Scatter(
                        x=[random_x.loc[k]],

                        y=[random_y1.loc[k]],

                        mode="markers",
                        marker=dict(color="blue", size=10))
                ])

                for k in range(len(random_x))

            ]
        )
        # 안전온도범위와 습도 위험범위 추가
        fig.add_hrect(y0=10, y1=34, line_width=0, fillcolor="green", opacity=0.1, annotation_text="안전 온도 범위",
                      annotation_position="top left", )
        fig.add_hrect(y0=75, y1=100, line_width=0, fillcolor="red", opacity=0.1, annotation_text="습도 위험 범위",
                      annotation_position="top left", )

        # 그래프 레이아웃 설정
        fig.update_layout(paper_bgcolor="#EAEAEA", margin=dict(l=10, r=10, t=60, b=10), )
        fig.update_layout(plot_bgcolor="#F6F6F6")  # 그래프 안쪽색
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_layout(xaxis_tickformat='%Y<br>%m-%d<br>%H시%M분')
        fig.update_xaxes(linecolor='gray', gridcolor='gray', mirror=True, zeroline=False)
        fig.update_yaxes(linecolor='gray', gridcolor='gray', mirror=True)
        fig.update_yaxes(title_text="습도(%)    온도(℃)", zeroline=False, range=[-10, 100])

        f_json.append(fig.to_json())
        dfs.append(th_frame)

    return f_json, dfs


def predict_visual():  # LSTM 예측한 것을 시각화 해주는 함수
    collection = db.zone
    z_result = list(collection.find({}))
    f_json = list()

    if z_result[0]['name'][-3:] == "무위사":  # 구역마다 기상청 api 코드가 다르므로 설정하고 RMSE값을 설정
        x_error = [2, 2, 2, 2, 2.5]
        y_error = [4.4, 7, 7, 4.5, 7]
        nx = '56'
        ny = '64'
    elif z_result[0]['name'][-3:] == "직지사":
        x_error = [2, 2, 2, 2.4, 2.4]
        y_error = [7, 5.5, 6, 8, 9]
        nx = '79'
        ny = '96'
    else:  # 여수 흥국사
        x_error = [1.7, 1.6, 1.5, 1.7, 1.6, 1.6]
        y_error = [6, 5.7, 5.5, 5.8, 5.4, 5.4]
        nx = '73'
        ny = '67'

    num_index = 0
    ssl._create_default_https_context = ssl._create_unverified_context
    warnings.filterwarnings('ignore')

    yesterday = datetime.today() - timedelta(1)

    yesterday = yesterday.strftime("%Y-%m-%d")

    r1 = yesterday[0:4]
    r2 = yesterday[5:7]
    r3 = yesterday[8:10]
    today_result = str(r1 + r2 + r3)

    serviceKey = 'Zy3WCPt243lyaq0PqqKVwGL%2F42nvD9qjxGcz%2FgVr3y01%2BxYJ%2BsmdjB1H01RWJsVgaQkayn32SBzLQDdMgpiVIg%3D%3D'  # api 키
    pageNo = '1'  # 페이지번호
    numOfRows = '290'  # 한 페이지 결과 수
    dataType = 'XML'  # 요청자료형식(XML/JSON) Default: XML
    base_date = '{}'.format(today_result)  # ‘22년 x월 x일 발표 최근껄로 해야지 오류가 안남
    # 날짜 설정 잘해야함 오류날 수 있음
    base_time = '2300'  # 06시30분 발표(30분 단위)
    # 마찬가지 오늘날짜 기준 잘 설정해야함

    url = 'https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst?serviceKey={}&pageNo={}&numOfRows={}&dataType={}&base_date={}&base_time={}&nx={}&ny={}'.format(
        serviceKey, pageNo, numOfRows, dataType, base_date, base_time, nx, ny)

    content = requests.get(url, verify=False).content
    dict_a = xmltodict.parse(content)

    jsonString = json.dumps(dict_a['response']['body']['items'], ensure_ascii=False)

    ls_dict = json.loads(jsonString)

    df = pd.DataFrame(ls_dict['item'])

    result_df = pd.DataFrame()
    for i in range(0, 24):  # 24시간 데이터를 가져오기 위해 반복
        if i < 10:
            iname = '0' + str(i)

            str_expr = "fcstTime.str.contains('{}00')".format(iname)  # 문자열에 '' 포함

        elif i >= 10:

            str_expr = "fcstTime.str.contains('{}00')".format(i)  # 문자열에 '' 포함

        tdf = df.query(str_expr)
        tdf.drop(['baseDate'], axis=1, inplace=True)  # 필요없는 데이터 삭제
        tdf.drop(['baseTime'], axis=1, inplace=True)
        tdf.drop(['fcstDate'], axis=1, inplace=True)
        tdf.drop(['fcstTime'], axis=1, inplace=True)
        tdf.drop(['nx'], axis=1, inplace=True)
        tdf.drop(['ny'], axis=1, inplace=True)

        tdf = tdf.transpose()  # 행렬 변환

        tdf = tdf.rename(columns=tdf.iloc[0])

        tdf.rename(columns={'TMP': 'ta'}, inplace=True)  # 요구에 맞게 이름을 변경
        tdf.rename(columns={'PCP': 'rn'}, inplace=True)
        tdf.rename(columns={'WSD': 'ws'}, inplace=True)
        tdf.rename(columns={'VEC': 'wd'}, inplace=True)
        tdf.rename(columns={'REH': 'hm'}, inplace=True)
        tdf.drop(['category'], axis=0, inplace=True)
        tdf = tdf.reset_index()
        tdf.drop(['index'], axis=1, inplace=True)

        tdf = tdf[['ta', 'wd', 'ws', 'rn', 'hm']]

        tdf.loc[tdf['rn'] == "강수없음", 'rn'] = 0

        tm = datetime.now().strftime('%Y-%m-%d') + " " + "{}:00".format(i)

        tdf.insert(0, 'tm', tm)

        result_df = result_df.append(tdf)

    result_df = result_df.astype({'rn': 'str'})
    result_df['rn'] = result_df['rn'].str.replace('mm', '')
    result_df.tm = pd.to_datetime(result_df.tm, format='%Y-%m-%d')  # datetime으로 변환
    result_df = result_df.set_index('tm')
    result_df = result_df.apply(pd.to_numeric)  # 값들은 숫자형으로
    random_x = result_df.index.tolist()  # 인덱스를 리스트화 해서 저장
    total_vars = ["ta", "rn", "ws", "wd", "hm"]
    input_vars = total_vars

    for i in z_result:
        # collection = db.thdata
        # th_result = list(
        #     collection.find({"zone_name": i['name'], "time": {"$regex": datetime.now().strftime('%Y-%m-%d')}}).sort(
        #         [("time", 1)]))
        # th_frame = pd.DataFrame(th_result)
        # th_frame.time = pd.to_datetime(th_frame.time)
        # th_frame = th_frame.set_index('time')
        # min_th_frame = th_frame.resample(rule='H').mean()
        # min_th_frame['utime'] = min_th_frame.index
        # min_th_frame = min_th_frame.fillna(method='pad')
        # min_th_frame = min_th_frame.round(1)
        # # 판다 데이터프레임에서 칼럼만 추출
        # # random_x = th_frame['time']
        # act_y0 = min_th_frame['temperature']
        # act_y1 = min_th_frame['humidity']
        fig = go.Figure()

        for j in range(1, 3):  # 온도 예측 모델과 습도예측모델이 따로 있으므로 따로 설정

            if j == 1:
                output_var = "temperature"
                name = "예측온도(℃)"
                color = 'firebrick'
                fill = 'rgba(255,0,0,0.2)'
            else:
                output_var = "humidity"
                name = "예측습도(%)"
                color = 'royalblue'
                fill = 'rgba(0,0,255,0.2)'

            training_set = result_df[input_vars].values  # 값만 가져옴

            m_name = i['name'] + output_var
            model = load_model("model/model24_%s.h5" % m_name)  # 미리 학습시켜둔 lstm 모델을 가져온다

            sc = joblib.load(i['name'] + output_var + "x")  # 미리 학습시켜둔 정규와 minmax모델도 가져온다
            sc_predict = joblib.load(i['name'] + output_var + "y")

            x_data = sc.transform(training_set[:, 0:])  # 정규화 실행

            X = np.array(x_data)  # 예측모델의 인풋으로 들어가게끔 변환
            X = X.reshape((X.shape[0], 1, X.shape[1]))

            y_pred = model.predict(X)  # 예측 실행
            y_pred_inv = sc_predict.inverse_transform(y_pred)  # 정규화값을 다시 역변환
            y_pred_inv = np.asarray(y_pred_inv)
            y_pred_inv = np.transpose(y_pred_inv)  # 행열 바꾸기
            y_pred_inv = y_pred_inv.tolist()  # list로
            # print(y_pred_inv)
            # print(random_x)
            random_y0 = y_pred_inv

            # 시각화 해주는 코드
            if j == 1:  # 오차범위 먼저 그려준다
                fig.add_trace(go.Scatter(
                    x=random_x + random_x[::-1],
                    y=([random_y0[0][i] + x_error[num_index] for i in range(len(random_y0[0]))]) + ([
                        random_y0[0][i] - x_error[num_index] for i in
                        range(len(random_y0[0]))])[
                                                                                                   ::-1],
                    fill='toself',
                    fillcolor=fill,
                    mode='lines',
                    line_color='rgba(255,255,255,0)',
                    name=name[2:4] + "오차범위(±" + str(x_error[num_index]) + ")",

                ))
            else:  # 오차범위 먼저 그려준다
                fig.add_trace(go.Scatter(
                    x=random_x + random_x[::-1],
                    y=([random_y0[0][i] + y_error[num_index] for i in range(len(random_y0[0]))]) + ([
                        random_y0[0][i] - y_error[num_index] for i in
                        range(len(random_y0[0]))])[
                                                                                                   ::-1],
                    fill='toself',
                    fillcolor=fill,
                    mode='lines',
                    line_color='rgba(255,255,255,0)',
                    name=name[2:4] + "오차범위(±" + str(y_error[num_index]) + ")",

                ))
            # 예측값 나온 것을 시각화
            fig.add_trace(go.Scatter(x=random_x, y=y_pred_inv[0],
                                     mode='lines+markers', line=dict(color=color, width=1.0),
                                     name=name))

        # fig.add_trace(go.Scatter(x=random_x, y=act_y0,
        #                          mode='lines', line=dict(color='black', width=1.0),
        #                          ))
        # fig.add_trace(go.Scatter(x=random_x, y=act_y1,
        #                          mode='lines', line=dict(color='black', width=1.0),
        #                          ))
        # 안전 온도범위와 습도 위험범위를 시각화
        fig.add_hrect(y0=10, y1=34, line_width=0, fillcolor="green", opacity=0.1, annotation_text="안전 온도 범위",
                      annotation_position="top left", )
        fig.add_hrect(y0=75, y1=100, line_width=0, fillcolor="red", opacity=0.1, annotation_text="습도 위험 범위",
                      annotation_position="top left", )
        # 그래프 레이아웃 설정
        fig.update_layout(paper_bgcolor="#EAEAEA", margin=dict(l=10, r=10, t=60, b=10), )  # 차트 바깥 색
        fig.update_layout(plot_bgcolor="#F6F6F6")  # 그래프 안쪽색
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_layout(xaxis_tickformat='%Y<br>%m-%d<br>%H시%M분')
        fig.update_xaxes(linecolor='gray', gridcolor='gray', mirror=True, zeroline=False)
        fig.update_yaxes(linecolor='gray', gridcolor='gray', mirror=True)
        fig.update_yaxes(title_text="습도(%)    온도(℃)", zeroline=False, range=[-10, 100])
        num_index = num_index + 1
        f_json.append(fig.to_json())

    return f_json


def visual(query1, query2):  # 평균 온습도 데이터 시각화 함수

    collection = db.zone
    z_result = list(collection.find())
    f_json = list()
    dfs = list()

    for i in z_result:  # 구역별로 온습도 데이터로 가져옴
        collection = db.thdata
        th_result = list(
            collection.find({"zone_name": i['name'], "time": {"$gte": query1, "$lte": query2}}).sort([("time", 1)]))
        if not th_result:
            return False, False
        th_frame = pd.DataFrame(th_result)
        th_frame.time = pd.to_datetime(th_frame.time)  # 시각을 데이트 타입으로 변경
        th_frame = th_frame.set_index('time')  # 인덱스로 설정
        min_th_frame = th_frame.resample(rule='H').mean()  # 한시간 평균으로 행 합치기
        min_th_frame['utime'] = min_th_frame.index  # 시간 데이터 전처리
        min_th_frame['utime'] = min_th_frame['utime'].astype('str')
        min_th_frame['utime'] = min_th_frame['utime'].str.slice(start=0, stop=16)
        min_th_frame = min_th_frame.fillna(method='pad')  # 만약 비어있는 값이 있다면 채운다
        min_th_frame = min_th_frame.round(1)  # 소수점 첫째까지 반올림

        # 판다 데이터프레임에서 칼럼만 추출
        random_x = min_th_frame['utime']
        random_y0 = min_th_frame['temperature']
        random_y1 = min_th_frame['humidity']

        # 그래프 그리기 파트
        fig = go.Figure(
            data=[
                go.Scatter(x=random_x, y=random_y0,
                           mode="lines",
                           name='온도(℃)', line=dict(color='firebrick', width=0.5)),
                go.Scatter(x=random_x, y=random_y1,
                           mode="lines",
                           name='습도(%)', line=dict(color='royalblue', width=1))]

        )
        # 안전 온도범위와 습도 위험범위를 시각화
        fig.add_hrect(y0=10, y1=34, line_width=0, fillcolor="green", opacity=0.1, annotation_text="안전 온도 범위",
                      annotation_position="top left", )
        fig.add_hrect(y0=75, y1=100, line_width=0, fillcolor="red", opacity=0.1, annotation_text="습도 위험 범위",
                      annotation_position="top left", )

        # 그래프 레이아웃 설정
        fig.update_layout(paper_bgcolor="#EAEAEA")  # 차트 바깥 색
        fig.update_layout(plot_bgcolor="#F6F6F6")  # 그래프 안쪽색
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_layout(xaxis_tickformat='%Y<br>%m-%d<br>%H시%M분')
        fig.update_xaxes(linecolor='gray', gridcolor='gray', mirror=True, zeroline=False)
        fig.update_yaxes(linecolor='gray', gridcolor='gray', mirror=True)
        fig.update_yaxes(title_text="습도(%)    온도(℃)", zeroline=False, range=[-10, 100])

        # 자세하게 볼수있는 슬라이더 애니메이션 추가
        fig.update_layout(

            margin=dict(l=10, r=10, t=60, b=10),
            xaxis=dict(
                # rangeselector_yanchor="auto",
                # rangeselector_xanchor="auto",
                # rangeselector_bgcolor="#000",

                rangeslider=dict(
                    visible=True,

                ),
                type="date"
            )
        )

        f_json.append(fig.to_json())
        dfs.append(min_th_frame)

    return f_json, dfs


def three_day_visual(query1, query2):  # Home 화면 시각화 함수 3일치 온습도 데이터를 시각화 해주는함수
    collection = db.zone
    z_result = list(collection.find())
    f_json = list()

    collection = db.thdata
    th_result = list(
        collection.find({"zone_name": z_result[0]['name'], "time": {"$gte": query1, "$lte": query2}}).sort(
            [("time", 1)]))

    if not th_result:
        return False, False
    th_frame = pd.DataFrame(th_result)
    th_frame.time = pd.to_datetime(th_frame.time)
    th_frame = th_frame.set_index('time')
    min_th_frame = th_frame.resample(rule='H').mean()
    min_th_frame['utime'] = min_th_frame.index
    min_th_frame = min_th_frame.fillna(method='pad')
    min_th_frame = min_th_frame.round(1)

    # 판다 데이터프레임에서 온습도, 시간 칼럼만 추출
    random_x = min_th_frame['utime']
    random_y0 = min_th_frame['temperature']
    random_y1 = min_th_frame['humidity']

    # 그래프 그리기 파트
    fig = go.Figure(
        data=[go.Scatter(x=random_x, y=random_y1,
                         mode="lines",
                         name='마크(습도)', line=dict(color='firebrick', width=0.5), showlegend=False),

              go.Scatter(x=random_x, y=random_y0,
                         mode="lines",
                         name='마크(온도)', line=dict(color='royalblue', width=0.5), showlegend=False),
              go.Scatter(x=random_x, y=random_y0,
                         mode="lines",
                         name='온도', line=dict(color='firebrick', width=0.5)),
              go.Scatter(x=random_x, y=random_y1,
                         mode="lines",
                         name='습도', line=dict(color='royalblue', width=1))],

    )
    fig.add_hrect(y0=10, y1=34, line_width=0, fillcolor="green", opacity=0.1, annotation_text="안전 온도 범위",
                  annotation_position="top left", )
    fig.add_hrect(y0=75, y1=100, line_width=0, fillcolor="red", opacity=0.1, annotation_text="습도 위험 범위",
                  annotation_position="top left", )

    # 배경 레이어색 파트
    fig.update_layout(paper_bgcolor="#EAEAEA", margin=dict(l=10, r=10, t=60, b=10), )  # 차트 바깥 색
    fig.update_layout(plot_bgcolor="#F6F6F6")  # 그래프 안쪽색
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_layout(xaxis_tickformat='%Y<br>%m-%d<br>%H시%M분')
    fig.update_xaxes(linecolor='gray', gridcolor='gray', mirror=True, zeroline=False)
    fig.update_yaxes(linecolor='gray', gridcolor='gray', mirror=True)
    fig.update_yaxes(title_text="습도(%)    온도(℃)", zeroline=False, range=[-10, 100])
    fig.update_layout(

        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(
            # rangeselector_yanchor="auto",
            # rangeselector_xanchor="auto",
            # rangeselector_bgcolor="#000",

            rangeslider=dict(
                visible=True,

            ),
            type="date"
        )
    )
    f_json.append(fig.to_json())

    return f_json


@app.route('/addManager.html')
def addManager():  # 관리자 회원가입하는 페이지 구성함수
    url = "https://goqual.io/openapi/homes"
    # api를 이용하여 현재 관리중인 유물공간 가져오기
    response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})

    homes_list = response.json()

    return render_template('addManager.html', homes_list=homes_list['result'])


@app.route('/addManagerAct', methods=['POST', 'GET'])
def addManagerAct():  # 관리자 회원가입 실행 함수
    if request.method == 'POST':  # Post 요청시
        global db
        db = client.wc_project
        collection = db.manager
        result = request.form.to_dict()  # 폼으로 값을 가져옴

        id_check = list(collection.find({'manager_id': result.get('manager_id')}))  # 혹시 아이디가 중복되는 확인

        if not id_check:

            url = "https://goqual.io/openapi/homes"
            response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})
            homes_list = response.json()
            for i in homes_list['result']:  # 유물공간의 homeId를 찾아서
                if i['name'] == result['manager_place']:
                    result['place_num'] = i['homeId']  # 넣고

            collection.insert_one(result)  # 몽고디비에 저장

            flash("회원가입이 완료되었습니다.")
            return redirect(url_for("login"))
        else:
            flash("아이디가 중복되었습니다.")
            return redirect(url_for("addManager"))


@app.route('/')
@app.route('/login.html')
def login():  # 홈페이지 첫 화면 로그인화면
    return render_template('login.html')


@app.route('/loginAct', methods=['POST', 'GET'])
def loginAct():  # 로그인 실행 함수
    if request.method == 'POST':  # post 요청시
        global db
        global d_where
        db = client.wc_project
        collection = db.manager
        result = request.form.to_dict()  # 폼으로 받은 로그인정보 가져오기
        if result['manager_id'] == "":  # 아이디 입력 안했을때 처리
            flash("아이디를 입력해주세요")
            return redirect(url_for("login"))

        if result['manager_password'] == "":  # 비밀번호 입력 안했을때 처리
            flash("비밀번호를 입력해주세요.")

            return redirect(url_for("login"))

        info = list(collection.find({'manager_id': result.get('manager_id')}))  # 아이디에 맞는 비밀번호를 가져옴
        if not info:  # 디비에 저장된 아이디가 아닐때
            flash("아이디가 존재하지 않습니다.")

            return redirect(url_for("login"))

        if result.get('manager_password') == info[0]['manager_password']:  # 비밀번호 일치하는지 확인하고
            session['id'] = result.get('manager_id')  # 세션을 저장함 페이지를 유동적으로 바꿔주기 위함
            session['place_num'] = info[0]['place_num']
            session['manager_place'] = info[0]['manager_place']
            d_where = info[0]['manager_place']
            if info[0]['manager_place'] == "김천 직지사":
                db = client.wc_project_직지사
            elif info[0]['manager_place'] == "여수 흥국사":
                db = client.wc_project_흥국사
            else:
                db = client.wc_project

            return redirect(url_for("index"))

        else:  # 비밀번호 불일치시
            flash("아이디 또는 비밀번호가 맞지 않습니다.")
            return redirect(url_for("login"))


@app.route('/logout')
def logout():  # 로그아웃 함수
    session.pop('id', None)  # 세션 전부 제거뒤에
    session.pop('place_num', None)
    session.pop('manager_place', None)
    global db
    db = client.wc_project  # 초기 DB로 변경하기
    return redirect(url_for("login"))


@app.route('/index.html')
def index():  # home 화면
    global db

    weather_url = ""
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            weather_url = "https://forecast.io/embed/#lat=36.1213&lon=128.1186&name=김천&color=&font=&units=si"
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            weather_url = "https://forecast.io/embed/#lat=34.7617&lon=127.6628&name=여수&color=&font=&units=si"
            db = client.wc_project_흥국사
        else:
            weather_url = "https://forecast.io/embed/#lat=34.6362&lon=126.7702&name=강진&color=&font=&units=si"
            db = client.wc_project

    collection = db.relic  # 각각의 유물, 센서, 구역의 갯수를 가져온다
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    number2 = collection.estimated_document_count()
    count_list = list((number1, number2, number3))
    now = time
    now_time = now.strftime('%Y-%m-%d')  # 현재시간 가져오기
    query2 = now_time
    query1 = (datetime.now() - relativedelta(days=3)).strftime('%Y-%m-%d')  # 3일전 시간 구하기

    fig_code = three_day_visual(query1, query2)  # 시각화 실행

    collection = db.timeline  # 타임라인 디비에서 오늘 알림 가져오기
    timeline_result = list(collection.find({"time": {"$regex": query2}}).sort([("time", 1)]))
    if not timeline_result:
        timeline_result.append({'time': '오늘 현재까지', 'zone_name': '모든 구역 안전합니다.', 'risk_name': 'success'})

    return render_template('index.html', count=count_list, fig_code=fig_code, timeline_result=timeline_result,
                           weather_url=weather_url)


@app.route('/about.html')
def about():  # about 페이지 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    collection = db.relic  # 각각의 유물, 센서, 구역의 갯수를 가져온다
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    number2 = collection.estimated_document_count()
    count_list = list((number1, number2, number3))

    return render_template('about.html', count=count_list)


@app.route('/visualization.html')
def visualization():  # Today 시각화 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    collection = db.relic  # 각각의 유물, 센서, 구역의 갯수를 가져온다
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    z_result = list(collection.find())
    number2 = collection.estimated_document_count()
    count_list = list((number1, number2, number3))

    now = time  # 5~7은 지금 현재시각을 가져오는 코드임 2022-08-22 20:48:57 이렇게
    query = now.strftime('%Y-%m-%d')
    th_tmp = list()
    th_num = list()
    th_slide = list()

    fig_code, dfs = realtime_visual(query)  # 시각화 함수 실행
    if not fig_code:  # 만약 온습도 데이터가 한개도 없을시
        flash("업데이트가 되지 않았습니다.")
        return redirect(url_for("index"))

    for i in range(0, len(z_result)):  # 구역별로
        tmp = dfs[i].values.tolist()  # 몽고디비에서 가져온 온습도 데이터를 list로 변환하고
        tmp2 = len(dfs[i])  # 길이구하기
        th_num.append(tmp2)
        tmp3 = math.ceil(tmp2 / 9)  # 길이를 구한 이유는 페이지 넘기는 페이지 수를 구하기 위함
        th_slide.append(tmp3)
        th_tmp.append(tmp)

    return render_template('visualization.html', count=count_list, z_result=z_result, th_result=th_tmp,
                           fig_code=fig_code, th_num=th_num, th_slide=th_slide)


@app.route('/THDatatable.html', methods=['POST', 'GET'])
def THDatatable():  # past 시각화 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    if request.method == 'POST':  # post 요청시 즉 기간을 조회했을때
        result = request.form.to_dict()  # 기간 조회 폼을 통해 데이터를 받고
        query1 = result['start']
        query2 = (datetime.strptime(result['end'], '%Y-%m-%d') + relativedelta(days=1)).strftime('%Y-%m-%d')
        collection = db.relic  # 각각의 유물, 센서, 구역의 갯수를 가져옴
        number3 = collection.estimated_document_count()
        collection = db.sensor
        number1 = collection.estimated_document_count()
        collection = db.zone
        z_result = list(collection.find())
        number2 = collection.estimated_document_count()
        count_list = list((number1, number2, number3))
        collection = db.thdata
        min_time = list(collection.find().sort([("time", 1)]).limit(1))  # select 최대 최소값을 설정하기 위해 가져옴
        max_time = list(collection.find().sort([("time", -1)]).limit(1))
        min_time = min_time[0]["time"]
        max_time = max_time[0]["time"]

        th_tmp = list()
        th_num = list()
        th_slide = list()

        fig_code, dfs = visual(query1, query2)  # 평균 온습도 시각화 함수 실행
        if not fig_code:  # 온습도 데이터가 존재하지 않을시
            flash("입력하신 날짜에 데이터가 존재하지 않습니다.")
            return redirect(url_for("THDatatable"))

        for i in range(0, len(z_result)):  # 구역별로
            tmp = dfs[i].values.tolist()  # 몽고디비에서 가져온 온습도 데이터를 list로 변환
            tmp2 = len(dfs[i])  # 길이 구하기
            th_num.append(tmp2)
            tmp3 = math.ceil(tmp2 / 9)  # 길이를 구한 이유는 페이지 넘기는 페이지 수를 구하기 위함
            th_slide.append(tmp3)
            th_tmp.append(tmp)

        return render_template('THDatatable.html', count=count_list, z_result=z_result, th_result=th_tmp,
                               day_re=result, fig_code=fig_code, th_num=th_num, th_slide=th_slide, min_time=min_time,
                               max_time=max_time, st_date=query1, en_date=result['end'])
    else:  # 초기 실행시
        collection = db.relic  # 각각의 유물, 센서, 구역의 갯수를 가져옴
        number3 = collection.estimated_document_count()
        collection = db.sensor
        number1 = collection.estimated_document_count()
        collection = db.zone
        z_result = list(collection.find())
        number2 = collection.estimated_document_count()
        count_list = list((number1, number2, number3))
        collection = db.thdata
        min_time = list(collection.find().sort([("time", 1)]).limit(1))
        max_time = list(collection.find().sort([("time", -1)]).limit(1))
        min_time = min_time[0]["time"]
        max_time = max_time[0]["time"]
        now = time  # 5~7은 지금 현재시각을 가져오는 코드임 2022-08-22 20:48:57 이렇게
        now_time = now.strftime('%Y-%m-%d')
        query2 = now_time
        query1 = (datetime.now() - relativedelta(months=3)).strftime('%Y-%m-%d')
        th_tmp = list()
        th_num = list()
        th_slide = list()
        fig_code, dfs = visual(query1, query2)  # 평균 온습도 시각화 함수 실행
        if not fig_code:  # 온습도 데이터가 존재하지 않을시
            flash("업데이트가 되지 않았습니다.")
            return redirect(url_for("index"))

        for i in range(0, len(z_result)):  # 구역별로
            tmp = dfs[i].values.tolist()
            tmp2 = len(dfs[i])  # 길이 구하기
            th_num.append(tmp2)
            tmp3 = math.ceil(tmp2 / 9)  # 길이를 구한 이유는 페이지 넘기는 페이지 수를 구하기 위함
            th_slide.append(tmp3)
            th_tmp.append(tmp)

        return render_template('THDatatable.html', count=count_list, z_result=z_result, th_result=th_tmp,
                               fig_code=fig_code,
                               th_num=th_num, th_slide=th_slide, min_time=min_time, max_time=max_time, st_date=query1,
                               en_date=query2)


@app.route('/ML.html')
def ML():  # data 분석 페이지 함수
    weather_url = ""
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 회원에 맞게 날씨 정보를 가져옴
        if session['manager_place'] == "김천 직지사":
            weather_url = "https://forecast.io/embed/#lat=36.1213&lon=128.1186&name=김천&color=&font=&units=si"
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            weather_url = "https://forecast.io/embed/#lat=34.7617&lon=127.6628&name=여수&color=&font=&units=si"
            db = client.wc_project_흥국사
        else:
            weather_url = "https://forecast.io/embed/#lat=34.6362&lon=126.7702&name=강진&color=&font=&units=si"
            db = client.wc_project

    collection = db.relic  # 각각의 유물, 센서, 구역의 갯수를 가져옴
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    number2 = collection.estimated_document_count()
    z_result = list(collection.find())
    count_list = list((number1, number2, number3))

    fig_code = predict_visual()  # 예측 시각화 함수 실행

    return render_template('ML.html', count=count_list, fig_code=fig_code, weather_url=weather_url, z_result=z_result)


@app.route('/riskalarm.html')
def riskalarm():  # 위험 알림 페이지 함수
    weather_url = ""
    ex = list()
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 회원에 맞게 날씨 정보를 가져옴
        if session['manager_place'] == "김천 직지사":
            weather_url = "https://forecast.io/embed/#lat=36.1213&lon=128.1186&name=김천&color=&font=&units=si"
            ex.append("45530039")
            ex.append("51651616")
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            weather_url = "https://forecast.io/embed/#lat=34.7617&lon=127.6628&name=여수&color=&font=&units=si"
            ex.append("36828887")
            ex.append("45453790")
            db = client.wc_project_흥국사
        else:
            weather_url = "https://forecast.io/embed/#lat=34.6362&lon=126.7702&name=강진&color=&font=&units=si"
            ex.append("43946807")
            ex.append("50563655")
            db = client.wc_project

    collection = db.relic  # 각각의 유물, 센서, 구역의 갯수를 가져옴
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    number2 = collection.estimated_document_count()
    z_result = list(collection.find())
    count_list = list((number1, number2, number3))
    now = time
    now_time = now.strftime('%Y-%m-%d')  # 현재 시각 가져오기
    collection = db.timeline  # 오늘 발생한 위험 감지 데이터를 가져온다
    timeline_result = list(collection.find({"time": {"$regex": now_time}}).sort([("time", 1)]))
    if not timeline_result:  # 발생한 위험감지가 없을시
        timeline_result.append({'time': '오늘 현재까지', 'zone_name': '모든 구역 안전합니다.', 'risk_name': 'success'})

    return render_template('riskalarm.html', count=count_list, z_result=z_result, ex=ex, weather_url=weather_url,
                           timeline_result=timeline_result)


@app.route('/risktable.html', methods=['POST', 'GET'])
def risktable():  # 전체 위험 감지 데이터를 보여주는 페이지의 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    if request.method == 'POST':
        result = request.form.to_dict()  # 폼으로 원하는 카테고리 양식을 가져와서
        tmp = list((result['zone_name'], result['risk_name']))
        collection = db.zone
        z_result = list(collection.find())
        collection = db.timeline  # 원하는 구역 or 원하는 위험도만 조회할수 있도록 폼에서 받은 양식에 맞춰서 데이터 가져오기
        if result['zone_name'] == "A" and result['risk_name'] != "A":
            timeline_result = list(
                collection.find({'risk_name': result['risk_name']}).sort([("time", 1)]))
        elif result['zone_name'] != "A" and result['risk_name'] == "A":
            timeline_result = list(
                collection.find({'zone_name': result['zone_name']}).sort([("time", 1)]))
        elif result['zone_name'] == "A" and result['risk_name'] == "A":
            timeline_result = list(
                collection.find().sort([("time", 1)]))
        else:
            timeline_result = list(
                collection.find({'risk_name': result['risk_name'], 'zone_name': result['zone_name']}).sort(
                    [("time", 1)]))

        return render_template('risktable.html', timeline_result=timeline_result, z_result=z_result, tmp=tmp)
    else:  # 초기 페이지일때 전체 위험 감지 데이터를 가져온다
        collection = db.zone
        tmp = list(("A", "A"))
        z_result = list(collection.find())
        collection = db.timeline
        timeline_result = list(collection.find().sort([("time", 1)]))

        return render_template('risktable.html', timeline_result=timeline_result, z_result=z_result, tmp=tmp)


@app.route('/Sensor_table.html')
def Sensor_table():  # 센서 테이블 페이지 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    page = request.args.get('page')  # 페이지 몇번째인지 확인
    if page is None:  # 들어온게 없다면 1로 설정
        page = 1
    page = int(page)
    collection = db.relic  # 각각의 유물, 구역 ,센서 가져오는데 센서를 10개씩 가져오기
    r_count = collection.estimated_document_count()

    collection = db.zone
    z_count = collection.estimated_document_count()
    z_result = list(collection.find())
    collection = db.sensor
    s_count = collection.estimated_document_count()
    page_count = s_count / 10

    page_count = math.ceil(page_count) + 1  # 센서 열개씩 끊어서 가져온다
    s_result = list(collection.find().skip((page - 1) * 10).limit(10))
    count = list((r_count, z_count))

    return render_template('Sensor_table.html', s_result=s_result, page_list=page_count, count=count, z_result=z_result)


@app.route('/addSensorAct', methods=['POST', 'GET'])
def addSensorAct():  # 센서 추가하는 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    if request.method == 'POST':  # post요청시
        collection = db.sensor
        result = request.form.to_dict()  # 폼으로 받은 데이터를 받아옴

        if result['name'] == '':  # 이름을 빈칸으로 제출시
            flash("센서 이름을 입력하여 주세요.")
            return redirect(url_for("Sensor_table"))
        elif result['responsible'] == '2017':  # 책임자 번호를 확인
            collection.insert_one(result)  # 디비에 새로운 센서 정보 저장
            flash("정상적으로 저장되었습니다.")
            return redirect(url_for("Sensor_table"))
        else:
            flash("책임자 번호가 일치하지않습니다.")
            return redirect(url_for("Sensor_table"))


@app.route('/update_sensor.html', methods=['POST', 'GET'])
def update_sensor():  # 센서 업데이트 페이지 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    if request.method == 'GET':  # get 요청을 이용해서 id 값을 받아옴
        num = request.args.get('id')
        # print(type(relic_num))
        collection = db.sensor
        result = list(collection.find({"_id": num}))  # DB에서 일치하는 데이터를 가져옴
        if not result:
            result = list(collection.find({"_id": ObjectId(num)}))

        collection = db.zone
        z_result = list(collection.find())

        return render_template('update_sensor.html', result=result, z_result=z_result)


@app.route('/update_sensor_act', methods=['POST', 'GET'])
def update_sensor_act():  # 센서 업데이트 실행 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    if request.method == 'POST':  # post 요청시
        sensor_num = request.args.get('id')
        result = request.form.to_dict()

        if result['name'] == '':  # 이름이 없다면 실패
            flash("센서 이름을 입력하여 주세요.")
            return redirect(url_for("Sensor_table"))
        elif result['responsible'] == '2017':  # 책임자 번호 확인후에
            collection = db.sensor
            if len(sensor_num) == 22:  # 업데이트 실행
                collection.update_one({"_id": sensor_num},
                                      {"$set": {"name": result['name'], "deviceType": result['deviceType'],
                                                "zone_number": result['zone_number']}})
            else:
                collection.update_one({"_id": ObjectId(sensor_num)},
                                      {"$set": {"name": result['name'], "deviceType": result['deviceType'],
                                                "zone_number": result['zone_number']}})

            flash("수정 완료")
            return redirect(url_for("Sensor_table"))
        else:
            flash("책임자 번호가 일치하지않습니다.")
            return redirect(url_for("Sensor_table"))


@app.route('/Zone_table.html')
def Zone_table():  # 구역 테이블 페이지 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    page = request.args.get('page')  # 페이지를 가져옴
    if page is None:  # 없다면 1로 설정
        page = 1
    page = int(page)
    collection = db.relic  # 각각의 유물, 구역, 센서의 갯수를 가져오는데 구역은 10개씩 페이지화를 해서 가져온다
    r_count = collection.estimated_document_count()

    collection = db.zone
    z_count = collection.estimated_document_count()
    z_result = list(collection.find().skip((page - 1) * 10).limit(10))
    page_count = z_count / 10

    page_count = math.ceil(page_count) + 1
    count = list((r_count, z_count))
    # 이것은 구역을 추가할때 공간 번호를 가져오기위해서 고퀄api를 사용
    url = "https://goqual.io/openapi/homes/" + str(session['place_num']) + "/rooms"
    response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})
    room_list = response.json()

    return render_template('Zone_table.html', z_result=z_result, page_list=page_count, count=count,
                           room_list=room_list['rooms'])


@app.route('/addZoneAct', methods=['POST', 'GET'])
def addZoneAct():  # 구역 추가 실행 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    if request.method == 'POST':  # post 요청시
        collection = db.zone
        result = request.form.to_dict()  # 폼으로 받은 구역 정보를 가져옴
        if result['name'] == '':  # 이름이 없다면 다시
            flash("구역 이름을 입력하여 주세요.")
            return redirect(url_for("Zone_table"))
        elif result['responsible'] == '2017':  # 책임자 번호 일치시
            collection.insert_one(result)  # DB에 저장
            flash("정상적으로 저장되었습니다.")
            return redirect(url_for("Zone_table"))
        else:
            flash("책임자 번호가 일치하지않습니다.")
            return redirect(url_for("Zone_table"))


@app.route('/update_zone.html', methods=['POST', 'GET'])
def update_zone():  # 구역 업데이트 페이지 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    if request.method == 'GET':
        num = request.args.get('id')  # 해당 구역의 id 즉 기본키를 가져옴
        # print(type(relic_num))
        collection = db.zone  # 해당 데이터가 있는지 가져옴
        result = list(collection.find({"_id": ObjectId(num)}))

        return render_template('update_zone.html', result=result)


@app.route('/update_zone_act', methods=['POST', 'GET'])
def update_zone_act():  # 구역 업데이트 실행 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    if request.method == 'POST':
        num = request.args.get('id')  # 해당 구역의 id 즉 기본키를 가져옴
        result = request.form.to_dict()  # 폼으로 구역 데이터 수정할 정보를 가져옴

        if result['name'] == '':  # 이름을 적지 않았다면
            flash("구역 이름을 입력하여 주세요.")
            return redirect(url_for("Zone_table"))
        elif result['responsible'] == '2017':  # 책임자 번호 일치시
            collection = db.zone  # 구역 정보 업데이트 실행
            collection.update_one({"_id": ObjectId(num)},
                                  {"$set": {"name": result['name'], "familyId": result['familyId']
                                            }})

            flash("수정 완료")
            return redirect(url_for("Zone_table"))
        else:
            flash("책임자 번호가 일치하지않습니다.")
            return redirect(url_for("Zone_table"))


@app.route('/Relic_table.html')
def Relic_table():  # 유물 테이블 페이지 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    page = request.args.get('page')  # 페이지 수를 가져옴
    if page is None:  # 없다면 첫번째 페이지로 설정
        page = 1
    page = int(page)
    collection = db.relic  # 각각의 유물, 구역, 센서의 갯수를 가져오는데 유물의 갯수는 페이지별로 10개씩 끊어서 가져옴
    number = collection.estimated_document_count()
    r_count = number
    r_result = list(collection.find().skip((page - 1) * 10).limit(10))
    page_count = number / 10

    page_count = math.ceil(page_count) + 1
    collection = db.zone
    z_count = collection.estimated_document_count()
    z_result = list(collection.find())
    count = list((r_count, z_count))
    return render_template('Relic_table.html', r_result=r_result, z_result=z_result, page_list=page_count,
                           count=count, )


@app.route('/addRelicAct', methods=['POST', 'GET'])
def addRelicAct():  # 유물 추가 실행 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    if request.method == 'POST':
        collection = db.relic
        result = request.form.to_dict()  # 폼으로 받은 유물 정보를 가져옴
        if result['relic_name'] == '':  # 이름을 입력하지 않았을때
            flash("유물 이름을 입력하여 주세요.")
            return redirect(url_for("Relic_table"))

        collection.insert_one(result)  # 정상적으로 DB에 저장

        flash("정상적으로 저장되었습니다.")
        return redirect(url_for("Relic_table"))


@app.route('/update_relic.html', methods=['POST', 'GET'])
def update_relic():  # 유물 수정 페이지 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    if request.method == 'GET':
        relic_num = request.args.get('id')  # 해당 유물의 id를 가져옴
        # print(type(relic_num))
        collection = db.relic  # 일치하는 유물 데이터를 가져옴
        result = list(collection.find({"_id": ObjectId(relic_num)}))
        # print(result)
        collection = db.zone  # 유물이 어디 구역에 있는지 입력 받기위해 구역 데이터 가져옴
        z_result = list(collection.find())
        return render_template('update_relic.html', result=result, z_result=z_result)


@app.route('/update_relic_act', methods=['POST', 'GET'])
def update_relic_act():  # 유물 업데이트 실행 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    if request.method == 'POST':
        result = request.form.to_dict()  # 폼으로 업데이트할 유물 정보 가져옴

        if result['relic_name'] == '':  # 이름을 입력 안했을때
            flash("유물 이름을 입력하여 주세요.")
            return redirect(url_for("Relic_table"))
        relic_num = request.args.get('id')  # 기본키를 get요청으로 가져온다
        collection = db.relic  # DB에 업데이트 실행
        collection.update_one({"_id": ObjectId(relic_num)},
                              {"$set": {"relic_name": result['relic_name'], "relic_type": result['relic_type'],
                                        "zone_name": result['zone_name']}})

        flash("수정완료되었습니다.")
        return redirect(url_for("Relic_table"))


@app.route('/data_del.html', methods=['POST', 'GET'])
def data_del():  # 데이터 삭제하는 함수
    global db
    if 'manager_place' in session:  # 현재 로그인 되어있는 관리자에 맞게 유동적으로 위치 변경
        if session['manager_place'] == "김천 직지사":
            db = client.wc_project_직지사
        elif session['manager_place'] == "여수 흥국사":
            db = client.wc_project_흥국사
        else:
            db = client.wc_project
    if request.method == 'GET':  # get요청시 유물을 삭제하는 요청임
        result = list()
        result.append(request.args.get('id'))  # 기본키를 가져오고
        result.append(request.args.get('data'))
        if result[1] == 'R':
            collection = db.relic  # 유물데이터를 삭제
            collection.delete_one({'_id': ObjectId(result[0])})
            flash("해당 유물 정보가 정상적으로 삭제되었습니다.")
            return redirect(url_for("Relic_table"))
        return render_template('data_del.html', result=result)
    if request.method == 'POST':  # post요청은 센서아니면 구역이다.
        result = request.form.to_dict()  # 폼으로 삭제할 데이터 정보를 가져오고
        num = request.args.get('id')  # get으로 기본키와 어떤 데이터인지 가져옴
        data = request.args.get('data')

        if result['responsible'] == '2017':  # 책임자 번호 일치시
            if data == "S":  # 센서 삭제 요청시
                collection = db.sensor
                if len(num) == 22:  # 삭제
                    collection.delete_one({"_id": num})
                else:
                    collection.delete_one({"_id": ObjectId(num)})

                flash("정상적으로 삭제되었습니다.")
                return redirect(url_for("Sensor_table"))
            else:  # 구역 삭제 요청시
                collection = db.zone  # 삭제
                collection.delete_one({'_id': ObjectId(num)})
                flash("정상적으로 삭제되었습니다.")
                return redirect(url_for("Zone_table"))

        else:  # 책임자 번호 불일치시
            flash("책임자 번호가 일치하지않습니다.")
            if data == "S":
                return redirect(url_for("Sensor_table"))
            else:
                return redirect(url_for("Zone_table"))


@app.route('/dloding.html')
def dloding():  # 시각화 로딩 페이지함수
    return render_template('dloding.html')


@app.route('/ploding.html')
def ploding():  # 예측 로딩 페이지 함수
    return render_template('ploding.html')


# 파이썬 명령어로 실행할 수 있음
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
    # app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)

