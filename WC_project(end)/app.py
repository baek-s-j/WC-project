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

app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')
app.secret_key = '2017'
global db
global d_where

client = MongoClient('mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')
db = client.wc_project
ac = 1
schedule_1 = BackgroundScheduler()
schedule_2 = BackgroundScheduler()


@schedule_1.scheduled_job('cron', hour='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23',
                          minute='0,10,20,30,40,50', id='thdata_job')
def thdata_job():
    exec(open("Thdataapi.py", encoding='utf-8').read())
    global db

    if d_where == "김천 직지사":
        db = client.wc_project_직지사
    elif d_where == "여수 흥국사":
        db = client.wc_project_흥국사
    else:
        db = client.wc_project


@schedule_2.scheduled_job('cron', hour='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23',
                          minute='0,30', second='5', id='risknotice_job')
def risknotice_job():
    global db
    exec(open("alarm_timeline.py", encoding='utf-8').read())
    if d_where == "김천 직지사":
        db = client.wc_project_직지사
    elif d_where == "여수 흥국사":
        db = client.wc_project_흥국사
    else:
        db = client.wc_project


schedule_1.start()
schedule_2.start()


def realtime_visual(query):
    collection = db.zone
    z_result = list(collection.find())
    f_json = list()
    dfs = list()

    for i in z_result:
        collection = db.thdata
        th_result = list(collection.find({"zone_name": i['name'], "time": {"$regex": query}}).sort([("time", 1)]))
        if not th_result:
            return False, False
        th_frame = pd.DataFrame(th_result)

        # 판다 데이터프레임에서 칼럼만 추출
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

        f_json.append(fig.to_json())
        dfs.append(th_frame)

    return f_json, dfs


def predict_visual():
    collection = db.zone
    z_result = list(collection.find({}))
    f_json = list()

    if z_result[0]['name'][-3:] == "무위사":
        x_error = [1.5, 1.5, 1.5, 1.7, 2.5]
        y_error = [4.4, 6, 6, 4, 6]
        nx = '56'
        ny = '64'
    elif z_result[0]['name'][-3:] == "직지사":
        x_error = [1.7, 1.6, 1.5, 2.4, 2.1]
        y_error = [6, 5, 5.1, 7, 7]
        nx = '79'
        ny = '96'
    else:
        x_error = [1.7, 1.6, 1.5, 1.7, 1.6, 1.6]
        y_error = [6, 5.5, 5, 5.7, 5.4, 5.4]
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
    for i in range(0, 24):
        if i < 10:
            iname = '0' + str(i)

            str_expr = "fcstTime.str.contains('{}00')".format(iname)  # 문자열에 '' 포함

        elif i >= 10:

            str_expr = "fcstTime.str.contains('{}00')".format(i)  # 문자열에 '' 포함

        tdf = df.query(str_expr)
        tdf.drop(['baseDate'], axis=1, inplace=True)
        tdf.drop(['baseTime'], axis=1, inplace=True)
        tdf.drop(['fcstDate'], axis=1, inplace=True)
        tdf.drop(['fcstTime'], axis=1, inplace=True)
        tdf.drop(['nx'], axis=1, inplace=True)
        tdf.drop(['ny'], axis=1, inplace=True)

        tdf = tdf.transpose()

        tdf = tdf.rename(columns=tdf.iloc[0])

        tdf.rename(columns={'TMP': 'ta'}, inplace=True)
        tdf.rename(columns={'PCP': 'rn'}, inplace=True)
        tdf.rename(columns={'WSD': 'ws'}, inplace=True)
        tdf.rename(columns={'VEC': 'wd'}, inplace=True)
        tdf.rename(columns={'REH': 'hm'}, inplace=True)
        tdf.drop(['category'], axis=0, inplace=True)
        tdf = tdf.reset_index()
        tdf.drop(['index'], axis=1, inplace=True)

        tdf = tdf[['ta', 'wd', 'ws', 'rn', 'hm']]

        tdf.loc[tdf['rn'] == "강수없음", 'rn'] = 0

        tm = str(int(base_date) + 1) + " " + "{}:00".format(i)

        r1 = tm[0:4]
        r2 = tm[4:6]
        r3 = tm[6:8]
        time12 = str(r1 + '-' + r2 + '-' + r3) + " " + "{}:00".format(i)

        tdf.insert(0, 'tm', time12)

        result_df = result_df.append(tdf)

    result_df.tm = pd.to_datetime(result_df.tm, format='%Y-%m-%d')
    result_df = result_df.set_index('tm')
    result_df = result_df.apply(pd.to_numeric)
    random_x = result_df.index.tolist()
    print(random_x)
    total_vars = ["ta", "rn", "ws", "wd", "hm"]
    input_vars = total_vars

    for i in z_result:

        fig = go.Figure()

        for j in range(1, 3):

            if j == 1:
                output_var = "temperature"
                name = "온도(℃)"
                color = 'firebrick'
                fill = 'rgba(255,0,0,0.2)'
            else:
                output_var = "humidity"
                name = "습도(%)"
                color = 'royalblue'
                fill = 'rgba(0,0,255,0.2)'

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
            print(random_x)
            random_y0 = y_pred_inv

            # 예측값 타입 알아봐야함
            if j == 1:
                fig.add_trace(go.Scatter(
                    x=random_x + random_x[::-1],
                    y=([random_y0[0][i] + x_error[num_index] for i in range(len(random_y0[0]))]) + ([
                        random_y0[0][i] - x_error[num_index] for i in
                        range(len(random_y0[0]))])[
                                                                                                   ::-1],
                    fill='toself',
                    fillcolor=fill,
                    mode='lines',
                    line=dict(line_color='rgba(255,255,255,0)'),
                    name=name[:2] + " 오차범위",

                ))
            else:
                fig.add_trace(go.Scatter(
                    x=random_x + random_x[::-1],
                    y=([random_y0[0][i] + y_error[num_index] for i in range(len(random_y0[0]))]) + ([
                        random_y0[0][i] - y_error[num_index] for i in
                        range(len(random_y0[0]))])[
                                                                                                   ::-1],
                    fill='toself',
                    fillcolor=fill,
                    mode='lines',
                    line=dict(line_color='rgba(255,255,255,0)'),
                    name=name[:2] + " 오차범위",

                ))

            fig.add_trace(go.Scatter(x=random_x, y=y_pred_inv[0],
                                     mode='lines+markers', line=dict(color=color, width=1.0),
                                     name=name))

        fig.add_hrect(y0=10, y1=34, line_width=0, fillcolor="green", opacity=0.1, annotation_text="안전 온도 범위",
                      annotation_position="top left", )
        fig.add_hrect(y0=75, y1=100, line_width=0, fillcolor="red", opacity=0.1, annotation_text="습도 위험 범위",
                      annotation_position="top left", )

        fig.update_layout(paper_bgcolor="#EAEAEA", margin=dict(l=10, r=10, t=60, b=10), )  # 차트 바깥 색
        fig.update_layout(plot_bgcolor="#F6F6F6")  # 그래프 안쪽색
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_layout(xaxis_tickformat='%Y<br>%m-%d<br>%H시%M분')
        fig.update_xaxes(linecolor='gray', gridcolor='gray', mirror=True, zeroline=False)
        fig.update_yaxes(linecolor='gray', gridcolor='gray', mirror=True)
        fig.update_yaxes(title_text="습도(%)    온도(℃)", zeroline=False, range=[-10, 100])

        f_json.append(fig.to_json())

    return f_json


def visual(query1, query2):  # 평균 온습도 데이터 시각화 함수

    collection = db.zone
    z_result = list(collection.find())
    f_json = list()
    dfs = list()

    for i in z_result:
        collection = db.thdata
        th_result = list(
            collection.find({"zone_name": i['name'], "time": {"$gte": query1, "$lte": query2}}).sort([("time", 1)]))
        if not th_result:
            return False, False
        th_frame = pd.DataFrame(th_result)
        th_frame.time = pd.to_datetime(th_frame.time)
        th_frame = th_frame.set_index('time')
        min_th_frame = th_frame.resample(rule='H').mean()
        min_th_frame['utime'] = min_th_frame.index
        min_th_frame['utime'] = min_th_frame['utime'].astype('str')
        min_th_frame['utime'] = min_th_frame['utime'].str.slice(start=0, stop=16)
        min_th_frame = min_th_frame.fillna(method='pad')
        min_th_frame = min_th_frame.round(1)

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

        fig.add_hrect(y0=10, y1=34, line_width=0, fillcolor="green", opacity=0.1, annotation_text="안전 온도 범위",
                      annotation_position="top left", )
        fig.add_hrect(y0=75, y1=100, line_width=0, fillcolor="red", opacity=0.1, annotation_text="습도 위험 범위",
                      annotation_position="top left", )

        # 배경 레이어색 파트
        fig.update_layout(paper_bgcolor="#EAEAEA")  # 차트 바깥 색
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
        dfs.append(min_th_frame)

    return f_json, dfs


def three_day_visual(query1, query2):  # 평균 온습도 데이터 시각화 함수
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

    # 판다 데이터프레임에서 칼럼만 추출
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
def addManager():
    url = "https://goqual.io/openapi/homes"

    response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})

    homes_list = response.json()

    return render_template('addManager.html', homes_list=homes_list['result'])


@app.route('/addManagerAct', methods=['POST', 'GET'])
def addManagerAct():
    if request.method == 'POST':
        global db
        db = client.wc_project
        collection = db.manager
        result = request.form.to_dict()

        id_check = list(collection.find({'manager_id': result.get('manager_id')}))

        if not id_check:

            url = "https://goqual.io/openapi/homes"
            response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})
            homes_list = response.json()
            for i in homes_list['result']:
                if i['name'] == result['manager_place']:
                    result['place_num'] = i['homeId']

            collection.insert_one(result)

            flash("회원가입이 완료되었습니다.")
            return redirect(url_for("login"))
        else:
            flash("아이디가 중복되었습니다.")
            return redirect(url_for("addManager"))


@app.route('/')
@app.route('/login.html')
def login():
    return render_template('login.html')


@app.route('/loginAct', methods=['POST', 'GET'])
def loginAct():
    if request.method == 'POST':
        global db
        global d_where
        db = client.wc_project
        collection = db.manager
        result = request.form.to_dict()
        if result['manager_id'] == "":
            flash("아이디를 입력해주세요")
            return redirect(url_for("login"))

        if result['manager_password'] == "":
            flash("비밀번호를 입력해주세요.")

            return redirect(url_for("login"))

        info = list(collection.find({'manager_id': result.get('manager_id')}))
        if not info:
            flash("아이디가 존재하지 않습니다.")

            return redirect(url_for("login"))

        if result.get('manager_password') == info[0]['manager_password']:
            session['id'] = result.get('manager_id')
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

        else:
            flash("아이디 또는 비밀번호가 맞지 않습니다.")
            return redirect(url_for("login"))


@app.route('/logout')
def logout():
    session.pop('id', None)
    session.pop('place_num', None)
    session.pop('manager_place', None)
    global db
    db = client.wc_project
    return redirect(url_for("login"))


@app.route('/index.html')
def index():
    global d_where

    weather_url = ""
    if 'manager_place' in session:
        d_where = session['manager_place']
        if session['manager_place'] == "김천 직지사":
            weather_url = "https://forecast.io/embed/#lat=36.1213&lon=128.1186&name=김천&color=&font=&units=si"

        elif session['manager_place'] == "여수 흥국사":
            weather_url = "https://forecast.io/embed/#lat=34.7617&lon=127.6628&name=여수&color=&font=&units=si"
        else:
            weather_url = "https://forecast.io/embed/#lat=34.6362&lon=126.7702&name=강진&color=&font=&units=si"

    collection = db.relic
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    number2 = collection.estimated_document_count()
    collection = db.manager
    number4 = collection.estimated_document_count()
    count_list = list((number1, number2, number3, number4))
    now = time
    now_time = now.strftime('%Y-%m-%d')
    query2 = now_time
    query1 = (datetime.now() - relativedelta(days=3)).strftime('%Y-%m-%d')

    fig_code = three_day_visual(query1, query2)

    collection = db.timeline
    timeline_result = list(collection.find({"time": {"$regex": query2}}).sort([("time", 1)]))
    if not timeline_result:
        timeline_result.append({'time': '오늘 현재까지', 'zone_name': '모든 구역 안전합니다.', 'risk_name': 'success'})

    return render_template('index.html', count=count_list, fig_code=fig_code, timeline_result=timeline_result,
                           weather_url=weather_url)


@app.route('/about.html')
def about():
    collection = db.relic
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    number2 = collection.estimated_document_count()
    collection = db.manager
    number4 = collection.estimated_document_count()
    count_list = list((number1, number2, number3, number4))

    return render_template('about.html', count=count_list)


@app.route('/visualization.html')
def visualization():
    collection = db.relic
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    z_result = list(collection.find())
    number2 = collection.estimated_document_count()
    collection = db.manager
    number4 = collection.estimated_document_count()
    count_list = list((number1, number2, number3, number4))

    now = time  # 5~7은 지금 현재시각을 가져오는 코드임 2022-08-22 20:48:57 이렇게
    query = now.strftime('%Y-%m-%d')
    th_tmp = list()
    th_num = list()
    th_slide = list()

    fig_code, dfs = realtime_visual(query)
    if not fig_code:
        flash("업데이트가 되지 않았습니다.")
        return redirect(url_for("index"))

    for i in range(0, len(z_result)):
        tmp = dfs[i].values.tolist()
        tmp2 = len(dfs[i])
        th_num.append(tmp2)
        tmp3 = math.ceil(tmp2 / 9)
        th_slide.append(tmp3)
        th_tmp.append(tmp)

    return render_template('visualization.html', count=count_list, z_result=z_result, th_result=th_tmp,
                           fig_code=fig_code, th_num=th_num, th_slide=th_slide)


@app.route('/THDatatable.html', methods=['POST', 'GET'])
def THDatatable():
    if request.method == 'POST':
        result = request.form.to_dict()
        query1 = result['start']
        query2 = (datetime.strptime(result['end'], '%Y-%m-%d') + relativedelta(days=1)).strftime('%Y-%m-%d')
        collection = db.relic
        number3 = collection.estimated_document_count()
        collection = db.sensor
        number1 = collection.estimated_document_count()
        collection = db.zone
        z_result = list(collection.find())
        number2 = collection.estimated_document_count()
        collection = db.manager
        number4 = collection.estimated_document_count()
        count_list = list((number1, number2, number3, number4))
        collection = db.thdata
        min_time = list(collection.find().sort([("time", 1)]).limit(1))
        max_time = list(collection.find().sort([("time", -1)]).limit(1))
        min_time = min_time[0]["time"]
        max_time = max_time[0]["time"]

        th_tmp = list()
        th_num = list()
        th_slide = list()

        fig_code, dfs = visual(query1, query2)
        if not fig_code:
            flash("입력하신 날짜에 데이터가 존재하지 않습니다.")
            return redirect(url_for("THDatatable"))

        for i in range(0, len(z_result)):
            tmp = dfs[i].values.tolist()
            tmp2 = len(dfs[i])
            th_num.append(tmp2)
            tmp3 = math.ceil(tmp2 / 9)
            th_slide.append(tmp3)
            th_tmp.append(tmp)

        return render_template('THDatatable.html', count=count_list, z_result=z_result, th_result=th_tmp,
                               day_re=result, fig_code=fig_code, th_num=th_num, th_slide=th_slide, min_time=min_time,
                               max_time=max_time, st_date=query1, en_date=result['end'])
    else:
        collection = db.relic
        number3 = collection.estimated_document_count()
        collection = db.sensor
        number1 = collection.estimated_document_count()
        collection = db.zone
        z_result = list(collection.find())
        number2 = collection.estimated_document_count()
        collection = db.manager
        number4 = collection.estimated_document_count()
        count_list = list((number1, number2, number3, number4))
        collection = db.thdata
        min_time = list(collection.find().sort([("time", 1)]).limit(1))
        max_time = list(collection.find().sort([("time", -1)]).limit(1))
        min_time = min_time[0]["time"]
        max_time = max_time[0]["time"]
        now = time  # 5~7은 지금 현재시각을 가져오는 코드임 2022-08-22 20:48:57 이렇게
        now_time = now.strftime('%Y-%m-%d')
        query2 = now_time
        query1 = (datetime.now() - relativedelta(months=4)).strftime('%Y-%m-%d')
        th_tmp = list()
        th_num = list()
        th_slide = list()
        fig_code, dfs = visual(query1, query2)
        if not fig_code:
            flash("업데이트가 되지 않았습니다.")
            return redirect(url_for("index"))

        for i in range(0, len(z_result)):
            tmp = dfs[i].values.tolist()
            tmp2 = len(dfs[i])
            th_num.append(tmp2)
            tmp3 = math.ceil(tmp2 / 9)
            th_slide.append(tmp3)
            th_tmp.append(tmp)

        return render_template('THDatatable.html', count=count_list, z_result=z_result, th_result=th_tmp,
                               fig_code=fig_code,
                               th_num=th_num, th_slide=th_slide, min_time=min_time, max_time=max_time, st_date=query1,
                               en_date=query2)


@app.route('/ML.html')
def ML():
    weather_url = ""
    if 'manager_place' in session:
        if session['manager_place'] == "김천 직지사":
            weather_url = "https://forecast.io/embed/#lat=36.1213&lon=128.1186&name=김천&color=&font=&units=si"
        elif session['manager_place'] == "여수 흥국사":
            weather_url = "https://forecast.io/embed/#lat=34.7617&lon=127.6628&name=여수&color=&font=&units=si"
        else:
            weather_url = "https://forecast.io/embed/#lat=34.6362&lon=126.7702&name=강진&color=&font=&units=si"

    collection = db.relic
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    number2 = collection.estimated_document_count()
    z_result = list(collection.find())
    collection = db.manager
    number4 = collection.estimated_document_count()
    count_list = list((number1, number2, number3, number4))

    fig_code = predict_visual()

    collection = db.timeline
    timeline_result = list(collection.find())

    return render_template('ML.html', count=count_list, fig_code=fig_code, timeline_result=timeline_result,
                           weather_url=weather_url, z_result=z_result)


@app.route('/riskalarm.html')
def riskalarm():
    weather_url = ""
    ex = list()
    if 'manager_place' in session:
        if session['manager_place'] == "김천 직지사":
            weather_url = "https://forecast.io/embed/#lat=36.1213&lon=128.1186&name=김천&color=&font=&units=si"
            ex.append("45530039")
            ex.append("51651616")
        elif session['manager_place'] == "여수 흥국사":
            weather_url = "https://forecast.io/embed/#lat=34.7617&lon=127.6628&name=여수&color=&font=&units=si"
            ex.append("36828887")
            ex.append("45453790")

        else:
            weather_url = "https://forecast.io/embed/#lat=34.6362&lon=126.7702&name=강진&color=&font=&units=si"
            ex.append("43946807")
            ex.append("50563655")

    collection = db.relic
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    number2 = collection.estimated_document_count()
    z_result = list(collection.find())
    collection = db.manager
    number4 = collection.estimated_document_count()
    count_list = list((number1, number2, number3, number4))
    now = time
    now_time = now.strftime('%Y-%m-%d')
    collection = db.timeline
    timeline_result = list(collection.find({"time": {"$regex": now_time}}).sort([("time", 1)]))
    if not timeline_result:
        timeline_result.append({'time': '오늘 현재까지', 'zone_name': '모든 구역 안전합니다.', 'risk_name': 'success'})

    return render_template('riskalarm.html', count=count_list, z_result=z_result, ex=ex, weather_url=weather_url,
                           timeline_result=timeline_result)


@app.route('/risktable.html', methods=['POST', 'GET'])
def risktable():
    if request.method == 'POST':
        result = request.form.to_dict()
        tmp = list((result['zone_name'], result['risk_name']))
        collection = db.zone
        z_result = list(collection.find())
        collection = db.timeline
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
    else:
        collection = db.zone
        tmp = list(("A", "A"))
        z_result = list(collection.find())
        collection = db.timeline
        timeline_result = list(collection.find().sort([("time", 1)]))

        return render_template('risktable.html', timeline_result=timeline_result, z_result=z_result, tmp=tmp)


@app.route('/Sensor_table.html')
def Sensor_table():
    page = request.args.get('page')
    if page is None:
        page = 1
    page = int(page)
    collection = db.relic
    r_count = collection.estimated_document_count()

    collection = db.zone
    z_count = collection.estimated_document_count()
    z_result = list(collection.find())
    collection = db.sensor
    s_count = collection.estimated_document_count()
    page_count = s_count / 10

    page_count = math.ceil(page_count) + 1
    s_result = list(collection.find().skip((page - 1) * 10).limit(10))
    count = list((r_count, z_count))

    return render_template('Sensor_table.html', s_result=s_result, page_list=page_count, count=count, z_result=z_result)


@app.route('/addSensorAct', methods=['POST', 'GET'])
def addSensorAct():
    if request.method == 'POST':
        collection = db.sensor
        result = request.form.to_dict()

        if result['name'] == '':
            flash("센서 이름을 입력하여 주세요.")
            return redirect(url_for("Sensor_table"))
        elif result['responsible'] == '2017':
            collection.insert_one(result)
            flash("정상적으로 저장되었습니다.")
            return redirect(url_for("Sensor_table"))
        else:
            flash("책임자 번호가 일치하지않습니다.")
            return redirect(url_for("Sensor_table"))


@app.route('/update_sensor.html', methods=['POST', 'GET'])
def update_sensor():
    if request.method == 'GET':
        num = request.args.get('id')
        # print(type(relic_num))
        collection = db.sensor
        result = list(collection.find({"_id": num}))
        if not result:
            result = list(collection.find({"_id": ObjectId(num)}))

        collection = db.zone
        z_result = list(collection.find())

        return render_template('update_sensor.html', result=result, z_result=z_result)


@app.route('/update_sensor_act', methods=['POST', 'GET'])
def update_sensor_act():
    if request.method == 'POST':
        sensor_num = request.args.get('id')
        result = request.form.to_dict()

        if result['name'] == '':
            flash("센서 이름을 입력하여 주세요.")
            return redirect(url_for("Sensor_table"))
        elif result['responsible'] == '2017':
            collection = db.sensor
            if len(sensor_num) == 22:
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
def Zone_table():
    page = request.args.get('page')
    if page is None:
        page = 1
    page = int(page)
    collection = db.relic
    r_count = collection.estimated_document_count()

    collection = db.zone
    z_count = collection.estimated_document_count()
    z_result = list(collection.find().skip((page - 1) * 10).limit(10))
    page_count = z_count / 10

    page_count = math.ceil(page_count) + 1
    count = list((r_count, z_count))

    url = "https://goqual.io/openapi/homes/" + str(session['place_num']) + "/rooms"
    response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})
    room_list = response.json()

    return render_template('Zone_table.html', z_result=z_result, page_list=page_count, count=count,
                           room_list=room_list['rooms'])


@app.route('/addZoneAct', methods=['POST', 'GET'])
def addZoneAct():
    if request.method == 'POST':
        collection = db.zone
        result = request.form.to_dict()
        if result['zone_name'] == '':
            flash("구역 이름을 입력하여 주세요.")
            return redirect(url_for("Zone_table"))
        elif result['responsible'] == '2017':
            collection.insert_one(result)
            flash("정상적으로 저장되었습니다.")
            return redirect(url_for("Zone_table"))
        else:
            flash("책임자 번호가 일치하지않습니다.")
            return redirect(url_for("Zone_table"))


@app.route('/update_zone.html', methods=['POST', 'GET'])
def update_zone():
    if request.method == 'GET':
        num = request.args.get('id')
        # print(type(relic_num))
        collection = db.zone
        result = list(collection.find({"_id": ObjectId(num)}))

        return render_template('update_zone.html', result=result)


@app.route('/update_zone_act', methods=['POST', 'GET'])
def update_zone_act():
    if request.method == 'POST':
        num = request.args.get('id')
        result = request.form.to_dict()

        if result['name'] == '':
            flash("구역 이름을 입력하여 주세요.")
            return redirect(url_for("Zone_table"))
        elif result['responsible'] == '2017':
            collection = db.zone
            collection.update_one({"_id": ObjectId(num)},
                                  {"$set": {"name": result['name'], "familyId": result['familyId']
                                            }})

            flash("수정 완료")
            return redirect(url_for("Zone_table"))
        else:
            flash("책임자 번호가 일치하지않습니다.")
            return redirect(url_for("Zone_table"))


@app.route('/Relic_table.html')
def Relic_table():
    page = request.args.get('page')
    if page is None:
        page = 1
    page = int(page)
    collection = db.relic
    number = collection.estimated_document_count()
    r_count = number
    r_result = list(collection.find().skip((page - 1) * 10).limit(10))
    page_count = number / 10

    page_count = math.ceil(page_count) + 1
    collection = db.zone
    z_count = collection.estimated_document_count()
    z_result = list(collection.find())
    count = list((r_count, z_count))
    collection = db.riskdata
    risk_result = list(collection.find())

    return render_template('Relic_table.html', r_result=r_result, z_result=z_result, page_list=page_count, count=count,
                           risk_result=risk_result)


@app.route('/addRelicAct', methods=['POST', 'GET'])
def addRelicAct():
    if request.method == 'POST':

        collection = db.relic
        result = request.form.to_dict()
        if result['relic_name'] == '':
            flash("유물 이름을 입력하여 주세요.")
            return redirect(url_for("Relic_table"))

        collection.insert_one(result)

        flash("정상적으로 저장되었습니다.")
        return redirect(url_for("Relic_table"))


@app.route('/update_relic.html', methods=['POST', 'GET'])
def update_relic():
    if request.method == 'GET':
        relic_num = request.args.get('id')
        # print(type(relic_num))
        collection = db.relic
        result = list(collection.find({"_id": ObjectId(relic_num)}))
        # print(result)
        collection = db.zone
        z_result = list(collection.find())
        collection = db.riskdata
        risk_result = list(collection.find())
        return render_template('update_relic.html', result=result, z_result=z_result, risk_result=risk_result)


@app.route('/update_relic_act', methods=['POST', 'GET'])
def update_relic_act():
    if request.method == 'POST':
        result = request.form.to_dict()

        if result['relic_name'] == '':
            flash("유물 이름을 입력하여 주세요.")
            return redirect(url_for("Relic_table"))
        relic_num = request.args.get('id')
        collection = db.relic
        collection.update_one({"_id": ObjectId(relic_num)},
                              {"$set": {"relic_name": result['relic_name'], "relic_type": result['relic_type'],
                                        "zone_name": result['zone_name']}})

        flash("수정완료되었습니다.")
        return redirect(url_for("Relic_table"))


@app.route('/data_del.html', methods=['POST', 'GET'])
def data_del():
    if request.method == 'GET':
        result = list()
        result.append(request.args.get('id'))
        result.append(request.args.get('data'))
        if result[1] == 'R':
            collection = db.relic
            collection.delete_one({'_id': ObjectId(result[0])})
            return redirect(url_for("Relic_table"))

        return render_template('data_del.html', result=result)
    if request.method == 'POST':
        result = request.form.to_dict()
        num = request.args.get('id')
        data = request.args.get('data')

        if result['responsible'] == '2017':
            if data == "S":
                collection = db.sensor
                if len(num) == 22:
                    collection.delete_one({"_id": num})
                else:
                    collection.delete_one({"_id": ObjectId(num)})

                flash("정상적으로 삭제되었습니다.")
                return redirect(url_for("Sensor_table"))
            else:
                collection = db.zone
                collection.delete_one({'_id': ObjectId(num)})
                flash("정상적으로 삭제되었습니다.")
                return redirect(url_for("Zone_table"))

        else:
            flash("책임자 번호가 일치하지않습니다.")
            return redirect(url_for("Zone_table"))


@app.route('/dloding.html')
def dloding():
    return render_template('dloding.html')


@app.route('/ploding.html')
def ploding():
    return render_template('ploding.html')


# 파이썬 명령어로 실행할 수 있음
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
    # app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
