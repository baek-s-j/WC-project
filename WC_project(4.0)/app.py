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

app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')
app.secret_key = '2017'

client = MongoClient('mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')
db = client.wc_project
ac = 1
schedule_1 = BackgroundScheduler()
schedule_2 = BackgroundScheduler()


@schedule_1.scheduled_job('cron', hour='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23',
                          minute='0,10,20,30,40,50', id='thdata_job')
def thdata_job():
    exec(open("Thdataapi.py", encoding='utf-8').read())

@schedule_2.scheduled_job('cron', hour='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23',
                          minute='0,30', id='risknotice_job')
def risknotice_job():
    exec(open("alarm_timeline.py", encoding='utf-8').read())


schedule_1.start()


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
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=random_x, y=random_y0,
                                 mode='lines', line=dict(color='firebrick', width=1.0),
                                 name='온도(℃)'), secondary_y=False)
        fig.add_trace(go.Scatter(x=random_x, y=random_y1,
                                 mode='lines', line=dict(color='royalblue', width=1.0),
                                 name='습도(%)'), secondary_y=True)
        fig.add_hrect(y0=20, y1=30, line_width=0, fillcolor="red", opacity=0.1, annotation_text="안전 온도",
                      annotation_position="top left", )
        fig.add_hrect(y0=70, y1=80, line_width=0, fillcolor="blue", opacity=0.1, annotation_text="안전 습도",
                      annotation_position="top left", )

        # 배경 레이어색 파트
        fig.update_layout(paper_bgcolor="#EAEAEA")  # 차트 바깥 색
        fig.update_layout(plot_bgcolor="#F6F6F6")  # 그래프 안쪽색
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_layout(xaxis_tickformat='%Y<br>%m-%d<br>%H시%M분')
        fig.update_xaxes(linecolor='gray', gridcolor='gray', mirror=True, zeroline=False)
        fig.update_yaxes(linecolor='gray', gridcolor='gray', mirror=True)
        fig.update_yaxes(title_text="온도(℃)", secondary_y=False, zeroline=False, range=[-10, 100])
        fig.update_yaxes(title_text="습도(%)", secondary_y=True, zeroline=False, range=[-10, 100])

        f_json.append(fig.to_json())
        dfs.append(th_frame)

    return f_json, dfs


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
        min_th_frame = min_th_frame.fillna(method='pad')
        min_th_frame = min_th_frame.round(1)

        # 판다 데이터프레임에서 칼럼만 추출
        random_x = min_th_frame['utime']
        random_y0 = min_th_frame['temperature']
        random_y1 = min_th_frame['humidity']

        # 그래프 그리기 파트
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=random_x, y=random_y0,
                                 mode='lines', line=dict(color='firebrick', width=0.5),
                                 name='온도(℃)'), secondary_y=False)
        fig.add_trace(go.Scatter(x=random_x, y=random_y1,
                                 mode='lines', line=dict(color='royalblue', width=0.5),
                                 name='습도(%)'), secondary_y=True)
        fig.add_hrect(y0=20, y1=30, line_width=0, fillcolor="red", opacity=0.1, annotation_text="안전 온도",
                      annotation_position="top left", )
        fig.add_hrect(y0=70, y1=80, line_width=0, fillcolor="blue", opacity=0.1, annotation_text="안전 습도",
                      annotation_position="top left", )

        # 배경 레이어색 파트
        fig.update_layout(paper_bgcolor="#EAEAEA")  # 차트 바깥 색
        fig.update_layout(plot_bgcolor="#F6F6F6")  # 그래프 안쪽색
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_layout(xaxis_tickformat='%Y<br>%m-%d<br>%H시%M분')
        fig.update_xaxes(linecolor='gray', gridcolor='gray', mirror=True, zeroline=False)
        fig.update_yaxes(linecolor='gray', gridcolor='gray', mirror=True)
        fig.update_yaxes(title_text="온도(℃)", secondary_y=False, zeroline=False, range=[-10, 100])
        fig.update_yaxes(title_text="습도(%)", secondary_y=True, zeroline=False, range=[-10, 100])
        fig.update_layout(

            margin=dict(l=10, r=10, t=60, b=10),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3d", step="day", stepmode="backward"),
                        # step 기준으로 범위를 지정 month니깐 count 1이면 1달 step이 year일때 count 1이면 1년
                        # day를 넣으면 1일 치 범위로 설정 근데 1day는 너무 짧은 범위라 생각보다 조작이 어려움 3day부터를 추천
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                ),

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
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=random_x, y=random_y0,
                             mode='lines', line=dict(color='firebrick', width=1.0),
                             name='온도(℃)'), secondary_y=False)
    fig.add_trace(go.Scatter(x=random_x, y=random_y1,
                             mode='lines', line=dict(color='royalblue', width=1.0),
                             name='습도(%)'), secondary_y=True)
    fig.add_hrect(y0=20, y1=30, line_width=0, fillcolor="red", opacity=0.1, annotation_text="안전 온도",
                  annotation_position="top left", )
    fig.add_hrect(y0=70, y1=80, line_width=0, fillcolor="blue", opacity=0.1, annotation_text="안전 습도",
                  annotation_position="top left", )

    # 배경 레이어색 파트
    fig.update_layout(paper_bgcolor="#EAEAEA")  # 차트 바깥 색
    fig.update_layout(plot_bgcolor="#F6F6F6")  # 그래프 안쪽색
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_layout(xaxis_tickformat='%Y<br>%m-%d<br>%H시%M분')
    fig.update_xaxes(linecolor='gray', gridcolor='gray', mirror=True, zeroline=False)
    fig.update_yaxes(linecolor='gray', gridcolor='gray', mirror=True)
    fig.update_yaxes(title_text="온도(℃)", secondary_y=False, zeroline=False, range=[-10, 100])
    fig.update_yaxes(title_text="습도(%)", secondary_y=True, zeroline=False, range=[-10, 100])
    f_json.append(fig.to_json())

    return f_json


@app.route('/addManager.html')
def addManager():
    url = "https://goqual.io/openapi/homes"

    response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})

    homes_list = response.json()
    for i in homes_list['result']:
        print(i['name'])

    return render_template('addManager.html', homes_list=homes_list['result'])


@app.route('/addManagerAct', methods=['POST', 'GET'])
def addManagerAct():
    if request.method == 'POST':

        collection = db.manager
        result = request.form.to_dict()
        print(result)
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
            return redirect(url_for("index"))
        else:
            flash("아이디가 중복되었습니다.")
            return redirect(url_for("addManager"))


@app.route('/login.html')
def login():
    return render_template('login.html')


@app.route('/loginAct', methods=['POST', 'GET'])
def loginAct():
    if request.method == 'POST':
        global db
        collection = db.manager
        result = request.form.to_dict()
        print(result)
        info = list(collection.find({'manager_id': result.get('manager_id')}))
        print(info)

        if result.get('manager_password') == info[0]['manager_password']:
            session['id'] = result.get('manager_id')
            session['place_num'] = info[0]['place_num']
            session['manager_place'] = info[0]['manager_place']

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
    return redirect(url_for("index"))


@app.route('/')
@app.route('/index.html')
def index():
    if 'place_num' in session:
        print(session['place_num'])

    print(ac)

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
    timeline_result = list(collection.find())

    return render_template('index.html', count=count_list, fig_code=fig_code, timeline_result=timeline_result)


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

    collection = db.thdata
    now = time  # 5~7은 지금 현재시각을 가져오는 코드임 2022-08-22 20:48:57 이렇게
    query = now.strftime('%Y-%m-%d')
    th_tmp = list()
    th_num = list()
    th_slide = list()
    slide_num = collection.count_documents({"time": {"$regex": query}})
    total_num = int(slide_num / number2)
    print(total_num)

    print(query)
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


@app.route('/THDatatable.html')
def THDatatable():
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

    print(th_num)
    print(th_slide)
    return render_template('THDatatable.html', count=count_list, z_result=z_result, th_result=th_tmp, fig_code=fig_code,
                           th_num=th_num, th_slide=th_slide, min_time=min_time, max_time=max_time, st_date=query1,
                           en_date=query2)


@app.route('/THDatatableAct.html', methods=['POST', 'GET'])
def THDatatableAct():
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

        return render_template('THDatatableAct.html', count=count_list, z_result=z_result, th_result=th_tmp,
                               day_re=result, fig_code=fig_code, th_num=th_num, th_slide=th_slide, min_time=min_time,
                               max_time=max_time, st_date=query1, en_date=result['end'])


@app.route('/riskalarm.html')
def riskalarm():
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
    collection = db.timeline
    timeline_result = list(collection.find())

    ex = "43946807"

    return render_template('riskalarm.html', count=count_list, z_result=z_result, ex=ex, timeline_result=timeline_result)


@app.route('/ML2.html')
def ML():
    collection = db.relic
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    number2 = collection.estimated_document_count()
    collection = db.manager
    number4 = collection.estimated_document_count()
    count_list = list((number1, number2, number3, number4))

    return render_template('ML2.html', count=count_list)


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
    print(page_count * 10)
    page_count = math.ceil(page_count) + 1
    s_result = list(collection.find().skip((page - 1) * 10).limit(10))
    count = list((r_count, z_count))

    return render_template('Sensor_table.html', s_result=s_result, page_list=page_count, count=count, z_result=z_result)


@app.route('/addSensorAct', methods=['POST', 'GET'])
def addSensorAct():
    if request.method == 'POST':
        collection = db.sensor
        result = request.form.to_dict()
        if result['sensor_name'] == '':
            flash("센서 이름을 입력하여 주세요.")
            return redirect(url_for("Sensor_table"))
        collection.insert_one(result)
        flash("성공")
        return redirect(url_for("Sensor_table"))


@app.route('/update_sensor.html', methods=['POST', 'GET'])
def update_sensor():
    if request.method == 'GET':
        num = request.args.get('id')
        # print(type(relic_num))
        collection = db.sensor
        result = list(collection.find({"_id": num}))
        # print(result)
        collection = db.zone
        z_result = list(collection.find())

        return render_template('update_sensor.html', result=result, z_result=z_result)


@app.route('/update_sensor_act', methods=['POST', 'GET'])
def update_sensor_act():
    if request.method == 'POST':
        sensor_num = request.args.get('id')
        result = request.form.to_dict()
        print(result)
        if result['sensor_name'] == '':
            flash("센서 이름을 입력하여 주세요.")
            return redirect(url_for("Sensor_table"))

        collection = db.senor
        collection.update_one({"_id": sensor_num},
                              {"$set": {"name": result['sensor_name'], "deviceType": result['sensor_type'],
                                        "zone_number": result['zone_number']}})

        flash("수정 완료")
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
    print(page_count * 10)
    page_count = math.ceil(page_count) + 1
    count = list((r_count, z_count))

    return render_template('Zone_table.html', z_result=z_result, page_list=page_count, count=count)


@app.route('/addZoneAct', methods=['POST', 'GET'])
def addZoneAct():
    if request.method == 'POST':
        collection = db.zone
        result = request.form.to_dict()
        if result['zone_name'] == '':
            flash("구역 이름을 입력하여 주세요.")
            return redirect(url_for("Zone_table"))
        collection.insert_one(result)
        flash("성공")
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
    print(page_count * 10)
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

        flash("성공")
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
        print(result)
        if result['relic_name'] == '':
            flash("유물 이름을 입력하여 주세요.")
            return redirect(url_for("Relic_table"))
        relic_num = request.args.get('id')
        collection = db.relic
        collection.update_one({"_id": ObjectId(relic_num)},
                              {"$set": {"relic_name": result['relic_name'], "relic_type": result['relic_type'],
                                        "zone_name": result['zone_name']}})

        flash("성공")
        return redirect(url_for("Relic_table"))


@app.route('/404.html')
def err():
    return render_template('404.html')


@app.route('/blank.html')
def blank():
    return render_template('blank.html')


@app.route('/button.html')
def button():
    return render_template('button.html')


@app.route('/chart.html')
def chart():
    return render_template('chart.html')


@app.route('/element.html')
def element():
    return render_template('element.html')


@app.route('/form.html')
def form():
    return render_template('form.html')


@app.route('/signin.html')
def signin():
    return render_template('signin.html')


@app.route('/signup.html')
def signup():
    return render_template('signup.html')


@app.route('/table.html')
def table():
    return render_template('table.html')


@app.route('/typography.html')
def typography():
    return render_template('typography.html')


@app.route('/widget.html')
def widget():
    return render_template('widget.html')


@app.route('/formtest', methods=['POST', 'GET'])
def test():
    result = request.form.to_dict()
    print(result)
    return redirect(url_for("widget"))


# 파이썬 명령어로 실행할 수 있음
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    # app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
