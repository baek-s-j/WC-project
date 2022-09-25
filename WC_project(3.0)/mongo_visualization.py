import plotly.express as px  # 이코드를 가지고 함수로 만들어서 매개변수에 날짜를 받고 만들면 가능하려나?
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
import time
from pymongo import MongoClient


def visual(query):
    client = MongoClient('127.0.0.1', 27017)  # 여기서 부터는 몽고디비에 데이터 넣는것
    db = client.wc_project
    collection = db.zone
    z_result = list(collection.find())
    f_json = list()

    for i in z_result:
        collection = db.thdata
        th_result = list(collection.find({"zone_name": i['name'], "time": {"$regex": query}}))
        th_frame = pd.DataFrame(th_result)

        # 판다 데이터프레임에서 칼럼만 추출
        random_x = th_frame['time']
        random_y0 = th_frame['temperature']
        random_y1 = th_frame['humidity']

        # 그래프 그리기 파트
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=random_x, y=random_y0,
                                 mode='lines+markers',
                                 name='온도'))
        fig.add_trace(go.Scatter(x=random_x, y=random_y1,
                                 mode='lines+markers',
                                 name='습도'))

        # 배경 레이어색 파트
        fig.update_layout(paper_bgcolor="black")
        fig.update_layout(plot_bgcolor="black")
        fig.update_xaxes(linecolor='red', gridcolor='gray', mirror=True)
        fig.update_yaxes(linecolor='red', gridcolor='gray', mirror=True)
        fig.update_yaxes(tickformat=',')  # 간단하게 , 형으로 변경
        f_json.append(fig.to_json())

    return f_json


f_json = visual()
