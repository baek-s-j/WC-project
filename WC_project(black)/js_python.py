import sys
from pymongo import MongoClient
from bson.objectid import ObjectId
import requests
from threading import Thread
import winsound as sd
import math
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

import plotly.express as px  # 이코드를 가지고 함수로 만들어서 매개변수에 날짜를 받고 만들면 가능하려나?
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def realtime_visual(query):
    client = MongoClient('mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')
    db = client.wc_project
    collection = db.zone
    z_result = list(collection.find())
    f_json = list()
    dfs = list()

    for i in z_result:
        collection = db.thdata
        th_result = list(collection.find({"zone_name": i['name'], "time": {"$regex": query}}).sort([("time", 1)]))
        if not th_result:
            return False
        th_frame = pd.DataFrame(th_result)

        # 판다 데이터프레임에서 칼럼만 추출
        random_x = th_frame['time']
        random_y0 = th_frame['temperature']
        random_y1 = th_frame['humidity']

        # 그래프 그리기 파트
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=random_x, y=random_y0,
                                 mode='lines',
                                 name='온도(℃)'), secondary_y=False)
        fig.add_trace(go.Scatter(x=random_x, y=random_y1,
                                 mode='lines',
                                 name='습도(%)'), secondary_y=True)
        fig.add_hrect(y0=20, y1=30, line_width=0, fillcolor="blue", opacity=0.1, annotation_text="안전 온도",
                      annotation_position="top left", )
        fig.add_hrect(y0=70, y1=80, line_width=0, fillcolor="red", opacity=0.1, annotation_text="안전 습도",
                      annotation_position="top left", )

        # 배경 레이어색 파트
        fig.update_layout(paper_bgcolor="black")  # 차트 바깥 색
        fig.update_layout(plot_bgcolor="black")  # 그래프 안쪽색
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_layout(xaxis_tickformat='%Y<br>%m-%d<br>%H시%M분')
        fig.update_xaxes(linecolor='gray', gridcolor='gray', mirror=True, zeroline=False)
        fig.update_yaxes(linecolor='gray', gridcolor='gray', mirror=True)
        fig.update_yaxes(title_text="온도(℃)", secondary_y=False, zeroline=False, range=[-10, 100])
        fig.update_yaxes(title_text="습도(%)", secondary_y=True, zeroline=False, range=[-10, 100])

        f_json.append(fig.to_json())
        dfs.append(th_frame)

    return f_json, dfs


if __name__ == '__main__':
    realtime_visual(sys.argv[1])
