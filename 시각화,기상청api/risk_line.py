from flask import Flask, render_template, request, redirect, url_for, flash
from pymongo import MongoClient
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime

plt.rc("font", family="Malgun Gothic")
sns.set(font="Malgun Gothic",
        rc={"axes.unicode_minus": False}, style='darkgrid')

tdf = pd.read_csv("C:/무위사아이디X.csv", encoding='cp949')


#tdf['업데이트 시간'] = pd.to_datetime(tdf['업데이트 시간'])

#tdf['업데이트 시간'] = tdf['업데이트 시간'].str[:10]
#tdf['업데이트 시간'] = pd.to_datetime(tdf['업데이트 시간'])
#tdf = tdf.sort_values(by='업데이트 시간',ascending=True)

str_expr = "센서.str.contains('내부 후불벽화-5번-무위사')" # 문자열에 '' 포함
tdf = tdf.query(str_expr)               # 조건 부합 데이터 추출


#판다 데이터프레임에서 칼럼만 추출
random_x = tdf['업데이트 시간']
random_y0 = tdf['온도']
random_y1 = tdf['습도']

print(random_y0)




# 그래프 그리기 파트
fig = go.Figure()
fig.add_trace(go.Scatter(x=random_x, y=random_y0,
                    mode='lines',
                    name='온도',line=dict(color='firebrick',width=0.5)))
fig.add_trace(go.Scatter(x=random_x, y=random_y1,
                    mode='lines',
                    name='습도',line=dict(color='royalblue',width=1)))

##선 색 바꾸는 법

#fig.add_treac 항목중 color 항목에 
#'royalblue 가 파란색 firevrick이 붉은색으로 설정하는 것
#그 외 색깔은 plotly 홈페이지에 색깔 명칭이 있으니 확인후 수정하면 다른색 설정가능




###
#그래프 위험선 그리기 파트
#원리는 데이터를 선형으로 받아서 보내주기
random_y0 = tdf[['온도']]
random_y0.loc[(random_y0.온도<100),'온도']='16'
random_y0 = random_y0['온도']
#데이터 갯수만큼 선을 그어야 하기 때문에 원래 데이터를 받아서 변환함
#모두 같은 값을 가져야 하기 때문에 같은 값으로 통일


fig.add_trace(go.Scatter(x=random_x, y=random_y0,
                    mode='lines',
                    name='온도위험범위 16',line=dict(color='royalblue',width=1)))

#아래쪽 선 그리기 파트임




#위쪽 선 그리기 파트 (온도)
random_y0 = tdf[['온도']]
random_y0.loc[(random_y0.온도<100),'온도']='24'
random_y0 = random_y0['온도']

fig.add_trace(go.Scatter(x=random_x, y=random_y0,
                    mode='lines',
                    name='온도위험범위 24',line=dict(color='royalblue',width=1)))



##### 모두 동일하게 설정
#습도기준은 명확하게 나와있음
#야외 목조문화재 상대습도 범위는 75%를 넘지 않는것을 추천함
#따라서 75%상대습도 범위 아래에 위치해야 보다 목재부후와 같은 열화가 일어나지 않음
#해당 논문은 같이 첨부

random_y0 = tdf[['온도']]
random_y0.loc[(random_y0.온도<100),'온도']='75'
random_y0 = random_y0['온도']

fig.add_trace(go.Scatter(x=random_x, y=random_y0,
                    mode='lines',
                    name='상대습도 위험도 75',line=dict(color='firebrick',width=1)))

random_y0 = tdf[['온도']]
random_y0.loc[(random_y0.온도<100),'온도']='100'
random_y0 = random_y0['온도']

fig.add_trace(go.Scatter(x=random_x, y=random_y0,
                    mode='lines',
                    name='상대습도 위험도 100',line=dict(color='firebrick',width=1)))




fig.add_hrect(y0=16, y1=24, line_width=0, fillcolor="blue", opacity=0.1, annotation_text="안전 온도",
              annotation_position="top left", )
fig.add_hrect(y0=75, y1=100, line_width=0, fillcolor="red", opacity=0.1, annotation_text="안전 습도",
              annotation_position="top left", )

fig.show()

import cufflinks as cf
from flask import Flask, render_template
import plotly.figure_factory as ff
import plotly
import json

import numpy as np
import pandas as pd

