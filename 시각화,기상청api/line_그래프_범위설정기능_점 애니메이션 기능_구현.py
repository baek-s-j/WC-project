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


fig = go.Figure(
    data=[go.Scatter(x=random_x, y=random_y0,
                     mode="lines",
                     name='마크',line=dict(color='firebrick',width=0.5)),
          ##점을 위한 선이 하나 더 존재해야함
          go.Scatter(x=random_x, y=random_y0,
                     mode="lines",
                     name='온도',line=dict(color='firebrick',width=0.5)),
          go.Scatter(x=random_x, y=random_y1,
                     mode="lines",
                     name='습도',line=dict(color='royalblue',width=1))],
    layout=go.Layout(
        title_text="실시간 온습도 포인트", hovermode="closest",
        ##레이아웃 텍스트 설정
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="온도",
                                        method="animate",
                                        args=[None])])]),
    ##버튼이름 버튼 타입 설정

    frames=[go.Frame(
        data=[go.Scatter(
            x=[random_x.loc[k]],
            ##점이 어디 축으로 움직이는 지 데이터 값을 넣어줘야함
            ##어떤 선을따라 움직여야 하기 때문에 X값이 변해야함
            ##pandas 데이터 접근 중 index를 통한 접근을 활용
            ##for문의 k값이 변하면 온습도 인덱스에 차례대로 접근하도록 설정
            ##데이터 순서가 역순이면 for문을 반대로 돌리거나 데이터를 sort한 뒤 돌려줘야함
            y=[random_y0.loc[k]],
            ##y축도 마찬가지
            mode="markers",
            marker=dict(color="blue", size=10))])
        ##점의 크기와 색깔 설정
        for k in range(len(random_x))
        ##for문으로 데이터의 인덱스로 접근함, 범위는 데이터 행렬의 갯수만큼 실행 행의 갯수가 곧 X축의 갯수만큼임
      ]
)


## 범위설정 기능 파트
##고대로 복사하면 사용가능
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=3, label="3d", step="day", stepmode="backward"),
                ## step 기준으로 범위를 지정 month니깐 count 1이면 1달 step이 year일때 count 1이면 1년
                ##day를 넣으면 1일 치 범위로 설정 근데 1day는 너무 짧은 범위라 생각보다 조작이 어려움 3day부터를 추천
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1,label="1y",step="year",stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)


fig.show()