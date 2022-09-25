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

tdf['업데이트 시간'] = tdf['업데이트 시간'].str[:10]
#tdf['업데이트 시간'] = pd.to_datetime(tdf['업데이트 시간'])
#tdf = tdf.sort_values(by='업데이트 시간',ascending=True)

tdf.groupby(['업데이트 시간'], as_index=False).mean()

print(tdf)
tdf.info()
# 연도 항목을 정수형으로 변환
#tdf['업데이트 시간'] = tdf['year'].astype(int)

'''
tdf = tdf.sort_values(by='업데이트 시간', ascending=True)
tdf.index = tdf.'업데이트시간'

tdf["업데이트 시간"] = tdf["업데이트 시간"].str.replace(pat='-', repl=r'', regex=True)
tdf["업데이트 시간"] = tdf["업데이트 시간"].str.replace(pat=':', repl=r'', regex=True)
tdf["업데이트 시간"] = tdf["업데이트 시간"].str.replace(pat=' ', repl=r'', regex=True)


print(tdf)

'''


fig = px.density_mapbox(tdf, lat='lat', lon='lon', z='온도', radius=10,
                        center=dict(lat=34.7387689, lon=126.686808), zoom=18,
                        mapbox_style="open-street-map",
                        animation_frame="업데이트 시간",


                        #range_x=[202205010000, 202208010000]
                        #range_x=[datetime.date(2021, 7, 3), datetime.date(2022, 8, 23)],
                        # range_y=[-80,90]
                        )

'''

fig.update_layout(
 mapbox_style="white-bg",
 mapbox_layers=[
 {
 "below": 'traces',
 "sourcetype": "raster",
 "sourceattribution": "stamen",
 "source": ["http://api.vworld.kr/req/wms?key=E93C8350-D30A-3F4C-9FFF-B4C119A44938&png" ]
 }
  ])





fig.update_layout(

    mapbox_style='whilte-bg' ,
    mapbox_layers=[
        {
            "source" : [

            ]
        }
    ]
)


'''

#fig.update_xaxes(range=[datetime.date(2020, 7, 3), datetime.date(2023, 8, 23)])
#fig.update_yaxes(range=[-80, 90])
# 배경 레이어색 파트

#기본 ex line 그래프 온도만 나타남

'''

'''
fig.update_layout(paper_bgcolor="black")
fig.update_layout(plot_bgcolor="black")
fig.update_xaxes(linecolor='red', gridcolor='gray', mirror=True)
fig.update_yaxes(linecolor='red', gridcolor='gray', mirror=True)
fig.update_yaxes(tickformat=',')  # 간단하게 , 형으로 변경

#fig.write_html('map.html')
fig.show()

