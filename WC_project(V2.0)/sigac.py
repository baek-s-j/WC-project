import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# csv파일 인코딩
plt.rc("font", family="Malgun Gothic")
sns.set(font="Malgun Gothic",
        rc={"axes.unicode_minus": False}, style='darkgrid')

tdf = pd.read_csv('C:/Users/qor27/PycharmProjects/WC_project/testdata.csv', encoding='utf-8')

'''
#기본 ex line 그래프 온도만 나타남
fig = px.line(tdf,
x="업데이트 시간",
y="온도")


#배경 레이어색 파트
fig.update_layout(paper_bgcolor ="black")
fig.update_layout(plot_bgcolor="black")
fig.update_xaxes(linecolor='red', gridcolor='gray',mirror=True)
fig.update_yaxes(linecolor='red', gridcolor='gray',mirror=True)
fig.update_yaxes(tickformat=',') # 간단하게 , 형으로 변경
fig.show()

'''

# 판다 데이터프레임에서 칼럼만 추출
random_x = tdf['업데이트 시간']
random_y0 = tdf['온도']
random_y1 = tdf['습도']

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

# html 파일로 저장하고 따로 위치장소를 태그 해야만 플라스크 웹에 띄울 수 있음 그외 방법은 아직 모르겠음
fig.write_image('si.png')
# 이 파일을
'''
<iframe src="../static/charts/line.html" width="1200" height="500" frameborder="0" framespacing="0" marginheight="0" marginwidth="0" scrolling="no" vspace="0"></iframe>
이 태그로 넣고 싶은 웹 태그 부분에 입력 src는 경로 나머지는 크기와 프레임 선 나타내기 옵션들
'''
# 그래프 보기

