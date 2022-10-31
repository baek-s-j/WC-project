from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus
import urllib
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import warnings
import xmltodict
import ssl
import matplotlib.pyplot as plt

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
nx = '56'  # 예보지점 X 좌표값 여수 73,67 무위사 56,64  김천 직지사 79 96
ny = '64'  # 예보지점 Y 좌표값



url = 'https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst?serviceKey={}&pageNo={}&numOfRows={}&dataType={}&base_date={}&base_time={}&nx={}&ny={}'.format(
    serviceKey, pageNo, numOfRows, dataType, base_date, base_time, nx, ny)


content = requests.get(url, verify=False).content
dict = xmltodict.parse(content)

jsonString = json.dumps(dict['response']['body']['items'], ensure_ascii=False)

ls_dict = json.loads(jsonString)

df = pd.DataFrame(ls_dict['item'])

result = pd.DataFrame()

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
    time = str(r1 + '-' + r2 + '-' + r3) + " " + "{}:00".format(i)

    tdf.insert(0, 'tm', time)

    result = result.append(tdf)

print(result)
result.tm = pd.to_datetime(result.tm, format='%Y-%m-%d')
result = result.set_index('tm')
result= result.apply(pd.to_numeric)
random_x = result.index.tolist()
print(random_x)
result.plot()
plt.title("Pandas의 Plot메소드 사용 예")
plt.xlabel("시간")
plt.ylabel("Data")
plt.show()

print(result)
