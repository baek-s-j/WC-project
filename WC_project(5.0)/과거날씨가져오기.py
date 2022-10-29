from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus
import urllib
import requests, xmltodict
import json
import pandas as pd
from pymongo import MongoClient
import math
import pandas as pd
from datetime import datetime
import numpy as np

url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?serviceKey=Zy3WCPt243lyaq0PqqKVwGL%2F42nvD9qjxGcz%2FgVr3y01%2BxYJ%2BsmdjB1H01RWJsVgaQkayn32SBzLQDdMgpiVIg%3D%3D&numOfRows=10&pageNo=1&dataCd=ASOS&dateCd=HR&stnIds=108&endDt=20200310&endHh=01&startHh=01&startDt=20190120'
# print(url)
# url 확인용


key = 'Zy3WCPt243lyaq0PqqKVwGL%2F42nvD9qjxGcz%2FgVr3y01%2BxYJ%2BsmdjB1H01RWJsVgaQkayn32SBzLQDdMgpiVIg%3D%3D'
# 공공데이터 포털에서 key값 받아오기


client = MongoClient('mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')
db = client.wc_project
collection = db.thdata
th_result = list(
    collection.find({"zone_name": "외부 전면-3번-무위사", }).sort([("time", 1)]))

min_time = list(collection.find().sort([("time", 1)]).limit(1))  # 데이터가 어디부터 어디까지 들어있는지 볼때
max_time = list(collection.find().sort([("time", -1)]).limit(1))

min_date = min_time[0]["time"][0:10]
max_date = max_time[0]["time"][0:10]
diff = datetime.strptime(max_time[0]["time"][0:10], "%Y-%m-%d") - datetime.strptime(min_time[0]["time"][0:10], "%Y-%m-%d")
print(diff.days)
num=diff.days+1
min_date = min_date.replace('-', '')
max_date = max_date.replace('-', '')
max_date=str(int(max_date)-1)


pageNo = 1  # 페이지 번호
dataCd = 'ASOS'  # 자료코드..?
dateCd = 'HR'  # 날짜코드?
stnIds = '259'  # 지점번호 어느지역 기상청인지 강진 259 여수 168 추풍령(김천) 135
endDt = max_date  # 종료날짜
endHh = '23'  # 종료시간
startHh = '00'
startDt = min_date
numOfRows = '999'  # 한 페이지의 결과수 320일*24시간 시작일 포함을해줘야함 차이구한후 +1

empty_df = pd.DataFrame()



for pageNo in range(1, math.ceil((num*24) / 999) + 1):
    # url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?serviceKey={}&numOfRows=10&pageNo=1&dataCd=ASOS&dateCd=HR&stnIds=108&endDt=20200310&endHh=01&startHh=01&startDt=20190120'.format(key)
    url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?serviceKey={}&numOfRows={}&pageNo={' \
          '}&dataCd={}&dateCd={}&stnIds={}&endDt={}&endHh={}&startHh={}&startDt={}'.format(
        key, numOfRows, pageNo, dataCd, dateCd, stnIds, endDt, endHh, startHh, startDt)
    # numOfRows부터 시작 원하는 데이터 행을 받고자하면 공공데이터 포털에서 해당 문서를 확인 한 후 날짜와 데이터 양식을 선택하여 요청함

    content = requests.get(url).content
    dict_content = xmltodict.parse(content)
    # items 하고 item 차이로 에러남
    # 데이터를 요청하여 xml로 받아옴

    jsonString = json.dumps(dict_content['response']['body']['items'], ensure_ascii=False)
    # json으로 한번 가공 아스키 코드까지 json형식으로 형태변환
    # print(jsonString)

    jsonObj = json.loads(jsonString)
    # 두번 가공하여야 함 json으로 다시 가공
    # print(jsonObj)

    # for item in jsonObj['item']:
    #    print(item)

    df = pd.DataFrame(jsonObj['item'])
    # 확인하기위해 pandas 데이터 프레임에 집어넣기
    # 여기까지 가공끝

    empty_df = empty_df.append(df)

print(empty_df)
print(len(empty_df))
print(empty_df.isnull().sum())

input_var = ['tm', 'ta', 'rn', 'ws', 'wd', 'hm']
# 시간, 온도, 강수량, 풍속, 풍향 ,습도

empty_df = empty_df[input_var]

empty_df.tm = pd.to_datetime(empty_df.tm)
empty_df = empty_df.set_index('tm')

result_df = empty_df
print(len(result_df))
print(result_df.isnull().sum())
result_df['rn'] = result_df['rn'].fillna(0)
result_df = result_df.apply(pd.to_numeric)
# corr_data = result_df.corr(method='pearson')
# corr_data.to_csv('corr_dataxx.csv')
result_df = result_df.dropna(axis=0)
result_df = result_df.reset_index()
result_df.to_csv('weather_무위사.csv')
