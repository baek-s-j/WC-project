from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus
import urllib
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import warnings
import xmltodict  # 결과가 xml 형식으로 반환된다. 이것을 dict 로 바꿔주는 라이브러리다
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

# 초단기예보조회
# 초단기예보정보를 조회하기 위해 발표일자, 발표시각, 예보지점 X 좌표, 예보지점 Y 좌표의 조회 조건으로 자료구분코드, 예보값, 발표일자, 발표시각, 예보지점 X 좌표, 예보지점 Y 좌표의 정보를 조회하는 기능

serviceKey = 'Zy3WCPt243lyaq0PqqKVwGL%2F42nvD9qjxGcz%2FgVr3y01%2BxYJ%2BsmdjB1H01RWJsVgaQkayn32SBzLQDdMgpiVIg%3D%3D'  # api 키
pageNo = '1'  # 페이지번호
numOfRows = '260'  # 한 페이지 결과 수
dataType = 'XML'  # 요청자료형식(XML/JSON) Default: XML
base_date = '20221027'  # ‘22년 x월 x일 발표 최근껄로 해야지 오류가 안남
# 날짜 설정 잘해야함 오류날 수 있음
base_time = '0200'  # 06시30분 발표(30분 단위)
# 마찬가지 오늘날짜 기준 잘 설정해야함
nx = '56'  # 예보지점 X 좌표값
ny = '64'  # 예보지점 Y 좌표값

# 위치좌표 엑셀파일로 첨부했습니다.

url = 'https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst?serviceKey={}&pageNo={}&numOfRows={}&dataType={}&base_date={}&base_time={}&nx={}&ny={}'.format(
    serviceKey, pageNo, numOfRows, dataType, base_date, base_time, nx, ny)

# 초단기,단기,실황조회
# getUltraSrtNcst 명칭만 바꿔주면 된다
#
# getUltraSrtNcst   초단기실황조회
# getUltraSrtFcst   초단기예보조회
# getVilageFcst   단기예보조회
# getFcstVersion   예보버전조회

# 각 조회버전마다 value값을 바꿔줘야 하니 그건 문서양식참조


content = requests.get(url, verify=False).content
dict = xmltodict.parse(content)

jsonString = json.dumps(dict['response']['body']['items'], ensure_ascii=False)
# json으로 한번 가공 아스키 코드까지 json형식으로 형태변환
# print(jsonString)

print(jsonString)

ls_dict = json.loads(jsonString)  # json문자열을 파이썬 객체로 변환한다.

df = pd.DataFrame(ls_dict['item'])

print(df)

# file = open("./weather.json", "w+")
# file.write(json.dumps(jsonObj['item']))
# 저장 필요하면 json파일 저장 몽고디비에 넣어보면 데이터 잘 들어와 있는걸 알 수 있음
