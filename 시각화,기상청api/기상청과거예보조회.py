from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus
import urllib
import requests, xmltodict
import json
import pandas as pd



#url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?serviceKey=Zy3WCPt243lyaq0PqqKVwGL%2F42nvD9qjxGcz%2FgVr3y01%2BxYJ%2BsmdjB1H01RWJsVgaQkayn32SBzLQDdMgpiVIg%3D%3D&numOfRows=10&pageNo=1&dataCd=ASOS&dateCd=HR&stnIds=108&endDt=20200310&endHh=01&startHh=01&startDt=20190120'
#print(url)
#url 확인용


key = 'Zy3WCPt243lyaq0PqqKVwGL%2F42nvD9qjxGcz%2FgVr3y01%2BxYJ%2BsmdjB1H01RWJsVgaQkayn32SBzLQDdMgpiVIg%3D%3D'
#공공데이터 포털에서 key값 받아오기

numOfRows = '999'
pageNo = '1'
dataCd = 'ASOS'
dateCd = 'HR'
stnIds = '131'
endDt = '20220905'
endHh = '01'
startHh = '01'
startDt = '20200101'


#url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?serviceKey={}&numOfRows=10&pageNo=1&dataCd=ASOS&dateCd=HR&stnIds=108&endDt=20200310&endHh=01&startHh=01&startDt=20190120'.format(key)
url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?serviceKey={}&numOfRows={}&pageNo={}&dataCd={}&dateCd={}&stnIds={}&endDt={}&endHh={}&startHh={}&startDt={}'.format(key,numOfRows,pageNo,dataCd,dateCd,stnIds,endDt,endHh,startHh,startDt)
#numOfRows부터 시작 원하는 데이터 행을 받고자하면 공공데이터 포털에서 해당 문서를 확인 한 후 날짜와 데이터 양식을 선택하여 요청함


content = requests.get(url).content
dict = xmltodict.parse(content)
print(dict['response']['body']['items'])
#items 하고 item 차이로 에러남
# 데이터를 요청하여 xml로 받아옴

jsonString = json.dumps(dict['response']['body']['items'],ensure_ascii=False)
# json으로 한번 가공 아스키 코드까지 json형식으로 형태변환
#print(jsonString)

jsonObj = json.loads(jsonString)
# 두번 가공하여야 함 json으로 다시 가공
#print(jsonObj)

#for item in jsonObj['item']:
#    print(item)

df = pd.DataFrame(jsonObj['item'])
#확인하기위해 pandas 데이터 프레임에 집어넣기
#여기까지 가공끝
print(df.count())

file = open("./weather.json", "w+")
file.write(json.dumps(jsonObj['item']))
#저장 필요하면 json파일 저장 몽고디비에 넣어보면 데이터 잘 들어와 있는걸 알 수 있음