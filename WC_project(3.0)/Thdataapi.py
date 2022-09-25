import requests
import time
from pymongo import MongoClient

now = time  # 5~7은 지금 현재시각을 가져오는 코드임 2022-08-22 20:48:57 이렇게
print(now.strftime('%Y-%m-%d %H:%M:%S'))
anytime = now.strftime('%Y-%m-%d %H:%M')
url = "https://goqual.io/openapi/43946807/room/50563655/devices-state?scope=openapi"  # 방에있는 센서의 상태 전부 조회
# 이거는 그 헤이홈 Open API 가이드에서 상세설명 에서 원하는 데이터 마다 URL이 달라

response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})
# 인증토큰? 이렇게 넣으면 답이 와

ex = response.json()  # 응답객체를 dict 형으로 받음. 여러개면 리스트형태이지만 원소를 접근하면 다 dict형임

client = MongoClient('127.0.0.1', 27017)  # 여기서 부터는 몽고디비에 데이터 넣는것

db = client.wc_project
collection = db.thdata



for i in ex:
    tmp = dict()
    tmp['time'] = anytime
    tmp['zone_name'] = i['name']
    tmp['temperature'] = i['deviceState']['temperature']
    tmp['humidity'] = i['deviceState']['humidity']
    collection.insert_one(tmp)
    print(tmp)

print(response.text)
# result['time'] = anytime  # 이렇게 컬럼을 추가해줄수 있어
# print(result)
# client = MongoClient('127.0.0.1', 27017)  # 여기서 부터는 몽고디비에 데이터 넣는것
# db = client.wc_project
# collection = db.thdata
# collection.insert_one(result)
# client.close()
