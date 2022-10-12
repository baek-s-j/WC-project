import requests  # 센서를을 몽고 디비에 추가하는 파이썬 파일
import json
from pymongo import MongoClient
from ast import literal_eval

# 무위사 homeID :43946807
url = "https://goqual.io/openapi/43946807/devices?scope=openapi"
# 이거는 그 헤이홈 Open API 가이드에서 상세설명 에서 원하는 데이터 마다 URL이 달라

response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})
# 인증토큰? 이렇게 넣으면 답이 와
ex = response.json()  # 응답객체를 dict 형으로 받음. 여러개면 리스트형태이지만 원소를 접근하면 다 dict형임
# 로컬 주소는 MongoClient('127.0.0.1', 27017)
client = MongoClient('mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')  # 여기서 부터는 몽고디비에 데이터 넣는것

db = client.wc_project
collection = db.sensor
for i in ex:
    i['_id'] = i.pop('id')
    i.pop('hasSubDevices')
    i.pop('category')
    i.pop('online')
    col = db.zone
    i['zone_number'] = list(col.find({"name": i['name']}, {"_id": 1}))[0]['_id']
    print(i)
    collection.insert_one(i)

client.close()
print(response.text)
# print(ex[0]['id'])
# print(ex)
