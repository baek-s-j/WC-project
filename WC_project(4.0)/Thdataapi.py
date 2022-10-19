import requests
import time
from pymongo import MongoClient

place = ['43946807']
room = ['50563655']
# 방찾는 url https://goqual.io/openapi/openapi/homes/${homeId}/rooms
client = MongoClient(
    'mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')  # 여기서 부터는 몽고디비에 데이터 넣는것

global db

for i in range(0, len(place)):
    if i == 0:
        db = client.wc_project_무위사

    url = "https://goqual.io/openapi/" + place[i] + "/room/" + room[i] + "/devices-state?scope=openapi"
    now = time  # 5~7은 지금 현재시각을 가져오는 코드임 2022-08-22 20:48:57 이렇게
    anytime = now.strftime('%Y-%m-%d %H:%M')
    # 이거는 그 헤이홈 Open API 가이드에서 상세설명 에서 원하는 데이터 마다 URL이 달라
    response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})
    # 인증토큰? 이렇게 넣으면 답이 와

    ex = response.json()  # 응답객체를 dict 형으로 받음. 여러개면 리스트형태이지만 원소를 접근하면 다 dict형임

    db = client.wc_project
    collection = db.thdata
    for i in ex:
        tmp = dict()
        tmp['time'] = anytime
        tmp['zone_name'] = i['name']
        tmp['temperature'] = i['deviceState']['temperature']
        tmp['humidity'] = i['deviceState']['humidity']
        collection.insert_one(tmp)

    # result['time'] = anytime  # 이렇게 컬럼을 추가해줄수 있어
    # print(result)
    # client = MongoClient('127.0.0.1', 27017)  # 여기서 부터는 몽고디비에 데이터 넣는것
    # db = client.wc_project
    # collection = db.thdata
    # collection.insert_one(result)
    # client.close()
