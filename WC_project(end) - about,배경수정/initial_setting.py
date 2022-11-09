import requests  # 구역을 몽고 디비에 추가하는 파이썬 파일
import json
from pymongo import MongoClient
from ast import literal_eval

# homeID {'name': '강진 무위사', 'homeId': 43946807} {'home': '강진 무위사', 'rooms': [{'name': '극락보전', 'room_id': 50563655}]}
# {'name': '김천 직지사', 'homeId': 45530039}
# {'home': '김천 직지사', 'rooms': [{'name': '대웅전', 'room_id': 51651616}]}
# {'name': '여수 흥국사', 'homeId': 36828887}
# {'home': '여수 흥국사', 'rooms': [{'name': '대웅전 내부', 'room_id': 45453790}, {'name': '대웅전 외부', 'room_id': 51746074}]}
place = ['43946807', '45530039','36828887']
client = MongoClient(
    'mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')  # 여기서 부터는 몽고디비에 데이터 넣는것

global db


for i in range(0, len(place)):

    if i == 0:
        db = client.wc_project_무위사
    elif i == 1:
        db = client.wc_project_직지사
    else:
        db = client.wc_project_흥국사

    url = "https://goqual.io/openapi/" + place[i] + "/devices?scope=openapi"
    # 이거는 그 헤이홈 Open API 가이드에서 상세설명 에서 원하는 데이터 마다 URL이 달라
    response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})
    # 인증토큰? 이렇게 넣으면 답이 와
    ex = response.json()  # 응답객체를 dict 형으로 받음. 여러개면 리스트형태이지만 원소를 접근하면 다 dict형임

    collection = db.zone
    for i in ex:  # 구역 데이터 넣기
        tmp = dict()
        tmp['name'] = i['name']
        tmp['familyId'] = i['familyId']
        collection.insert_one(tmp)
        print(i['name'])

    collection = db.sensor
    for i in ex:  # 센서 데이터 넣기
        i['_id'] = i.pop('id')
        i.pop('hasSubDevices')
        i.pop('category')
        i.pop('online')
        col = db.zone
        i['zone_number'] = list(col.find({"name": i['name']}, {"_id": 1}))[0]['_id']
        collection = db.sensor
        print(i)
        collection.insert_one(i)

    collection = db.riskdata #위험도 데이터
    ty = ["목제품(출토)", "목제품(전승)", "금속류", "토기류", "섬유류", "고문서류", "유화", "칠기류"]
    max_t = [20, 24, 24, 24, 24, 24, 24, 24]
    min_t = [15, 16, 16, 16, 16, 16, 16, 16]
    max_h = [80, 65, 50, 50, 65, 65, 65, 65]
    min_h = [75, 50, 40, 40, 50, 50, 50, 50]
    for i in range(0, 8):
        tmp = dict()
        tmp['relic_type'] = ty[i]
        tmp['max_temperature'] = max_t[i]
        tmp['min_temperature'] = min_t[i]
        tmp['max_humidity'] = max_h[i]
        tmp['min_humidity'] = min_h[i]
        collection.insert_one(tmp)
        print(tmp)


    print(response.text)
    # print(ex[0]['id'])
    # print(ex)
client.close()