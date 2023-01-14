import requests
import time
from pymongo import MongoClient

# homeID {'name': '강진 무위사', 'homeId': 43946807} {'home': '강진 무위사', 'rooms': [{'name': '극락보전', 'room_id': 50563655}]}
# {'name': '김천 직지사', 'homeId': 45530039}
# {'home': '김천 직지사', 'rooms': [{'name': '대웅전', 'room_id': 51651616}]}
# {'name': '여수 흥국사', 'homeId': 36828887}
# {'home': '여수 흥국사', 'rooms': [{'name': '대웅전 내부', 'room_id': 45453790}, {'name': '대웅전 외부', 'room_id': 51746074}]}
place = ['43946807', '45530039', '36828887']
room = ['50563655', '51651616', '45453790']
# 방찾는 url https://goqual.io/openapi/openapi/homes/${homeId}/rooms
client = MongoClient(
    'mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')  # 여기서 부터는 몽고디비에 데이터 넣는것

global db

for j in range(0, len(place)):
    if j == 0:
        db = client.wc_project
    elif j == 1:
        db = client.wc_project_직지사
    else:
        db = client.wc_project_흥국사

    url = "https://goqual.io/openapi/" + place[j] + "/room/" + room[j] + "/devices-state?scope=openapi"
    now = time  # 5~7은 지금 현재시각을 가져오는 코드임 2022-08-22 20:48:57 이렇게
    anytime = now.strftime('%Y-%m-%d %H:%M')
    # 정해져있는 url을 가지고 api로 요청하여 가져온다
    response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})

    ex = response.json()  # 응답객체를 dict 형으로 받음. 여러개면 리스트형태이지만 원소를 접근하면 다 dict형임

    collection = db.thdata
    for i in ex:  # 받아온 온습도 데이터를 DB에 저장

        tmp = dict()
        tmp['time'] = anytime
        tmp['zone_name'] = i['name']
        tmp['temperature'] = i['deviceState']['temperature']
        tmp['humidity'] = i['deviceState']['humidity']
        #collection.insert_one(tmp)
