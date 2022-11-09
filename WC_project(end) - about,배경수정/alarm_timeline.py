import requests
import time
from pymongo import MongoClient

place = ['43946807', '45530039', '36828887']
room = ['50563655', '51651616', '45453790']
# 방찾는 url https://goqual.io/openapi/openapi/homes/${homeId}/rooms
client = MongoClient(
    'mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')  # 여기서 부터는 몽고디비에 데이터 넣는것

global db

place = ['43946807', '45530039', '36828887']
room = ['50563655', '51651616', '45453790']
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
    # 이거는 그 헤이홈 Open API 가이드에서 상세설명 에서 원하는 데이터 마다 URL이 달라
    response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})
    # 인증토큰? 이렇게 넣으면 답이 와

    ex = response.json()  # 응답객체를 dict 형으로 받음. 여러개면 리스트형태이지만 원소를 접근하면 다 dict형임

    collection = db.timeline
    for i in ex:
        score = 0
        risk_name = ""
        risk_ment = list()

        if 34 < i['deviceState']['temperature']:
            score = score + 2
            risk_name = risk_name + " 고온"
            risk_ment.append("온도가 {0}℃로 고온".format(i['deviceState']['temperature']))

        elif i['deviceState']['temperature'] < 10:
            score = score + 2
            risk_ment.append("온도가 {0}℃로 저온".format(i['deviceState']['temperature']))

        else:
            score = score + 1

        if i['deviceState']['humidity'] >= 95:
            score = score + 8
            risk_ment.append("습도가 {0}%로 높아 부후균 생장 위험".format(i['deviceState']['humidity']))

        elif i['deviceState']['humidity'] <= 75:
            score = score + 1

        else:
            score = score + 4
            risk_ment.append("습도가 {0}%로, 표면오염균 생장 경고".format(i['deviceState']['humidity']))

        if i['deviceState']['battery'] > 50:
            score = score + 1

        elif i['deviceState']['battery'] <= 25:
            score = score + 6
            risk_ment.append("배터리가 {0}%로, 교체가 시급하여 빠른 교체바랍니다.".format(i['deviceState']['battery']))

        else:
            score = score + 3
            risk_ment.append("배터리가 {0}%로, 교체를 권고합니다.".format(i['deviceState']['battery']))

        if 5 <= score < 7:
            tmp = dict()
            tmp['time'] = anytime
            tmp['zone_name'] = i['name']
            tmp['risk_name'] = "warning"
            tmp['risk_ment'] = risk_ment

            collection.insert_one(tmp)

        elif score >= 7:
            tmp = dict()
            tmp['time'] = anytime
            tmp['zone_name'] = i['name']
            tmp['risk_name'] = "danger"
            tmp['risk_ment'] = risk_ment
            collection.insert_one(tmp)

        # tmp = dict()
        # tmp['time'] = anytime
        # tmp['zone_name'] = i['name']
        # tmp['temperature'] = i['deviceState']['temperature']
        # tmp['humidity'] = i['deviceState']['humidity']
        # collection.insert_one(tmp)
