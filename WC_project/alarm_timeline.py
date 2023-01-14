import requests
import time
from pymongo import MongoClient

place = ['43946807', '45530039', '36828887']  # 각각의 무위사, 직지사, 흥국사 유물공간 번호
room = ['50563655', '51651616', '45453790']  # 각각의 무위사, 직지사, 흥국사 룸 번호
# 방찾는 url https://goqual.io/openapi/openapi/homes/${homeId}/rooms
client = MongoClient(
    'mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')  # 몽고디비 연결

global db

for j in range(0, len(place)):



    url = "https://goqual.io/openapi/" + place[j] + "/room/" + room[j] + "/devices-state?scope=openapi"
    now = time  # 5~7은 지금 현재시각을 가져오는 코드임 2022-08-22 20:48:57 이렇게
    anytime = now.strftime('%Y-%m-%d %H:%M')
    # 정해져있는 url을 가지고 api로 요청하여 가져온다
    response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})

    ex = response.json()  # 응답객체를 dict 형으로 받음. 여러개면 리스트형태이지만 원소를 접근하면 다 dict형임

    collection = db.timeline  # 타임라인 콜랙션으로 변경하고
    if j == 0:  # 각각의 DB에 맞게 변경
        db = client.wc_project
        print("무")
    elif j == 1:
        db = client.wc_project_직지사
        print("직")
    else:
        db = client.wc_project_흥국사
        print("흥")
    for i in ex:
        score = 0
        risk_name = ""
        risk_ment = list()
        tmp = dict()

        if 34 < i['deviceState']['temperature']:  # 논문을 근거로 온도 위험도 계산
            score = score + 2
            risk_name = risk_name + " 고온"
            risk_ment.append("온도가 {0}℃로 고온".format(i['deviceState']['temperature']))

        elif i['deviceState']['temperature'] < 10:
            score = score + 2
            risk_ment.append("온도가 {0}℃로 저온".format(i['deviceState']['temperature']))

        else:
            score = score + 1

        if i['deviceState']['humidity'] >= 95:  # 논문을 근거로 습도 위험도 계산
            score = score + 8
            risk_ment.append("습도가 {0}%로 높아 부후균 생장 위험".format(i['deviceState']['humidity']))

        elif i['deviceState']['humidity'] <= 75:
            score = score + 1

        else:
            score = score + 4
            risk_ment.append("습도가 {0}%로, 표면오염균 생장 경고".format(i['deviceState']['humidity']))

        if i['deviceState']['battery'] > 50:  # 배터리 잔량 위험정도를 계산
            score = score + 1

        elif i['deviceState']['battery'] <= 25:
            score = score + 6
            risk_ment.append("배터리가 {0}%로, 교체가 시급하여 빠른 교체바랍니다.".format(i['deviceState']['battery']))

        else:
            score = score + 3
            risk_ment.append("배터리가 {0}%로, 교체를 권고합니다.".format(i['deviceState']['battery']))



        if 5 <= score < 7:  # 총 점수를 가지고 위험정도를 판단후 DB에 저장

            tmp['time'] = anytime
            tmp['zone_name'] = i['name']
            tmp['risk_name'] = "warning"
            tmp['risk_ment'] = risk_ment
            print(j)
            #collection.insert_one(tmp)

        elif score >= 7:

            tmp['time'] = anytime
            tmp['zone_name'] = i['name']
            tmp['risk_name'] = "danger"
            tmp['risk_ment'] = risk_ment
            print(j)
            #collection.insert_one(tmp)

        print(tmp)
        # tmp = dict()
        # tmp['time'] = anytime
        # tmp['zone_name'] = i['name']
        # tmp['temperature'] = i['deviceState']['temperature']
        # tmp['humidity'] = i['deviceState']['humidity']
        # collection.insert_one(tmp)
