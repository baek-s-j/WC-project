import requests
import time
from datetime import datetime, timedelta
from pymongo import MongoClient
from bson.objectid import ObjectId

client = MongoClient(
    'mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')  # 몽고디비 연결

# db = client.wc_project
# collection = db.thdata
# min_time = list(collection.find().sort([("time", 1)]).limit(1))  # select 최대 최소값을 설정하기 위해 가져옴
# max_time = list(collection.find().sort([("time", -1)]).limit(1))
# min_time = min_time[0]["time"]
# max_time = max_time[0]["time"]
# min_time = datetime.strptime(min_time, '%Y-%m-%d %H:%M')
# max_time = datetime.strptime(max_time, '%Y-%m-%d %H:%M')
# print(type(min_time))
# print(max_time - min_time)
# tmp = min_time
# ex=0
# while True:
#     if tmp == max_time:
#         break
#     th_result = list(collection.find({"time": str(tmp)[:16]}))
#     zone1 = 0
#     zone2 = 0
#     zone3 = 0
#     zone4 = 0
#     zone5 = 0
#     for i in th_result:
#         if i['zone_name'] == "내부 후불벽화-5번-무위사":
#             zone5 = zone5 + 1
#             if zone5 > 1:
#                 collection.delete_one({'_id': ObjectId(i['_id'])})
#
#         elif i['zone_name'] == "외부 후면-4번-무위사":
#             zone1 = zone1 + 1
#             if zone1 > 1:
#                 collection.delete_one({'_id': ObjectId(i['_id'])})
#         elif i['zone_name'] == "외부 전면-3번-무위사":
#             zone2 = zone2 + 1
#             if zone2 > 1:
#                 collection.delete_one({'_id': ObjectId(i['_id'])})
#         elif i['zone_name'] == "내부 우측-2번-무위사":
#             zone3 = zone3 + 1
#             if zone3 > 1:
#                 collection.delete_one({'_id': ObjectId(i['_id'])})
#         elif i['zone_name'] == "내부 좌측-1번-무위사":
#             zone4 = zone4 + 1
#             if zone4 > 1:
#                 collection.delete_one({'_id': ObjectId(i['_id'])})
#     tmp = tmp + timedelta(minutes=10)
#     print(str(tmp)[:16])


# db = client.wc_project_직지사
# collection = db.thdata
# min_time = list(collection.find().sort([("time", 1)]).limit(1))  # select 최대 최소값을 설정하기 위해 가져옴
# max_time = list(collection.find().sort([("time", -1)]).limit(1))
# min_time = min_time[0]["time"]
# max_time = max_time[0]["time"]
# min_time = datetime.strptime(min_time, '%Y-%m-%d %H:%M')
# max_time = datetime.strptime(max_time, '%Y-%m-%d %H:%M')
# print(type(min_time))
# print(max_time - min_time)
# tmp = datetime.strptime("2022-08-01 00:00", '%Y-%m-%d %H:%M')
# ex=0
# while True:
#     if tmp == max_time:
#         break
#     th_result = list(collection.find({"time": str(tmp)[:16]}))
#     zone1 = 0
#     zone2 = 0
#     zone3 = 0
#     zone4 = 0
#     zone5 = 0
#     for i in th_result:
#         if i['zone_name'] == "대웅전-내부 괘불-5번-직지사":
#             zone5 = zone5 + 1
#             if zone5 > 1:
#                 collection.delete_one({'_id': ObjectId(i['_id'])})
#
#         elif i['zone_name'] == "대웅전-내부 좌측-4번-직지사":
#             zone1 = zone1 + 1
#             if zone1 > 1:
#                 collection.delete_one({'_id': ObjectId(i['_id'])})
#         elif i['zone_name'] == "대웅전 내부 우측_직지사":
#             zone2 = zone2 + 1
#             if zone2 > 1:
#                 collection.delete_one({'_id': ObjectId(i['_id'])})
#         elif i['zone_name'] == "대웅전-외부 전면-2번-직지사":
#             zone3 = zone3 + 1
#             if zone3 > 1:
#                 collection.delete_one({'_id': ObjectId(i['_id'])})
#         elif i['zone_name'] == "대웅전-외부 후면-3번-직지사":
#             zone4 = zone4 + 1
#             if zone4 > 1:
#                 collection.delete_one({'_id': ObjectId(i['_id'])})
#     tmp = tmp + timedelta(minutes=10)
#     print(str(tmp)[:16])
db = client.wc_project_흥국사
collection = db.thdata
min_time = list(collection.find().sort([("time", 1)]).limit(1))  # select 최대 최소값을 설정하기 위해 가져옴
max_time = list(collection.find().sort([("time", -1)]).limit(1))
min_time = min_time[0]["time"]
max_time = max_time[0]["time"]
min_time = datetime.strptime(min_time, '%Y-%m-%d %H:%M')
max_time = datetime.strptime(max_time, '%Y-%m-%d %H:%M')
print(type(min_time))
print(max_time - min_time)
tmp = datetime.strptime("2022-08-01 00:00", '%Y-%m-%d %H:%M')
ex = 0
while True:
    if tmp == max_time:
        break
    th_result = list(collection.find({"time": str(tmp)[:16]}))
    zone1 = 0
    zone2 = 0
    zone3 = 0
    zone4 = 0
    zone5 = 0
    zone6 = 0
    for i in th_result:
        if i['zone_name'] == "대웅전 후면 벽면-6번-흥국사":
            zone5 = zone5 + 1
            if zone5 > 1:
                collection.delete_one({'_id': ObjectId(i['_id'])})

        elif i['zone_name'] == "대웅전 동쪽 벽면-5번-흥국사":
            zone1 = zone1 + 1
            if zone1 > 1:
                collection.delete_one({'_id': ObjectId(i['_id'])})
        elif i['zone_name'] == "대웅전 전면 벽면-4번-흥국사":
            zone2 = zone2 + 1
            if zone2 > 1:
                collection.delete_one({'_id': ObjectId(i['_id'])})
        elif i['zone_name'] == "대웅전 서쪽 벽면-3번-흥국사":
            zone3 = zone3 + 1
            if zone3 > 1:
                collection.delete_one({'_id': ObjectId(i['_id'])})
        elif i['zone_name'] == "벽화 좌측-2번-흥국사":
            zone4 = zone4 + 1
            if zone4 > 1:
                collection.delete_one({'_id': ObjectId(i['_id'])})
        elif i['zone_name'] == "벽화 우측-1번-흥국사":
            zone6 = zone6 + 1
            if zone6 > 1:
                collection.delete_one({'_id': ObjectId(i['_id'])})
    tmp = tmp + timedelta(minutes=10)
    print(str(tmp)[:16])
