import requests
import time
from pymongo import MongoClient

client = MongoClient('mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')   # 여기서 부터는 몽고디비에 데이터 넣는것

db = client.wc_project
collection = db.riskdata
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
