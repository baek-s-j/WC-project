from pymongo import MongoClient

client = MongoClient('127.0.0.1', 27017)  # 여기서 부터는 몽고디비에 데이터 넣는것
db = client.wc_project
collection = db.thdata
z_result = list(collection.find().sort([("time", -1)]).limit(1))
print(type(z_result[0]["time"]))
